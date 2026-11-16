import re, base64, subprocess, sys, os
from pathlib import Path

BASE_DIR = Path("/Users/fabian/GlaucomaDL")
MD_PATH = BASE_DIR / "informes" / "03_experimento_estadificacion_multietapa" / "INFORME_EXPERIMENTO_CLASIFICACION_MULTIETAPA.md"
HTML_PATH = BASE_DIR / "informes" / "03_experimento_estadificacion_multietapa" / "INFORME_EXPERIMENTO_CLASIFICACION_MULTIETAPA.html"
PDF_PATH = BASE_DIR / "informes" / "03_experimento_estadificacion_multietapa" / "INFORME_EXPERIMENTO_CLASIFICACION_MULTIETAPA.pdf"

def compile_doc():
    if not MD_PATH.exists():
        print(f"[ERROR] No existe {MD_PATH}")
        return False

    print("[INFO] Leyendo archivo Markdown del experimento multietapa...")
    md_content = MD_PATH.read_text(encoding="utf-8")

    fenced_blocks = []
    def save_fenced(m):
        idx = len(fenced_blocks)
        fenced_blocks.append(m.group(1))
        return f"\n@@FENCED_BLOCK_{idx}@@\n"

    text = re.sub(r"```(.*?)```", save_fenced, md_content, flags=re.DOTALL)

    math_blocks = []
    def save_math_display(m):
        idx = len(math_blocks)
        math_blocks.append(("display", m.group(1).strip()))
        return f"@@MATH_BLOCK_{idx}@@"

    def save_math_inline(m):
        idx = len(math_blocks)
        math_blocks.append(("inline", m.group(1).strip()))
        return f"@@MATH_BLOCK_{idx}@@"

    text = re.sub(r"\$\$(.*?)\$\$", save_math_display, text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", save_math_inline, text)

    inline_codes = []
    def save_code(m):
        idx = len(inline_codes)
        inline_codes.append(m.group(1))
        return f"@@INLINE_CODE_{idx}@@"

    text = re.sub(r"`([^`\n]+)`", save_code, text)

    def replace_image(match):
        alt = match.group(1)
        rel_path = match.group(2)
        img_file = (BASE_DIR / rel_path).resolve()
        if img_file.exists():
            b64 = base64.b64encode(img_file.read_bytes()).decode("ascii")
            return f'<figure class="report-figure"><img src="data:image/png;base64,{b64}" alt="{alt}"/><figcaption>{alt}</figcaption></figure>'
        print(f"[WARN] No se encontró imagen: {img_file}")
        return match.group(0)

    text = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_image, text)

    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)

    for i, c in enumerate(inline_codes):
        text = text.replace(f"@@INLINE_CODE_{i}@@", f"<code>{c}</code>")

    lines = text.splitlines()
    html_lines = []
    i = 0
    in_ul = False
    in_ol = False
    in_blockquote = False

    def close_lists():
        nonlocal in_ul, in_ol, in_blockquote
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False
        if in_blockquote:
            html_lines.append("</blockquote>")
            in_blockquote = False

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            close_lists()
            i += 1
            continue

        if stripped == "---":
            close_lists()
            html_lines.append("<hr/>")
            i += 1
            continue

        m_fenced = re.match(r"^@@FENCED_BLOCK_(\d+)@@$", stripped)
        if m_fenced:
            close_lists()
            raw_block = fenced_blocks[int(m_fenced.group(1))]
            block_lines = raw_block.splitlines()
            lang = block_lines[0].strip() if block_lines else ""
            code_text = "\n".join(block_lines[1:]) if len(block_lines) > 1 else raw_block
            code_text = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(f'<pre><code class="language-{lang}">{code_text}</code></pre>')
            i += 1
            continue

        if stripped.startswith("#"):
            close_lists()
            level = len(stripped.split()[0])
            header_text = stripped.lstrip("#").strip()
            html_lines.append(f"<h{level}>{header_text}</h{level}>")
            i += 1
            continue

        if stripped.startswith(">"):
            if not in_blockquote:
                close_lists()
                html_lines.append("<blockquote>")
                in_blockquote = True
            bq_text = stripped.lstrip(">").strip()
            html_lines.append(f"<p>{bq_text}</p>")
            i += 1
            continue
        elif in_blockquote:
            html_lines.append("</blockquote>")
            in_blockquote = False

        if stripped.startswith(("- ", "* ")):
            if not in_ul:
                close_lists()
                html_lines.append("<ul>")
                in_ul = True
            item_text = stripped[2:].strip()
            html_lines.append(f"<li>{item_text}</li>")
            i += 1
            continue

        m_ol = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m_ol:
            if not in_ol:
                close_lists()
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{m_ol.group(2).strip()}</li>")
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            close_lists()
            table_lines = [stripped]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_lines.append(lines[i].strip())
                i += 1

            if len(table_lines) >= 2:
                headers = [c.strip() for c in table_lines[0].strip("|").split("|")]
                rows = []
                for tl in table_lines[2:]:
                    row = [c.strip() for c in tl.strip("|").split("|")]
                    rows.append(row)

                tbl_html = ['<div class="table-container"><table><thead><tr>']
                for h in headers:
                    tbl_html.append(f"<th>{h}</th>")
                tbl_html.append("</tr></thead><tbody>")
                for r in rows:
                    tbl_html.append("<tr>")
                    for cell in r:
                        tbl_html.append(f"<td>{cell}</td>")
                    tbl_html.append("</tr>")
                tbl_html.append("</tbody></table></div>")
                html_lines.append("".join(tbl_html))
            continue

        close_lists()
        html_lines.append(f"<p>{stripped}</p>")
        i += 1

    close_lists()
    body_html = "\n".join(html_lines)

    for i, (mtype, formula) in enumerate(math_blocks):
        escaped_formula = formula.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        if mtype == "display":
            replacement = f'<div class="katex-display">\\[{escaped_formula}\\]</div>'
        else:
            replacement = f'<span class="katex-inline">\\({escaped_formula}\\)</span>'
        body_html = body_html.replace(f"@@MATH_BLOCK_{i}@@", replacement)

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Estadificación Multietapa y Progresión Clínica del Glaucoma</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{
        delimiters: [
            {{left: '$$', right: '$$', display: true}},
            {{left: '\\\\[', right: '\\\\]', display: true}},
            {{left: '$', right: '$', display: false}},
            {{left: '\\\\(', right: '\\\\)', display: false}}
        ]
    }});"></script>
<style>
    @page {{
        size: A4;
        margin: 20mm 18mm 20mm 18mm;
        @bottom-right {{
            content: "Página " counter(page) " de " counter(pages);
            font-size: 8pt;
            color: #6c757d;
        }}
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 10pt;
        line-height: 1.55;
        color: #1a1a1a;
        background-color: #ffffff;
        margin: 0;
        padding: 0;
    }}
    .document-header {{
        border-bottom: 3px double #0d6efd;
        padding-bottom: 12px;
        margin-bottom: 24px;
    }}
    .doc-meta {{
        font-size: 8.5pt;
        color: #495057;
        margin-top: 4px;
    }}
    h1 {{
        color: #0b3d91;
        font-size: 18pt;
        font-weight: 700;
        margin: 0 0 8px 0;
        letter-spacing: -0.3px;
    }}
    h2 {{
        color: #1e3a8a;
        font-size: 13pt;
        font-weight: 600;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 5px;
        margin-top: 22px;
        margin-bottom: 10px;
        page-break-after: avoid;
    }}
    h3 {{
        color: #1e40af;
        font-size: 11pt;
        font-weight: 600;
        margin-top: 14px;
        margin-bottom: 6px;
        page-break-after: avoid;
    }}
    p {{
        margin: 0 0 10px 0;
        text-align: justify;
    }}
    blockquote {{
        border-left: 4px solid #2563eb;
        background-color: #eff6ff;
        padding: 10px 14px;
        margin: 12px 0;
        font-style: italic;
        color: #1e3a8a;
        border-radius: 0 6px 6px 0;
    }}
    ul, ol {{
        margin: 0 0 10px 0;
        padding-left: 20px;
    }}
    li {{
        margin-bottom: 4px;
    }}
    .table-container {{
        width: 100%;
        overflow-x: auto;
        margin: 14px 0;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 8.5pt;
        page-break-inside: avoid;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 6px 9px;
        text-align: left;
    }}
    th {{
        background-color: #f1f5f9;
        color: #0f172a;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #f8fafc;
    }}
    figure.report-figure {{
        margin: 16px 0;
        text-align: center;
        page-break-inside: avoid;
    }}
    figure.report-figure img {{
        max-width: 95%;
        height: auto;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
    }}
    figcaption {{
        font-size: 8.5pt;
        color: #64748b;
        margin-top: 6px;
        font-style: italic;
    }}
    code {{
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 2px 5px;
        border-radius: 4px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 8.5pt;
    }}
    pre {{
        background-color: #0f172a;
        color: #f8fafc;
        padding: 12px;
        border-radius: 6px;
        overflow-x: auto;
        font-size: 8pt;
        line-height: 1.4;
        margin: 12px 0;
    }}
    pre code {{
        background-color: transparent;
        color: inherit;
        padding: 0;
    }}
    .badge {{
        display: inline-block;
        padding: 2px 7px;
        font-size: 7.5pt;
        font-weight: 600;
        border-radius: 4px;
        background-color: #e0e7ff;
        color: #3730a3;
    }}
</style>
</head>
<body>
<div class="document-header">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="badge">AUDITORÍA CLÍNICA DE APRENDIZAJE PROFUNDO</span>
        <span class="doc-meta">Fase 7: Estadificación Multietapa y Progresión Patológica</span>
    </div>
</div>
{body_html}
</body>
</html>"""

    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"[OK] HTML generado exitosamente en: {HTML_PATH}")

    # Compilar a PDF con Chrome headless
    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    if not os.path.exists(chrome_path):
        chrome_path = "google-chrome"

    print("[INFO] Compilando reporte a PDF con headless Chrome...")
    cmd = [
        chrome_path,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0 and PDF_PATH.exists():
        size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
        print(f"[EXITO] PDF compilado con éxito: {PDF_PATH} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"[ERROR] Falló compilación a PDF: {res.stderr}")
        return False

if __name__ == "__main__":
    compile_doc()

