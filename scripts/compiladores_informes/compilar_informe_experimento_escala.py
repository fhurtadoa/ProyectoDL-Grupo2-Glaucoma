import re, base64, subprocess, sys, os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
MD_PATH = BASE_DIR / "scripts" / "compiladores_informes" / "fuentes" / "INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.md"
HTML_PATH = BASE_DIR / "informes" / "01_experimento_invarianza_escala" / "INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.html"
PDF_PATH = BASE_DIR / "informes" / "01_experimento_invarianza_escala" / "INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.pdf"

def compile_doc():
    if not MD_PATH.exists():
        print(f"[ERROR] No existe {MD_PATH}")
        return False

    print("[INFO] Leyendo archivo Markdown del experimento...")
    md_content = MD_PATH.read_text(encoding="utf-8")

    # 1. Protect fenced code blocks ```...```
    fenced_blocks = []
    def save_fenced(m):
        idx = len(fenced_blocks)
        fenced_blocks.append(m.group(1))
        return f"\n@@FENCED_BLOCK_{idx}@@\n"

    text = re.sub(r"```(.*?)```", save_fenced, md_content, flags=re.DOTALL)

    # 2. Protect display math $$...$$ and inline math $...$
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

    # 3. Protect inline code `...`
    inline_codes = []
    def save_code(m):
        idx = len(inline_codes)
        inline_codes.append(m.group(1))
        return f"@@INLINE_CODE_{idx}@@"

    text = re.sub(r"`([^`\n]+)`", save_code, text)

    # 4. Embed images as base64 data URIs
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

    # 5. Convert markdown links
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)

    # 6. Bold and italic
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)

    # 7. Restore inline code
    for i, c in enumerate(inline_codes):
        text = text.replace(f"@@INLINE_CODE_{i}@@", f"<code>{c}</code>")

    # 8. Parse lines into structured HTML
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
            f_idx = int(m_fenced.group(1))
            code_content = fenced_blocks[f_idx]
            code_content_escaped = code_content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(f"<pre class='code-block'><code>{code_content_escaped}</code></pre>")
            i += 1
            continue

        if stripped.startswith("# "):
            close_lists()
            html_lines.append(f"<h1>{stripped[2:].strip()}</h1>")
            i += 1
            continue
        if stripped.startswith("## "):
            close_lists()
            html_lines.append(f"<h2>{stripped[3:].strip()}</h2>")
            i += 1
            continue
        if stripped.startswith("### "):
            close_lists()
            html_lines.append(f"<h3>{stripped[4:].strip()}</h3>")
            i += 1
            continue
        if stripped.startswith("#### "):
            close_lists()
            html_lines.append(f"<h4>{stripped[5:].strip()}</h4>")
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            close_lists()
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith("|") and lines[i].strip().endswith("|"):
                table_rows.append(lines[i].strip())
                i += 1
            
            if len(table_rows) >= 2:
                html_table = ["<div class='table-wrapper'><table>"]
                header_cells = [c.strip() for c in table_rows[0].split("|")[1:-1]]
                html_table.append("<thead><tr>" + "".join(f"<th>{c}</th>" for c in header_cells) + "</tr></thead>")
                html_table.append("<tbody>")
                for r in table_rows[2:]:
                    cells = [c.strip() for c in r.split("|")[1:-1]]
                    html_table.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
                html_table.append("</tbody></table></div>")
                html_lines.append("\n".join(html_table))
            continue

        if stripped.startswith(">"):
            if in_ul or in_ol:
                close_lists()
            if not in_blockquote:
                html_lines.append("<blockquote>")
                in_blockquote = True
            bq_text = re.sub(r"^>\s?", "", stripped)
            html_lines.append(f"<p>{bq_text}</p>")
            i += 1
            continue

        m_ol = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m_ol:
            if in_ul or in_blockquote:
                close_lists()
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            li_content = m_ol.group(2)
            sub_p = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("   ") and nxt.strip() and not re.match(r"^\s*[-*]\s+", nxt) and not re.match(r"^\s*\d+\.\s+", nxt):
                    sub_p.append(nxt.strip())
                    i += 1
                else:
                    break
            if sub_p:
                html_lines.append(f"<li>{li_content}<br/><span class='list-sub'>{' '.join(sub_p)}</span></li>")
            else:
                html_lines.append(f"<li>{li_content}</li>")
            continue

        m_ul = re.match(r"^[-*]\s+(.*)$", stripped)
        if m_ul:
            if in_ol or in_blockquote:
                close_lists()
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            li_content = m_ul.group(1)
            sub_p = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("  ") and nxt.strip() and not re.match(r"^\s*[-*]\s+", nxt) and not re.match(r"^\s*\d+\.\s+", nxt):
                    sub_p.append(nxt.strip())
                    i += 1
                else:
                    break
            if sub_p:
                html_lines.append(f"<li>{li_content}<br/><span class='list-sub'>{' '.join(sub_p)}</span></li>")
            else:
                html_lines.append(f"<li>{li_content}</li>")
            continue

        if stripped.startswith("<figure"):
            close_lists()
            html_lines.append(stripped)
            i += 1
            continue

        close_lists()
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if not nxt or nxt.startswith("#") or nxt.startswith("|") or nxt.startswith(">") or nxt.startswith("- ") or nxt.startswith("* ") or re.match(r"^\d+\.\s+", nxt) or nxt == "---" or nxt.startswith("@@FENCED_BLOCK_") or nxt.startswith("<figure"):
                break
            para_lines.append(nxt)
            i += 1
        html_lines.append(f"<p>{' '.join(para_lines)}</p>")

    close_lists()
    body_html = "\n".join(html_lines)

    for idx, (kind, expr) in enumerate(math_blocks):
        if kind == "display":
            body_html = body_html.replace(f"@@MATH_BLOCK_{idx}@@", f"\\[{expr}\\]")
        else:
            body_html = body_html.replace(f"@@MATH_BLOCK_{idx}@@", f"\\({expr}\\)")

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Informe Experimental: Data Augmentation Maligno de Escala y FOV</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {{
    renderMathInElement(document.body, {{
      delimiters: [
        {{left: '\\\\[', right: '\\\\]', display: true}},
        {{left: '\\\\(', right: '\\\\)', display: false}}
      ],
      throwOnError: false,
      ignoredTags: ["script", "noscript", "style", "textarea", "pre"]
    }});
  }});
</script>
<style>
  @page {{
    size: A4 portrait;
    margin: 15mm 14mm 15mm 14mm;
    @bottom-center {{
      content: counter(page);
      font-size: 8.5pt;
      color: #64748b;
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.55;
    font-size: 9.6pt;
  }}
  h1 {{
    font-size: 18pt;
    color: #0f172a;
    border-bottom: 2.5px solid #0284c7;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 10px;
    font-weight: 800;
    page-break-after: avoid;
    line-height: 1.25;
  }}
  h2 {{
    font-size: 13pt;
    color: #0369a1;
    border-bottom: 1.2px solid #cbd5e1;
    padding-bottom: 4px;
    margin-top: 18px;
    margin-bottom: 8px;
    font-weight: 700;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 10.8pt;
    color: #334155;
    margin-top: 14px;
    margin-bottom: 6px;
    font-weight: 600;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
  }}
  ul, ol {{
    margin-top: 0;
    margin-bottom: 8px;
    padding-left: 20px;
  }}
  li {{
    margin-bottom: 3.5px;
  }}
  .list-sub {{
    font-size: 9pt;
    color: #475569;
  }}
  blockquote {{
    border-left: 3.5px solid #0284c7;
    background: #f0f9ff;
    margin: 8px 0;
    padding: 6px 12px;
    color: #0369a1;
    font-style: italic;
    page-break-inside: avoid;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 14px 0;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8.4pt;
    background-color: #f1f5f9;
    color: #b91c1c;
    padding: 1.5px 3.5px;
    border-radius: 3px;
  }}
  pre.code-block {{
    background-color: #0f172a;
    color: #f8fafc;
    padding: 9px 12px;
    border-radius: 4.5px;
    overflow-x: auto;
    font-size: 8pt;
    margin: 8px 0;
    page-break-inside: avoid;
  }}
  pre.code-block code {{
    background-color: transparent;
    color: inherit;
    padding: 0;
  }}
  .table-wrapper {{
    margin: 10px 0;
    page-break-inside: avoid;
    width: 100%;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.4pt;
    page-break-inside: avoid;
  }}
  th {{
    background-color: #0369a1;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 5.5px 7px;
    border: 1px solid #0369a1;
  }}
  td {{
    padding: 5px 6.5px;
    border: 1px solid #cbd5e1;
    vertical-align: middle;
  }}
  tr:nth-child(even) td {{
    background-color: #f8fafc;
  }}
  .report-figure {{
    margin: 11px auto;
    text-align: center;
    page-break-inside: avoid;
  }}
  .report-figure img {{
    max-width: 98%;
    max-height: 380px;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .report-figure figcaption {{
    font-size: 8.4pt;
    color: #475569;
    font-style: italic;
    margin-top: 4px;
  }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    HTML_PATH.write_text(full_html, encoding="utf-8")
    print(f"[OK] Archivo HTML escrito en: {HTML_PATH}")

    chrome_bin = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    cmd = [
        chrome_bin,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--virtual-time-budget=10000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF_PATH}",
        f"file://{HTML_PATH}",
    ]
    print("[INFO] Ejecutando Google Chrome headless para generar PDF...")
    subprocess.run(cmd, check=True)
    print(f"[EXITO] PDF generado en: {PDF_PATH} ({PDF_PATH.stat().st_size / (1024*1024):.2f} MB)")
    return True

if __name__ == "__main__":
    compile_doc()
