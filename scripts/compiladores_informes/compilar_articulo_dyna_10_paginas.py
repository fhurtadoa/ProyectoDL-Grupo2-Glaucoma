#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilador del Artículo Científico Formato Revista DYNA (~10 páginas):
Evaluación de la generalización de clasificadores profundos de glaucoma
ante heterogeneidad de adquisición de datos en retinografías.

Facultad de Minas — Universidad Nacional de Colombia
"""

import os, sys, re, base64, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MD_PATH = BASE_DIR / "ARTICULO_DYNA_GENERALIZACION_GLAUCOMA.md"
HTML_PATH = BASE_DIR / "ARTICULO_DYNA_GENERALIZACION_GLAUCOMA.html"
PDF_PATH = BASE_DIR / "ARTICULO_DYNA_GENERALIZACION_GLAUCOMA.pdf"
FIGURES_DIR = BASE_DIR / "figures"

def get_browser():
    candidates = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def compile_article():
    if not MD_PATH.exists():
        print(f"[ERROR] No existe {MD_PATH}")
        return False

    print(f"[*] 1. Leyendo manuscrito Markdown: {MD_PATH.name}")
    md_content = MD_PATH.read_text(encoding="utf-8")

    # Separar título/cabecera y cuerpo
    # Salvar bloques matemáticos
    math_blocks = []
    def save_math_disp(m):
        idx = len(math_blocks)
        math_blocks.append(("disp", m.group(1).strip()))
        return f"@@MATH_DISP_{idx}@@"

    def save_math_inline(m):
        idx = len(math_blocks)
        math_blocks.append(("inline", m.group(1).strip()))
        return f"@@MATH_INLINE_{idx}@@"

    text = re.sub(r"\$\$(.*?)\$\$", save_math_disp, md_content, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", save_math_inline, text)

    # Sustituir imágenes por base64
    def replace_img(match):
        alt = match.group(1)
        rel_path = match.group(2)
        img_path = (BASE_DIR / rel_path).resolve()
        if not img_path.exists():
            img_path = (FIGURES_DIR / Path(rel_path).name).resolve()
        
        if img_path.exists():
            b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            ext = img_path.suffix.lower().replace(".", "")
            mime = "image/png" if ext == "png" else "image/jpeg"
            return f'<figure class="dyna-figure"><img src="data:{mime};base64,{b64}" alt="{alt}"/><figcaption><strong>Figura:</strong> {alt}</figcaption></figure>'
        print(f"[WARN] No se encontró imagen: {rel_path}")
        return f'<div class="img-placeholder">[Imagen: {alt}]</div>'

    text = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_img, text)

    # Convertir tablas Markdown a HTML con clases DYNA
    lines = text.split("\n")
    processed_lines = []
    in_table = False
    table_lines = []

    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            in_table = True
            table_lines.append(line.strip())
        else:
            if in_table:
                # Renderizar tabla
                if len(table_lines) >= 2:
                    header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                    rows = []
                    for r in table_lines[2:]:
                        rows.append([c.strip() for c in r.split("|")[1:-1]])
                    
                    tbl_html = '<div class="dyna-table-wrapper"><table class="dyna-table"><thead><tr>'
                    for h in header:
                        tbl_html += f'<th>{h}</th>'
                    tbl_html += '</tr></thead><tbody>'
                    for row in rows:
                        tbl_html += '<tr>'
                        for cell in row:
                            tbl_html += f'<td>{cell}</td>'
                        tbl_html += '</tr>'
                    tbl_html += '</tbody></table></div>'
                    processed_lines.append(tbl_html)
                table_lines = []
                in_table = False
            processed_lines.append(line)

    text = "\n".join(processed_lines)

    # Encabezados
    text = re.sub(r"^# (.*?)$", r'<header class="article-header"><h1 class="article-title">\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r"^\*\*Evaluation of the Generalization.*?\*\*$", r'<p class="article-subtitle">\g<0></p></header>', text, flags=re.MULTILINE)
    text = re.sub(r"^## (.*?)$", r'<h2 class="dyna-h2">\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r"^### (.*?)$", r'<h3 class="dyna-h3">\1</h3>', text, flags=re.MULTILINE)

    # Formatos inline
    text = re.sub(r"\*\*(.*?)\*\*", r'<strong>\1</strong>', text)
    text = re.sub(r"\*(.*?)\*", r'<em>\1</em>', text)
    text = re.sub(r"`([^`\n]+)`", r'<code>\1</code>', text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)

    # Párrafos
    paras = text.split("\n\n")
    html_body = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        if p.startswith("<header") or p.startswith("<h2") or p.startswith("<h3") or p.startswith("<div") or p.startswith("<figure"):
            html_body.append(p)
        elif p.startswith("### Resumen") or p.startswith("### Abstract"):
            html_body.append(f'<div class="abstract-box">{p}</div>')
        elif p.startswith("* ") or p.startswith("- "):
            items = [re.sub(r"^[\*\-]\s*", "", it).strip() for it in p.split("\n") if it.strip()]
            ul = '<ul class="dyna-list">' + "".join([f"<li>{it}</li>" for it in items]) + "</ul>"
            html_body.append(ul)
        elif re.match(r"^\d+\.\s", p):
            items = [re.sub(r"^\d+\.\s*", "", it).strip() for it in p.split("\n") if it.strip()]
            ol = '<ol class="dyna-list">' + "".join([f"<li>{it}</li>" for it in items]) + "</ol>"
            html_body.append(ol)
        elif p.startswith("---"):
            html_body.append('<hr class="dyna-hr"/>')
        else:
            html_body.append(f'<p class="dyna-p">{p}</p>')

    rendered_body = "\n".join(html_body)

    # Restaurar KaTeX
    for idx, (mtype, formula) in enumerate(math_blocks):
        if mtype == "disp":
            rendered_body = rendered_body.replace(f"@@MATH_DISP_{idx}@@", f'<div class="katex-disp">\\[{formula}\\]</div>')
        else:
            rendered_body = rendered_body.replace(f"@@MATH_INLINE_{idx}@@", f'\\({formula}\\)')

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Evaluación de la generalización de clasificadores profundos de glaucoma — Revista DYNA</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '\\\\[', right: '\\\\]', display: true}},
        {{left: '$', right: '$', display: false}},
        {{left: '\\\\(', right: '\\\\)', display: false}}
    ]}});"></script>
<style>
  @page {{
    size: letter portrait;
    margin: 18mm 15mm 18mm 15mm;
    @bottom-center {{
      content: "Revista DYNA — Facultad de Minas, Universidad Nacional de Colombia";
      font-size: 8pt;
      font-family: 'Times New Roman', serif;
      color: #666;
    }}
  }}

  body {{
    font-family: 'Times New Roman', Times, Georgia, serif;
    font-size: 9.5pt;
    line-height: 1.35;
    color: #111;
    background: #fff;
    margin: 0;
    padding: 0;
  }}

  .article-header {{
    text-align: center;
    margin-bottom: 18px;
    border-bottom: 1.5px solid #000;
    padding-bottom: 12px;
  }}

  .article-title {{
    font-size: 15.5pt;
    font-weight: bold;
    line-height: 1.25;
    margin: 0 0 6px 0;
    color: #000;
  }}

  .article-subtitle {{
    font-size: 11pt;
    font-style: italic;
    color: #333;
    margin: 0;
  }}

  .abstract-box {{
    background: #fafafa;
    border: 1px solid #e5e5e5;
    padding: 8px 12px;
    margin-bottom: 14px;
    font-size: 8.8pt;
    text-align: justify;
    line-height: 1.3;
  }}

  .dyna-columns {{
    column-count: 2;
    column-gap: 7mm;
    column-rule: 0.5px solid #e0e0e0;
    text-align: justify;
  }}

  .dyna-h2 {{
    font-size: 11pt;
    font-weight: bold;
    text-transform: uppercase;
    color: #000;
    margin: 14px 0 6px 0;
    border-bottom: 0.8px solid #333;
    padding-bottom: 2px;
    break-after: avoid;
  }}

  .dyna-h3 {{
    font-size: 10pt;
    font-weight: bold;
    font-style: italic;
    margin: 10px 0 4px 0;
    color: #222;
    break-after: avoid;
  }}

  .dyna-p {{
    margin: 0 0 7px 0;
    text-indent: 1.2em;
    text-justify: inter-word;
  }}

  .dyna-list {{
    margin: 4px 0 8px 16px;
    padding: 0;
    font-size: 9.2pt;
  }}

  .dyna-list li {{
    margin-bottom: 3px;
  }}

  .dyna-table-wrapper {{
    margin: 10px 0;
    page-break-inside: avoid;
    break-inside: avoid;
    width: 100%;
  }}

  .dyna-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    line-height: 1.2;
    margin: 4px 0;
  }}

  .dyna-table th {{
    border-top: 1.2px solid #000;
    border-bottom: 1px solid #000;
    padding: 4px 3px;
    font-weight: bold;
    text-align: left;
    background: #f7f7f7;
  }}

  .dyna-table td {{
    border-bottom: 0.5px solid #ddd;
    padding: 3.5px 3px;
  }}

  .dyna-table tr:last-child td {{
    border-bottom: 1.2px solid #000;
  }}

  .dyna-figure {{
    margin: 10px 0;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
    width: 100%;
  }}

  .dyna-figure img {{
    max-width: 100%;
    height: auto;
    border: 0.5px solid #ccc;
  }}

  .dyna-figure figcaption {{
    font-size: 8pt;
    font-style: italic;
    color: #333;
    margin-top: 4px;
  }}

  .katex-disp {{
    margin: 6px 0;
    font-size: 9pt;
  }}

  .dyna-hr {{
    border: none;
    border-top: 0.5px solid #ccc;
    margin: 10px 0;
  }}

  a {{
    color: #003366;
    text-decoration: none;
  }}
</style>
</head>
<body>

{rendered_body.split('<h2 class="dyna-h2">1. Introducción')[0]}

<div class="dyna-columns">
<h2 class="dyna-h2">1. Introducción{rendered_body.split('<h2 class="dyna-h2">1. Introducción')[1]}
</div>

</body>
</html>
"""

    print(f"[*] 2. Guardando HTML autónomo en: {HTML_PATH.name}")
    HTML_PATH.write_text(full_html, encoding="utf-8")

    browser = get_browser()
    if not browser:
        print("[WARN] No se encontró Chromium en el sistema para compilar a PDF.")
        return False

    print(f"[*] 3. Compilando PDF vía Chromium ({Path(browser).name})...")
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH)
    ]
    subprocess.run(cmd, check=True)

    if PDF_PATH.exists():
        size_mb = PDF_PATH.stat().st_size / (1024 * 1024)
        print(f"[EXITO] Artículo compilado: {PDF_PATH.name} ({size_mb:.2f} MB)")
        return True
    return False

if __name__ == "__main__":
    compile_article()
