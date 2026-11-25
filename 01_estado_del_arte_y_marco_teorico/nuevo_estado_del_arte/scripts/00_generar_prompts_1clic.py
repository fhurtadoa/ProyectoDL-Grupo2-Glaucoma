"""
Genera los 6 prompts (R1-R6) + R7 opcional 100% autocontenidos (listos para copiar en 1 clic)
a partir de 01_prompts_gemini_deep_research.md, sin marcadores pendientes.
Crea además la estructura de recepción en nuevo_estado_del_arte/deep_research/.
"""
from pathlib import Path
import re

BASE_DIR = Path(__file__).resolve().parent.parent
SRC_MD = BASE_DIR / "01_prompts_gemini_deep_research.md"
OUT_MD = BASE_DIR / "01_prompts_listos_1clic.md"
DR_DIR = BASE_DIR / "deep_research"
PROMPTS_DIR = DR_DIR / "prompts_1clic"

DR_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

text = SRC_MD.read_text(encoding="utf-8")

# Extraer bloque común
common_match = re.search(
    r"## Bloque de contexto común.*?```text\n(.*?)\n```",
    text,
    flags=re.DOTALL,
)
assert common_match, "No se encontró el bloque de contexto común"
common_block = common_match.group(1).strip()

# Reforzar formato exacto de DOI para cruce automático con Crossref
common_block += (
    "\n- Incluye siempre el DOI exacto en formato canónico (10.xxxx/...) en la columna correspondiente; "
    "si el trabajo no tiene DOI o no pudiste comprobarlo, escribe \"sin DOI verificado\"."
)

# Extraer cada sección R1..R7
pattern = re.compile(
    r"## (R\d[^#\n]*)\n+```text\n(.*?)\n```",
    flags=re.DOTALL,
)

slugs = {
    "R1": "R1_validacion_externa",
    "R2": "R2_catalogo_bases",
    "R3": "R3_variacion_imagen",
    "R4": "R4_variacion_etiqueta_estadios",
    "R5": "R5_umbral_calibracion_prevalencia",
    "R6": "R6_estrategias_mitigacion",
    "R7": "R7_opcional_epidemiologia",
}

assembled_sections = []
for header, body in pattern.findall(text):
    code = header.split(".")[0].split(" ")[0].strip()  # R1..R7
    full_prompt = body.replace("[Pega aquí el BLOQUE DE CONTEXTO COMÚN]", common_block).strip()
    slug = slugs.get(code, code)
    txt_path = PROMPTS_DIR / f"{slug}.txt"
    txt_path.write_text(full_prompt + "\n", encoding="utf-8")
    assembled_sections.append((code, header.strip(), slug, full_prompt))

md_lines = [
    "# Prompts listos en 1 clic para Gemini Deep Research (R1–R6 + R7 opcional)",
    "",
    "Cada bloque ya incluye el **Bloque de contexto común** integrado. No hay que reemplazar ni pegar nada adicional.",
    "",
    "## Flujo rápido (6 pestañas en paralelo)",
    "1. Abre 6 pestañas de Gemini con **Deep Research** activado.",
    "2. Copia `R1` a `R6` (con el botón de copiar de cada bloque o abriendo los `.txt` en `deep_research/prompts_1clic/` con `Ctrl+A` → `Ctrl+C`).",
    "3. Al revisar el plan que propone Gemini en cada pestaña, verifica que priorice artículos revisados por pares (PubMed, PMC, Nature, IEEE, Elsevier, Springer) y aprueba.",
    "4. Cuando terminen, guarda o exporta los informes en `01_estado_del_arte_y_marco_teorico/nuevo_estado_del_arte/deep_research/` con nombres `R1.md` (o `.docx` / `.pdf` / `.txt`) a `R6.md`.",
    "",
    "---",
    "",
]

for code, header, slug, full_prompt in assembled_sections:
    md_lines.append(f"## {header}")
    md_lines.append(f"Archivo directo: `deep_research/prompts_1clic/{slug}.txt`")
    md_lines.append("")
    md_lines.append("```text")
    md_lines.append(full_prompt)
    md_lines.append("```")
    md_lines.append("")

OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")

readme_dr = DR_DIR / "README.md"
readme_dr.write_text(
    "# Recepción de informes de Gemini Deep Research\n\n"
    "Guarda aquí los informes exportados de las 6 pestañas paralelas (aceptamos `.md`, `.docx`, `.pdf` o `.txt`):\n\n"
    "- `R1_validacion_externa.(md|docx|pdf|txt)`\n"
    "- `R2_catalogo_bases.(md|docx|pdf|txt)`\n"
    "- `R3_variacion_imagen.(md|docx|pdf|txt)`\n"
    "- `R4_variacion_etiqueta_estadios.(md|docx|pdf|txt)`\n"
    "- `R5_umbral_calibracion_prevalencia.(md|docx|pdf|txt)`\n"
    "- `R6_estrategias_mitigacion.(md|docx|pdf|txt)`\n"
    "- *(Opcional)* `R7_epidemiologia.(md|docx|pdf|txt)`\n\n"
    "Los prompts autocontenidos listos para `Ctrl+A` -> `Ctrl+C` están en `prompts_1clic/`.\n",
    encoding="utf-8",
)
print(f"Generados {len(assembled_sections)} prompts autocontenidos en {PROMPTS_DIR} y {OUT_MD}")
