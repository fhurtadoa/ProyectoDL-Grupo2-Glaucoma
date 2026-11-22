#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilador del Estado del Arte Oficial (Versión Reescrita con Figuras Embebidas):
Evaluación de la generalización de clasificadores profundos de glaucoma
ante heterogeneidad de adquisición de datos en retinografías.

Facultad de Minas — Universidad Nacional de Colombia
"""

import os, sys, re, base64, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ESTADO_DIR = BASE_DIR / "estado_del_arte"
FIGURAS_DIR = BASE_DIR / "Mauricio" / "figuras_estado_del_arte"
if not FIGURAS_DIR.exists():
    FIGURAS_DIR = ESTADO_DIR / "figuras"

MD_OUTPUT_PATH = ESTADO_DIR / "estado_del_arte.md"
HTML_OUTPUT_PATH = ESTADO_DIR / "estado_del_arte.html"
PDF_OUTPUT_PATH = ESTADO_DIR / "estado_del_arte.pdf"

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def get_b64(img_name):
    p = FIGURAS_DIR / img_name
    if not p.exists():
        p = BASE_DIR / "estado_del_arte" / "figuras" / img_name
    if p.exists():
        data = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"
    print(f"[WARN] No se encontró imagen: {img_name}")
    return ""

def generate_markdown():
    """Genera el markdown consolidado de estado_del_arte basado en estado_del_arte_reescrito.md"""
    md_content = """# Estado del arte: de la clasificación de retinografías a la validez entre fuentes

**Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías**  
*Deep Learning (2026-2) — Facultad de Minas, Universidad Nacional de Colombia — Sede Medellín*

---

La clasificación automática de glaucoma en retinografías ha pasado de medidas diseñadas a mano, como la relación copa-disco, a representaciones aprendidas por redes profundas. Pero una pregunta más exigente que la precisión en una base es qué parte de lo aprendido corresponde a signos de enfermedad conservados entre poblaciones y qué parte depende de cómo se obtuvieron y etiquetaron las fotografías. La literatura demuestra discriminación bajo condiciones definidas; ofrece evidencia más desigual sobre su transporte a otras fuentes.

## De la prueba de concepto a la prueba entre bases

El aprendizaje por transferencia permitió entrenar clasificadores con conjuntos relativamente pequeños al ajustar redes preentrenadas en imágenes generales. Gómez-Valverde et al. compararon configuraciones convolucionales, tamaños de muestra y estrategias de transferencia en tres conjuntos; su mejor configuración utilizó VGG19 ajustada y la combinación de datos mejoró el resultado dentro del diseño estudiado [1]. Diaz-Pinto et al. extendieron esta línea al comparar cinco redes preentrenadas y, de forma especialmente pertinente, probarlas tanto mediante validación cruzada como en bases completas que no se usaron para entrenar [2]. Leídos juntos, estos trabajos sostienen que el preentrenamiento es una opción práctica y que aumentar la diversidad de datos puede ser útil. No permiten concluir que VGG19, Xception u otra arquitectura sea intrínsecamente superior para cualquier cámara o población: cambiaron el conjunto, la composición de casos y el procedimiento experimental. En particular, Gómez-Valverde et al. incluyeron tanto sospechosos como casos diagnosticados en la clase positiva [1]; esa definición afecta qué aprende la red y cómo debe interpretarse su rendimiento.

La diferencia entre validar dentro de una base y hacerlo fuera es sustantiva. Una partición aleatoria puede medir discriminación entre pacientes no vistos, pero conserva buena parte del proceso de adquisición y anotación. La prueba entre bases de Diaz-Pinto et al. enfrenta otras imágenes y decisiones de curación [2]. REFUGE estableció un conjunto y reglas comunes para comparar métodos de clasificación y segmentación del disco y la copa [3]. La comparación dentro de un desafío reduce diferencias de protocolo entre equipos; no reproduce por sí misma el paso a un hospital independiente. Una segmentación puede servir como insumo diagnóstico, pero su evaluación geométrica no valida la decisión clínica final.

![Fig. 1. Diagrama de flujo PRISMA 2020: Identificación, cribado y selección de estudios primarios sobre generalización de clasificadores de glaucoma en retinografías.](fig00_prisma_2020_flujo_escala_grises.png)

![Fig. 2. Frecuencia de protocolos de evaluación diagnóstica en los estudios primarios (partición interna aleatoria vs. evaluación entre bases independientes).](fig03_protocolos_evaluacion_escala_grises.png)

![Fig. 3. Frecuencia de uso de bases de datos públicas de glaucoma en la literatura analizada.](fig02_bases_datos_frecuencia_escala_grises.png)

## Qué cambia cuando cambia la fuente

Una retinografía de otro centro puede variar en encuadre, escala de la papila, iluminación, color, resolución, artefactos y calidad. También pueden cambiar la mezcla de edades, comorbilidades, gravedad y prevalencia. Por último, dos bases que usan la palabra «glaucoma» pueden haber aplicado criterios distintos para construir la etiqueta. Li et al. entrenaron un sistema para detectar neuropatía óptica glaucomatosa remitible a partir de rasgos evaluados en fotografías; entre sus errores identificaron ojos con miopía y otras afecciones [4]. Esa observación muestra que la dificultad no se limita a la cámara: hay anatomías y enfermedades que modifican la interpretación de los signos papilares. PAPILA, al aportar imágenes de ambos ojos y datos clínicos, hace visible la necesidad de separar pacientes al evaluar y de examinar el origen del diagnóstico [5]. Si ojos de una misma persona aparecen en entrenamiento y prueba, o si los criterios de caso cambian entre fuentes, una cifra aislada de ROC-AUC adquiere otro significado.

Una caída al pasar de una base a otra demuestra que el desempeño no se transportó bajo ese conjunto de cambios; no demuestra que la causara el campo visual, un fabricante o el balance de color. Atribuir el efecto a una variable requiere comparaciones controladas. AIROGS acercó la evaluación al cribado al incluir imágenes no gradables y entradas inesperadas de numerosos centros, además de pruebas en otras bases [6]. Su aporte es recordar que un sistema debe reconocer cuándo la imagen no permite una decisión fiable, algo que no mide una prueba binaria con fotografías seleccionadas por su calidad.

![Fig. 4. Magnitud del colapso diagnóstico inter-dominio (caída Delta-AUC) reportado en estudios primarios al evaluar en cohortes externas no vistas.](fig05_caida_delta_auc_por_estudio_escala_grises.png)

## Estrategias de generalización y alcance de su evidencia

Las propuestas para reducir la dependencia de la fuente operan sobre distintos niveles. Reunir imágenes de varios orígenes expone al modelo a mayor diversidad, pero solo constituye evidencia de generalización si alguna fuente permanece íntegramente ajena al ajuste y a la elección del umbral. El aumento geométrico o fotométrico intenta desacoplar la predicción de escala, posición o apariencia; su plausibilidad no basta para probar beneficio externo, y una transformación excesiva puede alterar información clínica. La armonización procura presentar al modelo entradas más uniformes. Hemelings et al. aplicaron un procesamiento común centrado en el disco y evaluaron un modelo de regresión de riesgo en dos cohortes poblacionales y once bases públicas [7]. El estudio muestra que una evaluación externa extensa es posible y que un modelo entrenado en una fuente puede conservar discriminación en otras. También impide una lectura simple: el diseño combina una salida continua, una forma concreta de preprocesamiento y etiquetas heterogéneas. Su resultado no permite adjudicar toda la generalización a la armonización, ni demuestra que el entrenamiento multifuente sea siempre necesario o siempre ventajoso.

La adaptación de dominio introduce otra distinción: puede aprovechar imágenes del destino durante el desarrollo, mientras que la generalización estricta exige un destino no visto. Ambas estrategias pueden ser pertinentes, pero responden a preguntas distintas y deben informarse por separado. El preentrenamiento retiniano de RETFound sugiere una fuente adicional de representaciones para tareas oftalmológicas [8]; aún debe comprobarse, bajo un mismo protocolo de clasificación de glaucoma, cuánto añade frente a alternativas convencionales en bases externas. El interés de estas líneas no reside en proponer una técnica ganadora, sino en precisar qué clase de variación aborda cada una y con qué prueba se ha evaluado.

![Fig. 5. Frecuencia de estrategias de mitigación frente a la heterogeneidad de adquisición y cambio de dominio.](fig04_estrategias_mitigacion_escala_grises.png)

![Fig. 6. Distribución temporal de estudios primarios incluidos por año en la revisión sistemática.](fig01_publicaciones_por_ano_escala_grises.png)

## Etiquetas, grados y decisión clínica

La clasificación binaria suele reunir objetivos que no son equivalentes: detectar neuropatía visible, identificar un sospechoso, recomendar remisión o reconocer glaucoma confirmado. Ahn et al. estudiaron casos tempranos y avanzados por separado [9], lo que apunta a una limitación de las evaluaciones que agregan todas las severidades. La clasificación ordinal exige un criterio explícito de estadio y datos clínicos adecuados para adjudicarlo; dividir artificialmente una puntuación o reutilizar etiquetas de procedencia incierta no resuelve ese requisito. Además de la exactitud global, interesa saber qué grados se confunden y si la dificultad se concentra en los casos tempranos. La validez de esas conclusiones depende tanto de la definición clínica como de la red.

Finalmente, ROC-AUC resume la capacidad de ordenar casos a través de umbrales, pero una decisión de cribado se toma en un punto operativo. Por ello, la sensibilidad y la especificidad deben informarse con el umbral, sus denominadores e incertidumbre. Hemelings et al. observaron que los umbrales apropiados variaban entre fuentes y definiciones de caso [7]. Si se elige un umbral con las etiquetas de la base externa, esa base ya participó en el desarrollo; conviene distinguir tal reajuste de la aplicación de un umbral fijado previamente. Así, el vacío que articula esta literatura no es la falta de otro clasificador de alta AUC interna, sino la necesidad de pruebas que separen variación de imagen, variación de etiqueta y cambio del punto de decisión. Solo esa separación permite valorar qué estrategia mejora realmente el transporte del sistema.

## Estudios clave de referencia

| Estudio | Tarea y datos | Evaluación | Aporte y límite |
|---|---|---|---|
| Li et al. [4] | Neuropatía remitible; retinografías clínicas | Validación externa en cohorte multicéntrica | Demuestra alta discriminación pero revela errores críticos en ojos con miopía y opacidades. |
| Gómez-Valverde et al. [1] | Clasificación binaria; retinografías | Comparación de backbones convolucionales y transferencia | Evalúa VGG19 y transferencia desde ImageNet; no aísla factores de adquisición entre cámaras. |
| Diaz-Pinto et al. [2] | Clasificación binaria; cinco bases públicas | Validación cruzada y pruebas cruzadas inter-base | Primer benchmark sistemático inter-dominio que expone la brecha de generalización externa. |
| Orlando et al. [3] | Clasificación y segmentación papilar (REFUGE) | Competencia internacional con protocolo unificado | Estandariza la evaluación comparativa, aunque dentro de las condiciones acotadas del reto. |
| de Vente et al. [6] | Cribado e imágenes no gradables (AIROGS) | Evaluación en múltiples centros clínicos no vistos | Introduce el manejo de calidad óptica y rechazo de retinografías no interpretables. |
| Hemelings et al. [7] | Regresión ordinal de riesgo; 13 fuentes externas | Validación externa masiva con armonización | Demuestra la viabilidad de generalización mediante preprocesamiento y alineación papilar. |
| Kovalyk et al. [5] | Cohorte binocular y metadatos clínicos (PAPILA) | Auditoría de procedencia y partición por paciente | Resalta el riesgo de fuga de datos entre ojos y la necesidad de separar pacientes. |
| Zhou et al. [8] | Modelo fundacional retiniano (RETFound) | Preentrenamiento autosupervisado multimodal | Abre la ruta de representaciones específicas para retina; pendiente evaluar en clasificación fina. |
| Ahn et al. [9] | Detección estratificada temprana vs. avanzada | Evaluación por estadio de gravedad | Demuestra el crítico subdiagnóstico del estadio temprano frente al avanzado. |

## Referencias verificadas

[1] Gómez-Valverde, J. J. et al. «Automatic glaucoma classification using color fundus images based on convolutional neural networks and transfer learning». *Biomedical Optics Express* 10(2), 892–913 (2019). https://doi.org/10.1364/BOE.10.000892

[2] Diaz-Pinto, A., Morales, S., Naranjo, V., Köhler, T., Mossi, J. M. y Navea, A. «CNNs for automatic glaucoma assessment using fundus images: an extensive validation». *BioMedical Engineering OnLine* 18, 29 (2019). https://doi.org/10.1186/s12938-019-0649-y

[3] Orlando, J. I. et al. «REFUGE Challenge: A unified framework for evaluating automated methods for glaucoma assessment from fundus photographs». *Medical Image Analysis* 59, 101570 (2020). https://doi.org/10.1016/j.media.2019.101570

[4] Li, Z., He, Y., Keel, S., Meng, W., Chang, R. T. y He, M. «Efficacy of a Deep Learning System for Detecting Glaucomatous Optic Neuropathy Based on Color Fundus Photographs». *Ophthalmology* 125(8), 1199–1206 (2018). https://doi.org/10.1016/j.ophtha.2018.01.023

[5] Kovalyk, O., Morales-Sánchez, J., Verdú-Monedero, R., Sellés-Navarro, I., Palazón-Cabanes, A. y Sancho-Gómez, J.-L. «PAPILA: Dataset with fundus images and clinical data of both eyes of the same patient for glaucoma assessment». *Scientific Data* 9, 291 (2022). https://doi.org/10.1038/s41597-022-01388-1

[6] de Vente, C. et al. «AIROGS: Artificial Intelligence for Robust Glaucoma Screening Challenge». *IEEE Transactions on Medical Imaging* 43(1), 542–557 (2024). https://doi.org/10.1109/TMI.2023.3313786

[7] Hemelings, R. et al. «A generalizable deep learning regression model for automated glaucoma screening from fundus images». *npj Digital Medicine* 6, 112 (2023). https://doi.org/10.1038/s41746-023-00857-0

[8] Zhou, Y. et al. «A foundation model for generalizable disease detection from retinal images». *Nature* 622, 156–163 (2023). https://doi.org/10.1038/s41586-023-06555-x

[9] Ahn, J. M., Kim, S., Ahn, K. S., Cho, S. H., Lee, K. B. y Kim, U. S. «A deep learning model for the detection of both advanced and early glaucoma using fundus photography». *PLOS ONE* 13(11), e0207982 (2018). https://doi.org/10.1371/journal.pone.0207982
"""
    return md_content

def build_pdf():
    print("[*] 1. Generando markdown estructurado...")
    md_text = generate_markdown()

    # Pre-codificar figuras en base64
    fig_map = {
        "fig00_prisma_2020_flujo_escala_grises.png": get_b64("fig00_prisma_2020_flujo_escala_grises.png"),
        "fig01_publicaciones_por_ano_escala_grises.png": get_b64("fig01_publicaciones_por_ano_escala_grises.png"),
        "fig02_bases_datos_frecuencia_escala_grises.png": get_b64("fig02_bases_datos_frecuencia_escala_grises.png"),
        "fig03_protocolos_evaluacion_escala_grises.png": get_b64("fig03_protocolos_evaluacion_escala_grises.png"),
        "fig04_estrategias_mitigacion_escala_grises.png": get_b64("fig04_estrategias_mitigacion_escala_grises.png"),
        "fig05_caida_delta_auc_por_estudio_escala_grises.png": get_b64("fig05_caida_delta_auc_por_estudio_escala_grises.png"),
    }

    # Reemplazar figuras con etiquetas HTML responsivas
    def replace_img(match):
        alt = match.group(1)
        fname = match.group(2)
        b64_data = fig_map.get(fname, "")
        if b64_data:
            return f'''<div class="figure-container">
  <img src="{b64_data}" alt="{alt}" />
  <p class="caption"><strong>{alt.split(':')[0] if ':' in alt else 'Figura'}:</strong> {alt.split(':', 1)[1] if ':' in alt else alt}</p>
</div>'''
        return f'<div class="img-placeholder">[Imagen: {alt}]</div>'

    html_body = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_img, md_text)

    # Renderizar tablas
    lines = html_body.split("\n")
    processed_lines = []
    in_table = False
    table_lines = []

    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            in_table = True
            table_lines.append(line.strip())
        else:
            if in_table:
                if len(table_lines) >= 2:
                    header = [c.strip() for c in table_lines[0].split("|")[1:-1]]
                    rows = []
                    for r in table_lines[2:]:
                        rows.append([c.strip() for c in r.split("|")[1:-1]])
                    
                    tbl_html = '<div class="table-container"><table class="academic-table"><thead><tr>'
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

    html_body = "\n".join(processed_lines)

    # Convertir encabezados Markdown a HTML
    html_body = re.sub(r"^# (.*?)$", r'<h1 class="main-title">\1</h1>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^## (.*?)$", r'<h2 class="section-title">\1</h2>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^### (.*?)$", r'<h3 class="subsection-title">\1</h3>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^\*\*(.*?)\*\*", r'<strong>\1</strong>', html_body)
    html_body = re.sub(r"^\* (.*?)$", r'<li>\1</li>', html_body, flags=re.MULTILINE)
    html_body = re.sub(r"^(<li>.*?</li>\n?)+", r'<ul class="academic-list">\g<0></ul>', html_body, flags=re.MULTILINE)

    # Párrafos
    p_lines = []
    for line in html_body.split("\n"):
        line_s = line.strip()
        if line_s and not line_s.startswith("<h") and not line_s.startswith("<div") and not line_s.startswith("<p") and not line_s.startswith("<ul") and not line_s.startswith("<li") and not line_s.startswith("</") and not line_s.startswith("---") and not line_s.startswith("|"):
            p_lines.append(f'<p class="academic-p">{line}</p>')
        elif line_s == "---":
            p_lines.append('<hr class="academic-divider"/>')
        else:
            p_lines.append(line)
    html_body = "\n".join(p_lines)

    # Estilos CSS de alta calidad académica
    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Estado del Arte: Clasificación de Glaucoma y Validez entre Fuentes</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,300;1,400&family=Montserrat:wght@400;500;600;700&display=swap');

  @page {{
    size: letter portrait;
    margin: 22mm 18mm 22mm 18mm;
    @top-right {{
      content: "Facultad de Minas — Universidad Nacional de Colombia";
      font-family: 'Montserrat', sans-serif;
      font-size: 7.5pt;
      color: #06784F;
      font-weight: 600;
    }}
    @bottom-center {{
      content: "Estado del Arte • Deep Learning (2026-2) • Página " counter(page) " de " counter(pages);
      font-family: 'Montserrat', sans-serif;
      font-size: 7.5pt;
      color: #64748B;
    }}
  }}

  body {{
    font-family: 'Merriweather', Georgia, 'Times New Roman', serif;
    font-size: 9.8pt;
    line-height: 1.55;
    color: #1E293B;
    background: #FFFFFF;
    margin: 0;
    padding: 0;
  }}

  .main-title {{
    font-family: 'Montserrat', sans-serif;
    font-size: 18pt;
    font-weight: 700;
    color: #06784F;
    line-height: 1.25;
    margin-top: 0;
    margin-bottom: 6px;
    border-bottom: 2.5px solid #06784F;
    padding-bottom: 8px;
  }}

  .section-title {{
    font-family: 'Montserrat', sans-serif;
    font-size: 12.5pt;
    font-weight: 700;
    color: #06784F;
    margin-top: 24px;
    margin-bottom: 10px;
    border-bottom: 1px solid #CBD5E1;
    padding-bottom: 4px;
    page-break-after: avoid;
    break-after: avoid;
  }}

  .academic-p {{
    margin-top: 0;
    margin-bottom: 10px;
    text-align: justify;
    text-justify: inter-word;
  }}

  .academic-divider {{
    border: none;
    border-top: 1px solid #E2E8F0;
    margin: 14px 0;
  }}

  .figure-container {{
    margin: 16px auto;
    text-align: center;
    page-break-inside: avoid;
    break-inside: avoid;
    max-width: 95%;
  }}

  .figure-container img {{
    max-width: 100%;
    max-height: 380px;
    height: auto;
    border: 1px solid #E2E8F0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }}

  .caption {{
    font-family: 'Montserrat', sans-serif;
    font-size: 8.5pt;
    color: #475569;
    margin-top: 6px;
    margin-bottom: 0;
    line-height: 1.35;
    text-align: center;
  }}

  .table-container {{
    margin: 18px 0;
    page-break-inside: avoid;
    break-inside: avoid;
    width: 100%;
    overflow-x: auto;
  }}

  .academic-table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'Montserrat', sans-serif;
    font-size: 8.2pt;
    line-height: 1.3;
  }}

  .academic-table th {{
    background-color: #06784F;
    color: #FFFFFF;
    font-weight: 600;
    text-align: left;
    padding: 6px 8px;
    border: 1px solid #06784F;
  }}

  .academic-table td {{
    padding: 6px 8px;
    border: 1px solid #E2E8F0;
    vertical-align: top;
  }}

  .academic-table tr:nth-child(even) {{
    background-color: #F8FAFC;
  }}

  .academic-list {{
    margin-top: 4px;
    margin-bottom: 10px;
    padding-left: 20px;
  }}

  .academic-list li {{
    margin-bottom: 4px;
    text-align: justify;
  }}

  a {{
    color: #06784F;
    text-decoration: none;
    word-break: break-all;
  }}

  a:hover {{
    text-decoration: underline;
  }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    print(f"[*] 2. Guardando Markdown depurado en: {MD_OUTPUT_PATH}")
    MD_OUTPUT_PATH.write_text(md_text, encoding="utf-8")

    print(f"[*] 3. Guardando HTML autónomo en: {HTML_OUTPUT_PATH}")
    HTML_OUTPUT_PATH.write_text(full_html, encoding="utf-8")

    print(f"[*] 4. Compilando PDF vía Google Chrome...")
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF_OUTPUT_PATH}",
        str(HTML_OUTPUT_PATH)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if PDF_OUTPUT_PATH.exists():
        size_mb = PDF_OUTPUT_PATH.stat().st_size / (1024 * 1024)
        print(f"[EXITO] PDF generado: {PDF_OUTPUT_PATH.name} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"[ERROR] Error al generar PDF: {res.stderr}")
        return False

if __name__ == "__main__":
    build_pdf()
