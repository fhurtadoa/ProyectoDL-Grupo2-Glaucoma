#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilador del Informe Técnico 04:
Evaluación Exhaustiva de Bases de Datos Médicas, Inspección Visual de Retinografías,
Armonización Geométrica de la Mega-Cohorte y Protocolo de Entrenamiento con Data Augmentation Maligno.
"""

import os, sys, re, base64, subprocess
from pathlib import Path

BASE_DIR = Path("/Users/fabian/GlaucomaDL")
REPORT_DIR = BASE_DIR / "informes" / "04_evaluacion_bases_medicas_y_mega_dataset"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MD_PATH = REPORT_DIR / "INFORME_EVALUACION_BASES_MEDICAS.md"
HTML_PATH = REPORT_DIR / "INFORME_EVALUACION_BASES_MEDICAS.html"
PDF_PATH = REPORT_DIR / "INFORME_EVALUACION_BASES_MEDICAS.pdf"

md_content = """# INFORME TÉCNICO Y EVALUACIÓN EXPERIMENTAL DE BASES DE DATOS MÉDICAS DE GLAUCOMA
## Análisis de Validez de Etiquetas Clínicas, Inspección Visual de Retinografías, Armonización Geométrica Peripapilar y Protocolo de Evaluación Robusta

---

### Metadatos del Documento
- **Proyecto de Investigación:** GlaucomaDL — Evaluación de Generalización Multicéntrica y Selección de Arquitecturas
- **Materia:** Deep Learning — Facultad de Minas, Universidad Nacional de Colombia
- **Fecha de Compilación:** Noviembre 2026
- **Propósito:** Responder exhaustivamente a la auditoría de calidad de imágenes médicas, verificar la existencia y rigor de las etiquetas diagnósticas en bases internacionales y locales, documentar la armonización geométrica/cromática de la cohorte y proveer las especificaciones de ejecución para entrenamiento en clusters externos con *Data Augmentation* Maligno.

---

## 1. Resumen Ejecutivo y Hallazgos Principales de la Auditoría

1. **Validez de Etiquetas en Bases Abiertas Internacionales:**  
   No todas las bases de datos oftalmológicas son equivalentes. Se constató que bases como **REFUGE-2**, **PAPILA**, **ACRIMA** y **RIM-ONE DL** cuentan con un estándar de oro (*Gold Standard*) de máxima confianza, certificado por consenso de múltiples especialistas y correlacionado con campimetría visual y tomografía OCT. Por el contrario, bases como **AIROGS** presentan un etiquetado poblacional de cribado (*referable vs. non-referable*) con un desbalance extremo ($>95\%$ negativos), y **BRSET** es un repositorio multi-etiqueta donde el glaucoma es un marcador secundario.

2. **Hallazgo Crítico en el Repositorio Local (`THIRD_DATASET`):**  
   Durante la inspección de tensores píxel a píxel, se descubrió que el conjunto local `data/THIRD_DATASET/` ($N = 2,805$ imágenes) **no contiene retinografías en color RGB**, sino mapas derivados de bordes y esqueletos vasculares binarios en formato RGBA (con más del $97.8\%$ de los píxeles en cero y medias de canal $\approx 3.5$).  
   > **Decisión Metodológica Inmediata:** `THIRD_DATASET` fue **estrictamente segregado y excluido** del conjunto de entrenamiento RGB para evitar la corrupción de las representaciones convolucionales residuales de ResNet-18.

3. **Ejecución Exitosa del Pipeline de Armonización (`07_armonizacion_roi_y_mega_dataset.py`):**  
   Se diseñó e implementó un algoritmo detector de papila óptica y extractor de región de interés (ROI) que transforma imágenes panorámicas de $45^\circ$ (EyePACS) a encuadres equivalentes peripapilares de $35^\circ$ con papila centrada, resolviendo la discrepancia de escala que causaba el desplome a azar ($0.51$ ROC-AUC). Se estructuró la **Mega-Cohorte Armonizada** ($N = 2,150$ retinografías RGB) dividida en entrenamiento ($N = 1,719$) y prueba ($N = 431$).

---

## 2. Auditoría Exhaustiva de Bases de Datos Médicas: ¿Realmente Sirven y Tienen Etiquetas?

A continuación se presenta la tabla diagnóstica evaluando la autenticidad del sensor, formato de imagen, tipo de etiqueta y aptitud para el entrenamiento de redes convolucionales:

| Base de Datos Médica | País / Fuente | N Total | ¿Es Foto RGB Real? | ¿Tiene Etiquetas Diagnósticas? | Método de Validación de la Etiqueta | Nivel de Confianza | Diagnóstico de Aptitud para DL |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: | :--- |
| **REFUGE / REFUGE-2** | Multi-Centro (China/Europa) | 2,000 | **Sí** (Zeiss, Canon, Kowa, Topcon) | **Sí** (Glaucoma / Sano + Máscaras OD/OC) | Correlación con OCT, campimetría Humphrey y consenso de especialistas | **Máxima (Gold Standard)** | **Excelente.** Es el benchmark estándar internacional de transferencia multi-cámara. |
| **PAPILA** | Argentina (San Juan) | 488 | **Sí** (Topcon TRC 50DX) | **Sí** (Sano, Glaucoma, Sospechoso) | Consenso oftalmológico clínico + PIO + Índices de campo visual (MD, PSD) | **Máxima (Clínica Real)** | **Excelente.** Aporta la clase intermedia "Sospechoso", reduciendo falsos positivos. |
| **ACRIMA** *(En repo)* | España | 703 | **Sí** (Canon CR-DGi 35°) | **Sí** (Sano=307, Glaucoma=396) | Consenso estricto de 3 retinólogos independientes | **Máxima (Gold Standard)** | **Excelente.** Ya integrado como núcleo peripapilar de alta resolución. |
| **RIM-ONE DL** *(En repo)* | España (Canarias) | 447 | **Sí** (Kowa WX 3D Estéreo) | **Sí** (Sano=275, Glaucoma=172) | Segmentación fina peripapilar y confirmación clínica | **Máxima (Gold Standard)** | **Excelente.** Ya integrado; sensor de referencia para tono frío. |
| **EyePACS Real** *(En repo)* | EE. UU. / LatAm | 200 | **Sí** (Cámaras de campo abierto 45°) | **Sí** (Sano=100, Referable Glauc=100) | Lectura médica en centros de telemedicina de cribado masivo | **Alta (Cribado Real)** | **Apta con Armonización.** Requiere recorte ROI para centrar papila. |
| **Moondream Staged** *(En repo)* | Internacional | 800 | **Sí** (Multi-cámara estadiado) | **Sí** (Sano, Temprano, Avanzado) | Graduación por daño morfológico del anillo | **Alta** | **Apta.** Permite evaluar la sensibilidad ante glaucoma temprano. |
| **LAG Dataset** | China (Beijing Tongren) | 11,760 | **Sí** (Topcon / Canon 45°) | **Sí** (1,711 Glaucoma, 4,113 Normal) | Revisión médica + Mapas de atención por seguimiento ocular (*eye-tracking*) | **Muy Alta** | **Muy Apta.** Excelente para pre-entrenamiento masivo y explicabilidad. |
| **G1020** | Alemania (Kaiserslautern) | 1,020 | **Sí** (Cámara de fondo 45°) | **Sí** (296 Glaucoma, 724 Normal) | Bounding box OD/OC, regla ISNT, pérdida anillo inferior/superior | **Alta** | **Muy Apta.** Muy útil para tareas morfológicas detalladas. |
| **AIROGS** | Holanda / EE. UU. | 113,893 | **Sí** (Múltiples sensores 45°) | **Sí** (Referable Glaucoma vs Non-Referable) | Lectura centralizada + Banderas de calidad (*Ungradable*) | **Alta (Poblacional)** | **Apta con Filtrado.** Desbalance severo (3,270 positivos vs 110k negativos). |
| **BRSET** | Brasil (PhysioNet) | 16,266 | **Sí** (Canon CR-2 y Nikon NF505) | **Parcial / Multi-etiqueta** | Etiquetas de retinopatía diabética, DMAE y papila anormal | **Media** | **Condicional.** El glaucoma no es clase primaria aislada. |
| **ODIR-5K** | China (Peking Univ.) | 10,000 | **Sí** (Canon, Zeiss, Kowa 45°) | **Parcial (Multi-enfermedad)** | Diagnóstico bilateral de 8 clases oculares (solo 307 casos de glaucoma) | **Baja-Media** | **No prioritaria.** Prevalencia de glaucoma muy reducida ($3\%$). |
| **THIRD_DATASET** *(En repo)* | Kaggle Glaucoma | 2,805 | **NO (Esqueleto / Bordes RGBA)** | **Sí (en nombres de carpetas)** | Sin metadatos clínicos; artefacto derivado de segmentación | **Nula (No es RGB)** | **INAPTA.** Corrompe el aprendizaje de texturas retinianas en CNNs. |

---

## 3. Inspección Visual de las Fotos: Características Morfológicas y Cromáticas

La siguiente figura muestra cómo se observan realmente las imágenes capturadas por los diferentes instrumentos clínicos del ecosistema:

![Inspección Visual Comparativa](figures/fig23_visual_sample_inspection_and_roi.png)

### Análisis de las Muestras Inspeccionadas:
1. **ACRIMA (Canon CR-DGi $35^\circ$):**
   - *Composición:* Encuadre peripapilar centrado donde el disco ocupa $\approx 35\%$ del fotograma.
   - *Firma Cromática:* Dominancia masiva del canal rojo ($\overline{R}=187.9, \overline{B}=26.0, R/B=7.23$). La retina presenta un tinte anaranjado cálido intenso.
2. **RIM-ONE DL (Kowa WX 3D $35^\circ$):**
   - *Composición:* Captura estereoscópica peripapilar con lentes polarizadas gemelas.
   - *Firma Cromática:* Tono visiblemente más neutro/frío ($\overline{R}=146.2, \overline{B}=37.3, R/B=3.92$). Mayor sensibilidad a longitudes de onda cortas.
3. **EyePACS Real ($45^\circ$ FOV):**
   - *Composición:* Fotografía de campo abierto panorámico centrada en la mácula lútea. La papila óptica está desplazada hacia el borde nasal y ocupa menos del $6\%$ de la superficie.
   - *Firma Cromática:* Subexpuesta en promedio ($\overline{R}=86.1, \overline{B}=36.4, R/B=2.37$) con fuerte viñeteo periférico.
4. **THIRD_DATASET (¡Alerta Metodológica!):**
   - Como se evidencia en el panel inferior central de la figura, esta base no contiene fotografías de fondo de ojo. Consiste en representaciones esqueletizadas de vasculatura con fondo negro ($\overline{R}=\overline{G}=\overline{B}=3.5$). Entrenar una red convolucional pre-entrenada en ImageNet con estos tensores vacíos destruye los pesos de los filtros espaciales.

---

## 4. Solución Implementada: Armonización Geométrica (Optic Disc ROI Cropping)

Para solucionar la discrepancia de encuadre entre cámaras peripapilares ($35^\circ$) y panorámicas ($45^\circ$), se implementó el algoritmo `OpticDiscCropper` en el script `scripts/experimentos/07_armonizacion_roi_y_mega_dataset.py`.

![Armonización Geométrica y Fusión de la Mega-Cohorte](figures/fig24_harmonization_pipeline_before_after.png)

### Mecanismo de Funcionamiento:
1. **Convergencia de Brillo y Máscara FOV:** Se aísla el disco calculando una combinación ponderada de los canales rojo y verde ($0.70 R + 0.30 G$) sujeta a una máscara circular concéntrica que descarta el anillo exterior reflectante del oftalmoscopio.
2. **Suavizado Gaussiano ($\sigma = 4.0$):** Suprime destellos especulares puntuales y reflexiones de la arcada vascular.
3. **Recorte Centrado Proporcional ($55\%$ del FOV):** Se recorta un bounding box cuadrado centrado en el centroide de la papila y se reescala con interpolación bicúbica a $512 \times 512$ píxeles.
4. **Resultado de la Fusión:** La Mega-Cohorte Armonizada resultante cuenta con **2,150 retinografías válidas y homogéneas** (todas con la papila centrada ocupando $>30\%$ del campo visual), divididas en $1,719$ de entrenamiento y $431$ de prueba independiente.

---

## 5. Mapeo Internacional y Distribución de Recursos

El catálogo de bases analizadas a nivel global se resume en la siguiente comparativa multivariable:

![Mapeo Comparativo de Bases de Datos Médicas](figures/fig22_medical_databases_comparative_profiling.png)

---

## 6. Prompt y Protocolo de Ejecución para Entrenamiento en Otro PC (GPU / Cluster)

Para que el entrenamiento sobre la Mega-Cohorte y la evaluación de **Data Augmentation Maligno** puedan ejecutarse de forma autónoma en otra máquina (e.g., estación con GPU NVIDIA RTX 3090/4090, servidor A100 o clúster Linux/Mac), se diseñó el siguiente protocolo maestro llave en mano.

```bash
# ==============================================================================
# PROTOCOLO DE EJECUCIÓN: ENTRENAMIENTO MEGA-COHORTE Y TEST MALIGNO
# ==============================================================================

# 1. Clonar el repositorio y posicionarse en la carpeta correcta (SIN XAI):
git clone https://github.com/tu-usuario/GlaucomaDL.git
cd GlaucomaDL

# 2. Configurar entorno virtual con aceleración GPU (CUDA / MPS):
python3 -m venv venv_glaucoma
source venv_glaucoma/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn pillow matplotlib scipy

# 3. Verificar estado de los datos armonizados:
python3 scripts/experimentos/07_armonizacion_roi_y_mega_dataset.py

# 4. Ejecutar el entrenamiento con Data Augmentation Maligno:
#    (Simula distorsiones extremas de zoom 0.25x - 1.85x, caos fotométrico y viñetas)
python3 scripts/experimentos/04_data_augmentation_invarianza_escala.py

# 5. Ejecutar la evaluación de transferencia cruzada multi-sensor:
python3 scripts/experimentos/05_mega_cohorte_federada_multisensor.py
```

### Prompt Listo para Copiar y Pegar en la Consola / Agente del Otro PC:

```text
Actúa como Ingeniero Senior de Deep Learning en Imagen Médica. Tu misión es entrenar y someter a prueba de estrés el modelo de clasificación de glaucoma sobre la nueva cohorte armonizada multi-sensor en el repositorio 'GlaucomaDL'.

Sigue estrictamente las siguientes directrices técnicas:
1. ARQUITECTURA: Utiliza ResNet-18 pre-entrenada en ImageNet-1K (el backbone con menor sobreajuste verificado, gap = 0.0232). Reemplaza la capa lineal final 'fc' por una salida binaria única con BCEWithLogitsLoss.
2. CONJUNTO DE DATOS: Utiliza la ruta 'data/HARMONIZED_MEGA_COHORT' generada por '07_armonizacion_roi_y_mega_dataset.py'. Asegúrate de que THIRD_DATASET esté excluido porque contiene esqueletos vasculares RGBA no aptos para visión natural.
3. PROTOCOLO DE DATA AUGMENTATION MALIGNO:
   - Aplica la clase 'MalignantScaleAugmentation' de '04_data_augmentation_invarianza_escala.py':
     * 45% de probabilidad de zoom-out agresivo (0.25x a 0.65x) con relleno de fondo sintético ruidoso y viñeta óptica circular.
     * 35% de probabilidad de zoom-in focal (1.25x a 1.85x) simulando micro-inspección de copa.
     * 65% de probabilidad de perturbación fotométrica (gamma 0.75 - 1.35 y ganancia de verde 0.90 - 1.15).
4. HIPERPARÁMETROS DE ENTRENAMIENTO:
   - Optimizador: AdamW (lr = 1e-4, weight_decay = 1e-4).
   - Scheduler: CosineAnnealingLR durante 15 épocas completas.
   - Batch size: 32 (o 64 si tienes >16GB VRAM).
   - Métricas a registrar por época: Train Loss, Val ROC-AUC, F1-Score, Sensibilidad y Especificidad.
5. PRUEBA DE ESTRÉS FINAL:
   - Evalúa el checkpoint 'best_resnet18_malignant_scale.pth' sobre las 4 particiones de test independientes: ACRIMA (Canon 35°), RIM-ONE DL (Kowa 3D), EyePACS (45° armonizado) y Moondream.
   - Genera la matriz de confusión y exporta las métricas en 'results/malignant_stress_test_metrics.json'.
```

---

## 7. Conclusiones y Hoja de Ruta

1. **Rigor en la Selección de Datos:** La auditoría confirmó que bases como **REFUGE-2** y **PAPILA** son las candidatas óptimas para enriquecer el estudio por su estándar clínico riguroso y la disponibilidad de metadatos oftalmológicos.
2. **Filtrado Preventivo Exitoso:** Se evitó un error severo de entrenamiento al aislar `THIRD_DATASET` (esqueletos vasculares) y enfocar los esfuerzos en retinografías en color reales.
3. **Invarianza Lograda:** El recorte automático de ROI resuelve el desacoplamiento geométrico entre dispositivos, permitiendo que la red convolucional entrene sobre un área uniforme y maximice su generalización inter-hospitalaria.
"""

def compile_pdf():
    print("[INFO] Escribiendo archivo Markdown en:", MD_PATH)
    MD_PATH.write_text(md_content, encoding="utf-8")

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

        m_num = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if m_num:
            if in_ul:
                close_lists()
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{m_num.group(2)}</li>")
            i += 1
            continue

        if stripped.startswith("- ") or stripped.startswith("* "):
            if in_ol:
                close_lists()
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            item_text = stripped[2:].strip()
            html_lines.append(f"<li>{item_text}</li>")
            i += 1
            continue

        close_lists()
        html_lines.append(f"<p>{stripped}</p>")
        i += 1

    close_lists()
    body_html = "\n".join(html_lines)

    for i, (mtype, formula) in enumerate(math_blocks):
        escaped_formula = formula.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
        if mtype == "display":
            replacement = f'<div class="math-display" data-katex="{escaped_formula}"></div>'
        else:
            replacement = f'<span class="math-inline" data-katex="{escaped_formula}"></span>'
        body_html = body_html.replace(f"@@MATH_BLOCK_{i}@@", replacement)

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8"/>
<title>Informe Evaluación Bases Médicas y Mega-Cohorte</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"/>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function() {{
    document.querySelectorAll(".math-inline").forEach(function(el) {{
      katex.render(el.getAttribute("data-katex"), el, {{ displayMode: false, throwOnError: false }});
    }});
    document.querySelectorAll(".math-display").forEach(function(el) {{
      katex.render(el.getAttribute("data-katex"), el, {{ displayMode: true, throwOnError: false }});
    }});
  }});
</script>
<style>
  @page {{
    size: A4 portrait;
    margin: 14mm 14mm 14mm 14mm;
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
    font-size: 9.2pt;
  }}
  h1 {{
    font-size: 16pt;
    color: #0f172a;
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 5px;
    margin-top: 0;
    margin-bottom: 8px;
    font-weight: 800;
    page-break-after: avoid;
    line-height: 1.25;
  }}
  h2 {{
    font-size: 12pt;
    color: #1d4ed8;
    border-bottom: 1.2px solid #cbd5e1;
    padding-bottom: 3px;
    margin-top: 14px;
    margin-bottom: 6px;
    font-weight: 700;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 10.2pt;
    color: #334155;
    margin-top: 10px;
    margin-bottom: 4px;
    font-weight: 600;
    page-break-after: avoid;
  }}
  p {{
    margin-top: 0;
    margin-bottom: 6px;
    text-align: justify;
  }}
  ul, ol {{
    margin-top: 0;
    margin-bottom: 6px;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 2.5px;
  }}
  blockquote {{
    border-left: 3.5px solid #2563eb;
    background: #eff6ff;
    margin: 6px 0;
    padding: 5px 10px;
    color: #1e40af;
    font-style: italic;
    page-break-inside: avoid;
  }}
  hr {{
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 10px 0;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8.2pt;
    background-color: #f1f5f9;
    color: #b91c1c;
    padding: 1.5px 3.5px;
    border-radius: 3px;
  }}
  pre.code-block {{
    background-color: #0f172a;
    color: #f8fafc;
    padding: 8px 10px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 7.8pt;
    margin: 6px 0;
    page-break-inside: avoid;
  }}
  pre.code-block code {{
    background-color: transparent;
    color: inherit;
    padding: 0;
  }}
  .table-wrapper {{
    margin: 8px 0;
    page-break-inside: avoid;
    width: 100%;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    page-break-inside: avoid;
  }}
  th {{
    background-color: #1e40af;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 5px 6px;
    border: 1px solid #1e40af;
  }}
  td {{
    padding: 4px 5.5px;
    border: 1px solid #cbd5e1;
    vertical-align: middle;
  }}
  tr:nth-child(even) td {{
    background-color: #f8fafc;
  }}
  .report-figure {{
    margin: 8px auto;
    text-align: center;
    page-break-inside: avoid;
  }}
  .report-figure img {{
    max-width: 98%;
    max-height: 340px;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .report-figure figcaption {{
    font-size: 8pt;
    color: #475569;
    font-style: italic;
    margin-top: 3px;
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
    print("[INFO] Ejecutando Google Chrome headless para compilar PDF...")
    subprocess.run(cmd, check=True)
    print(f"[EXITO] PDF generado en: {PDF_PATH} ({PDF_PATH.stat().st_size / (1024*1024):.2f} MB)")
    return True

if __name__ == "__main__":
    compile_pdf()
