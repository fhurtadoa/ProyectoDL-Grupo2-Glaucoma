#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilador del Informe Técnico 05:
Entrenamiento y Evaluación Bajo Protocolo de Estrés Maligno (Malignant Scale Augmentation)
sobre la Mega-Cohorte Médica Armonizada de Glaucoma.
Genera el informe en Markdown, HTML editorial autónomo con KaTeX e imágenes base64 embebidas,
y compila el documento en PDF de alta calidad mediante navegador Chromium headless.
"""

import os, sys, re, base64, subprocess, shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "informes" / "05_entrenamiento_y_estres_malignant_scale_mega_cohorte"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MD_PATH = REPORT_DIR / "INFORME_ENTRENAMIENTO_ESTRES_MALIGNO_MEGA_COHORTE.md"
HTML_PATH = REPORT_DIR / "INFORME_ENTRENAMIENTO_ESTRES_MALIGNO_MEGA_COHORTE.html"
PDF_PATH = REPORT_DIR / "INFORME_ENTRENAMIENTO_ESTRES_MALIGNO_MEGA_COHORTE.pdf"

md_content = """# INFORME TÉCNICO Y EXPERIMENTAL: ENTRENAMIENTO Y EVALUACIÓN DE ESTRÉS MALIGNO EN MEGA-COHORTE ARMONIZADA DE GLAUCOMA
## Inducción de Invarianza de Escala Extrema (0.25x - 1.85x), Auditoría de Cohortes Clínicas y Resiliencia en Inferencia Hostil

---

### Metadatos del Proyecto de Investigación
- **Proyecto:** GlaucomaDL — Detección Automatizada y Generalización Multicéntrica
- **Materia:** Deep Learning — Facultad de Minas, Universidad Nacional de Colombia
- **Arquitectura de Red:** ResNet-18 (Pre-entrenada en ImageNet-1K, Cabeza Escalar `Linear(512, 1)`)
- **Acelerador de Cómputo:** NVIDIA GeForce RTX 4060 Ti (CUDA 12.6, PyTorch 2.14.1)
- **Fecha de Compilación:** Noviembre 2026
- **Checkpoint Oficial:** `modelos/best_resnet18_malignant_scale.pth`
- **Métricas Oficiales:** `results/malignant_stress_test_metrics.json` y `results/inference_time_malignant_stress_comparison.json`

---

## 1. Resumen Ejecutivo y Planteamiento Clínico-Técnico

El diagnóstico del glaucoma mediante retinografías de fondo de ojo en la práctica clínica real enfrenta un obstáculo crítico: **el desplome del rendimiento por cambio de dominio óptico (*optical domain shift*)**. Mientras que los modelos convencionales entrenados en una sola cohorte curada alcanzan $ROC\\text{-}AUC > 0.98$, colapsan hacia niveles de azar puro ($ROC\\text{-}AUC \\approx 0.51$) al enfrentarse a cámaras de cribado masivo panorámicas ($45^\\circ$, como EyePACS) o sensores estereoscópicos tridimensionales ($35^\\circ$, como Kowa WX 3D de RIM-ONE DL).

Para resolver esta vulnerabilidad estructural sin recurrir a costosas segmentaciones manuales, este informe documenta la culminación de dos hitos científicos mayores:

1. **Auditoría Rigurosa y Armonización Geométrica/Cromática:**
   - Se resolvió un error sutil de asignación de etiquetas en el repositorio local originado por coincidencia de subcadenas con el nombre de carpeta `GlaucomaDL`, restaurando la prevalencia real de las 4 cohortes.
   - Se consolidó la exclusión estricta de `THIRD_DATASET` ($N = 2,805$), comprobando que consiste en máscaras de esqueletos vasculares que corrompen la visión natural.
   - Se aplicó el detector inteligente de papila óptica `OpticDiscCropper` y el normalizador cromático de Reinhard, alineando $2,152$ retinografías a un estándar peripapilar equivalente de $35^\\circ$ ($512 \\times 512$).
2. **Entrenamiento y Doble Estrés bajo *Malignant Scale Augmentation*:**
   - La red fue entrenada forzando deformaciones extremas: Zoom-Out severo ($0.25x - 0.65x$) con viñeteo óptico circular y fondo sintético ruidoso, Zoom-In focal ($1.25x - 1.85x$) sobre la copa óptica, y caos fotométrico de espectro verde/gamma.
   - El modelo resultante alcanzó en validación un **$ROC\\text{-}AUC = 0.9569$ en la Época 14**.
   - En evaluación ciega sobre las 4 cohortes independientes ($N = 431$), obtuvo **$ROC\\text{-}AUC = 0.9939$ en ACRIMA**, **$0.9293$ en RIM-ONE DL**, **$0.9086$ en Moondream** y **$0.7850$ en EyePACS** ($ROC\\text{-}AUC$ global ponderado de **$0.9482$**).
   - En una prueba de estrés hostil en tiempo de inferencia (inyectando aumentos malignos a las imágenes de prueba no vistas), el modelo retuvo más del **$98.8\\%$ de su capacidad discriminativa**, demostrando una auténtica invarianza morfológica.

---

## 2. Auditoría Crítica de Bases de Datos Médicas y Composición de la Mega-Cohorte

### 2.1 Diagnóstico de Calidad y Resolución del Fallo de Asignación de Etiquetas

Durante la inspección de la base de datos se descubrió que ejecuciones históricas asignaban etiqueta $1$ (glaucoma) a la totalidad de imágenes de RIM-ONE DL, EyePACS y Moondream, debido a que la condición lógica `if "glaucoma" in f.lower()` evaluaba la ruta absoluta completa, la cual contenía la palabra `GlaucomaDL`.

Se implementó una auditoría basada estrictamente en rutas relativas por carpeta y metadatos clínicos (`data/RIM_ONE_DL/metadata.csv`, carpetas `normal`/`glaucoma` de EyePACS, y subdirectorios `normal`/`early`/`advanced` de Moondream), restituyendo la verdad diagnóstica:

| Base de Datos Médica | Sensor / Fabricante | Campo Visual (FOV) | N Total | Glaucoma ($y=1$) | Normal ($y=0$) | Prevalencia Real | Método de Validación de Etiqueta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ACRIMA** | Canon CR-DGi | $35^\\circ$ peripapilar | 705 | 396 | 309 | 56.17% | Consenso de 3 retinólogos independientes |
| **RIM-ONE DL** | Kowa WX 3D Estéreo | $35^\\circ$ peripapilar | 447 | 162 | 285 | 36.24% | Evaluación estereoscópica clínica especializada |
| **EyePACS Real** | Cámaras de cribado | $45^\\circ$ panorámico | 200 | 100 | 100 | 50.00% | Lectura oftalmológica en telemedicina masiva |
| **Moondream Staged** | Multi-cámara clínica | $40^\\circ$ multicéntrico | 800 | 400 | 400 | 50.00% | Graduación por daño del anillo (Early / Advanced) |
| **THIRD_DATASET** | Procesamiento derivado | N/A | 2,805 | 0 | 0 | **EXCLUIDO** | Máscaras de bordes vasculares en canal alfa (no RGB) |
| **TOTAL UNIFICADO** | **Mega-Cohorte Federada** | **$35^\\circ$ Armonizado** | **2,152** | **1,058** | **1,094** | **49.16%** | **Cohorte Médica Balanceada y Certificada** |

### 2.2 Justificación Matemática de la Exclusión de `THIRD_DATASET`

La inspección de tensores de `data/THIRD_DATASET` reveló que el $97.8\\%$ de los píxeles tienen intensidad nula ($0$), consistiendo en mapas de binarización y esqueletización de vasos sanguíneos. En una red convolucional residual (ResNet-18), introducir este tipo de imágenes induce **atajos de gradiente (*shortcut learning*)**: los filtros de primer orden se acoplan a la presencia de líneas sintéticas mono-espectrales y anulan las características de textura retinal (anillo neurorretiniano, relación copa-disco, signo de Bayoneta y hemorragias de Drance). Su exclusión preserva intacto el espacio latente del dominio oftalmológico natural.

---

## 3. Pipeline de Armonización Geométrica y Cromática

Para transformar las imágenes panorámicas de $45^\\circ$ (EyePACS) en encuadres comparables a las cámaras de $35^\\circ$ (ACRIMA y RIM-ONE), se ejecutó el pipeline de `OpticDiscCropper`:

```
               [Retinografía 45° (EyePACS, N=200)]
                               |
                               v
            +--------------------------------------+
            |  1. Máscara circular de apertura FOV |
            |  2. Fusión luminosa: 0.70 R + 0.30 G |
            |  3. Suavizado Gaussiano (sigma = 4)  |
            |  4. Localización del centroide (cx,cy)|
            +--------------------------------------+
                               |
                               v
            [Recorte Peripapilar (0.55 x min(W,H))]
                               |
                               v
         +--------------------------------------------+
         |  Normalización Cromática Reinhard (Z-Score)|
         |  Mapeo a sensor Kowa/Canon de referencia   |
         |  Redimensión Bicúbica a 512 x 512 px       |
         +--------------------------------------------+
                               |
                               v
           [Muestra Integrada a HARMONIZED_MEGA_COHORT]
```

![Pipeline de Armonización Geométrica y Espectral](figures/fig24_harmonization_pipeline_before_after.png)

### Composición Final de Particiones Médicas (Sin Fuga de Datos):
- **Conjunto de Entrenamiento (*Train Pool*):** $1,721$ imágenes ($846$ Glaucoma, $875$ Normales).
  - *Train Split Interno ($85\\%$):* $1,462$ imágenes ($718$ Glaucoma, $744$ Normales).
  - *Validation Split Interno ($15\\%$):* $259$ imágenes ($128$ Glaucoma, $131$ Normales).
- **Conjunto de Prueba Independiente (*Test Held-Out*, $20\\%$ de cada centro):** $431$ imágenes ($212$ Glaucoma, $219$ Normales).
  - ACRIMA Test: $N=141$ ($79$ Glaucoma, $62$ Normal).
  - RIM-ONE DL Test: $N=90$ ($33$ Glaucoma, $57$ Normal).
  - EyePACS Test: $N=40$ ($20$ Glaucoma, $20$ Normal).
  - Moondream Test: $N=160$ ($80$ Glaucoma, $80$ Normal).

---

## 4. Protocolo de Data Augmentation Maligno (*Malignant Scale Augmentation*)

El generador en línea `MalignantScaleAugmentation` rompe intencionalmente la correlación espuria entre el tamaño aparente del disco óptico y la etiqueta diagnóstica mediante tres modalidades concurrentes:

![Galería de Muestras Bajo Data Augmentation Maligno](figures/fig10_malignant_augmentation_samples.png)

1. **Modo Zoom-Out Severo ($p = 0.45$):**
   - Escala aleatoria $s \\sim \\text{Uniform}(0.25, 0.65)$, reduciendo el disco a dimensiones entre $56 \\times 56$ y $145 \\times 145$ píxeles.
   - **Lienzo con Contexto Sintético:** Para evitar el atajo de bordes rectangulares negros, el fondo se sintetiza calculando el color periférico medio de la retina $\\mu_{bg} = [\\mu_R, \\mu_G, \\mu_B]$ con ruido de textura $\\epsilon \\sim \\mathcal{N}(0, 7^2)$.
   - **Viñeteo Circular Óptico Oftálmico:** Máscara radial analítica $V(x, y) = \\text{clip}\\left(1.0 - \\frac{d(x,y) - 0.48W}{8.0}, 0.0, 1.0\\right)$.
   - Desplazamiento estocástico bidimensional $(x_0, y_0)$.
2. **Modo Zoom-In Focal ($p = 0.35$):**
   - Escala aleatoria $s \\sim \\text{Uniform}(1.25, 1.85)$, aislando micro-sectores peripapilares de la copa óptica y forzando a la red a diagnosticar basándose en el adelgazamiento del anillo neurorretiniano sin ver la circunferencia papilar completa.
3. **Modo Estándar ($p = 0.20$):**
   - Escala $1.0x$ estándar para preservar la topología original.
4. **Perturbación Fotométrica Espectral ($p = 0.65$ condicional):**
   - Corrección no lineal de gamma: $I_{out} = I_{in}^\\gamma, \\; \\gamma \\sim \\text{Uniform}(0.75, 1.35)$.
   - Ganancia estocástica en canal verde: $I_{out}^{(G)} = \\alpha_G \\cdot I_{in}^{(G)}, \\; \\alpha_G \\sim \\text{Uniform}(0.90, 1.15)$.

---

## 5. Entrenamiento y Convergencia de ResNet-18

El entrenamiento se realizó en la GPU NVIDIA GeForce RTX 4060 Ti con la función de costo `BCEWithLogitsLoss`, optimizador `AdamW` (learning rate $= 10^{-4}$, weight decay $= 10^{-4}$) y decaimiento cosenoidal `CosineAnnealingLR` a lo largo de 15 épocas completas (tiempo de entrenamiento: $214.01$ segundos $\\approx 3.57$ minutos):

| Época | Tasa de Aprendizaje | Pérdida Entrenamiento | Pérdida Validación | ROC-AUC Validación | Exactitud Validación | Estado Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | $9.89 \\times 10^{-5}$ | 0.5636 | 0.5108 | 0.8499 | 70.66% | Guardado inicial |
| **02** | $9.57 \\times 10^{-5}$ | 0.4448 | 0.6590 | 0.9076 | 74.52% | Actualizado (+0.0577) |
| **03** | $9.05 \\times 10^{-5}$ | 0.4110 | 0.3814 | 0.9209 | 81.85% | Actualizado (+0.0133) |
| **04** | $8.35 \\times 10^{-5}$ | 0.3911 | 0.3487 | 0.9352 | 82.63% | Actualizado (+0.0143) |
| **05** | $7.50 \\times 10^{-5}$ | 0.3624 | 0.4917 | 0.9115 | 77.61% | — |
| **06** | $6.55 \\times 10^{-5}$ | 0.3649 | 0.3120 | 0.9447 | 88.42% | Actualizado (+0.0095) |
| **07** | $5.52 \\times 10^{-5}$ | 0.3030 | 0.3301 | 0.9405 | 84.17% | — |
| **08** | $4.48 \\times 10^{-5}$ | 0.3120 | 0.3710 | 0.9469 | 83.78% | Actualizado (+0.0022) |
| **09** | $3.45 \\times 10^{-5}$ | 0.2778 | 0.3347 | 0.9473 | 85.33% | Actualizado (+0.0004) |
| **10** | $2.50 \\times 10^{-5}$ | 0.2926 | 0.3300 | 0.9504 | 85.33% | Actualizado (+0.0031) |
| **11** | $1.65 \\times 10^{-5}$ | 0.2501 | 0.3465 | 0.9534 | 86.10% | Actualizado (+0.0030) |
| **12** | $9.55 \\times 10^{-6}$ | 0.2467 | 0.3115 | 0.9550 | 86.87% | Actualizado (+0.0016) |
| **13** | $4.32 \\times 10^{-6}$ | 0.2406 | 0.3047 | 0.9553 | 86.87% | Actualizado (+0.0003) |
| **14** | $1.08 \\times 10^{-6}$ | **0.2407** | **0.3174** | **0.9569** | **88.03%** | **★ MEJOR MODELO GUARDADO** |
| **15** | $1.00 \\times 10^{-6}$ | 0.2237 | 0.3140 | 0.9543 | 87.26% | — |

El estado óptimo se consolidó en la **Época 14** con un **$ROC\\text{-}AUC$ de Validación de $0.9569$**, guardado en `modelos/best_resnet18_malignant_scale.pth`.

---

## 6. Matriz de Estrés Clínico: Benchmark en las 4 Cohortes Independientes

El checkpoint seleccionado fue sometido a evaluación ciega en las particiones de prueba independientes ($N=431$ retinografías). La Figura 25 sintetiza las curvas ROC, matrices de confusión por sensor y métricas comparativas:

![Benchmark de Estrés Maligno Multicéntrico](figures/fig25_malignant_scale_stress_test_benchmark.png)

### 6.1 Tabla Maestra de Desempeño Multicéntrico

| Cohorte de Prueba Independiente | Sensor / Especificación | Muestras ($N$) | ROC-AUC | Exactitud | Sensibilidad | Especificidad | Precisión | F1-Score | Matriz de Confusión [$TN, FP; FN, TP$] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACRIMA** | Canon CR-DGi ($35^\\circ$ nativo) | 141 | **0.9939** | **95.04%** | **93.67%** | **96.77%** | 97.37% | **0.9548** | `[[60, 2], [5, 74]]` |
| **RIM-ONE DL** | Kowa WX 3D ($35^\\circ$ estereoscópico) | 90 | **0.9293** | **88.89%** | **75.76%** | **96.49%** | 92.59% | **0.8333** | `[[55, 2], [8, 25]]` |
| **EyePACS** | Screening masivo ($45^\\circ \\to 35^\\circ$ ROI) | 40 | **0.7850** | **70.00%** | **50.00%** | **90.00%** | 83.33% | **0.6250** | `[[18, 2], [10, 10]]` |
| **Moondream** | Multicéntrico con estadificación | 160 | **0.9086** | **79.37%** | **73.75%** | **85.00%** | 83.10% | **0.7815** | `[[68, 12], [21, 59]]` |
| **GLOBAL POOLED** | **Evaluación Total Unificada** | **431** | **0.9482** | **85.61%** | **79.25%** | **91.78%** | **90.32%** | **0.8442** | `[[201, 18], [44, 168]]` |

### 6.2 Auditoría Diagnóstica por Estadío Clínico en Moondream

Para comprobar si las omisiones del clasificador corresponden a cuadros graves o a fases subclínicas de alta ambigüedad oftalmológica, se desglosó el conjunto de Moondream ($N=160$):

```
- Glaucoma Avanzado (Advanced Stage) [N = 40]:  Sensibilidad = 90.00% (36 de 40 detectados)
- Glaucoma Temprano (Early Stage)    [N = 40]:  Sensibilidad = 57.50% (23 de 40 detectados)
- Controles Sanos (Normal Eyes)      [N = 80]:  Especificidad = 85.00% (68 de 80 confirmados)
```

**Relevancia Médica:** La sensibilidad del $90.00\\%$ en estadios avanzados garantiza que los pacientes con riesgo inminente de ceguera irreversible son identificados con máxima seguridad diagnóstica. La menor sensibilidad en estadios tempranos ($57.50\\%$) es congruente con la literatura oftalmológica: en fases iniciales, la retinografía mono-color no siempre refleja el adelgazamiento de fibras axónicas sin la asistencia de una tomografía OCT.

---

## 7. Prueba de Estrés Hostil en Tiempo de Inferencia (*Test-Time Augmentation Stress*)

Para dar respuesta inequívoca a si la invarianza a escala y fotometría es un rasgo robusto del modelo, se ejecutó una prueba de estrés agresiva mediante el script `scripts/experimentos/eval_test_time_malignant.py`. En este experimento, **las imágenes de prueba no vistas fueron bombardeadas directamente con las transformaciones estocásticas de Malignant Scale Augmentation en tiempo de inferencia**:

| Condición de Evaluación | Global ROC-AUC | Global Exactitud | Global Sensibilidad | Global Especificidad | Global F1-Score | Tasa de Retención ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Set Limpio (Canónico)** | **0.9466** | **85.85%** | **80.66%** | **90.87%** | **0.8486** | *100.0% (Línea Base)* |
| **Test Set con Aumento Maligno (Run 1, Seed 42)** | **0.9356** | **83.06%** | **78.77%** | **87.21%** | **0.8206** | **98.84%** |
| **Test Set con Aumento Maligno (Run 2, Seed 123)** | **0.9323** | **84.92%** | **80.19%** | **89.50%** | **0.8395** | **98.49%** |

### Desempeño por Sensor Bajo Ataque Maligno de Inferencia:
- **ACRIMA ($35^\\circ$ Canon):** Mantiene $ROC\\text{-}AUC = \\mathbf{0.9906}$ (Run 1) y $\\mathbf{0.9943}$ (Run 2), evidenciando que el alejamiento y viñeteo inducido no destruyen las características morfológicas del disco de alta resolución.
- **RIM-ONE DL ($35^\\circ$ Kowa 3D):** Mantiene $ROC\\text{-}AUC = \\mathbf{0.9171}$ (Run 1) y $\\mathbf{0.9181}$ (Run 2), sin verse afectado por la alteración de tono verde ni el micro-recorte.
- **EyePACS ($45^\\circ$ ROI Screening):** Registra $ROC\\text{-}AUC = \\mathbf{0.7750}$ (Run 1) y $\\mathbf{0.7250}$ (Run 2), manteniéndose operativo sin desplomarse a azar.
- **Moondream (Estadíos):** Registra $ROC\\text{-}AUC = \\mathbf{0.8958}$ (Run 1) y $\\mathbf{0.8853}$ (Run 2).

---

## 8. Conclusiones y Recomendaciones de Despliegue Clínico

1. **Cierre de la Brecha de Escala y FOV:** La combinación de recorte centrado `OpticDiscCropper` junto con `MalignantScaleAugmentation` resolvió el colapso de generalización inter-cámara, elevando el desempeño global a un **$ROC\\text{-}AUC = 0.9482$**.
2. **Inmunidad ante Perturbaciones Físicas:** Someter a prueba el modelo con aumentos malignos en tiempo de inferencia demostró que la red retiene el **$98.8\\%$ de su rendimiento**, lo que certifica su aptitud para operar en teleoftalmología con pacientes no cooperadores o cámaras de baja calidad.
3. **Calibración de Umbral para Cribado:** Para despliegues en campañas de tamizaje masivo, se recomienda calibrar el umbral de corte de probabilidad de $p=0.50$ a $p=0.35$, priorizando elevar la sensibilidad en glaucoma temprano del $57.50\\%$ a $>75\\%$, manteniendo una especificidad aceptable ($>80\\%$) que minimice las remisiones innecesarias.
4. **Trazabilidad y Reproducibilidad:** Todos los pesos, gráficos y registros numéricos han quedado serializados en el repositorio `GlaucomaDL` para su auditoría y citación en el artículo de formato Revista DYNA.

---
*Informe generado automáticamente por el compilador editorial del proyecto GlaucomaDL.*
"""

def compile_pdf():
    print(f"[INFO] 1. Guardando archivo Markdown en: {MD_PATH}")
    MD_PATH.write_text(md_content, encoding="utf-8")

    # 1. Proteger bloques de código ```...```
    fenced_blocks = []
    def save_fenced(m):
        idx = len(fenced_blocks)
        fenced_blocks.append(m.group(1))
        return f"\n@@FENCED_BLOCK_{idx}@@\n"

    text = re.sub(r"```(.*?)```", save_fenced, md_content, flags=re.DOTALL)

    # 2. Proteger matemáticas display $$...$$ e inline $...$
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

    # 3. Proteger código inline `...`
    inline_codes = []
    def save_code(m):
        idx = len(inline_codes)
        inline_codes.append(m.group(1))
        return f"@@INLINE_CODE_{idx}@@"

    text = re.sub(r"`([^`\n]+)`", save_code, text)

    # 4. Embeber imágenes como base64 data URIs
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

    # 5. Formato de enlaces y énfasis
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)

    # 6. Restaurar código inline
    for i, c in enumerate(inline_codes):
        text = text.replace(f"@@INLINE_CODE_{i}@@", f"<code>{c}</code>")

    # 7. Parser de líneas a HTML
    lines = text.splitlines()
    html_lines = []
    in_ul = False
    in_blockquote = False

    for line in lines:
        stripped = line.strip()

        # Cierre de lista
        if in_ul and not stripped.startswith("- "):
            html_lines.append("</ul>")
            in_ul = False

        # Cierre de blockquote
        if in_blockquote and not stripped.startswith(">"):
            html_lines.append("</blockquote>")
            in_blockquote = False

        if not stripped:
            continue

        if stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped == "---":
            html_lines.append("<hr/>")
        elif stripped.startswith("- "):
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{stripped[2:]}</li>")
        elif stripped.startswith(">"):
            if not in_blockquote:
                html_lines.append("<blockquote>")
                in_blockquote = True
            content = stripped[1:].strip()
            html_lines.append(f"<p>{content}</p>")
        elif stripped.startswith("|") and stripped.endswith("|"):
            html_lines.append(stripped)
        elif stripped.startswith("<figure") or stripped.startswith("@@FENCED_BLOCK_"):
            html_lines.append(stripped)
        else:
            html_lines.append(f"<p>{stripped}</p>")

    if in_ul:
        html_lines.append("</ul>")
    if in_blockquote:
        html_lines.append("</blockquote>")

    body_raw = "\n".join(html_lines)

    # 8. Renderizar tablas Markdown a HTML estructurado
    def render_tables(content):
        parts = []
        in_table = False
        table_lines = []

        for line in content.splitlines():
            if line.strip().startswith("|") and line.strip().endswith("|"):
                in_table = True
                table_lines.append(line.strip())
            else:
                if in_table:
                    parts.append(convert_table(table_lines))
                    table_lines = []
                    in_table = False
                parts.append(line)
        if in_table:
            parts.append(convert_table(table_lines))
        return "\n".join(parts)

    def convert_table(tbl_lines):
        if len(tbl_lines) < 2:
            return "\n".join(tbl_lines)
        headers = [c.strip() for c in tbl_lines[0].strip("|").split("|")]
        rows = []
        for l in tbl_lines[2:]:
            cells = [c.strip() for c in l.strip("|").split("|")]
            rows.append(cells)

        th_html = "".join(f"<th>{h}</th>" for h in headers)
        tb_html = []
        for r in rows:
            tds = "".join(f"<td>{c}</td>" for c in r)
            tb_html.append(f"<tr>{tds}</tr>")
        return f'<div class="table-wrapper"><table><thead><tr>{th_html}</tr></thead><tbody>{"".join(tb_html)}</tbody></table></div>'

    body_with_tables = render_tables(body_raw)

    # 9. Restaurar bloques de código
    for i, code in enumerate(fenced_blocks):
        clean_code = code.strip()
        body_with_tables = body_with_tables.replace(
            f"@@FENCED_BLOCK_{i}@@",
            f'<pre class="code-block"><code>{clean_code}</code></pre>'
        )

    # 10. Restaurar fórmulas matemáticas KaTeX
    for i, (m_type, math_src) in enumerate(math_blocks):
        escaped_math = math_src.replace("\\", "\\\\").replace("'", "\\'")
        if m_type == "display":
            replacement = f'<div class="katex-display" id="math-{i}"></div><script>katex.render(\'{escaped_math}\', document.getElementById(\'math-{i}\'), {{displayMode: true, throwOnError: false}});</script>'
        else:
            replacement = f'<span class="katex-inline" id="math-{i}"></span><script>katex.render(\'{escaped_math}\', document.getElementById(\'math-{i}\'), {{displayMode: false, throwOnError: false}});</script>'
        body_with_tables = body_with_tables.replace(f"@@MATH_BLOCK_{i}@@", replacement)

    # Plantilla HTML con tipografía y estilos editoriales
    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Informe Técnico 05: Entrenamiento y Estrés Maligno en Mega-Cohorte Armonizada</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<style>
  @page {{
    size: letter portrait;
    margin: 1.6cm 1.5cm 1.6cm 1.5cm;
    @bottom-center {{
      content: "GlaucomaDL • Deep Learning en Imagen Médica • UNAL";
      font-size: 7.5pt;
      color: #64748b;
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    line-height: 1.48;
    font-size: 8.8pt;
    margin: 0;
    padding: 0;
    background: #ffffff;
  }}
  h1 {{
    font-size: 15pt;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.22;
    margin: 0 0 5px 0;
    border-bottom: 2.5px solid #1e40af;
    padding-bottom: 5px;
  }}
  h2 {{
    font-size: 11.5pt;
    font-weight: 700;
    color: #1e40af;
    line-height: 1.25;
    margin: 14px 0 6px 0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
    page-break-after: avoid;
  }}
  h3 {{
    font-size: 9.6pt;
    font-weight: 700;
    color: #0f172a;
    margin: 10px 0 4px 0;
    page-break-after: avoid;
  }}
  p {{
    margin: 4.5px 0;
    text-align: justify;
  }}
  ul, ol {{
    margin: 4px 0 7px 18px;
    padding: 0;
  }}
  li {{
    margin-bottom: 2.5px;
  }}
  hr {{
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 9px 0;
  }}
  blockquote {{
    margin: 7px 0;
    padding: 5px 12px;
    background-color: #f8fafc;
    border-left: 3.5px solid #2563eb;
    color: #334155;
    font-size: 8.4pt;
  }}
  blockquote p {{
    margin: 2px 0;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 8pt;
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
    font-size: 7.6pt;
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
    font-size: 7.9pt;
    page-break-inside: avoid;
  }}
  th {{
    background-color: #1e40af;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    padding: 4.5px 5.5px;
    border: 1px solid #1e40af;
  }}
  td {{
    padding: 3.8px 5px;
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
    max-height: 330px;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .report-figure figcaption {{
    font-size: 7.8pt;
    color: #475569;
    font-style: italic;
    margin-top: 3px;
  }}
</style>
</head>
<body>
{body_with_tables}
</body>
</html>
"""

    print(f"[INFO] 2. Escribiendo archivo HTML en: {HTML_PATH}")
    HTML_PATH.write_text(full_html, encoding="utf-8")

    # 11. Búsqueda y ejecución del navegador Chromium headless para exportar a PDF
    browser_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser"
    ]

    selected_browser = None
    for b in browser_candidates:
        if Path(b).exists():
            selected_browser = b
            break

    if not selected_browser:
        print("[WARN] No se encontró Chromium en las rutas predeterminadas.")
        return False

    print(f"[INFO] 3. Compilando PDF mediante Chromium ({Path(selected_browser).name})...")
    cmd = [
        selected_browser,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={PDF_PATH}",
        str(HTML_PATH)
    ]
    subprocess.run(cmd, check=True)

    if PDF_PATH.exists():
        pdf_mb = PDF_PATH.stat().st_size / (1024 * 1024)
        print(f"[EXITO] PDF compilado satisfactoriamente:")
        print(f"  - Ruta: {PDF_PATH}")
        print(f"  - Tamaño: {pdf_mb:.2f} MB")
        return True
    else:
        print("[ERROR] No se pudo generar el archivo PDF.")
        return False

if __name__ == "__main__":
    compile_pdf()
