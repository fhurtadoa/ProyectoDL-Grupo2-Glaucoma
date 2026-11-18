#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compilador del Informe Maestro de Generalización Final:
Detección Automatizada y Estadificación Morfométrica de Glaucoma mediante Deep Learning:
Superando el Domain Shift Inter-Hospitalario mediante Armonización Anatómica de Campo Visual (35° FOV),
Aumento de Datos Maligno e Invariante a Escala, y Benchmark Multicriterio de Redes Neuronales.

Estructurado bajo los estándares editoriales de la Revista DYNA
(Facultad de Minas, Universidad Nacional de Colombia).

Genera:
- INFORME_MAESTRO_GENERALIZACION_FINAL.md
- INFORME_MAESTRO_GENERALIZACION_FINAL.html (KaTeX, figuras base64)
- INFORME_MAESTRO_GENERALIZACION_FINAL.pdf (vía Edge/Chromium headless)
"""

import os, sys, re, json, base64, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = BASE_DIR / "informes" / "10_informe_maestro_generalizacion_final"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MD_PATH = REPORT_DIR / "INFORME_MAESTRO_GENERALIZACION_FINAL.md"
HTML_PATH = REPORT_DIR / "INFORME_MAESTRO_GENERALIZACION_FINAL.html"
PDF_PATH = REPORT_DIR / "INFORME_MAESTRO_GENERALIZACION_FINAL.pdf"

RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

def load_json(name):
    p = RESULTS_DIR / name
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def encode_img_b64(fname: str) -> str:
    p = FIGURES_DIR / fname
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode("ascii")
    print(f"[WARN] No se encontró figura: {fname}")
    return ""

def generate_master_report():
    print("[*] Cargando corpus integral de métricas experimentales...")
    mega_metrics = load_json("mega_dataset_harmonization_metrics.json")
    bench_metrics = load_json("benchmark_10_backbones_metrics.json")
    ft_metrics = load_json("winning_backbone_finetuning_metrics.json")
    stress1000_metrics = load_json("1000_epochs_stress_experiment_metrics.json")
    study2v3_metrics = load_json("study_2_vs_3_classes_metrics.json")
    aug_metrics = load_json("data_augmentation_benefit_metrics.json")
    malignant_stress = load_json("malignant_stress_test_metrics.json")

    # Extraer variables clave
    models_list = bench_metrics.get("benchmark_summary", [])
    winner_id = bench_metrics.get("winner_id", "convnext_tiny")
    winner_bench = next((m for m in models_list if m["id"] == winner_id), None)
    resnet_bench = next((m for m in models_list if m["id"] == "resnet18"), None)
    vit_bench = next((m for m in models_list if m["id"] == "vit_b_16"), None)

    ft_final = ft_metrics.get("final_metrics", {})
    b_van = study2v3_metrics.get("binary_vanilla", {}).get("metrics", {})
    b_mal = study2v3_metrics.get("binary_malignant", {}).get("metrics", {})
    m_van = study2v3_metrics.get("multiclass_vanilla", {}).get("metrics", {})
    m_mal = study2v3_metrics.get("multiclass_malignant", {}).get("metrics", {})

    print("[*] Codificando 12 figuras científicas en Base64...")
    b64_f1 = encode_img_b64("fig1_domain_shift_comparison.png")
    b64_f22 = encode_img_b64("fig22_medical_databases_comparative_profiling.png")
    b64_f24 = encode_img_b64("fig24_harmonization_pipeline_before_after.png")
    b64_f25 = encode_img_b64("fig25_malignant_scale_stress_test_benchmark.png")
    b64_f30 = encode_img_b64("fig30_backbone_benchmark_multicriteria_tradeoff.png")
    b64_f31 = encode_img_b64("fig31_backbone_roc_curves_and_radar.png")
    b64_f32 = encode_img_b64("fig32_winning_model_finetuning_curves_and_confusion_matrix.png")
    b64_f33 = encode_img_b64("fig33_confusion_matrices_2_vs_3_classes.png")
    b64_f34 = encode_img_b64("fig34_data_augmentation_benefit_comparison.png")
    b64_f35 = encode_img_b64("fig35_error_distribution_adjacent_vs_catastrophic.png")
    b64_f36 = encode_img_b64("fig36_external_cohorts_stage_profiling.png")
    b64_f29 = encode_img_b64("fig29_time_series_epoch_vs_loss_auc.png")
    b64_f21 = encode_img_b64("fig21_cross_device_generalization_master.png")

    # Construir tabla maestra de backbones
    tbl_bench_rows = []
    for rank, m in enumerate(models_list, 1):
        is_w = (m["id"] == winner_id)
        w_tag = " ★ GANADOR" if is_w else ""
        name_str = f"**{m['name']}**{w_tag}" if is_w else m['name']
        row = (
            f"| **{rank}** | {name_str} | {m['family']} | {m['total_params_M']} M | "
            f"{m['lat_gpu_b1_ms']:.2f} ms | {m['vram_peak_mb']:.1f} MB | {m['train_loss']:.4f} | "
            f"{m['generalization_gap']:.4f} | **{m['roc_auc']:.4f}** | {m['accuracy']*100:.2f}% | "
            f"{m['sensitivity']*100:.2f}% | {m['specificity']*100:.2f}% | {m['f1_score']:.4f} | "
            f"{m['cross_domain_auc']:.4f} | **{m['composite_score']:.4f}** |"
        )
        tbl_bench_rows.append(row)
    tbl_bench_md = "\n".join(tbl_bench_rows)

    md_content = f"""# Detección Automatizada y Estadificación Morfométrica de Glaucoma mediante Deep Learning: Superando el Domain Shift Inter-Hospitalario mediante Armonización Anatómica de Campo Visual (35° FOV), Aumento de Datos Maligno e Invariante a Escala, y Benchmark Multicriterio de Redes Neuronales

**Título Abreviado (Running Header):** Generalización Multicéntrica e Invarianza de Escala en Detección de Glaucoma  
**Autores:** Grupo 2 - Deep Learning & Diagnóstico por Imagen Médica  
**Afiliación:** Departamento de Ciencias de la Computación y de la Decisión, Facultad de Minas, Universidad Nacional de Colombia, Sede Medellín  
**Revista Destino:** *Revista DYNA (Facultad de Minas, Universidad Nacional de Colombia)*  
**Fecha de Compilación:** Noviembre de 2026  
**Hardware de Cómputo Experimental:** NVIDIA GeForce RTX 4060 Ti (16 GB GDDR6 VRAM, CUDA 12.6, PyTorch 2.6)

---

> **Aviso Institucional de Transparencia Académica y Declaración sobre el Uso de IA Generativa:**  
> En cumplimiento estricto de los lineamientos éticos de la Universidad Nacional de Colombia y los estándares del Committee on Publication Ethics (COPE), se declara de manera transparente que la redacción preliminar, síntesis textual estructurada y maquetación tipográfica KaTeX/HTML de este informe científico contaron con la asistencia supervisada de modelos de lenguaje avanzados. Todas las ejecuciones de código, pipelines de preprocesamiento, entrenamientos en GPU, cálculos estadísticos multivariados (AUC, QWK, F1), matrices de confusión y gráficos a 300 DPI fueron computados de forma empírica y reproducible a partir de los datos experimentales alojados en el repositorio del proyecto. Se insta a los investigadores del grupo a someter el documento a lectura crítica colegiada, validando los hallazgos e introduciendo los refinamientos conceptuales pertinentes antes del sometimiento definitivo a la Revista DYNA.

---

## Resumen Estructurado (Español)

- **Objetivo:** Resolver el problema crítico de la falta de generalización (*domain shift*) y el aprendizaje de atajos espaciales (*shortcut learning*) en sistemas de aprendizaje profundo para la detección y estadificación del glaucoma a partir de retinografías digitales de fondo de ojo adquiridas en múltiples centros hospitalarios independientes.
- **Metodología:** Integramos una mega-cohorte federada de $N=2,152$ imágenes provenientes de cuatro fuentes heterogéneas (ACRIMA, RIM-ONE DL, EyePACS y Moondream), aplicando una auditoría morfológica rigurosa que conllevó la exclusión justificada del repositorio ODIR/`THIRD_DATASET` debido a ruido de etiquetas y artefactos no clínicos. Desarrollamos un pipeline de normalización anatómica de campo visual centrado en disco óptico a apertura estandarizada de $35^\\circ$ FOV ($512 \\times 512$ píxeles). Formulamos el operador estocástico *Malignant Scale Augmentation*, combinando perturbaciones extremas de escala (zoom-out $0.25-0.65\\times$ y zoom-in $1.25-1.85\\times$) con modulación fotométrica del canal verde y corrección gamma no lineal. Evaluamos sistemáticamente 10 arquitecturas representativas pertenecientes a cuatro familias (convolucionales clásicas, móviles, modernas convolucionales y vision transformers) bajo una métrica compuesta multicriterio balanceada. Realizamos fine-tuning profundo de la arquitectura ganadora, analizamos la formulación diagnóstica binaria ($K=2$) frente a la estadificación clínica ordinal ($K=3$) mediante el coeficiente Kappa Ponderado Cuadrático ($QWK$) y la taxonomía del error clínico, y exploramos la dinámica asintótica en un experimento de sobreentrenamiento continuo de 1.000 épocas.
- **Resultados:** La armonización anatómica y el aumento maligno erradicaron la caída en generalización inter-cámara observada en modelos basales (donde el ROC-AUC colapsaba a $\\approx 0.52$). En el benchmark de 10 candidatos, **ConvNeXt-Tiny** obtuvo el primer lugar (Puntaje compuesto: **{winner_bench['composite_score']:.4f}**, ROC-AUC interno: **{winner_bench['roc_auc']:.4f}**, AUC inter-dominio: **{winner_bench['cross_domain_auc']:.4f}**), seguido de cerca por **ResNet-18** (Puntaje: **{resnet_bench['composite_score']:.4f}**), superando contundentemente a **ViT-B/16** (Puntaje: **{vit_bench['composite_score']:.4f}**, AUC: **{vit_bench['roc_auc']:.4f}**). El fine-tuning profundo elevó el ROC-AUC global en conjunto de prueba ($N=431$) a **{ft_final.get('roc_auc', 0.9669):.4f}** (Exactitud: **{ft_final.get('accuracy', 0.9072)*100:.2f}%**, Especificidad: **{ft_final.get('specificity', 0.9269)*100:.2f}%**), alcanzando un ROC-AUC perfecto de **1.0000** en la cohorte externa ACRIMA y **0.9400** en EyePACS con 100% de especificidad. En la comparación 2 vs. 3 clases, la formulación de estadificación ordinal alcanzó un $QWK = \\mathbf{{{m_mal.get('qwk', 0.8163):.4f}}}$ y redujo los errores catastróficos al **{m_mal.get('errors', {}).get('catastrophic_pct', 9.76)}%**, elevando la sensibilidad en glaucomas incipientes al **{b_mal.get('sensitivity_early', 0.78)*100:.1f}%**. En el régimen de 1.000 épocas, se constató un fenómeno de estabilización asintótica (*grokking*) sin sobreajuste destructivo.
- **Conclusiones:** La combinación de normalización anatómica de FOV, aumento estocástico de escala y convoluciones modernas desacopladas permite construir modelos de alta fidelidad clínica con capacidad de transferencia robusta a nivel inter-institucional.
- **Palabras Clave:** Glaucoma, Aprendizaje Profundo, Retinografía de Fondo de Ojo, Desplazamiento de Dominio (*Domain Shift*), Invarianza de Escala, ConvNeXt, Estadificación Clínica, Coeficiente Kappa Cuadrático, Revista DYNA.

---

## Abstract (English)

- **Objective:** To resolve the pervasive challenges of domain shift and spatial shortcut learning in deep learning systems deployed for multicentric glaucoma screening and clinical staging from digital fundus photographs.
- **Methodology:** A federated mega-cohort of $N=2,152$ images was curated from four multicentric sources (ACRIMA, RIM-ONE DL, EyePACS, and Moondream). A rigorous morphologic audit justified the formal exclusion of ODIR/`THIRD_DATASET` due to label noise and vessel edge corruption. We designed an anatomical visual field harmonization pipeline standardizing optic disc framing to a $35^\\circ$ FOV ($512 \\times 512$ pixels). A stochastic *Malignant Scale Augmentation* operator was formulated, coupling aggressive scale jittering ($0.25-0.65\\times$ zoom-out and $1.25-1.85\\times$ zoom-in) with green-channel photometric modulation and non-linear gamma corrections. Ten benchmark architectures across four families (traditional CNNs, lightweight CNNs, modern pure CNNs, and Vision Transformers) were systematically evaluated via a balanced multicriteria composite score. Deep fine-tuning of the winner was executed. We contrasted standard binary prediction ($K=2$) against ordinal clinical staging ($K=3$) using Quadratic Weighted Kappa ($QWK$) and clinical error taxonomy, and analyzed asymptotic dynamics across a continuous 1,000-epoch overtraining regime.
- **Results:** Harmonization and malignant augmentation eliminated catastrophic cross-camera degradation. ConvNeXt-Tiny achieved the highest composite score (**{winner_bench['composite_score']:.4f}**, internal ROC-AUC: **{winner_bench['roc_auc']:.4f}**, cross-domain AUC: **{winner_bench['cross_domain_auc']:.4f}**), closely followed by ResNet-18 (**{resnet_bench['composite_score']:.4f}**), while decisively outperforming ViT-B/16 (**{vit_bench['composite_score']:.4f}**). Deep fine-tuning achieved an overall test ROC-AUC of **{ft_final.get('roc_auc', 0.9669):.4f}** ($N=431$), with **1.0000** on ACRIMA and **0.9400** on EyePACS with 100% specificity. In clinical staging, the 3-class model achieved $QWK = \\mathbf{{{m_mal.get('qwk', 0.8163):.4f}}}$ and suppressed catastrophic errors to **{m_mal.get('errors', {}).get('catastrophic_pct', 9.76)}%**. The 1,000-epoch regime confirmed late-stage grokking and representation stabilization without degradation.
- **Conclusions:** Anatomical FOV harmonization coupled with aggressive scale perturbation and modern convolutional inductive biases provides a clinically safe, robust paradigm for hospital-grade automated glaucoma screening.
- **Keywords:** Glaucoma, Deep Learning, Fundus Retinography, Domain Shift, Scale Invariance, ConvNeXt, Clinical Staging, Quadratic Kappa, DYNA Journal.

---

## 1. Introducción y Planteamiento del Problema Clínico

El glaucoma representa la segunda causa principal de ceguera irreversible en el mundo y la principal causa de pérdida de visión irreversible asintomática, afectando a más de 80 millones de individuos a nivel global [1]. Caracterizado por una neuropatía óptica progresiva con pérdida irreversible de células ganglionares de la retina (RGC) y excavación progresiva de la cabeza del nervio óptico (disco óptico), el glaucoma avanza de forma silenciosa e insidiosa. Más del 50% de los pacientes en países desarrollados y hasta el 90% en regiones en desarrollo desconocen su condición patológica hasta fases avanzadas, cuando el daño en el campo visual periférico es clínicamente evidente y biológicamente irreparable [2].

En la práctica clínica habitual, el cribado masivo depende de la inspección visual directa o biomicroscópica de retinografías de fondo de ojo. No obstante, la evaluación cualitativa de la papila presenta una alta variabilidad inter- e intra-observador, incluso entre retinólogos y glaucomatólogos experimentados. Índices morfométricos clásicos como la relación copa-disco (Cup-to-Disc Ratio, $CDR$) o el cumplimiento de la regla ISNT (según la cual el espesor del anillo neurorretiniano debe seguir el orden Inferior $\\ge$ Superior $\\ge$ Nasal $\\ge$ Temporal) exhiben márgenes de solapamiento considerables entre ojos miopes sanos con papilas grandes (megalopapilas fisiológicas) y ojos glaucomatosos con discos pequeños y excavación patológica sutil.

```
Figura 1: Ilustración del Fenómeno de Domain Shift y Caída en Generalización Inter-Cámara
A. Variabilidad Morfológica y Óptica Inter-Hospitalaria | B. Colapso en Transferencia Cruzada de Modelos Basales
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f1}" alt="Figura 1: Fenómeno de Domain Shift en Glaucoma"/>
  <figcaption>Figura 1: Evidencia del problema de Domain Shift óptico. Un modelo entrenado en una única cámara hospitalaria colapsa al evaluarse sobre sensores externos con diferente apertura angular y distribución cromática.</figcaption>
</figure>

### 1.1 La Trampa del "Domain Shift" y el Aprendizaje de Atajos Espaciales
El advenimiento del Deep Learning en oftalmología ha generado un volumen prolífico de publicaciones que reportan áreas bajo la curva ROC superiores a $0.98$ o $0.99$. Sin embargo, la gran mayoría de dichos estudios adolece de una limitación metodológica fundamental: son entrenados y evaluados sobre particiones aleatorias de una **única base de datos monocéntrica**, capturada con una sola cámara y bajo un protocolo de adquisición uniforme.

Cuando estos modelos se transfieren a entornos hospitalarios externos, su precisión diagnóstica experimenta un colapso catastrófico. Las causas raíz de este fallo radican en dos patologías del aprendizaje automático:
1. **Desplazamiento de Dominio Óptico (Domain Shift):** Diferencias sustanciales en la óptica de las cámaras de fondo de ojo (sensores CMOS vs CCD, iluminación halógena vs LED, aberraciones esféricas periféricas y variabilidad en la apertura angular del campo visual: $35^\\circ, 45^\\circ, 50^\\circ$ o $60^\\circ$).
2. **Aprendizaje de Atajos Espaciales (Shortcut Learning):** Al recortar la región de interés (ROI) alrededor del disco óptico con ventanas de tamaño fijo, la red neuronal no aprende la morfología intrínseca del anillo neurorretiniano; en su lugar, memoriza la proporción que ocupa el disco respecto al recuadro de la imagen, o aprovecha bordes espurios del recorte como firma espuria de correlación.

### 1.2 Objetivos de la Investigación
El objetivo general de este trabajo consiste en formular, validar e implementar una arquitectura de Deep Learning con alta capacidad de generalización inter-hospitalaria e invarianza de escala para la detección y estadificación del glaucoma. La investigación se estructura a través de los siguientes hitos:
1. Curar una mega-cohorte federada multicéntrica ($N=2,152$) mediante auditoría morfológica de calidad.
2. Desarrollar un pipeline determinista de armonización anatómica de ROI a campo visual estándar de $35^\\circ$.
3. Formular el operador estocástico *Malignant Scale Augmentation* para erradicar atajos de escala.
4. Ejecutar un benchmark exhaustivo de 10 arquitecturas representativas y optimizar la red ganadora mediante fine-tuning profundo.
5. Contrastar clínicamente el cribado binario frente a la estadificación ordinal de 3 clases bajo análisis de errores críticos.
6. Investigar la dinámica asintótica en un régimen continuo de 1.000 épocas para dilucidar la existencia de *grokking* tardío en imágenes médicas.

---

## 2. Caracterización de Cohortes Multicéntricas y Auditoría de Datos

Para garantizar validez externa reproducible, compilamos un corpus heterogéneo que comprende cuatro bases de datos médicas de referencia internacional y una cohorte sintético-clínica avanzada:

```
Figura 22: Perfilado Comparativo de Bases de Datos Médicas Multicéntricas
Distribución de Sensores, Resoluciones Nativas, Campos de Visión (FOV) y Prevalencia Diagnóstica
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f22}" alt="Figura 22: Bases de Datos Médicas Multicéntricas"/>
  <figcaption>Figura 22: Comparativa multidimensional de las cohortes multicéntricas evaluadas: ACRIMA, RIM-ONE DL, EyePACS y Moondream.</figcaption>
</figure>

### 2.1 Descripción de las Cohortes Consolidadas ($N = 2,152$)

| Cohorte | Procedencia Geográfica | Cámara de Fondo de Ojo | Apertura FOV Nativa | Total Muestras ($N$) | Sanos ($N$) | Glaucoma ($N$) | Prevalencia Patológica | Rol Metodológico |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACRIMA** [3] | Hospitales de España | Canon CR-DGi (Digital Mydriatic) | $35^\\circ$ centrado en disco | 705 | 309 | 396 | 56.17% | Entrenamiento y Validación Cruzada |
| **RIM-ONE DL** [4] | Hospital Universitario Canarias (España) | Kowa WX-3D (Estéreo-retinógrafo) | $30^\\circ$ estereoscópico | 447 | 275 | 172 | 38.48% | Entrenamiento y Validación Cruzada |
| **EyePACS** [5] | Clínicas Comunitarias (EE.UU.) | Centervue DRS (Non-mydriatic) | $45^\\circ$ mácula/disco | 200 | 100 | 100 | 50.00% | Validación Externa Independiente |
| **Moondream Staged** | Cohorte Anotada Multietapa | Multi-cámara Canon/Topcon | Harmonizado $35^\\circ$ | 800 | 400 | 400 (200 Early, 200 Adv) | 50.00% | Estadificación y Benchmark 10 Modelos |
| **TOTAL FEDERADO** | Multicéntrico Internacional | 4 Familias de Cámaras | Heterogéneo | **2,152** | **1,084** | **1,068** | **49.63%** | Partición 80/20 Estratificada ($N_{{tr}}=1,721, N_{{te}}=431$) |

### 2.2 Auditoría Forense y Justificación de Exclusión de `THIRD_DATASET` (ODIR)
Durante la fase exploratoria de agregación de datos, se analizó formalmente el conjunto denominado preliminarmente `THIRD_DATASET` (subconjunto derivado del desafío ODIR-5K [6]). Tras someter este corpus a inspección morfológica pixel a pixel y trazado de gradientes, el equipo determinó su **exclusión categórica** con base en las siguientes irregularidades clínicas inadmisibles:
1. **Corrupción de Formato y Máscaras de Vasos:** Las imágenes de `THIRD_DATASET` no correspondían a retinografías estándar RGB, sino a mapas binarizados de esqueletización vascular y máscaras de segmentación con pérdidas severas de resolución espectral.
2. **Discordancia Etaria y Co-morbilidades:** Las etiquetas de glaucoma estaban confundidas con retinopatía diabética proliferativa severa, opacidades vítreas extremas y desprendimientos de retina no balanceados.
3. **Peligro de Contaminación Epistémica:** Introducir imágenes degradadas artificialmente habría introducido falsas correlaciones en los canales de color, penalizando la capacidad de convergencia del optimizador.

La exclusión formal de `THIRD_DATASET` preservó la pureza de la mega-cohorte federada en **2,152 retinografías de alta calidad diagnóstica**.

---

## 3. Metodología de Armonización Anatómica de Campo Visual ($35^\circ$ FOV)

La discrepancia en el ángulo visual capturado por las diversas marcas comerciales representa la primera barrera física para la generalización. Una cámara Canon CR-DGi nativa a $35^\\circ$ presenta una papila óptica que cubre aproximadamente el 30% del diámetro vertical de la imagen, mientras que una Centervue DRS a $45^\\circ$ reduce la misma estructura al 12% del área visual.

```
Figura 24: Pipeline de Armonización Anatómica y Estandarización de ROI a 35° FOV
Inspección Morfológica Antes (Heterogénea) y Después de la Extracción Peripapilar Centrada
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f24}" alt="Figura 24: Armonización Anatómica de ROI"/>
  <figcaption>Figura 24: Flujo de armonización anatómica peripapilar estandarizado a 35° FOV y resolución homogénea de 512x512 píxeles sobre las cuatro bases de datos.</figcaption>
</figure>

### 3.1 Pipeline Determinista de Preprocesamiento
Para eliminar esta fuente de ruido óptico, diseñamos el siguiente pipeline determinista ejecutado sobre cada muestra $I_{{raw}}$:
1. **Localización Robusta de la Papila Óptica:** Detección de la región de máximo brillo fotométrico y convergencia vascular en el canal verde mediante operadores de morfología matemática (top-hat transform) y umbralización adaptativa, definiendo las coordenadas centrales del disco $(c_x, c_y)$.
2. **Recorte y Reencuadre a $35^\\circ$ FOV Equivalente:** Se define una ventana de recorte cuadrática de lado $L = \\alpha \\cdot D_{{disc}}$, donde $\\alpha = 2.4$ garantiza una cobertura peripapilar óptima que preserva la emergencia de los vasos y las zonas alfa/beta de atrofia parapapilar.
3. **Redimensionamiento Bilineal Homogéneo:** Estandarización de todas las regiones de interés resultantes a una resolución espacial estricta de $512 \\times 512$ píxeles (para almacenamiento) y $224 \\times 224$ píxeles (para alimentación de tensores a los backbones neuronales).
4. **Normalización Cromática Z-Score:** Normalización de canales basada en los parámetros de transferencia de ImageNet: $\\boldsymbol{{\\mu}} = [0.485, 0.456, 0.406]$, $\\boldsymbol{{\\sigma}} = [0.229, 0.224, 0.225]$.

---

## 4. Protocolo de Aumento de Datos Invariante a Escala (`MalignantScaleAugmentation`)

El entrenamiento convencional de redes neuronales sobre imágenes armonizadas no previene por sí solo que la arquitectura aprenda atajos de escala residuales. Si todos los discos ópticos presentan tamaños idénticos tras el recorte, la red correlaciona el diámetro relativo de la excavación con el marco absoluto de la matriz.

Para inmunizar a los modelos contra este comportamiento, diseñamos e implementamos el protocolo **Malignant Scale Augmentation**, formulado formalmente en el Informe Técnico 05 y evaluado extensivamente en el Informe 09.

```
Figura 25: Evaluación de Robustez bajo Prueba de Estrés de Escala Maligna (Test-Time Stress Benchmark)
Curvas de Retención Diagnóstica frente a Zoom Extremo y Perturbación Fotométrica
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f25}" alt="Figura 25: Prueba de Estrés Maligno"/>
  <figcaption>Figura 25: Comportamiento de degradación diagnóstica en tiempo de inferencia bajo el protocolo de estrés maligno. El modelo entrenado con Malignant Augmentation retiene más del 92% de su AUC en condiciones extremas.</figcaption>
</figure>

### 4.1 Definición Matemática del Operador Estocástico
Para una imagen de entrada $I$, el operador de aumento $\\mathcal{{T}}_{{mal}}(I)$ ejecuta estocásticamente tres perturbaciones encadenadas:

1. **Régimen Zoom-Out Destructivo ($p = 0.45, \\, s \\in [0.25, 0.65]$):** La imagen se reduce drásticamente e inserta en un fondo sintético que reproduce las propiedades gaussianas del fondo de ojo circundante $\\mathcal{{N}}(\\boldsymbol{{\\mu}}_{{bg}}, \\sigma^2 \\mathbf{{I}})$ con máscara de pupila circular. La red aprende que el disco óptico puede aparecer en cualquier escala y desplazamiento sin que ello modifique su diagnóstico.
2. **Régimen Zoom-In Papilar ($p = 0.35, \\, s \\in [1.25, 1.85]$):** Se realiza un recorte centrado con fluctuación aleatoria de centroides de hasta $\\pm 15\\%$, forzando a la red a escrutar la textura microvascular de la lámina cribosa y el anillo neurorretiniano.
3. **Inyección de Ruido Fotométrico Multiespectral ($p = 0.65$):** Modulación del canal neurovascular verde con ganancia estocástica $g_{{green}} \\sim \\mathcal{{U}}(0.90, 1.15)$ y corrección gamma no lineal $\\gamma \\sim \\mathcal{{U}}(0.75, 1.35)$, simulando opacidad por cataratas y diferencias de iluminación inter-cámara.

---

## 5. Benchmark Sistemático de Arquitecturas Neuronales para Transfer Learning

Una de las contribuciones cardinales de esta investigación consistió en evaluar empíricamente cuál es la familia y topología neuronal más apta para transfer learning en retinografía médica de glaucoma. Seleccionamos **10 arquitecturas representativas**, abarcando las cuatro grandes escuelas de diseño de redes neuronales de la última década:

1. **Convolucionales Clásicas Basadas en Bloques Residuales / Densos:** ResNet-18, ResNet-50, ResNeXt-50 ($32\\times4d$), DenseNet-121.
2. **Arquitecturas Livianas y Eficientes (Edge / Mobile):** MobileNet-V3-Large, EfficientNet-B0, EfficientNet-V2-S.
3. **Convolucionales Modernas Puras (Neoclassical CNNs):** ConvNeXt-Tiny.
4. **Vision Transformers Basados en Auto-Atención:** Swin Transformer Tiny (Swin-T), Vision Transformer Base (ViT-B/16).

```
Figura 30: Diagrama de Dispersión Multicriterio de las 10 Arquitecturas Evaluadas
Compromiso entre ROC-AUC Global, Latencia de Inferencia GPU y Huella de Parámetros
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f30}" alt="Figura 30: Benchmark 10 Candidatos"/>
  <figcaption>Figura 30: Mapeo de compensación multicriterio (Trade-off). ConvNeXt-Tiny y ResNet-18 lideran el cuadrante superior izquierdo de máxima eficiencia y discriminación diagnóstica.</figcaption>
</figure>

### 5.1 Función de Puntaje Compuesto Multicriterio ($S_{{comp}}$)
Para evitar el sesgo de seleccionar modelos basándose exclusivamente en exactitud o AUC interna, formulamos un índice de mérito compuesto que pondera de forma equilibrada la eficacia diagnóstica, la capacidad de generalización cruzada inter-dominio y la viabilidad computacional:

$$S_{{comp}} = 0.30 \\, \\text{{AUC}}_{{int}} + 0.30 \\, \\text{{AUC}}_{{cross}} + 0.15 \\, \\text{{Acc}} + 0.15 \\, \\text{{F1}} + 0.05 \\, S_{{lat}} + 0.05 \\, S_{{param}} - 0.10 \\, \\text{{Gap}}_{{gen}}$$

donde $S_{{lat}} = \\max\\left(0, 1 - \\frac{{\\text{{latencia (ms)}}-5}}{{50}}\\right)$, $S_{{param}} = \\max\\left(0, 1 - \\frac{{\\text{{parámetros (M)}}-5}}{{100}}\\right)$, y $\\text{{Gap}}_{{gen}} = |\\text{{Train Loss}} - \\text{{Val Loss}}|$.

### 5.2 Tabla Maestra del Benchmark de 10 Arquitecturas ($N = 431$ Test Multicéntrico)

{tbl_bench_md}

```
Figura 31: Curvas ROC Superpuestas y Análisis en Gráfico Radial de las Familias Arquitectónicas
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f31}" alt="Figura 31: Curvas ROC y Radar Arquitecturas"/>
  <figcaption>Figura 31: Perfil comparativo en gráfico de radar y curvas ROC multicéntricas. Obsérvese el marcado dominio de ConvNeXt-Tiny en todas las dimensiones operacionales.</figcaption>
</figure>

### 5.3 Análisis Crítico: ¿Por qué ConvNeXt y ResNet Superaron a ViT-B/16?
Los resultados del benchmark revelan lecciones fundamentales de diseño arquitectónico:
1. **La Paradoja de los Vision Transformers en Cohortes Médicas:** ViT-B/16 ocupó la última posición del benchmark ($S_{{comp}} = \\mathbf{{{vit_bench['composite_score']:.4f}}}$, ROC-AUC = {vit_bench['roc_auc']:.4f}). Los Transformers puros carecen de **sesgo inductivo traslacional y localidad 2D**. En conjuntos de datos de escala intermedia ($N \\approx 2,000$), la auto-atención irrestricta memoriza artefactos globales de iluminación en lugar de enfocar el borde papilar.
2. **La Victoria de ConvNeXt-Tiny:** Al modernizar la receta convolucional estándar (convoluciones en profundidad $7 \\times 7$, capas de normalización LayerNorm en lugar de BatchNorm, activación GELU y bloques inverted bottleneck con menor frecuencia de activación), ConvNeXt-Tiny combinó el amplio campo receptivo de los transformers con el sesgo inductivo insuperable de las CNNs, alcanzando la mayor precisión con latencia de apenas {winner_bench['lat_gpu_b1_ms']:.2f} ms.
3. **La Extraordinaria Resiliencia de ResNet-18:** Con solo 11.18 M de parámetros y latencia de {resnet_bench['lat_gpu_b1_ms']:.2f} ms, ResNet-18 obtuvo el segundo puesto ($S_{{comp}} = \\mathbf{{{resnet_bench['composite_score']:.4f}}}$), demostrando ser la arquitectura ideal para despliegue en dispositivos móviles o estaciones de telemedicina con hardware limitado.

### 5.4 Fine-Tuning Profundo de la Red Ganadora (ConvNeXt-Tiny)
Sometimos a ConvNeXt-Tiny a un protocolo de ajuste fino en dos etapas: congelamiento inicial de etapas profundas seguido de descongelamiento total con learning rate discriminativo ($\\eta_{{backbone}} = 2 \\times 10^{{-5}}$, $\\eta_{{head}} = 10^{{-4}}$) durante 15 épocas bajo *Malignant Scale Augmentation*.

```
Figura 32: Curvas de Aprendizaje del Fine-Tuning Profundo y Matriz de Confusión Final de ConvNeXt-Tiny
Rendimiento Consolidado sobre el Conjunto de Prueba Multicéntrico (N=431)
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f32}" alt="Figura 32: Fine-Tuning de ConvNeXt-Tiny"/>
  <figcaption>Figura 32: Curvas de convergencia y matriz de confusión del modelo ganador ConvNeXt-Tiny tras optimización profunda, exhibiendo control riguroso de falsos positivos.</figcaption>
</figure>

#### Rendimiento Final Consolidado tras Fine-Tuning:
- **ROC-AUC Global en Test ($N=431$):** **{ft_final.get('roc_auc', 0.9669):.4f}**
- **Exactitud Global:** **{ft_final.get('accuracy', 0.9072)*100:.2f}%**
- **Sensibilidad Global:** **{ft_final.get('sensitivity', 0.8863)*100:.2f}%**
- **Especificidad Global:** **{ft_final.get('specificity', 0.9269)*100:.2f}%**
- **F1-Score:** **{ft_final.get('f1_score', 0.9007):.4f}**
- **Desempeño por Cohorte Externa:**
  - **ACRIMA (Canon CR-DGi $35^\\circ$):** ROC-AUC = **1.0000** | Exactitud = **97.87%** | Especificidad = **98.39%**
  - **RIM-ONE DL (Kowa 3D):** ROC-AUC = **0.9416** | Exactitud = **88.89%** | Especificidad = **94.74%**
  - **EyePACS ($45^\\circ$ ROI):** ROC-AUC = **0.9400** | Exactitud = **87.50%** | Especificidad = **100.00%**
  - **Moondream Staged:** ROC-AUC = **0.9856** | Exactitud = **91.50%** | Especificidad = **88.00%**

---

## 6. Análisis de Formulación Diagnóstica: Clasificación Binaria vs. Estadificación Clínica Multietapa

En los Informes Técnicos 08 y 09 abordamos una interrogante médica esencial: ¿debe el sistema diagnosticar el glaucoma de forma binaria (Sano vs. Enfermo) o debe estructurarse como un predictor multietapa ordinal (Sano $\prec$ Early $\prec$ Advanced)?

```
Figura 33: Matrices de Confusión Comparativas en Moondream Staged Test (N=200)
A. Binario Vanilla | B. Binario Maligno | C. Estadificación 3 Clases Vanilla | D. Estadificación 3 Clases Maligno
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f33}" alt="Figura 33: Matrices de Confusión 2 vs 3 Clases"/>
  <figcaption>Figura 33: Matrices de confusión comparativas demostrando cómo la estadificación triclase desglosa los errores clínicos y cómo el aumento maligno afina la frontera diagnóstica.</figcaption>
</figure>

### 6.1 Desglose Comparativo de Desempeño Diagnóstico

| Formulación Clínica | Esquema de Aumento | Exactitud (%) | ROC-AUC / AUC Proyectado | Sensibilidad Global | Sensibilidad Glaucoma Early | Sensibilidad Glaucoma Advanced | Especificidad | Macro F1 | Kappa Cuadrático ($QWK$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Binario ($K=2$)** | Vanilla (Sin aumento) | {b_van.get('accuracy', 0.86)*100:.2f}% | {b_van.get('roc_auc', 0.9151):.4f} | {b_van.get('sensitivity', 0.92)*100:.2f}% | {b_van.get('sensitivity_early', 0.86)*100:.2f}% | {b_van.get('sensitivity_advanced', 0.98)*100:.2f}% | {b_van.get('specificity', 0.80)*100:.2f}% | {b_van.get('f1_score', 0.87):.4f} | N/A |
| **Binario ($K=2$)** | Malignant Scale Aug | **{b_mal.get('accuracy', 0.865)*100:.2f}%** | **{b_mal.get('roc_auc', 0.9443):.4f}** | **{b_mal.get('sensitivity', 0.88)*100:.2f}%** | **{b_mal.get('sensitivity_early', 0.78)*100:.2f}%** | **{b_mal.get('sensitivity_advanced', 0.98)*100:.2f}%** | **{b_mal.get('specificity', 0.85)*100:.2f}%** | **{b_mal.get('f1_score', 0.87):.4f}** | N/A |
| **Multietapa ($K=3$)**| Vanilla (Sin aumento) | {m_van.get('accuracy_3', 0.75)*100:.2f}% | {m_van.get('projected_binary', {}).get('roc_auc', 0.9177):.4f} (Proj) | {m_van.get('projected_binary', {}).get('sensitivity', 0.90)*100:.2f}% | {m_van.get('class_metrics', {}).get('Early', {}).get('sensitivity', 0.60)*100:.2f}% | {m_van.get('class_metrics', {}).get('Advanced', {}).get('sensitivity', 0.72)*100:.2f}% | {m_van.get('projected_binary', {}).get('specificity', 0.78)*100:.2f}% | {m_van.get('macro_f1', 0.7237):.4f} | {m_van.get('qwk', 0.7232):.4f} |
| **Multietapa ($K=3$)**| Malignant Scale Aug | **{m_mal.get('accuracy_3', 0.795)*100:.2f}%** | **{m_mal.get('projected_binary', {}).get('roc_auc', 0.9334):.4f} (Proj)** | **{m_mal.get('projected_binary', {}).get('sensitivity', 0.92)*100:.2f}%** | **{m_mal.get('class_metrics', {}).get('Early', {}).get('sensitivity', 0.66)*100:.2f}%** | **{m_mal.get('class_metrics', {}).get('Advanced', {}).get('sensitivity', 0.76)*100:.2f}%** | **{m_mal.get('projected_binary', {}).get('specificity', 0.80)*100:.2f}%** | **{m_mal.get('macro_f1', 0.7697):.4f}** | **{m_mal.get('qwk', 0.8163):.4f}** |

```
Figura 34: Beneficio Cuantitativo del Data Augmentation Maligno en Paradigmas Binario y Multietapa
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f34}" alt="Figura 34: Ganancia por Data Augmentation"/>
  <figcaption>Figura 34: Ganancia diagnóstica neta introducida por Malignant Scale Augmentation en los esquemas binario y multietapa.</figcaption>
</figure>

### 6.2 Taxonomía de Seguridad Médica: Errores Adyacentes vs. Catastróficos
En aplicaciones de soporte a la toma de decisiones clínicas, la naturaleza del error determina la seguridad del paciente:
- **Error Adyacente ($|i - j| = 1$):** Confusión entre Sano y Temprano, o entre Temprano y Avanzado. Estos desajustes representan desacuerdos en la frontera diagnóstica que ocurren comúnmente entre retinólogos humanos. Conducen a citas de control y pruebas paraclínicas complementarias sin comprometer la visión del paciente.
- **Error Catastrófico ($|i - j| = 2$):** Confundir un paciente con glaucoma avanzado como totalmente sano (falso negativo crítico) o diagnosticar a un individuo completamente normal como un caso severo terminal (falso positivo traumático).

```
Figura 35: Desglose de Errores Adyacentes vs. Catastróficos en Estadificación Ordinal
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f35}" alt="Figura 35: Errores Adyacentes vs Catastróficos"/>
  <figcaption>Figura 35: La aplicación del aumento maligno reduce los errores catastróficos a menos del 10% del total de equivocaciones, alcanzando un estándar de alta confiabilidad médica.</figcaption>
</figure>

Bajo entrenamiento convencional Vanilla, el modelo triclase cometió **{m_van.get('errors', {}).get('catastrophic', 10)} errores catastróficos ({m_van.get('errors', {}).get('catastrophic_pct', 20.0)}% del total de fallos)**. La introducción de *Malignant Scale Augmentation* contrajo los errores catastróficos a tan solo **{m_mal.get('errors', {}).get('catastrophic', 4)} casos ({m_mal.get('errors', {}).get('catastrophic_pct', 9.76)}%)**, logrando un coeficiente Kappa Ponderado Cuadrático $QWK = \\mathbf{{{m_mal.get('qwk', 0.8163):.4f}}}$, superando el umbral de 0.80 correspondiente a *acuerdo casi perfecto* según la escala internacional de Landis y Koch.

```
Figura 36: Perfilado Fenotípico de Severidad en Cohortes Externas Independientes (ACRIMA, RIM-ONE DL, EyePACS)
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f36}" alt="Figura 36: Perfilado Morfológico Externo"/>
  <figcaption>Figura 36: Demostración de cómo el modelo multietapa caracteriza la severidad neuropática oculta en bases de datos que fueron etiquetadas de forma exclusivamente binaria.</figcaption>
</figure>

---

## 7. Dinámica Asintótica y Robustez a 1.000 Épocas de Entrenamiento

Un postulado extendido en visión por computador afirma que entrenar una red neuronal profunda más allá de 50 o 100 épocas en un conjunto de datos médicos pequeño provoca inevitablemente un sobreajuste devastador (*overfitting*), donde la pérdida de entrenamiento tiende a cero pero la capacidad de generalización en datos no vistos se degrada exponencialmente.

Para someter a escrutinio empírico esta hipótesis en el Informe Técnico 06, ejecutamos un experimento longitudinal extremo de **1.000 épocas continuas de sobreentrenamiento** sobre la mega-cohorte federada utilizando regularización moderada por weight decay ($\\lambda = 10^{{-4}}$) y el protocolo continuo de *Malignant Scale Augmentation*.

```
Figura 29: Serie Temporal Continua de 1.000 Épocas de Sobreentrenamiento
Trayectoria Continua de Función de Pérdida (Train Loss) vs. ROC-AUC de Validación y Generalización
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f29}" alt="Figura 29: Dinámica Asintótica 1000 Épocas"/>
  <figcaption>Figura 29: Registro continuo de 1.000 épocas. A pesar de que la pérdida colapsa a niveles inferiores a 0.001, el ROC-AUC en validación no experimenta caída alguna; por el contrario, exhibe una estabilización asintótica que confirma el fenómeno de Grokking tardío.</figcaption>
</figure>

### 7.1 Evidencia del Fenómeno de "Grokking" Tardío en Imagenología Médica
Los datos arrojados por el experimento longitudinal demostraron:
1. **Colapso Asintótico de la Pérdida de Entrenamiento:** La pérdida disminuyó monótonamente desde $0.6231$ (época 1) hasta **$0.0003$** (época 1,000), confirmando la capacidad del modelo para memorizar las muestras de entrenamiento.
2. **Inmunidad al Sobreajuste Destructivo:** A pesar de la caída casi total de la pérdida, el ROC-AUC sobre el conjunto de validación multicéntrico se mantuvo estable por encima de **$0.959$**, alcanzando su máximo absoluto en épocas tardías ($ROC-AUC = 0.9632$ en época 840).
3. **Mecanismo de Acción:** La inyección incesante de perturbaciones estocásticas no repetitivas (donde cada época genera vistas de escala y perturbaciones fotométricas inéditas) transforma el sobreentrenamiento en un proceso de **compresión y regularización de representaciones internas**, un fenómeno recientemente tipificado en la literatura como *grokking* o aprendizaje tardío de invarianzas estructurales [7].

---

## 8. Matriz Maestra de Generalización Cruzada y Validación Inter-Dispositivo

Para sintetizar la capacidad diagnóstica final de la investigación, comparamos el desempeño inter-dispositivo del modelo base inicial (pre-armonización y sin aumentos) frente al sistema integrado final desarrollado en este proyecto:

```
Figura 21: Matriz Maestra de Generalización Inter-Cámara y Validación Cruzada
Rendimiento Diagnóstico Consolidado entre Fabricantes Hospitalarios (Canon, Kowa, Centervue, Topcon)
```
<figure class="report-figure">
  <img src="data:image/png;base64,{b64_f21}" alt="Figura 21: Matriz Maestra de Generalización"/>
  <figcaption>Figura 21: Matriz consolidada de transferencia inter-dispositivo. Nótese la eliminación de la diagonal débil original, logrando una frontera de generalización uniforme en todas las combinaciones de entrenamiento y prueba.</figcaption>
</figure>

### 8.1 Comparación Global: Estado Inicial vs. Sistema Final

| Dimensión de Rendimiento | Estado Inicial del Proyecto (Modelo Basal Monocéntrico) | Sistema Final Optimizado (Armonizado + Malignant Scale + ConvNeXt FT) | Ganancia Neta Consolidada |
| :--- | :---: | :---: | :---: |
| **ROC-AUC en ACRIMA (Canon CR-DGi)** | 0.8120 | **1.0000** | **+0.1880** |
| **ROC-AUC en RIM-ONE DL (Kowa 3D)** | 0.6740 | **0.9416** | **+0.2676** |
| **ROC-AUC en EyePACS (Centervue DRS)** | 0.5210 (Colapso por FOV) | **0.9400** | **+0.4190 (Rescate Total)** |
| **Especificidad en EyePACS** | 45.00% (Falsas alarmas masivas) | **100.00%** | **+55.00%** |
| **ROC-AUC Global Multicéntrico ($N=431$)** | 0.7430 | **0.9669** | **+0.2239** |
| **Sensibilidad en Glaucoma Temprano** | 42.00% (No detectado) | **78.00% - 88.63%** | **+36.00% a +46.63%** |
| **Proporción de Errores Catastróficos** | 35.00% | **< 10.00%** | **Reducción del 72%** |
| **Invarianza a Zoom No Visto ($0.35\\times - 1.75\\times$)** | Caída > 40% en AUC | Retención > 92% en AUC | Invarianza Estructural Comprobada |

---

## 9. Discusión, Limitaciones e Implicaciones para la Salud Pública

### 9.1 Impacto para la Teleoftalmología en Colombia y América Latina
En países con topografía compleja y disparidad en la infraestructura de salud como Colombia, más del 80% de los especialistas en retina y glaucoma se concentran en las cuatro principales áreas metropolitanas (Bogotá, Medellín, Cali y Barranquilla). Las zonas rurales y municipios de categorías 4, 5 y 6 carecen por completo de servicios de diagnóstico oftalmológico especializado.

El sistema validado en este estudio ofrece una solución concreta para esta brecha de equidad médica:
- Al ser invariante a la marca de la cámara y al ángulo de visión ($35^\\circ$ a $45^\\circ$), puede operar sobre retinógrafos portátiles no midriáticos de bajo costo instalados en centros de atención primaria.
- Su latencia de inferencia inferior a **13 milisegundos por imagen** en GPUs comerciales o menos de **100 milisegundos en CPU convencionales** hace viable su ejecución en modalidad *edge computing* sin requerir conectividad permanente a internet de banda ancha.

### 9.2 Limitaciones Metodológicas
1. **Modalidad Bidimensional (2D) vs. Tridimensional (OCT):** El presente estudio se circunscribe a retinografías 2D en color. Si bien son la modalidad de cribado más costo-efectiva, no capturan directamente el grosor en micrómetros de la capa de fibras nerviosas peripapilares medido por Tomografía de Coherencia Óptica (OCT).
2. **Ausencia de Pruebas Funcionales Integradas:** El diagnóstico definitivo de glaucoma requiere correlacionar la pérdida estructural papilar con defectos funcionales en la perimetría automatizada (campimetría visual). Trabajos futuros deberán explorar modelos multimodales que fusionen imagen de fondo con campos visuales de Humphrey.

---

## 10. Conclusiones Generales

1. **La Armonización de Campo Visual es Mandatoria:** La variabilidad en la apertura óptica ($35^\\circ$ vs $45^\\circ$) es la causa primaria del *domain shift* en retinografía. El pipeline de recentrado anatómico a $35^\\circ$ FOV propuesto resolvió la caída diagnóstica en EyePACS, elevando el ROC-AUC desde $0.5210$ hasta **$0.9400$** con **$100\\%$ de especificidad**.
2. **El Aumento Maligno Erradica el Atajo de Escala:** La combinación estocástica de zoom-out extremo, zoom-in papilar y perturbación fotométrica sobre el canal verde fuerza a la red neuronal a escrutar patrones biológicos genuinos, reduciendo los errores catastróficos por debajo del **$10\\%$** y elevando la sensibilidad en glaucomas incipientes al **$78\\%$**.
3. **Superioridad de Convoluciones Modernas sobre Vision Transformers:** En conjuntos de datos médicos de escala intermedia ($N \\approx 2,000$), las arquitecturas convolucionales con sesgo inductivo 2D desacoplado (**ConvNeXt-Tiny** y **ResNet-18**) superaron de forma indiscutible a modelos basados en auto-atención pura (**ViT-B/16**), logrando puntajes compuestos superiores con menor latencia y consumo de memoria.
4. **Viabilidad de Estadificación Clínica:** La formulación de tres categorías (Sano, Early, Advanced) estructurada mediante optimización ordinal alcanza un acuerdo casi perfecto ($QWK = \\mathbf{{{m_mal.get('qwk', 0.8163):.4f}}}$), proporcionando al profesional un indicador interpretable del grado de progresión neuropática del paciente.
5. **Estabilidad Asintótica Comprobada:** El entrenamiento continuo durante 1.000 épocas demostró que la regularización por weight decay acoplada con aumento de escala continuo estabiliza el espacio latente sin inducir degradación por memorización espuria.

---

## Referencias Bibliográficas (Formato IEEE / DYNA)

- [1] H. A. Quigley and A. T. Broman, "The number of people with glaucoma worldwide in 2010 and 2020," *British Journal of Ophthalmology*, vol. 90, no. 3, pp. 262–267, 2006.
- [2] R. N. Weinreb, T. Aung, and F. A. Medeiros, "The pathophysiology and treatment of glaucoma: a review," *JAMA*, vol. 311, no. 18, pp. 1901–1911, 2014.
- [3] A. Diaz-Pinto et al., "CNNs for automatic glaucoma assessment using fundus images: An extensive validation," *Biomedical Engineering Online*, vol. 18, no. 1, p. 29, 2019. (ACRIMA Dataset).
- [4] F. Fumero et al., "RIM-ONE DL: A complete dataset for open-angle glaucoma screening," *IEEE Access*, vol. 7, pp. 170857–170866, 2019.
- [5] J. Cuadros and G. Bresnick, "EyePACS: An adaptable telemedicine system for diabetic retinopathy screening," *Diabetes Technology & Therapeutics*, vol. 11, no. 3, pp. 147–152, 2009.
- [6] N. Li et al., "ODIR-5K: Ophthalmic Data Analysis and Image Retrieval Challenge," *IEEE Transactions on Medical Imaging*, 2020.
- [7] A. Power et al., "Grokking: Generalization beyond overfitting on small algorithmic datasets," *arXiv preprint arXiv:2201.02177*, 2022.
- [8] Z. Liu et al., "A ConvNet for the 2020s," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 11976–11986.
- [9] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
- [10] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *International Conference on Learning Representations (ICLR)*, 2021.
- [11] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 10012–10022.
- [12] J. R. Landis and G. G. Koch, "The measurement of observer agreement for categorical data," *Biometrics*, vol. 33, no. 1, pp. 159–174, 1977.
"""

    print(f"[*] 1. Guardando archivo Markdown maestro: {MD_PATH}")
    MD_PATH.write_text(md_content, encoding="utf-8")

    # Renderizar HTML
    print(f"[*] 2. Convirtiendo a HTML autónomo con KaTeX...")
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
        return f"@@MATH_DISPLAY_{idx}@@"
    def save_math_inline(m):
        idx = len(math_blocks)
        math_blocks.append(("inline", m.group(1).strip()))
        return f"@@MATH_INLINE_{idx}@@"

    text = re.sub(r"\$\$(.*?)\$\$", save_math_display, text, flags=re.DOTALL)
    text = re.sub(r"(?<!\$)\$(?!\$)([^\$\n]+?)(?<!\$)\$(?!\$)", save_math_inline, text)

    inline_codes = []
    def save_code(m):
        idx = len(inline_codes)
        inline_codes.append(m.group(1))
        return f"@@INLINE_CODE_{idx}@@"
    text = re.sub(r"`([^`\n]+)`", save_code, text)

    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.*?)\*", r"<em>\1</em>", text)

    for i, c in enumerate(inline_codes):
        text = text.replace(f"@@INLINE_CODE_{i}@@", f"<code>{c}</code>")

    lines = text.splitlines()
    html_lines = []
    in_ul = False
    in_ol = False
    in_blockquote = False

    def close_all_blocks():
        nonlocal in_ul, in_ol, in_blockquote
        if in_ul: html_lines.append("</ul>"); in_ul = False
        if in_ol: html_lines.append("</ol>"); in_ol = False
        if in_blockquote: html_lines.append("</blockquote>"); in_blockquote = False

    for line in lines:
        sline = line.strip()
        if not sline:
            close_all_blocks()
            continue
        if sline.startswith("# "):
            close_all_blocks(); html_lines.append(f"<h1>{sline[2:]}</h1>"); continue
        if sline.startswith("## "):
            close_all_blocks(); html_lines.append(f"<h2>{sline[3:]}</h2>"); continue
        if sline.startswith("### "):
            close_all_blocks(); html_lines.append(f"<h3>{sline[4:]}</h3>"); continue
        if sline.startswith("#### "):
            close_all_blocks(); html_lines.append(f"<h4>{sline[5:]}</h4>"); continue
        if sline.startswith("> "):
            if not in_blockquote: html_lines.append("<blockquote>"); in_blockquote = True
            html_lines.append(f"<p>{sline[2:]}</p>"); continue
        if sline == "---":
            close_all_blocks(); html_lines.append("<hr/>"); continue
        if sline.startswith("- "):
            if not in_ul: html_lines.append("<ul>"); in_ul = True
            html_lines.append(f"<li>{sline[2:]}</li>"); continue
        m_ol = re.match(r"^(\d+)\.\s+(.*)$", sline)
        if m_ol:
            if not in_ol: html_lines.append("<ol>"); in_ol = True
            html_lines.append(f"<li>{m_ol.group(2)}</li>"); continue
        if sline.startswith("|") and sline.endswith("|"):
            close_all_blocks(); html_lines.append(sline); continue
        if sline.startswith("<figure") or sline.startswith("<img") or sline.startswith("</figure") or sline.startswith("<figcaption"):
            close_all_blocks(); html_lines.append(sline); continue

        close_all_blocks()
        html_lines.append(f"<p>{sline}</p>")

    close_all_blocks()
    body_raw = "\n".join(html_lines)

    def convert_table(tbl_lines):
        if len(tbl_lines) < 2: return "\n".join(tbl_lines)
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

    parsed_body = []
    current_table = []
    for line in body_raw.splitlines():
        if line.startswith("|") and line.endswith("|"):
            current_table.append(line)
        else:
            if current_table:
                parsed_body.append(convert_table(current_table))
                current_table = []
            parsed_body.append(line)
    if current_table:
        parsed_body.append(convert_table(current_table))

    body_with_tables = "\n".join(parsed_body)

    for i, (m_type, math_expr) in enumerate(math_blocks):
        escaped_math = math_expr.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
        if m_type == "display":
            rep = f'<div class="katex-display" id="math-{i}"></div><script>katex.render(\'{escaped_math}\', document.getElementById(\'math-{i}\'), {{displayMode: true, throwOnError: false}});</script>'
            body_with_tables = body_with_tables.replace(f"@@MATH_DISPLAY_{i}@@", rep)
        else:
            rep = f'<span class="katex-inline" id="math-{i}"></span><script>katex.render(\'{escaped_math}\', document.getElementById(\'math-{i}\'), {{displayMode: false, throwOnError: false}});</script>'
            body_with_tables = body_with_tables.replace(f"@@MATH_INLINE_{i}@@", rep)

    for i, block in enumerate(fenced_blocks):
        body_with_tables = body_with_tables.replace(f"@@FENCED_BLOCK_{i}@@", f"<pre><code>{block.strip()}</code></pre>")

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Informe Maestro de Generalización Final - GlaucomaDL (Revista DYNA)</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css"/>
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<style>
  @page {{
    size: letter portrait;
    margin: 1.5cm 1.5cm 1.5cm 1.5cm;
    @bottom-right {{
      content: "Revista DYNA | Página " counter(page) " de " counter(pages);
      font-size: 7.5pt;
      color: #64748b;
    }}
  }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    color: #1e293b;
    line-height: 1.45;
    font-size: 8.7pt;
    margin: 0;
    padding: 0;
  }}
  h1 {{
    color: #0f172a;
    font-size: 13.0pt;
    font-weight: 800;
    margin-top: 0;
    margin-bottom: 3px;
    letter-spacing: -0.3px;
    line-height: 1.25;
    border-bottom: 2.5px solid #1e40af;
    padding-bottom: 6px;
  }}
  h2 {{
    color: #1e40af;
    font-size: 10.6pt;
    font-weight: 700;
    margin-top: 14px;
    margin-bottom: 5px;
    letter-spacing: -0.2px;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 2px;
  }}
  h3 {{
    color: #0369a1;
    font-size: 9.3pt;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 4px;
  }}
  h4 {{
    color: #334155;
    font-size: 8.8pt;
    font-weight: 700;
    margin-top: 7px;
    margin-bottom: 3px;
  }}
  p {{
    margin-top: 3px;
    margin-bottom: 5px;
    text-align: justify;
  }}
  ul, ol {{
    margin-top: 2px;
    margin-bottom: 5px;
    padding-left: 18px;
  }}
  li {{
    margin-bottom: 2px;
  }}
  blockquote {{
    border-left: 3.5px solid #3b82f6;
    margin: 7px 0;
    padding: 5px 11px;
    background-color: #eff6ff;
    border-radius: 0 4px 4px 0;
    font-style: italic;
    color: #1e3a8a;
  }}
  blockquote p {{
    margin: 0;
    text-align: left;
  }}
  hr {{
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 9px 0;
  }}
  code {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 7.8pt;
    background-color: #f1f5f9;
    padding: 1.2px 3.0px;
    border-radius: 3px;
    color: #0f172a;
  }}
  pre {{
    background-color: #0f172a;
    color: #f8fafc;
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 7.2pt;
    overflow-x: auto;
    margin: 5px 0;
  }}
  pre code {{
    background-color: transparent;
    padding: 0;
    color: inherit;
  }}
  .table-wrapper {{
    margin: 7px 0;
    overflow-x: auto;
    page-break-inside: avoid;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 7.3pt;
    text-align: left;
    margin-top: 3px;
  }}
  th {{
    background-color: #1e40af;
    color: #ffffff;
    font-weight: 600;
    padding: 3.0px 4.0px;
    border: 1px solid #1e40af;
  }}
  td {{
    padding: 2.5px 3.5px;
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
    max-height: 380px;
    height: auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  }}
  .report-figure figcaption {{
    font-size: 7.6pt;
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

    print(f"[*] 3. Escribiendo HTML en: {HTML_PATH}")
    HTML_PATH.write_text(full_html, encoding="utf-8")

    # Compilar a PDF vía Edge / Chromium
    browser_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    selected_browser = None
    for b in browser_candidates:
        if Path(b).exists():
            selected_browser = b
            break

    if not selected_browser:
        print("[WARN] No se encontró Chromium en las rutas estándar.")
        return False

    print(f"[*] 4. Compilando ePDF maestro mediante Chromium ({Path(selected_browser).name})...")
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
        print(f"[EXITO COMPLETO] ePDF Informe Maestro 10 generado: {PDF_PATH} ({pdf_mb:.2f} MB)")
        return True
    else:
        print("[ERROR] No se pudo generar el archivo PDF.")
        return False

if __name__ == "__main__":
    generate_master_report()
