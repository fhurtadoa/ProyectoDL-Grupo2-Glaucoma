# Evaluación de la Capacidad de Generalización en Redes Convolucionales Profundas para la Detección de Glaucoma ante Heterogeneidad de Dispositivos y Condiciones de Adquisición

**Proyecto de Investigación — Deep Learning**  
**Facultad de Minas — Universidad Nacional de Colombia**  
**Materia:** Deep Learning  
**Formato de Publicación del Informe Final:** Plantilla de Artículo Científico — Revista DYNA (Facultad de Minas, Universidad Nacional de Colombia)  

---

> ⚠️ **Aviso de Transparencia sobre Asistencia con IA:**  
> Parte de la documentación, estructuración de scripts y compilación de informes técnicos de este repositorio contó con el apoyo de herramientas de Inteligencia Artificial Generativa (LLMs). Por este motivo, es posible que se presenten eventuales imprecisiones terminológicas, ligeras redundancias o expresiones con matices poco precisos en algunas secciones descriptivas. Ante cualquier discrepancia técnica o conceptual, se aconseja contrastar directamente la información con las implementaciones en `scripts/`, las métricas consolidadas en `results/` y los checkpoints de pesos en `modelos/`.

---

## 🧭 Guía de Inicio Rápido (Quickstart para Integrantes del Equipo)

Si acabas de clonar el repositorio y necesitas ponerte al día en 5 minutos sin tener que revisar todo el código desde cero, sigue estos pasos:

### 1. Requisitos y Entorno
- **Python:** 3.10 o superior (probado en Python 3.11 / 3.12 / 3.14).
- **GPU (Opcional para inferencia, requerida para entrenar):** Compatible con CUDA (ejecutado originalmente en NVIDIA GeForce RTX 4060 Ti 16GB).
- **Instalación de dependencias básicas:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install numpy pandas matplotlib scikit-learn pillow pymupdf requests beautifulsoup4
  ```

### 2. ¿Dónde están los modelos listos para usar?
No necesitas volver a entrenar los modelos para probarlos o extraer resultados. Todos los pesos entrenados están guardados en la carpeta `modelos/`:
- **`modelos/best_resnet18_1000ep_val_auc.pth`:** Es el **modelo principal y más potente** de todo el proyecto (Época 720, Val ROC-AUC = **0.9829**). Es el que se recomienda citar en el paper y usar para inferencia.
- **`modelos/final_resnet18_1000ep.pth`:** Pesos de la época 1000 (alcanza la máxima especificidad en test: **94.98%**, con solo 11 falsos positivos en 431 imágenes).
- **`modelos/best_resnet18_malignant_scale.pth`:** Checkpoint de línea base entrenado a 15 épocas bajo augmentación maligna (ROC-AUC test = **0.9482**).

### 3. ¿Dónde están los informes y gráficas para el artículo Revista DYNA?
- **Informes en PDF listos para leer:**
  - `informes/06_experimento_1000_epocas_sobreentrenamiento/INFORME_EXPERIMENTO_1000_EPOCAS_SOBREENTRENAMIENTO.pdf` *(Informe completo de 7 páginas con la serie temporal, análisis de Grokking y tablas comparativas).*
  - `informes/05_entrenamiento_y_estres_malignant_scale_mega_cohorte/INFORME_ENTRENAMIENTO_ESTRES_MALIGNO_MEGA_COHORTE.pdf` *(Informe del protocolo de data augmentation y armonización).*
- **Figuras en alta resolución (300 DPI):**
  Están en `figures/` con nombres correlativos (`fig19` a `fig29`). Para la sección de resultados del paper, las figuras clave son:
  - `fig26_1000_epochs_loss_and_auc_curves.png` (Curvas de entrenamiento a 1000 épocas).
  - `fig27_1000_epochs_cross_cohort_comparison.png` (Comparativa de métricas entre épocas 15, 720 y 1000).
  - `fig28_1000_epochs_confusion_matrices.png` (Matrices de confusión por cohorte médica).
  - `fig29_time_series_epoch_vs_loss_auc.png` (Serie temporal continua de épocas y tiempo en minutos vs. Loss y AUC).

---

## 📌 Resumen Técnico del Proyecto

El diagnóstico automatizado de glaucoma mediante retinografías de fondo de ojo suele reportar rendimientos altos ($\text{ROC-AUC} > 0.98$) cuando se entrena y evalúa dentro del mismo centro médico. Sin embargo, al probar esos modelos en imágenes capturadas con cámaras de otros fabricantes, diferentes resoluciones angulares ($35^\circ$ vs. $45^\circ$) o diferente iluminación, el rendimiento cae de forma severa por cambio de dominio (*domain shift* óptico).

Este proyecto investiga y resuelve este desafío a través de seis ejes:

1. **Selección justificada de arquitectura base:** Evaluación comparativa entre ResNet-18, ResNet-50, EfficientNet-B0 y MobileNet-V2 con transferencia de aprendizaje desde ImageNet.
2. **Estudio sistemático de generalización multicéntrica:** Análisis de transferencia cruzada entrenando con un solo dataset (monomodal), dos datasets (bi-céntrico) y federando múltiples fuentes clínicas.
3. **Módulo de adquisición web y filtro de calidad:** Descarga de imágenes web (*in-the-wild*), detector automático de validez retiniana para eliminar esquemas o fotografías no médicas, y prueba de inferencia.
4. **Armonización de escala y conformación de la Mega-Cohorte Médica ($N=2.152$):** Estandarización a un campo visual común de $35^\circ$ (ROI peripapilar) entre ACRIMA, RIM-ONE DL, EyePACS y Moondream, con exclusión justificada de fuentes no aptas (`THIRD_DATASET`).
5. **Data Augmentation Maligno (`MalignantScaleAugmentation`) y pruebas de estrés:** Perturbaciones severas de escala (zoom-out con viñeteo óptico y ruido sintético, zoom-in focal) y color para forzar a la red a no depender de bordes o encuadres fijos.
6. **Experimento de entrenamiento asintótico a 1.000 épocas:** Demostración empírica de que el entrenamiento prolongado bajo augmentación estocástica continua no produce sobreajuste clásico, sino que refina la capacidad de generalización en pruebas externas ($90.72\%$ exactitud, $94.98\%$ especificidad y $0.9657$ ROC-AUC).

---

## 📁 Estructura del Repositorio y Ubicación de Archivos

```
GlaucomaDL/
├── data/                                         # Datasets de imágenes organizados por cohorte
│   ├── Database/Images/                          # ACRIMA (España, N=705, Canon CR-DGi, 35°)
│   ├── RIM_ONE_DL/                               # RIM-ONE DL (Canarias, N=447, Kowa WX 3D estereoscópico)
│   ├── EYEPACS_REAL/                             # EyePACS (N=200, tamizaje en campo abierto, 45° FOV)
│   ├── MOONDREAM_STAGED/                         # Moondream (N=800, cohorte multicéntrica graduada por estadío)
│   ├── WEB_IN_THE_WILD/                          # Imágenes web (descarga raw, filtradas y descartadas)
│   ├── HARMONIZED_MEGA_COHORT/                   # Mega-cohorte unificada y recortada a ROI 35° (N=2.152)
│   │   ├── train/                                # 1.721 imágenes (846 glaucoma, 875 normal)
│   │   └── test/                                 # 431 imágenes de prueba independiente (212 glaucoma, 219 normal)
│   └── THIRD_DATASET/                            # AVISO: Excluido del entrenamiento (ver sección Eje 4)
│
├── modelos/                                      # Checkpoints de pesos entrenados (PyTorch .pth)
│   ├── best_resnet18_1000ep_val_auc.pth          # RECOMENDADO: Mejor modelo histórico (Época 720, Val AUC: 0.9829)
│   ├── final_resnet18_1000ep.pth                 # Checkpoint final época 1000 (Especificidad test: 94.98%)
│   ├── best_resnet18_malignant_scale.pth         # Modelo con data augmentation maligno a 15 épocas
│   ├── best_resnet18_acrima.pth                  # Modelo monomodal (entrenado solo en ACRIMA)
│   ├── best_resnet18_multidomain.pth             # Modelo bi-céntrico (ACRIMA + RIM-ONE)
│   ├── best_resnet18_megacohort.pth              # Modelo federado inicial (sin recorte homogéneo de ROI)
│   ├── best_resnet18_multistage.pth              # Modelo de clasificación en 3 estadíos clínicos
│   └── checkpoint_resnet18_epoch_{...}.pth       # Hitos intermedios (épocas 1, 5, 15, 50, 100, 250, 500, 750, 1000)
│
├── scripts/                                      # Código fuente modular y documentado
│   ├── experimentos/
│   │   ├── 01_seleccion_arquitectura_backbone.py      # Benchmark comparativo de arquitecturas
│   │   ├── 02_estudio_generalizacion_multicentrica.py # Evaluación cruzada entre cámaras
│   │   ├── 03_adquisicion_web_y_filtro_calidad.py     # Pipeline web y detector de validez
│   │   ├── 04_data_augmentation_invarianza_escala.py  # Primera prueba de escala
│   │   ├── 05_mega_cohorte_federada_multisensor.py    # Entrenamiento federado inicial
│   │   ├── 06_estadificacion_clinica_multietapa.py   # Clasificación en 3 niveles de severidad
│   │   ├── 07_armonizacion_roi_y_mega_dataset.py      # Estandarización a ROI 35° y armado de mega-cohorte
│   │   ├── 08_entrenamiento_malignant_scale_mega_cohorte.py # Entrenamiento a 15 épocas con augmentación maligna
│   │   ├── 09_experimento_1000_epocas_sobreentrenamiento.py # Entrenamiento a 1.000 épocas (RTX 4060 Ti)
│   │   ├── eval_test_time_malignant.py               # Prueba bajo perturbaciones en tiempo de inferencia
│   │   ├── generate_fig28.py                         # Generación de matrices de confusión comparativas
│   │   └── generate_fig29_timeseries.py              # Generación de la serie temporal continua (Figura 29)
│   └── compiladores_informes/                         # Compiladores automáticos a HTML y ePDF
│       ├── compilar_informe_bases_medicas.py
│       ├── compilar_informe_estres_malignant_mega_cohorte.py
│       └── compilar_informe_experimento_1000_epocas.py
│
├── figures/                                      # Figuras diagnósticas a 300 DPI
│   ├── fig19_backbone_comparison_benchmark.png        # Comparativa de backbones
│   ├── fig20_web_images_filtering_pipeline.png        # Pipeline del detector de calidad web
│   ├── fig21_cross_device_generalization_master.png   # Matriz maestra de generalización cruzada
│   ├── fig24_harmonization_pipeline_before_after.png  # Comparación de campo visual antes y después de ROI
│   ├── fig25_malignant_scale_stress_test_benchmark.png# Desempeño bajo augmentación maligna
│   ├── fig26_1000_epochs_loss_and_auc_curves.png      # Curvas de pérdida y ROC-AUC a 1.000 épocas
│   ├── fig27_1000_epochs_cross_cohort_comparison.png  # Comparativa inter-cohorte (15 vs 720 vs 1000 épocas)
│   ├── fig28_1000_epochs_confusion_matrices.png       # Matrices de confusión en las 4 cohortes externas
│   └── fig29_time_series_epoch_vs_loss_auc.png        # Serie temporal continua (épocas y tiempo vs loss/auc)
│
├── results/                                      # Resúmenes cuantitativos (JSON y CSV)
│   ├── backbone_selection_benchmark.json              # Métricas de redes candidatas
│   ├── cross_dataset_generalization_matrix.json       # Resultados de evaluación cruzada
│   ├── web_quality_filter_metrics.json                # Rendimiento del filtro detector web
│   ├── harmonized_mega_cohort_index.csv               # Índice de las 2.152 imágenes con centro y etiqueta
│   ├── malignant_stress_test_metrics.json             # Métricas de prueba de estrés a 15 épocas
│   ├── inference_time_malignant_stress_comparison.json# Métricas con test-time augmentation
│   └── 1000_epochs_stress_experiment_metrics.json     # Registro detallado de las 1.000 épocas
│
└── informes/                                     # Informes técnicos en PDF, HTML y Markdown
    ├── 01_experimento_invarianza_escala/
    ├── 02_experimento_mega_cohorte_multisensor/
    ├── 03_experimento_estadificacion_multietapa/
    ├── 04_evaluacion_bases_medicas_y_mega_dataset/
    ├── 05_entrenamiento_y_estres_malignant_scale_mega_cohorte/
    └── 06_experimento_1000_epocas_sobreentrenamiento/
```

---

## 🔬 Eje 1: Selección de la Arquitectura Base

Se evaluaron cuatro arquitecturas convolucionales utilizando pesos pre-entrenados en ImageNet-1K:

| Arquitectura Base | Parámetros | Memoria | Latencia (Batch 32) | ROC-AUC (Test) | Exactitud (Test) | Brecha (|Train - Val|) | Conclusión Técnica |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **ResNet-18** | **11.18 M** | **42.64 MB** | **147.15 ms** | **0.9982** | **98.58%** | **0.0232** | **Seleccionada:** Menor brecha de sobreajuste, bajo consumo de memoria y rápida convergencia. |
| **ResNet-50** | 23.51 M | 89.68 MB | 285.40 ms | 0.9965 | 97.87% | 0.0514 | Duplica tamaño y tiempo de cómputo sin mejora apreciable en AUC. |
| **EfficientNet-B0** | 4.01 M | 15.29 MB | 112.30 ms | 0.9912 | 96.45% | 0.0380 | Eficiente, pero con mayor dispersión de atención en bordes del disco. |
| **MobileNet-V2** | 2.23 M | 8.49 MB | 88.50 ms | 0.9845 | 95.04% | 0.0425 | Muy ligera, pero pierde sensibilidad en lesiones glaucomatosas finas. |

```bash
python scripts/experimentos/01_seleccion_arquitectura_backbone.py
```

---

## 🌐 Eje 2: Generalización Multicéntrica y Evaluación Cruzada

Se demostró cuantitativamente el impacto de entrenar en un único centro hospitalario vs. federar datos:

1. **Monomodal (Solo ACRIMA, Canon 35°):** Alcanza $0.9960$ ROC-AUC en ACRIMA, pero cae a $0.5828$ en RIM-ONE DL (Kowa 3D) y a $0.5197$ en EyePACS (nivel de azar).
2. **Bi-céntrico (ACRIMA + RIM-ONE DL):** El modelo aprende a transferir entre Canon y Kowa, subiendo en RIM-ONE a $0.9219$, pero sigue bajo en EyePACS ($0.5512$).
3. **Mega-Cohorte Multi-Sensor inicial:** Al federar las 4 fuentes sin procesar, EyePACS sube a $0.7314$ y ACRIMA preserva $1.0000$ ROC-AUC.

```bash
python scripts/experimentos/02_estudio_generalizacion_multicentrica.py
```

---

## 🔍 Eje 3: Adquisición Web In-The-Wild y Filtro de Calidad

Se simuló la descarga de imágenes web desde Google Images para evaluar su utilidad clínica:
- **Ruido detectado:** De 74 imágenes descargadas, el $45.9\%$ correspondió a esquemas anatómicos, diagramas vectoriales, cortes de tomografía OCT o fotografías de instrumental.
- **Filtro detector implementado:** Analiza la relación de color ($R/B > 1.15$), detecta fondos blancos planos ($>230$) y comprueba el viñeteo circular del oftalmoscopio.
- **Resultado:** **100% de precisión**, eliminando todo el contenido no retiniano sin descartar fondos de ojo legítimos.
- **Conclusión para el artículo:** El raspado web no supervisado introduce casi un $50\%$ de ruido; un filtro detector es indispensable, aunque las cohortes clínicas protocolizadas continúan siendo el estándar de oro.

```bash
python scripts/experimentos/03_adquisicion_web_y_filtro_calidad.py
```

---

## 🧩 Eje 4: Armonización a ROI 35° y Mega-Cohorte Médica ($N=2.152$)

### El Problema de la Disparidad de Campo Visual:
En EyePACS (cámaras de tamizaje panorámico a $45^\circ$), el disco óptico ocupa apenas entre el $5\%$ y el $10\%$ del área de la imagen. En cambio, en ACRIMA y RIM-ONE DL ($35^\circ$), el disco ocupa entre el $30\%$ y el $50\%$. Esta diferencia geométrica provocaba que los modelos aprendieran atajos de tamaño en lugar de características clínicas del anillo neurorretiniano.

### Protocolo de Armonización:
1. **Recorte de ROI peripapilar a $35^\circ$:** Se extrajo la región de interés centrada en la papila en las imágenes panorámicas de EyePACS y Moondream.
2. **Máscara circular homogénea:** Redimensionado a $224 \times 224$ píxeles con viñeteo óptico unificado.
3. **Composición de la Mega-Cohorte Armonizada:**
   - Total: **$N = 2.152$ imágenes** balanceadas ($1.058$ glaucoma, $1.094$ normales).
   - Entrenamiento: **$1.721$ imágenes** ($846$ glaucoma, $875$ normales).
   - Prueba independiente: **$431$ imágenes** ($212$ glaucoma, $219$ normales), con representación equilibrada de las 4 cohortes.
   - El índice maestro está guardado en `results/harmonized_mega_cohort_index.csv`.

> ⚠️ **Importante — Exclusión de `data/THIRD_DATASET/`:**  
> Al inspeccionar los datos se identificó que `THIRD_DATASET` contiene **máscaras binarizadas de segmentación vascular (esqueletos de vasos)** y no imágenes RGB naturales de fondo de ojo. Su inclusión alteraría negativamente el espacio de color y el aprendizaje convolucional. **Por tanto, debe permanecer estrictamente excluida.**

```bash
python scripts/experimentos/07_armonizacion_roi_y_mega_dataset.py
```

---

## ⚡ Eje 5: Data Augmentation Maligno (`MalignantScaleAugmentation`) y Pruebas de Estrés

Para forzar a la red a ser invariante a la escala y la iluminación, se implementó el generador estocástico `MalignantScaleAugmentation`:
- **45% Zoom-out severo ($0.25\times$ a $0.65\times$):** La retina se reduce a una miniatura sobre fondo con ruido gaussiano sintético y viñeteo circular.
- **35% Zoom-in focal ($1.25\times$ a $1.85\times$):** Recorta de cerca la copa óptica simulando tomas macro.
- **20% Escala estándar ($224 \times 224$).**
- **65% Perturbaciones fotométricas:** Modificaciones de gamma ($\gamma \in [0.75, 1.35]$) y ganancia en el canal verde ($[0.90, 1.15]$).

### Resultados del Modelo a 15 Épocas (`best_resnet18_malignant_scale.pth`):
- **Test limpio ($N=431$):** ROC-AUC: **0.9482**, Exactitud: **85.61%**, Sensibilidad: **79.25%**, Especificidad: **91.78%**.
- **Prueba bajo estrés en tiempo de inferencia (*Test-Time Augmentation*):** Al someter el conjunto de prueba a las perturbaciones malignas durante la evaluación, el modelo conservó el **98.8%** de su ROC-AUC ($0.9466 \to 0.9356$), confirmando su resistencia ante alteraciones geométricas.

```bash
# Entrenamiento a 15 épocas con augmentación maligna
python scripts/experimentos/08_entrenamiento_malignant_scale_mega_cohorte.py

# Evaluación bajo estrés en inferencia
python scripts/experimentos/eval_test_time_malignant.py
```

---

## 📈 Eje 6: Experimento de Entrenamiento Asintótico a 1.000 Épocas

En aprendizaje profundo médico existe la costumbre de aplicar parada temprana (*early stopping*) hacia la época 15 o 20 en cuanto la pérdida de validación sube ligeramente. Se planteó la hipótesis de evaluar qué ocurre al entrenar durante **1.000 épocas completas** bajo el flujo incesante de datos aumentados estocásticamente (~2.35 horas en GPU RTX 4060 Ti con planificador Cosine Annealing, $\eta = 10^{-4} \to 10^{-6}$).

### Hallazgo Central:
La red **no sufrió sobreajuste clásico**. Gracias a las infinitas combinaciones aleatorias de zoom, desplazamiento, ruido y color, la red nunca observó la misma entrada dos veces. Aunque la pérdida de entrenamiento se redujo a $0.00028$, las métricas sobre las 4 cohortes de prueba independientes mejoraron significativamente:

### Comparativa: Época 15 (Convencional) vs. Época 1.000 (Régimen Asintótico)

| Métrica Global ($N=431$ Test) | Época 15 (Convencional) | Época 720 (Mejor Val AUC) | Época 1000 (Final) | Diferencia Neta |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC Global** | 0.9490 | 0.9592 | **0.9657** | **+0.0167** |
| **Exactitud Global** | 86.54% | 89.10% | **90.72%** | **+4.18%** |
| **Especificidad Global** | 85.84% | 94.06% | **94.98%** | **+9.14%** |
| **Sensibilidad Global** | **87.26%** | 83.96% | 86.32% | -0.94% |
| **F1-Score** | 0.8645 | 0.8834 | **0.9015** | **+0.0370** |
| **Falsos Positivos Totales** | 31 | 13 | **11** | **-64.5% (de 31 a 11)** |

### Resultados por Cohorte Externa:
- **ACRIMA (Canon 35°):** Mantuvo rendimiento casi perfecto con **0.9939 ROC-AUC**, Exactitud de **97.87%** y Especificidad de **98.39%**.
- **RIM-ONE DL (Kowa 3D):** El ROC-AUC subió de $0.8995$ a **0.9479** (+0.0484) y la exactitud de $83.33\%$ a **92.22%** (+8.89%).
- **EyePACS Real (45°):** El ROC-AUC subió de $0.8500$ a **0.9125** (+0.0625) con una especificidad del **90.00%**.
- **Moondream (Estadíos):** El ROC-AUC mejoró de $0.9123$ a **0.9382**, y la especificidad aumentó del $77.50\%$ al **91.25%** (+13.75%).

### Análisis de la Serie Temporal (Figura 29):
Se incorporó la serie temporal continua (Figura 29 en `figures/`) que relaciona épocas y tiempo real de cómputo ($0 \to 141.13$ minutos):
- **Desacoplamiento entre pérdida y ranking:** Entre las épocas 50 y 75 ($t \approx 7-11$ min), la pérdida de validación repunta levemente (lo que normalmente asusta al analista y dispara el *early stopping*), pero el ROC-AUC se mantiene firme ($>0.960$).
- **Salto tardío de aprendizaje (*Grokking*):** En el minuto 84 (época 600) y minuto 101 (época 720), la red alcanza el récord histórico del proyecto: **Val ROC-AUC = 0.9829** y **Exactitud = 93.05%**.

```bash
# Entrenamiento a 1000 épocas (requiere GPU)
python scripts/experimentos/09_experimento_1000_epocas_sobreentrenamiento.py

# Generación de matrices de confusión comparativas (Fig. 28)
python scripts/experimentos/generate_fig28.py

# Generación de la serie temporal continua (Fig. 29)
python scripts/experimentos/generate_fig29_timeseries.py
```

---

## 💾 Guía de Pesos y Checkpoints en `modelos/`

| Archivo | Época | Uso Recomendado |
| :--- | :---: | :--- |
| **`best_resnet18_1000ep_val_auc.pth`** | 720 | **Modelo principal recomendado:** Mayor capacidad de discriminación diagnóstica (Val AUC = $0.9829$). |
| **`final_resnet18_1000ep.pth`** | 1000 | Pesos al completar las 1.000 épocas. Máxima especificidad en test ($94.98\%$) y menor tasa de falsos positivos. |
| **`best_resnet18_malignant_scale.pth`** | 15 | Modelo base entrenado con aumentación maligna a 15 épocas ($AUC = 0.9482$). |
| `best_resnet18_acrima.pth` | 20 | Modelo monomodal (solo ACRIMA). |
| `best_resnet18_multidomain.pth` | 20 | Modelo bi-céntrico (ACRIMA + RIM-ONE). |
| `best_resnet18_megacohort.pth` | 20 | Modelo federado inicial sin recorte de ROI. |
| `best_resnet18_multistage.pth` | 25 | Modelo de clasificación en 3 clases clínicas. |
| `checkpoint_resnet18_epoch_{...}.pth` | Varios | Checkpoints intermedios guardados en épocas 1, 5, 15, 50, 100, 250, 500, 750, 1000. |

---

## 📑 Informes Técnicos Compilados

En la carpeta `informes/` se encuentran los informes compilados en formato artículo científico:

- **Informe 05 — Datasets y Data Augmentation Maligno (PDF / HTML):**  
  `informes/05_entrenamiento_y_estres_malignant_scale_mega_cohorte/INFORME_ENTRENAMIENTO_ESTRES_MALIGNO_MEGA_COHORTE.pdf`
- **Informe 06 — Experimento de 1.000 Épocas y Serie Temporal (PDF / HTML):**  
  `informes/06_experimento_1000_epocas_sobreentrenamiento/INFORME_EXPERIMENTO_1000_EPOCAS_SOBREENTRENAMIENTO.pdf`

*Para compilar o regenerar los PDFs mediante Chromium/Edge headless:*
```bash
python scripts/compiladores_informes/compilar_informe_estres_malignant_mega_cohorte.py
python scripts/compiladores_informes/compilar_informe_experimento_1000_epocas.py
```

---

## ❓ Preguntas Frecuentes y Solución de Problemas (FAQ)

### ¿Tengo que re-entrenar las 1.000 épocas para hacer pruebas?
**No.** El entrenamiento de 1.000 épocas ya fue ejecutado y todos sus pesos y métricas están guardados en `modelos/` y `results/`. Puedes cargar directamente `modelos/best_resnet18_1000ep_val_auc.pth` en PyTorch con `model.load_state_dict(torch.load(...))` para evaluar o hacer inferencias inmediatas.

### ¿Qué hago si obtengo error de memoria CUDA (Out of Memory)?
Si decides re-entrenar algún script en una GPU con menos de 8 GB de VRAM, abre el script correspondiente y cambia `BATCH_SIZE = 32` por `BATCH_SIZE = 16` o `8`. Los scripts ya incluyen aceleración por precisión mixta (`torch.amp.autocast`).

### ¿Por qué `THIRD_DATASET` está excluida del entrenamiento?
Porque no contiene fotografías RGB de fondo de ojo. Son máscaras binarizadas de segmentación vascular (esqueletos vasculares en blanco y negro). Entrenar con ellas distorsiona los pesos convolucionales que dependen del color retiniano y la morfología de la excavación papilar.

### ¿Cómo compilar los PDFs si no tengo Google Chrome instalado?
Los scripts de compilación buscan automáticamente navegadores basados en Chromium en las siguientes rutas:
- Microsoft Edge en Windows (`C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`).
- Google Chrome en Windows, macOS o Linux.
Con tener instalado Microsoft Edge (que viene predeterminado en Windows) o Google Chrome, el compilador genera el PDF automáticamente con el comando `--headless=new --print-to-pdf`.

---

## 🚀 Resumen Secuencial de Comandos para Reproducción

```bash
# 1. Comparar arquitecturas base
python scripts/experimentos/01_seleccion_arquitectura_backbone.py

# 2. Reproducir matriz de generalización cruzada
python scripts/experimentos/02_estudio_generalizacion_multicentrica.py

# 3. Probar adquisición web y filtro de calidad
python scripts/experimentos/03_adquisicion_web_y_filtro_calidad.py

# 4. Generar la mega-cohorte armonizada a ROI 35°
python scripts/experimentos/07_armonizacion_roi_y_mega_dataset.py

# 5. Entrenar modelo a 15 épocas con augmentación maligna
python scripts/experimentos/08_entrenamiento_malignant_scale_mega_cohorte.py

# 6. Evaluar resistencia bajo estrés en inferencia
python scripts/experimentos/eval_test_time_malignant.py

# 7. Ejecutar entrenamiento a 1000 épocas (requiere GPU)
python scripts/experimentos/09_experimento_1000_epocas_sobreentrenamiento.py

# 8. Generar matrices de confusión (Figura 28)
python scripts/experimentos/generate_fig28.py

# 9. Generar serie temporal continua (Figura 29)
python scripts/experimentos/generate_fig29_timeseries.py

# 10. Compilar informe técnico 06 a ePDF
python scripts/compiladores_informes/compilar_informe_experimento_1000_epocas.py
```

---
*Facultad de Minas — Universidad Nacional de Colombia. Repositorio académico con fines docentes y de investigación.*
