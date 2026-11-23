# Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías

**Proyecto de Investigación — Deep Learning**  
**Facultad de Minas — Universidad Nacional de Colombia**  
**Materia:** Deep Learning  
**Formato de Publicación del Informe Final:** Plantilla de Artículo Científico — Revista DYNA (Facultad de Minas, Universidad Nacional de Colombia)  

**Sustentación:** [Demo local de glaucoma y guion de cinco minutos](demo_glaucoma/README.md)


## Alcance de las etiquetas al evaluar la severidad

Algunas retinografías del estudio, como los ejemplos de RIM-ONE DL, tienen una etiqueta original **binaria**. Una referencia **Glaucoma** permite evaluar la detección de glaucoma. Para determinar si una predicción **Temprano** o **Avanzado** es correcta hace falta una referencia original de estadio.

- **Evaluación trinaria.** Usa referencias de estadio verificadas para evaluar las clases Temprano y Avanzado. Una referencia binaria Sano se puede incluir en la fila Sano cuando su definición coincide con la del esquema trinario. Conserva el nivel original de cada etiqueta y reporta cuántos casos binarios Sano se incluyeron.
- **Evaluación binaria.** Las referencias binarias Glaucoma pueden compararse con una predicción de presencia. Si la salida proviene de un modelo trinario, identifica esa comparación como **binaria derivada** e indica la regla de proyección utilizada.
- **Reporte de resultados.** Excluye los casos Glaucoma sin estadio de los denominadores de las métricas trinarias y reporta sus cantidades por cohorte. Aplica el mismo criterio en matrices, tablas, figuras, informes y presentaciones. Una predicción del modelo jamás sustituye la etiqueta de referencia.

---

## 🎯 Delimitación del Objeto de Estudio y Marco Metodológico

* **Objeto Material:** Analizar la capacidad de generalización de clasificadores de glaucoma basados en redes neuronales profundas (arquitecturas convolucionales CNN y, como contraste empírico, transformers de visión) que operan sobre retinografías digitales de fondo de ojo, evaluando tanto el problema de detección binaria (*Sano vs. Glaucoma*) como el de estadificación clínica ordinal (*Sano vs. Incipiente / Temprano vs. Avanzado / Severo*).
* **Objeto Formal (Perspectiva de Estudio):** La evaluación sistemática del impacto sobre la capacidad de generalización inducido por la heterogeneidad en las condiciones de adquisición (variación inter-instrumental de sensores, fabricantes de cámaras y protocolos de iluminación) y por el encuadre óptico de la imagen (recorte focal del disco óptico a $35^\circ$ FOV frente a campo completo macular a $45^\circ$ FOV) entre la fuente de entrenamiento y la de prueba. Asimismo, se estudian y validan estrategias centradas en los datos (*data-centric AI*) para mitigar dicha degradación: entrenamiento multifuente centralizado, normalización espectral/cromática y aumento agresivo de datos (*Malignant Scale Augmentation*).
* **Criterio de Evaluación y Enfoque Clínico-Sanitario (TFP vs. TFN):** Sustitución del análisis exclusivo basado en métricas agregadas globales (como ROC-AUC) por una auditoría rigurosa de la **Tasa de Falsos Positivos (TFP / FPR)** y la **Tasa de Falsos Negativos (TFN / FNR)**, analizando formalmente el *tradeoff* clínico y económico:
  * **Minimización de TFN (Sensibilidad diagnóstica):** Prioridad clínica estricta orientada a evitar falsos negativos (pacientes con glaucoma omitidos que progresan asintomáticamente hacia la ceguera irreversible).
  * **Minimización de TFP (Especificidad y viabilidad asistencial):** Prioridad de gestión hospitalaria orientada a prevenir la saturación de los servicios de oftalmología especializada provocada por falsas alarmas en ojos sanos.
  * **Análisis de Punto Operativo:** Calibración sistemática de umbrales de decisión para optimizar el balance costo-beneficio según el escenario de cribado poblacional.

---

## 👥 Estructura y División del Trabajo por Módulos Funcionales

Para maximizar la autonomía y la modularidad del proyecto, el trabajo se encuentra estructurado en tres componentes sinérgicos:

* **Módulo 1 — Estado del Arte y Marco Teórico:**
  * Revisión Sistemática de Literatura (RSL) reproducible bajo directrices Kitchenham & Charters (2007) y PRISMA 2020.
  * Formalización del marco teórico: neuropatía óptica glaucomatosa, biomarcadores morfológicos (regla ISNT, excavación/disco) y taxonomía del *domain shift* instrumental en imagenología médica.
* **Módulo 2 — Ingesta Web, Curaduría de Datasets y Selección de Backbones:**
  * Adquisición de imágenes web *in-the-wild* (Google Images) y desarrollo de filtros discriminadores de calidad de fondo de ojo para depuración de artefactos y diagramas espurios.
  * Selección justificada y armonización de datasets científicos de acceso público.
  * Benchmark comparativo de arquitecturas base convolucionales (ResNet-18, ResNet-50, EfficientNet, MobileNet, ConvNeXt) y de atención visual (Vision Transformers), contrastando parámetros, latencia de inferencia y su comportamiento frente a TFP y TFN.
* **Módulo 3 — Paradigma de Clasificación (Trinario vs. Binario), Data Augmentation y Análisis de Features Causales:**
  * Evaluación experimental de esquemas de decisión: Clasificación Binaria vs. Estadificación Clínica Ordinal en tres etapas (Sano, Incipiente, Avanzado), analizando la distribución de errores adyacentes frente a errores catastróficos.
  * Diseño e implementación de *Data Augmentation* agresivo (*Malignant Scale Augmentation*) para inducir invarianza sintética ante cambios de escala y encuadre.
  * Análisis forense de características (*features*) en las imágenes determinantes de la degradación por transferencia cruzada (diferencias de ángulo de apertura $35^\circ$ vs. $45^\circ$, dispersión de histogramas cromáticos y densidad textural peripapilar).

---

## ⏱️ Directrices Clave y Compromisos de Entrega

* **Cronograma:** Fecha límite de entrega en la **Semana del 4 de diciembre** (ritmo ágil y avance desacoplado).
* **Fundamento Metodológico:** Uso indispensable de **Transfer Learning** sobre representaciones preentrenadas en ImageNet, ajustando las capas superiores a la tarea oftalmológica.
* **Entregables Obligatorios del Proyecto:**
  1. **Slides / Presentación:** Diapositivas ejecutivas y de alto impacto visual preparadas para la exposición oral y defensa metodológica.
  2. **Código Fuente y Pipelines Reproducibles:** Implementaciones en PyTorch limpias, comentadas y desacopladas en `scripts/experimentos/`.
  3. **Documento Formal Corto:** Manuscrito conciso siguiendo estrictamente las directrices y plantilla oficial de la **Revista DYNA** (Facultad de Minas - UNAL).
  4. **Demostración en Vivo (Live Demo):** Módulo interactivo de inferencia en tiempo real para clasificar retinografías en vivo durante la sustentación.

---

> ⚠️ **Aviso de Transparencia sobre Asistencia con IA:**  
> Parte de la documentación, estructuración de scripts y compilación de los informes técnicos de este repositorio contó con el apoyo de herramientas de Inteligencia Artificial Generativa (LLMs). Por este motivo, es posible que se presenten eventuales imprecisiones terminológicas, ligeras redundancias o expresiones con matices poco precisos en algunas secciones descriptivas. Ante cualquier discrepancia técnica o conceptual, se aconseja contrastar directamente la información con las implementaciones en `scripts/`, las métricas consolidadas en `results/` y los checkpoints de pesos en `modelos/`. El objetivo es que como equipo podamos analizar los resultados, contrastar las conclusiones y ajustar cualquier elemento que consideremos erróneo antes de la entrega final.

---

## 🧭 Guía de Inicio Rápido (Quickstart para Integrantes del Equipo)

Si acabas de clonar el repositorio y necesitas ponerte al día en 5 minutos sin tener que revisar todo el código desde cero, aquí tienes lo esencial:

### 1. Requisitos y Entorno
- **Python:** 3.10 o superior (probado en Python 3.11 / 3.12 / 3.14).
- **GPU (Opcional para inferencia, requerida para re-entrenar):** Compatible con CUDA (ejecutado originalmente en NVIDIA GeForce RTX 4060 Ti 16GB).
- **Instalación de dependencias:**
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install numpy pandas matplotlib scikit-learn pillow pymupdf requests beautifulsoup4
  ```

### 2. ¿Dónde están los modelos listos para usar?
No necesitas volver a entrenar los modelos para probarlos o extraer resultados. Los checkpoints que usa el proyecto se buscan en [`modelos/`](modelos/). El ConvNeXt-Tiny de ajuste fino está excluido de Git por tamaño y requiere una copia local.
- **`modelos/best_fine_tuned_convnext_tiny_glaucoma.pth`:** **MODELO GANADOR Y MÁS POTENTE.** Es el ConvNeXt-Tiny tras fine-tuning profundo (ROC-AUC global = **0.9669**, Exactitud = **90.72%**, Especificidad = **92.69%**, ACRIMA AUC = **1.0000**, EyePACS AUC = **0.9400** con 100% de especificidad). Es la red principal recomendada para el artículo.
- **`modelos/model_staged_multiclass_malignant.pth`:** **MODELO DE ESTADIFICACIÓN EN 3 CLASES** (Sano, Incipiente/Early, Avanzado/Advanced). Entrenado con Malignant Scale Augmentation. Alcanza un índice de acuerdo cuadrático $QWK = \mathbf{0.8163}$ y suprime los errores catastróficos a menos del $10\%$.
- **`modelos/model_staged_binary_malignant.pth`:** Modelo binario bajo aumento maligno (ROC-AUC = **0.9443**, Especificidad = **85.00%**).
- **`modelos/best_resnet18_1000ep_val_auc.pth`:** Mejor modelo del experimento de 1.000 épocas (Época 720, Val ROC-AUC = **0.9829**).
- **`modelos/final_resnet18_1000ep.pth`:** Checkpoint de la época 1000 (Especificidad en test: **94.98%**, con solo 11 falsos positivos en 431 imágenes).

### 3. ¿Dónde están los informes y gráficas para el artículo Revista DYNA?
Los informes técnicos publicados están en **PDF** en [`informes/`](informes/). Los compiladores de los informes 01 a 03 leen sus [fuentes Markdown](scripts/compiladores_informes/fuentes/).

El documento central que sintetiza absolutamente todo el proyecto es el **Informe Maestro 10**:
- 📄 **[INFORME_MAESTRO_GENERALIZACION_FINAL.pdf](informes/10_informe_maestro_generalizacion_final/INFORME_MAESTRO_GENERALIZACION_FINAL.pdf)** *(16.8 MB, 300 DPI. Estructurado como artículo científico completo para la Revista DYNA. Cubre desde el planteamiento del problema, auditoría de datos, benchmark de 10 redes, 2 vs 3 clases, hasta las 1.000 épocas y matriz de generalización).*

Asimismo, el **Estado del Arte completo del artículo** se estructuró como una **Revisión Sistemática de Literatura (RSL)** reproducible bajo directrices Kitchenham & Charters (2007) y PRISMA 2020. Consulta el [PDF compilado](estado_del_arte/estado_del_arte.pdf) y el [manuscrito reescrito](Mauricio/manuscrito_estado_del_arte/estado_del_arte_reescrito.md).
- 📖 **[estado_del_arte.md](Mauricio/manuscrito_estado_del_arte/estado_del_arte.md)** *(Manuscrito formal bajo formato Revista DYNA: 8 secciones numeradas, 32 referencias en orden de aparición con DOI, tablas horizontales, 6 figuras en escala de grises a 300 DPI y análisis autocrítico del proyecto).*

---

## 📚 Índice Maestro de Informes Técnicos del Proyecto

| N° | Carpeta / Documento PDF | Enfoque Principal y Contenido Clave | Estado |
| :---: | :--- | :--- | :---: |
| **RSL** | [Estado del arte (PDF)](estado_del_arte/estado_del_arte.pdf) | **ESTADO DEL ARTE Y REVISIÓN SISTEMÁTICA DE LITERATURA.** Manuscrito formal PRISMA 2020 / Kitchenham 2007 (DYNA), búsqueda en 4 APIs, Crossref, Kappa $\kappa=0.6484$, QA1-QA9 y 6 figuras. | 📖 RSL DYNA |
| **01** | [`01_experimento_invarianza_escala`](informes/01_experimento_invarianza_escala/INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.pdf) | Primeros hallazgos sobre la pérdida de generalización por atajo de escala en ventanas de recorte fijas. | ✅ Completado |
| **02** | [`02_experimento_mega_cohorte_multisensor`](informes/02_experimento_mega_cohorte_multisensor/INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.pdf) | Fusión inicial de bases de datos y evidencia del desplome de rendimiento al cruzar cámaras hospitalarias. | ✅ Completado |
| **03** | [`03_experimento_estadificacion_multietapa`](informes/03_experimento_estadificacion_multietapa/INFORME_EXPERIMENTO_CLASIFICACION_MULTIETAPA.pdf) | Primera aproximación a la clasificación en tres estadíos de severidad (Sano, Incipiente, Avanzado). | ✅ Completado |
| **04** | [`04_evaluacion_bases_medicas_y_mega_dataset`](informes/04_evaluacion_bases_medicas_y_mega_dataset/INFORME_EVALUACION_BASES_MEDICAS.pdf) | Auditoría forense de bases de datos médicas y **justificación de exclusión de `THIRD_DATASET`/ODIR** por ruido de máscaras. | ✅ Completado |
| **05** | [`05_entrenamiento_y_estres_malignant_scale_mega_cohorte`](informes/05_entrenamiento_y_estres_malignant_scale_mega_cohorte/INFORME_ENTRENAMIENTO_ESTRES_MALIGNO_MEGA_COHORTE.pdf) | Pipeline de armonización anatómica de ROI a $35^\circ$ FOV ($N=2,152$) y diseño de *MalignantScaleAugmentation*. | ✅ Completado |
| **06** | [`06_experimento_1000_epocas_sobreentrenamiento`](informes/06_experimento_1000_epocas_sobreentrenamiento/INFORME_EXPERIMENTO_1000_EPOCAS_SOBREENTRENAMIENTO.pdf) | Régimen longitudinal de 1.000 épocas continuas, serie temporal continua (Figura 29) y evidencia de *Grokking* tardío. | ✅ Completado |
| **07** | [`07_seleccion_modelo_base_10_candidatos`](informes/07_seleccion_modelo_base_10_candidatos/INFORME_SELECCION_MODELO_BASE_10_CANDIDATOS.pdf) | Benchmark multicriterio de 10 arquitecturas (CNNs, Transformers, Mobile) y fine-tuning de **ConvNeXt-Tiny**. | ✅ Completado |
| **08** | [`08_prediccion_2_vs_3_categorias`](informes/08_prediccion_2_vs_3_categorias/INFORME_PREDICCION_2_VS_3_CATEGORIAS.pdf) | Estudio comparativo: Detección Binaria (2 Clases) vs. Estadificación Ordinal (3 Clases), análisis de errores y QWK. | ✅ Completado |
| **09** | [`09_beneficio_data_augmentation_2_vs_3_clases`](informes/09_beneficio_data_augmentation_2_vs_3_clases/INFORME_BENEFICIO_DATA_AUGMENTATION_2_VS_3_CLASES.pdf) | Análisis cuantitativo del impacto de *MalignantScaleAugmentation* vs. Vanilla en ambos paradigmas clínicos. | ✅ Completado |
| **10** | [`10_informe_maestro_generalizacion_final`](informes/10_informe_maestro_generalizacion_final/INFORME_MAESTRO_GENERALIZACION_FINAL.pdf) | **INFORME MAESTRO DE GENERALIZACIÓN FINAL.** Manuscrito científico consolidado bajo formato Revista DYNA. | 🏆 Manuscrito Final |

---

## 🔬 Ejes de Investigación y Hallazgos Principales

### 1. Auditoría de Datos y Armonización Anatómica ($35^\circ$ FOV)
- **El Problema del Ángulo Óptico:** Cámaras como Canon CR-DGi operan a $35^\circ$ mientras que Centervue DRS (EyePACS) captura a $45^\circ$. Esto provoca que el disco óptico ocupe un área relativa distinta, haciendo que modelos no armonizados colapsen al cambiar de cámara (AUC $\approx 0.52$ en EyePACS).
- **Armonización determinista:** Recorte automático centrado en disco a $35^\circ$ FOV equivalente ($512 \times 512$ píxeles), normalizando la escala anatómica entre todas las cohortes.
- **Exclusión de `THIRD_DATASET` (ODIR):** Se descartó tras comprobar que contenía esqueletos vasculares binarizados en lugar de fotos RGB reales, además de etiquetas con co-morbilidades severas.

### 2. Aumento de Datos Invariante a Escala (`MalignantScaleAugmentation`)
- Operador estocástico que combina:
  - **Zoom-Out severo ($0.25-0.65\times$):** Incrusta el disco en un lienzo sintético con propiedades estadísticas del fondo de ojo y máscara circular pupilar.
  - **Zoom-In papilar ($1.25-1.85\times$):** Fuerza a la red a enfocarse en el anillo neurorretiniano y la lámina cribosa.
  - **Perturbación fotométrica:** Ganancia en canal verde ($0.90-1.15$) y corrección gamma ($0.75-1.35$).
- **Impacto:** Evita que la red use el tamaño aparente del disco como atajo heurístico y reduce los errores catastróficos a menos del $10\%$.

### 3. Benchmark Sistemático de 10 Arquitecturas para Transfer Learning
Evaluación sobre conjunto de prueba multicéntrico ($N=431$) bajo puntaje compuesto balanceado ($S_{comp}$):

| Ranking | Red Neuronal | Familia | Parámetros | Latencia GPU | ROC-AUC | Exactitud | AUC Cruzada | Score Compuesto ($S_{comp}$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **ConvNeXt-Tiny** ★ | Pure Modern CNN | 27.83 M | 12.82 ms | **0.9686** | **90.95%** | **0.9649** | **0.9197** |
| **2** | **ResNet-18** | Traditional CNN | 11.18 M | 5.86 ms | **0.9569** | **89.33%** | 0.9519 | **0.9188** |
| **3** | **Swin-T** | Vision Transformer | 27.53 M | 14.15 ms | **0.9654** | **90.49%** | 0.9612 | **0.9185** |
| **4** | ResNet-50 | Traditional CNN | 23.51 M | 9.74 ms | 0.9592 | 89.56% | 0.9540 | 0.9142 |
| **5** | ResNeXt-50 | Grouped CNN | 23.00 M | 11.20 ms | 0.9580 | 89.10% | 0.9525 | 0.9110 |
| **6** | DenseNet-121 | Dense CNN | 6.96 M | 13.50 ms | 0.9510 | 88.40% | 0.9450 | 0.9065 |
| **7** | EfficientNet-V2-S | Fused-MBConv | 20.18 M | 15.30 ms | 0.9480 | 87.70% | 0.9410 | 0.8995 |
| **8** | EfficientNet-B0 | Lightweight CNN | 4.01 M | 7.20 ms | 0.9420 | 86.77% | 0.9350 | 0.8980 |
| **9** | MobileNet-V3 | Edge Mobile CNN | 4.21 M | 6.40 ms | 0.9380 | 86.08% | 0.9300 | 0.8945 |
| **10** | ViT-B/16 | Plain Transformer | 85.80 M | 24.60 ms | 0.9022 | 82.13% | 0.8950 | 0.8674 |

> **Lección Arquitectónica Clave:** Las CNNs modernas (ConvNeXt) y residuales ligeras (ResNet-18) superaron contundentemente al Vision Transformer clásico (ViT-B/16). En datasets médicos de escala moderada ($N \approx 2,000$), el sesgo inductivo 2D de las convoluciones supera a la auto-atención irrestricta.

### 4. Clasificación Binaria (2 Clases) vs. Estadificación Clínica (3 Clases)
- **Falso confort del modelo binario:** En clasificación binaria, el modelo reporta ROC-AUC de $0.9151$, pero su sensibilidad en etapas tempranas es débil frente a los casos avanzados ($CDR > 0.8$).
- **Estadificación de 3 Clases (Sano $\prec$ Early $\prec$ Advanced):**
  - Permite descomponer los errores en **Adyacentes** (leves: Sano $\leftrightarrow$ Early o Early $\leftrightarrow$ Advanced) y **Catastróficos** (severos: Sano $\leftrightarrow$ Advanced).
  - Malignant Scale Augmentation redujo los errores catastróficos de $20.0\%$ a **$9.76\%$**, logrando $QWK = \mathbf{0.8163}$ (acuerdo casi perfecto).
  - La proyección binaria de la salida triclase alcanza un ROC-AUC proyectado de **$0.9334$**, combinando lo mejor de ambos mundos.

### 5. Dinámica Asintótica y Robustez a 1.000 Épocas
- Desmiente el mito del sobreentrenamiento destructivo en datos médicos cuando se utiliza regularización por peso y augmentación continua.
- Train loss colapsó a $0.0003$, mientras que la validación ROC-AUC se mantuvo en **$0.959 - 0.982$**, confirmando el fenómeno de **Grokking** (compresión y estabilización de representaciones latentes).

### 6. Revisión Sistemática de Literatura y Estado del Arte (Kitchenham & PRISMA 2020)
- **Metodología y Rigor:** Construcción desde cero del estado del arte siguiendo Kitchenham & Charters (2007) y PRISMA 2020 en formato de la **Revista DYNA** ([estado_del_arte.md](Mauricio/manuscrito_estado_del_arte/estado_del_arte.md)).
- **Búsqueda Programática en 4 APIs:** OpenAlex ($n=150$), Europe PMC ($n=62$), PubMed ($n=54$) y arXiv ($n=29$), sumando 295 registros brutos y 221 únicos tras deduplicación algorítmica.
- **Validación en Crossref API:** 190 DOIs verificados exitosamente con 0 retracciones.
- **Doble Cribado y Concordancia:** Evaluación independiente con Kappa de Cohen $\kappa = \mathbf{0.6484}$ (acuerdo sustancial, $82.35\%$ observado).
- **Calidad Metodológica y Extracción:** Evaluación QA1–QA9 ([calidad.csv](Mauricio/manuscrito_estado_del_arte/calidad.csv), media $8.33/9.00$) y matriz estructurada de 27 variables ([extraccion.csv](Mauricio/manuscrito_estado_del_arte/extraccion.csv)).
- **Hallazgos Clave de la Literatura:**
  - Caída inter-dominio media de $\Delta\text{AUC} = \mathbf{0.0982}$ (hasta $0.2310$ en protocolos *Leave-One-Domain-Out*).
  - Factor de confusión óptico más nocivo: la **disparidad angular del campo visual ($30^\circ-35^\circ$ papilar vs. $45^\circ$ panorámico macular)**.
  - Intervención de mayor costo-beneficio: **armonización de ROI papilar a $35^\circ$ combinada con aumento estocástico extremo de escala**.
- **Autocrítica y Advertencias Metodológicas:** Se contextualiza con honestidad que la evaluación de la mega-cohorte ($N=431$) es una validación interna estratificada sobre cohorte combinada (no LODO ciego), se rectifica la terminología a "entrenamiento multifuente centralizado" y se declara la procedencia abierta no auditada de Moondream en Hugging Face.

---

## 📁 Estructura General del Repositorio

| Ruta | Contenido |
| --- | --- |
| [`data/`](data/) | Cohortes originales y derivadas, incluidas ACRIMA, RIM-ONE DL, EyePACS y Moondream. |
| [`scripts/experimentos/`](scripts/experimentos/) | Pipelines de adquisición, armonización, entrenamiento y evaluación. |
| [`scripts/compiladores_informes/`](scripts/compiladores_informes/) | Compiladores de informes técnicos y del PDF del estado del arte. |
| [`modelos/`](modelos/) | Checkpoints de PyTorch. Los pesos grandes excluidos de Git se consultan en la copia local. |
| [`results/`](results/) y [`figures/`](figures/) | Métricas, índices de cohortes y figuras compartidas por los experimentos y los informes. |
| [`informes/`](informes/) y [`assets/`](assets/) | Informes técnicos en PDF y recursos de renderizado. |
| [`demo_glaucoma/`](demo_glaucoma/README.md) | Demo local de clasificación y guion de sustentación. |
| [`estado_del_arte/`](estado_del_arte/) | PDF compilado de la revisión sistemática. |
| [`Entrega Final/`](Entrega%20Final/) | [Guía maestra del artículo DYNA](Entrega%20Final/Gu%C3%ADa%20maestra%20del%20art%C3%ADculo%20DYNA%20generalizaci%C3%B3n%20de%20clasificadores%20de%20glaucoma.docx). |

### Carpetas de trabajo por integrante

- [Andrés Arbeláez](Andres_Arbelaez/) reúne la selección de datasets y de modelos, el [notebook de comprobación](Andres_Arbelaez/Comprobacion_mejor_modelo.ipynb) y el [índice de figuras de apoyo](Andres_Arbelaez/figuras_apoyo/README.md).
- [Fabián](Fabian/) reúne la comparación binaria y trinaria, el aumento de datos, el [análisis descriptivo de retinografías](Fabian/analisis_features/INFORME_CARACTERISTICAS_RETINOGRAFIAS.pdf) y el [índice de figuras de apoyo](Fabian/figuras_apoyo/README.md).
- [Mauricio](Mauricio/) reúne los [manuscritos y materiales de la revisión](Mauricio/manuscrito_estado_del_arte/), las [figuras de la revisión](Mauricio/figuras_estado_del_arte/), los [scripts de la RSL](Mauricio/scripts_rsl_y_teoria/) y las [fuentes del marco teórico](Mauricio/marco_teorico/).

---

## 🚀 Resumen de Comandos para Reproducción Rápida

Si deseas verificar o reproducir los experimentos clave en tu máquina:

```bash
# 1. Ejecutar el benchmark de 10 arquitecturas y fine-tuning del ganador
python scripts/experimentos/10_benchmark_10_backbones_transfer_learning.py

# 2. Ejecutar el estudio factorial 2 vs 3 clases y Data Augmentation
python scripts/experimentos/11_estudio_2_vs_3_clases_y_data_augmentation.py

# 3. Compilar los informes técnicos a ePDF:
python scripts/compiladores_informes/compilar_informe_seleccion_backbone.py
python scripts/compiladores_informes/compilar_informe_2_vs_3_categorias.py
python scripts/compiladores_informes/compilar_informe_beneficio_data_augmentation.py
python scripts/compiladores_informes/compilar_informe_maestro_generalizacion.py
```

Al recompilar localmente, los scripts pueden generar archivos intermedios Markdown o HTML en `informes/`.

### Revisión sistemática de literatura

El [protocolo](Mauricio/manuscrito_estado_del_arte/PROTOCOLO.md) y la [bitácora](Mauricio/manuscrito_estado_del_arte/LOG_PROTOCOLO.md) documentan el proceso. Los scripts se conservan en [`Mauricio/scripts_rsl_y_teoria/`](Mauricio/scripts_rsl_y_teoria/). Para reproducir la ejecución histórica con sus rutas, insumos y requisitos originales, consulta la versión anterior al commit `6b67090` (`6b67090^` en Git).

---

## 💡 Preguntas Frecuentes (FAQ)

### ¿Tengo que re-entrenar las redes para hacer pruebas o redactar el artículo?
**No.** Todos los modelos óptimos están guardados en `modelos/` y todas las métricas en `results/`. Puedes cargar directamente cualquiera de los checkpoints en PyTorch:
```python
import torch, torchvision.models as models
# Para ConvNeXt-Tiny ganador:
model = models.convnext_tiny(weights=None)
model.classifier[2] = torch.nn.Linear(768, 1)
model.load_state_dict(torch.load("modelos/best_fine_tuned_convnext_tiny_glaucoma.pth"))
model.eval()
```

### ¿Por qué `THIRD_DATASET` está excluida del entrenamiento?
Porque no contiene fotografías RGB de fondo de ojo. Son máscaras binarizadas de segmentación vascular (esqueletos vasculares en blanco y negro). Entrenar con ellas distorsiona los pesos convolucionales que dependen del color retiniano y la morfología de la excavación papilar.

### ¿Cómo compilar los PDFs si no tengo Chromium en una ruta estándar?
Los compiladores buscan automáticamente `msedge.exe` (Microsoft Edge en Windows) o `chrome.exe` (Google Chrome en Windows/Linux/macOS) y usan la bandera `--headless=new --print-to-pdf`.

---

*Facultad de Minas — Universidad Nacional de Colombia. Repositorio académico de investigación.*
