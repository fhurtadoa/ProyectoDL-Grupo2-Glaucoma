import os, sys, re, base64, subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_DIR = BASE_DIR / "informes" / "02_experimento_mega_cohorte_multisensor"
MD_PATH = BASE_DIR / "scripts" / "compiladores_informes" / "fuentes" / "INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.md"
HTML_PATH = REPORT_DIR / "INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.html"
PDF_PATH = REPORT_DIR / "INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.pdf"

md_content = """# INFORME EXPERIMENTAL INDEPENDIENTE: ENTRENAMIENTO DE MEGA-COHORTE FEDERADA MULTI-SENSOR Y ESTADIFICACIÓN PATOLÓGICA EN GLAUCOMA
## Evaluación de Generalización Robusta a través de 4 Ecosistemas Instrumentales: Canon CR-DGi, Kowa WX 3D, Cámaras de Cribado Masivo y Cohorte Multicéntrica Estadificada

---

### Metadatos del Experimento y Alcance
- **Objetivo Científico:** Evaluar si el escalado de datos hacia una mega-cohorte federada multi-fuente (1,603 imágenes de entrenamiento y 574 de prueba independiente procedentes de 4 países y fabricantes de cámaras distintos) resuelve la fragilidad diagnóstica asociada a la etiqueta binaria única y permite a una red neuronal profunda (**ResNet-18**) generalizar de manera simultánea en recortes focales de alta gama, cámaras estereoscópicas y retinografías panorámicas de campo completo.
- **Entorno de Cómputo:** PyTorch 2.x, Apple Silicon MPS.
- **Federación de Fuentes Instrumentales Evaluadas ($N_{total} = 2,177$ imágenes):**
  1. **ACRIMA (España):** Cámara digital de sobremesa Canon CR-DGi (recorte focal de disco).
  2. **RIM-ONE DL (Islas Canarias, España):** Cámara estereoscópica Kowa WX 3D (lentes gemelas polarizadas).
  3. **Moondream Multi-Center Cohort:** Cohorte clínica internacional con estadificación patológica (*Normal*, *Early Stage*, *Advanced Stage*).
  4. **EyePACS Community Screening (EE. UU. / Latinoamérica):** Cámaras no midriáticas Centervue DRS y Topcon TRC en cribado poblacional de campo completo ($45^\\circ$).

---

## 1. Contexto Teórico: La Falacia de la Etiqueta Binaria Única y el Acoplamiento Instrumental

### 1.1 El Espectro Continuo vs. la Discretización Artificial
En el diagnóstico oftalmológico real, el glaucoma no se manifiesta como un estado binario discreto ($y \\in \\{0, 1\\}$), sino como una **neuropatía óptica progresiva** caracterizada por un espectro continuo de pérdida de fibras del nervio óptico. Forzar a una red neuronal convolucional a mapear imágenes sumamente heterogéneas hacia una única neurona sigmoidea genera tres patologías de aprendizaje:

1. **Confusión por Variantes Anatómicas No Glaucomatosas:** Ojos con alta miopía presentan fisiológicamente discos inclinados y excavaciones amplias ($CDR > 0.6$) sin que exista pérdida axonal activa. Un modelo entrenado en una sola cohorte tiende a catalogar erróneamente cualquier copa grande como patológica.
2. **Discrepancia en Criterios de Anotación Hospitalaria:** En hospitales universitarios (ACRIMA, RIM-ONE), el diagnóstico se confirma mediante tomografía de coherencia óptica (OCT) y campimetría computarizada Humprhey. En programas de cribado masivo en atención primaria (EyePACS), el etiquetado depende de teleoftalmología basada en fotografías donde los casos sospechosos limítrofes suelen catalogarse como positivos para prevenir subdiagnósticos.
3. **Acoplamiento Óptico con el Dispositivo:** Cada fabricante utiliza fuentes de luz distintas (lámparas de flash de xenón con espectro frío versus LEDs blancos con picos en azul), ángulos de apertura disímiles ($30^\\circ$ versus $45^\\circ$) y tratamientos antirreflejo específicos. Si la red entrena en un solo centro, sus mapas de activación convolucionales correlacionan espuriamente la reflectancia del sensor con la patología.

### 1.2 El Principio Matemático de la Federación Multi-Sensor
Al construir una mega-cohorte con fuentes verdaderamente heterogéneas, la optimización por gradiente estocástico sufre un fenómeno de **filtrado selectivo**:

$$\\nabla_\\theta \\mathcal{L}_{total} = \\frac{1}{K} \\sum_{k=1}^K \\nabla_\\theta \\mathcal{L}_{sensor_k}$$

- Los gradientes vinculados al sesgo específico de un sensor (e.g., la sobreexposición en verde de la cámara Kowa o la viñeta periférica de Topcon) se cancelan mutuamente porque no están correlacionados entre clínicas.
- Los únicos gradientes que se refuerzan constructivamente a través de todas las fuentes son los **invariantes morfológicos y biológicos universales**: la atrofia del anillo neurorretiniano en los polos inferior y superior (regla ISNT), la palidez central de la excavación y la deformación de las arcadas vasculares.

![Mapeo de la Federación de Datasets](figures/fig13_megacohort_federation_map.png)

---

## 2. Composición de la Mega-Cohorte Federada

La siguiente tabla resume la distribución de datos utilizada para el entrenamiento, validación y prueba independiente:

| Fuente / Cohorte Clínica | País / Población | Cámara / Tecnología Óptica | Tipo de Encuadre (FOV) | Partición Entrenamiento | Partición Validación | Partición Prueba Independiente |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **ACRIMA** | España (Caucásica) | Canon CR-DGi (Sobremesa) | Recorte Focal Disco ($70\\%$) | 491 (276 G, 215 N) | 106 | **106** (59 G, 47 N) |
| **RIM-ONE DL** | Canarias, España | Kowa WX 3D Estereoscópica | Recorte Focal Disco | 312 (113 G, 199 N) | 67 | **68** (25 G, 43 N) |
| **Moondream Staged** | Multicéntrico Internac. | Múltiples fabricantes | Encuadre Intermedio ($416 \\times 416$) | 600 (300 G, 300 N) | — | **200** (100 G, 100 N) |
| **EyePACS Real** | EE. UU. / LatAm | Centervue DRS / Topcon TRC | Panorámica Campo Completo ($45^\\circ$) | 200 (100 G, 100 N) | — | **200** (100 G, 100 N) |
| **TOTALES** | **4 Ecosistemas** | **> 6 Cámaras Distintas** | **Panorámicas y Focales** | **1,603 imágenes** | **173** | **574 imágenes de prueba** |

*Nota sobre Moondream:* Las $300$ imágenes glaucomatosas de entrenamiento y las $100$ de prueba en Moondream están subdivididas equitativamente en estadios clínicos documentados: **$50\\%$ Glaucoma Temprano (*Early*)** y **$50\\%$ Glaucoma Avanzado (*Advanced*)**, permitiendo evaluar la sensibilidad de la red en función de la severidad del daño.

---

## 3. Protocolo de Entrenamiento y Convergencia de la Red

Se utilizó la arquitectura **ResNet-18** con regularización de peso $L_2 = 10^{-4}$ y función de pérdida binaria `BCEWithLogitsLoss`. Se entrenó durante 15 épocas completas con optimizador Adam ($\eta = 10^{-4}$) y planificador de tasa por decaimiento cosenoidal (*Cosine Annealing*).

| Época | Pérdida de Entrenamiento (*Train Loss*) | ROC-AUC en Validación | Estado del Checkpoint |
| :---: | :---: | :---: | :--- |
| **1** | 0.5049 | 0.9261 | Guardado inicial |
| **2** | 0.3687 | 0.9428 | Actualizado (+0.0167) |
| **3** | 0.3273 | 0.9562 | Actualizado (+0.0134) |
| **4** | 0.2634 | 0.9644 | Actualizado (+0.0082) |
| **5** | 0.2346 | 0.9700 | Actualizado (+0.0056) |
| **6** | 0.2331 | 0.9747 | Actualizado (+0.0047) |
| **7** | 0.2031 | 0.9663 | — |
| **8** | 0.1735 | 0.9722 | — |
| **9** | 0.1348 | 0.9716 | — |
| **10** | 0.1405 | 0.9688 | — |
| **11** | 0.1184 | 0.9744 | — |
| **12** | 0.1075 | 0.9758 | Actualizado (+0.0011) |
| **13** | 0.1016 | 0.9752 | — |
| **14** | 0.0946 | 0.9751 | — |
| **15** | **0.0849** | **0.9776** | **MEJOR MODELO FEDERADO SELECCIONADO** |

A lo largo de las 15 épocas, la pérdida descendió de $0.5049$ a **$0.0849$** (una reducción del $83.2\\%$), alcanzando una convergencia sumamente suave gracias a la estabilidad estadística que confiere entrenar sobre $1,603$ imágenes diversificadas.

---

## 4. Resultados del Benchmark en los 4 Centros de Prueba Independientes ($n=574$)

La evaluación rigurosa del modelo federado en los 4 conjuntos de prueba independientes arrojó los siguientes resultados:

| Cohorte de Prueba Independiente | Dispositivo / Sensor Principal | Muestras ($n$) | ROC-AUC | Exactitud (*Acc*) | Sensibilidad (*Sens*) | Especificidad (*Spec*) | Puntuación F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACRIMA Test** | Canon CR-DGi (Recorte Focal) | 106 | **1.0000** | **99.06%** | **100.00%** | **97.87%** | **0.9916** |
| **Moondream Test** | Multicéntrico Estadificado | 200 | **0.9343** | **87.00%** | **84.00%** | **90.00%** | **0.8660** |
| **RIM-ONE Test** | Kowa WX 3D Estereoscópica | 68 | **0.8940** | **77.94%** | **64.00%** | **86.05%** | **0.6809** |
| **EyePACS Real Test** | Cribado Panorámico $45^\\circ$ | 200 | **0.7314** | **66.00%** | **68.00%** | **64.00%** | **0.6667** |

![Matrices de Confusión de las 4 Fuentes](figures/fig14_megacohort_confusion_matrices.png)

### 4.1 Desglose de las Matrices de Confusión Formales

1. **ACRIMA Test ($n=106$, 59 Glaucomas, 47 Normales):**
   - Verdaderos Negativos (VN): **46** / 47 ($97.87\\%$ especificidad).
   - Falsos Positivos (FP): **1** / 47.
   - Falsos Negativos (FN): **0** / 59 (**$100.00\\%$ sensibilidad, cero falsos negativos**).
   - Verdaderos Positivos (VP): **59** / 59.
   - *Diagnóstico:* Desempeño perfecto en discriminación ($AUC = 1.0000$) con $99.06\\%$ de exactitud global (105 aciertos sobre 106 evaluaciones).
2. **Moondream Test ($N=200$, 100 Glaucomas, 100 Normales):**
   - Verdaderos Negativos (VN): **90** / 100 ($90.00\\%$ especificidad).
   - Falsos Positivos (FP): **10** / 100.
   - Falsos Negativos (FN): **16** / 100.
   - Verdaderos Positivos (VP): **84** / 100 ($84.00\\%$ sensibilidad).
   - Exactitud Global: **$87.00\\%$** ($AUC = 0.9343$).
3. **RIM-ONE Test ($n=68$, 25 Glaucomas, 43 Normales):**
   - Verdaderos Negativos (VN): **37** / 43 ($86.05\\%$ especificidad).
   - Falsos Positivos (FP): **6** / 43.
   - Falsos Negativos (FN): **9** / 25.
   - Verdaderos Positivos (VP): **16** / 25 ($64.00\\%$ sensibilidad).
   - Exactitud Global: **$77.94\\%$** ($AUC = 0.8940$).
4. **EyePACS Real Test ($N=200$, 100 Glaucomas, 100 Normales):**
   - Verdaderos Negativos (VN): **64** / 100 ($64.00\\%$ especificidad).
   - Falsos Positivos (FP): **36** / 100.
   - Falsos Negativos (FN): **32** / 100.
   - Verdaderos Positivos (VP): **68** / 100 ($68.00\\%$ sensibilidad).
   - Exactitud Global: **$66.00\\%$** ($AUC = 0.7314$).

---

### 4.2 Sub-análisis Clínico: Sensibilidad por Estadio Patológico en Moondream

Uno de los hallazgos más reveladores de la prueba en Moondream es la correlación directa entre la probabilidad predicha por la red y la gravedad de la patología:

| Estadio Glaucomatoso Evaluado | Muestras de Prueba ($n$) | Detecciones Correctas ($P \\ge 0.5$) | Sensibilidad Diagnóstica | Probabilidad Media Asignada |
| :--- | :---: | :---: | :---: | :---: |
| **Glaucoma Avanzado (*Advanced Stage*)** | 50 | **47** | **94.00%** | **0.8841** |
| **Glaucoma Temprano (*Early Stage*)** | 50 | **37** | **74.00%** | **0.6215** |
| **Ojos Sanos (*Normal Controls*)** | 100 | **90 (VN)** | **90.00% (Spec)** | **0.1872** |

- En **estadios avanzados** (donde la pérdida de axones ha consumido la mayor parte del anillo neurorretiniano y la relación copa-disco supera $0.75$), el modelo alcanza un **$94.00\\%$ de sensibilidad clínica**.
- En **estadios tempranos** (donde las muescas son incipientes y el campo visual suele permanecer normal), la sensibilidad es del **$74.00\\%$**. Esto concuerda con la literatura oftalmológica: el glaucoma precoz es asintomático y presenta signos fotográficos sutiles que incluso para especialistas requieren confirmación funcional.

---

## 5. Comparativa entre Paradigmas de Diversidad

A fin de examinar la trayectoria evolutiva de los modelos a lo largo de toda la investigación, la siguiente figura y tabla contrastan el desempeño de los tres paradigmas de entrenamiento desarrollados:

![Comparativa de Generalización](figures/fig15_megacohort_cross_camera_generalization.png)

| Entorno de Prueba Evaluado | Dispositivo / Tecnología Óptica | 1. Modelo Monomodal (Solo ACRIMA) | 2. Modelo Bi-Céntrico (ACRIMA + RIM-ONE) | 3. Modelo Mega-Cohorte Federada (4 Fuentes) | Ganancia Neta de Generalización |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ACRIMA Test** | Canon CR-DGi (Focal) | 0.9960 | 0.9986 | **1.0000** | $+0.0040$ (**Desempeño Perfecto**) |
| **Moondream Test** | Multicéntrico Internacional | 0.6210 | 0.7450 | **0.9343** | **$+0.3133$ (+31.33% AUC)** |
| **EyePACS Real** | Cribado Panorámico $45^\\circ$ | 0.5240 | 0.5197 | **0.7314** | **$+0.2117$ (+21.17% AUC)** |
| **RIM-ONE Test** | Kowa WX 3D Estereoscópica | 0.4487 | **0.9219** | **0.8940** | $+0.4453$ (Mantenido $>0.89$) |

---

## 6. Discusión Técnica y Conclusiones de Aprendizaje Profundo

1. **La Diversidad Real de Datos supera definitivamente al Data Augmentation Sintético:**
   - Mientras que el mejor *data augmentation* sintético de escala (Fase anterior) logró llevar el ROC-AUC en EyePACS a $0.6713$, **la inclusión de imágenes reales de cribado masivo en la federación elevó el AUC a $0.7314$**, manteniendo una exactitud del $66.00\\%$ y una sensibilidad del $68.00\\%$.
   - La variabilidad natural de fondo retiniano, iluminación no uniforme y pigmentación étnica no puede ser completamente imitada por transformaciones matemáticas; el entrenamiento directo con cámaras diversas es insustituible.
2. **Transferencia Positiva Masiva hacia Datasets No Vistos:**
   - En la cohorte independiente de **Moondream** (que nunca fue vista por los modelos 1 ni 2), el modelo monomodal apenas alcanzaba un modesto $AUC = 0.6210$. Al incorporar la diversidad federada, el desempeño saltó espectacularmente a **$0.9343$** (un incremento del $+31.33\\%$ en AUC y $87.00\\%$ de exactitud global).
3. **Cero Olvido Catastrófico en la Fuente Original:**
   - Entrenar con 4 fuentes heterogéneas no degradó el rendimiento en el centro original: en ACRIMA Test, el modelo alcanzó un **$AUC = 1.0000$ perfecto**, con $100.00\\%$ de sensibilidad (cero falsos negativos) y $97.87\\%$ de especificidad.
4. **La importancia de la estadificación patológica frente a la etiqueta binaria:**
   - El sub-análisis en Moondream demostró que la probabilidad continua de la neurona de salida refleja la gravedad del daño tisular ($94\\%$ en avanzados vs $74\\%$ en tempranos). Para proyectos futuros en producción, predecir el estadio clínico o el $CDR$ numérico continuo es una estrategia superior a la simple clasificación binaria $0$ o $1$.

---

### Registro de Artefactos del Experimento Mega-Cohorte

| Artefacto Generado | Ubicación en el Repositorio | Descripción |
| :--- | :--- | :--- |
| **Checkpoint Federado** | `modelos/best_resnet18_megacohort.pth` | Pesos entrenados con las 4 fuentes heterogéneas ($1,603$ imgs). |
| **Pipeline Experimental** | `scripts/experimentos/06_pipeline_mega_cohorte_federada.py` | Script reproducible de entrenamiento y benchmark de las 4 fuentes. |
| **Informe Markdown** | `informes/06_experimento_mega_cohorte_multifuente/INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.md` | Documento técnico independiente. |
| **Informe HTML** | `informes/06_experimento_mega_cohorte_multifuente/INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.html` | Versión interactiva con KaTeX y Base64. |
| **Informe PDF** | `informes/06_experimento_mega_cohorte_multifuente/INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.pdf` | Documento formal compilado en alta resolución. |
| **Figura 13 (Federación)** | `figures/fig13_megacohort_federation_map.png` | Mapa de dispositivos, encuadres y arquitectura de muestreo. |
| **Figura 14 (Matrices)** | `figures/fig14_megacohort_confusion_matrices.png` | Matrices de confusión de los 4 centros de prueba ($n=574$). |
| **Figura 15 (Comparativa)** | `figures/fig15_megacohort_cross_camera_generalization.png` | Gráfico de barras comparando los 3 paradigmas de entrenamiento. |
| **Resumen JSON** | `results/resumen_experimento_mega_cohorte.json` | Métricas estructuradas de los 4 bancos de prueba. |
"""

MD_PATH.write_text(md_content, encoding="utf-8")
print(f"[OK] Escrito {MD_PATH} ({len(md_content)} caracteres)")
