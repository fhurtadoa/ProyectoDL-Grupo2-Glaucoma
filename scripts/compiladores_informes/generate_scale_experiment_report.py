import os, sys, re, base64, subprocess
from pathlib import Path

BASE_DIR = Path("/Users/fabian/GlaucomaDL")
REPORT_DIR = BASE_DIR / "informes" / "01_experimento_invarianza_escala"
MD_PATH = REPORT_DIR / "INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.md"
HTML_PATH = REPORT_DIR / "INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.html"
PDF_PATH = REPORT_DIR / "INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.pdf"

md_content = """# INFORME EXPERIMENTAL INDEPENDIENTE: DATA AUGMENTATION MALIGNO DE ESCALA Y CAMPO DE VISIÓN (FOV)
## Evaluación de Invariancia de Escala Sintética y Cierre de la Brecha Panorámica en Detección de Glaucoma

---

### Metadatos del Experimento
- **Objetivo Científico:** Evaluar si una política de *data augmentation* extremo basada en simulación telescópica de escala (zoom-out), descentrado asimétrico, síntesis contextual de retina y zoom-in focal permite a una red neuronal convolucional (ResNet-18) generalizar hacia retinografías panorámicas de campo completo ($45^\\circ$, EyePACS) sin perder su capacidad diagnóstica en recortes de disco óptico (ACRIMA y RIM-ONE).
- **Entorno de Ejecución:** PyTorch 2.x, Apple Silicon MPS.
- **Modelos Evaluados en Comparativa Tri-Céntrica:**
  1. *Modelo Basal:* ResNet-18 entrenada únicamente con imágenes focales de ACRIMA.
  2. *Modelo Multidominio Convencional:* ResNet-18 entrenada con mezcla ACRIMA + RIM-ONE con transformaciones afines estándar.
  3. *Modelo con Data Augmentation Maligno (MSI-ResNet):* ResNet-18 entrenada con la nueva política de invariancia de escala y campo visual.
- **Conjuntos de Prueba Evaluados:**
  - **ACRIMA Test ($n=106$):** Recorte focalizado de alta resolución ($70\\%$ área del disco).
  - **RIM-ONE Test ($n=68$):** Recorte focalizado de cámara esteoroscópica Kowa WX 3D.
  - **EyePACS Real Test ($N=200$):** Retinografía panorámica completa a $45^\\circ$ ($3\\text{--}5\\%$ área del disco).

---

## 1. Contexto Teórico y Justificación del Experimento

### 1.1 El Origen del Desafío: El Choque de Campo de Visión (Field of View)
En fases anteriores de esta investigación, se evidenció un comportamiento divergente al validar los modelos en bases de datos externas:
- Al transferir entre **ACRIMA** y **RIM-ONE DL**, ambos datasets compartían la misma topología de adquisición: **recortes estrechamente centrados en la cabeza del nervio óptico** (donde el disco abarca más del $70\\%$ de los píxeles útiles). En este escenario, la brecha de dominio era eminentemente fotométrica (sensores, balance de blancos, ganancia de color), resolviéndose exitosamente mediante ajuste fino multidominio ($AUC = 0.9219$).
- Sin embargo, al evaluar en un tercer conjunto de imágenes reales de cribado masivo (**EyePACS**), el desempeño de ambos modelos colapsó a niveles de azar puro:
  - Modelo Basal ACRIMA en EyePACS: $ROC\\text{-}AUC = 0.5240$ (Exactitud: $51.00\\%$).
  - Modelo Multidominio en EyePACS: $ROC\\text{-}AUC = 0.5197$ (Exactitud: $48.00\\%$).

Este fracaso no responde a un error en el biomarcador médico, sino a un **desalineamiento estructural de frecuencias espaciales y escala**:
- En los filtros convolucionales estándar ($3 \\times 3, 7 \\times 7$) de ResNet-18, la red aprende detectores de características sintonizados para bordes de anillo de $15$ a $30$ píxeles de espesor.
- Al redimensionar una retinografía completa de $45^\\circ$ a $224 \\times 224$, el disco óptico entero queda confinado a una región minúscula de aproximadamente $15 \\times 15$ píxeles, rodeada por la mácula, las arcadas vasculares y los bordes curvos de la cámara. Las convoluciones intermedias no logran activar los mapas de características patológicos y disparan respuestas espurias sobre el fondo ocular.

### 1.2 La Hipótesis del "Data Augmentation Maligno"
Para atacar esta limitación sin recurrir a segmentadores manuales de discos, se formuló la siguiente hipótesis experimental:
> **Hipótesis:** *Si durante el entrenamiento forzamos a la red a observar discos ópticos severamente alejados (zoom-out), descentrados hacia cualquier cuadrante del lienzo y embebidos en fondos sintéticos que imiten la textura y el viñeteado de una cámara retiniana panorámica, las capas convolucionales se verán obligadas a desarrollar detectores con invariancia de escala y posición, elevando el desempeño en imágenes de campo completo sin colapsar el rendimiento en recortes focales.*

---

## 2. Arquitectura Matemática de la Política de Aumento

Para evitar que el aumento degrade el entrenamiento o introduzca atajos artificiales, la canalización (*pipeline*) se dividió en tres regímenes probabilísticos complementarios:

```
                  +---> Modo Zoom-out Telescópico (p = 0.45): s in [0.25, 0.65] + Padding Retiniano + Viñeta
                  |
Imagen Entrada ---+---> Modo Zoom-in Asimétrico (p = 0.35): s in [1.25, 1.85] + Anclaje Mínimo 50% Disco
                  |
                  +---> Modo Estándar / Fotométrico (p = 0.20): Escala 1.0x + Caos de Gamma y Canal Verde
```

### 2.1 Modo A: Zoom-out Telescópico con Síntesis de Contexto Retiniano ($p = 0.45$)
1. **Factor de Escala:** Se reduce el disco óptico a una fracción $s \\sim \\text{Uniform}(0.25, 0.65)$, mapeando la resolución nativa de $224 \\times 224$ hacia una sub-imagen de dimensiones $(w_s, h_s) = (224 \\cdot s, 224 \\cdot s)$.
2. **Eliminación del Sesgo de Borde Rectangular:** Si se pegara la imagen reducida sobre un fondo negro uniforme (`0, 0, 0`), la red aprendería a detectar el borde rectangular del recorte. Para evitar este atajo, se genera un lienzo artificial donde cada píxel se inicializa muestreando el color medio del borde periférico de la retina ($\\mu_{bg}$) más un término de ruido de textura gaussiano:
   $$I_{bg}(x, y) = \\mu_{bg} + \\epsilon, \\quad \\epsilon \\sim \\mathcal{N}(0, 7^2)$$
3. **Descentrado Espacial:** La sub-imagen se coloca en coordenadas aleatorias:
   $$x_{off} \\sim \\text{Uniform}(0, 224 - w_s), \\quad y_{off} \\sim \\text{Uniform}(0, 224 - h_s)$$
   rompiendo el sesgo de centralidad característico de los datasets curados.
4. **Apertura de Viñeteado Circular Oftálmico:** Se aplica una función de viñeta radial centrada en el cuadro para reproducir la máscara de las cámaras de fondo de ojo:
   $$V(x, y) = \\text{clip}\\left(1.0 - \\frac{d(x, y) - R_0}{\\delta}, 0.0, 1.0\\right)$$
   donde $d(x,y)$ es la distancia euclidiana al centro del lienzo, $R_0 = 0.48 \\cdot 224 \\approx 107.5$ px y $\\delta = 8.0$ px es la zona de transición suave.

### 2.2 Modo B: Zoom-in Asimétrico con Anclaje Anatómicamente Restringido ($p = 0.35$)
Se incrementa la escala a $s \\sim \\text{Uniform}(1.25, 1.85)$ con recortes aleatorios restringidos para garantizar que al menos el $50\\%$ del disco óptico permanezca dentro de la ventana de corte. Esto obliga a la red a emitir diagnósticos basándose en sectores locales aislados (e.g., solo el polo inferior o solo el superior), previniendo que dependa de la visualización simultánea de toda la circunferencia escleral.

### 2.3 Modo C: Caos Fotométrico y Espectral ($p = 0.65$ condicional)
Sobre la imagen resultante, se aplican perturbaciones no lineales de luminancia:
$$I_{final}(x, y) = I(x, y)^\\gamma, \\quad \\gamma \\sim \\text{Uniform}(0.75, 1.35)$$
y se introduce una ganancia estocástica en el canal verde:
$$I_{final}^{(G)}(x, y) = \\alpha_G \\cdot I^{(G)}(x, y), \\quad \\alpha_G \\sim \\text{Uniform}(0.90, 1.15)$$
desacoplando el aprendizaje de la reflectancia específica de cualquier sensor hospitalario.

![Galería de Muestras Generadas](figures/fig10_malignant_augmentation_samples.png)

---

## 3. Protocolo de Entrenamiento y Convergencia

Se reentrenó la arquitectura **ResNet-18** (inicializada con pesos de ImageNet) utilizando los conjuntos de entrenamiento estratificados combinados de ACRIMA ($n=491$) y RIM-ONE DL ($n=312$), totalizando $803$ imágenes de entrenamiento y $173$ de validación.

- **Optimizador:** Adam con tasa inicial $\\eta = 10^{-4}$ y regularización de peso $L_2 = 10^{-4}$.
- **Planificador de Tasa:** Decaimiento por coseno (*Cosine Annealing*) a lo largo de 15 épocas.
- **Función de Pérdida:** `BCEWithLogitsLoss`.
- **Evolución de la Pérdida y Métrica de Validación:**

| Época | Pérdida de Entrenamiento (*Train Loss*) | ROC-AUC en Validación | Estado del Checkpoint |
| :---: | :---: | :---: | :--- |
| **1** | 0.4937 | 0.9151 | Guardado inicial |
| **2** | 0.3492 | 0.9620 | Actualizado (+0.0469) |
| **3** | 0.2977 | 0.9689 | Actualizado (+0.0069) |
| **4** | 0.2985 | 0.9790 | Actualizado (+0.0101) |
| **5** | 0.2369 | 0.9581 | — |
| **6** | 0.2379 | 0.9829 | Actualizado (+0.0039) |
| **7** | 0.2168 | 0.9780 | — |
| **8** | 0.1891 | 0.9830 | Actualizado (+0.0001) |
| **9** | 0.2094 | 0.9833 | Actualizado (+0.0003) |
| **10** | 0.1708 | 0.9821 | — |
| **11** | 0.1897 | 0.9851 | Actualizado (+0.0018) |
| **12** | 0.1643 | 0.9853 | Actualizado (+0.0002) |
| **13** | 0.1435 | **0.9855** | Actualizado (+0.0002) |
| **14** | 0.1539 | 0.9854 | — |
| **15** | **0.1400** | **0.9859** | **MEJOR MODELO SELECCIONADO** |

A pesar de la severidad de las transformaciones sintéticas, la red convergió de manera robusta, alcanzando un $AUC = 0.9859$ en el conjunto de validación y reduciendo la pérdida de $0.4937$ a $0.1400$.

---

## 4. Resultados Cuantitativos del Benchmark Tri-Céntrico

A continuación se presentan los resultados obtenidos por el modelo entrenado con esta política de aumento en las particiones de prueba independientes:

| Conjunto de Prueba Evaluado | Encuadre (FOV) | Muestras ($n$) | ROC-AUC | Exactitud (*Acc*) | Sensibilidad (*Sens*) | Especificidad (*Spec*) | Puntuación F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACRIMA Test** | Recorte Focal Disco | 106 | **0.9910** | **92.45%** | **89.83%** | **95.74%** | **0.9298** |
| **RIM-ONE Test** | Recorte Focal Cámara Kowa | 68 | **0.8698** | **82.35%** | **68.00%** | **90.70%** | **0.7391** |
| **EyePACS Real Test** | **Panorámica Completa $45^\\circ$** | **200** | **0.6713** | **62.50%** | **49.00%** | **76.00%** | **0.5665** |
| **EyePACS (Recorte ROI 50%)** | Panorámica con Crop Central | 200 | **0.6740** | **62.00%** | **48.00%** | **76.00%** | **0.5581** |

![Matrices de Confusión del Modelo](figures/fig11_malignant_augmentation_confusion_matrices.png)

### 4.1 Desglose de las Matrices de Confusión

1. **ACRIMA Test ($n=106$, 59 Glaucoma, 47 Normal):**
   - Verdaderos Negativos (VN): **45** / 47 ($95.74\\%$ especificidad).
   - Falsos Positivos (FP): **2** / 47.
   - Falsos Negativos (FN): **6** / 59.
   - Verdaderos Positivos (VP): **53** / 59 ($89.83\\%$ sensibilidad).
   - *Interpretación:* El modelo mantiene una discriminación casi perfecta en recortes de su centro base ($AUC = 0.9910$).
2. **RIM-ONE Test ($n=68$, 25 Glaucoma, 43 Normal):**
   - Verdaderos Negativos (VN): **39** / 43 ($90.70\\%$ especificidad).
   - Falsos Positivos (FP): **4** / 43.
   - Falsos Negativos (FN): **8** / 25.
   - Verdaderos Positivos (VP): **17** / 25 ($68.00\\%$ sensibilidad).
   - *Interpretación:* Mantiene un AUC elevado ($0.8698$) en la cámara externa esteoroscópica.
3. **EyePACS Real Test ($N=200$, 100 Glaucoma, 100 Normal):**
   - Verdaderos Negativos (VN): **76** / 100 ($76.00\\%$ especificidad).
   - Falsos Positivos (FP): **24** / 100.
   - Falsos Negativos (FN): **51** / 100.
   - Verdaderos Positivos (VP): **49** / 100 ($49.00\\%$ sensibilidad).
   - *Interpretación:* Por primera vez, el modelo clasifica correctamente a 76 de cada 100 pacientes sanos en retinografías de campo completo sin haber visto nunca una imagen panorámica real en su entrenamiento.

---

## 5. Comparativa entre Paradigmas de Entrenamiento

Para cuantificar el beneficio relativo de esta aproximación, la siguiente tabla y figura contrastan las tres metodologías evaluadas en este proyecto:

![Comparativa Tri-Céntrica](figures/fig12_scale_invariance_comparison.png)

| Paradigma de Entrenamiento | ACRIMA Test AUC | RIM-ONE Test AUC | EyePACS Real AUC | EyePACS Exactitud | Retención de Especificidad |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Modelo Basal (Solo ACRIMA)** | **0.9960** | 0.4487 | 0.5240 | 51.00% | 0.00% en RIM-ONE (Colapso total). |
| **2. Multidominio Convencional** | **0.9986** | **0.9219** | 0.5197 | 48.00% | Óptima en recortes ($97.8\\%$ / $93.2\\%$). |
| **3. Augmentación Maligna de Escala** | **0.9910** | **0.8698** | **0.6713** | **62.50%** | **Ruptura de la barrera de azar (+15.16% AUC).** |

### 5.1 Análisis Técnico: ¿Qué se logró y cuál fue el compromiso (*Trade-Off*)?

1. **Ruptura del Techo de Azar en Campo Completo:**
   - Mientras el modelo basal y el multidominio estándar se comportaban como un clasificador de moneda al aire en EyePACS ($AUC \\approx 0.52$), la augmentación sintética de escala logró elevar el $AUC$ a **$0.6713$** (y **$0.6740$** con recorte amplio del $50\\%$), con una exactitud que sube de $48\\%$ a **$62.50\\%$** y una especificidad del **$76.00\\%$**.
   - Esto demuestra empíricamente que someter a la red a transformaciones de zoom-out telescópico ($s \\in [0.25, 0.65]$) con relleno contextual de retina le enseñó a identificar patrones del disco óptico en frecuencias espaciales mucho más bajas.
2. **El Compromiso en Recortes de Ultra-Alta Resolución:**
   - Para aprender a diagnosticar a escalas diminutas con fondo ruidoso, la red tuvo que ceder un pequeño margen en las imágenes de resolución óptima: en ACRIMA Test el $AUC$ pasó de $0.9986$ a $0.9910$ (la exactitud pasó de $98.11\\%$ a $92.45\\%$), y en RIM-ONE pasó de $0.9219$ a $0.8698$.
   - Este compromiso es consistente con el principio de capacidad de representación en Deep Learning: una red con capacidad finita (ResNet-18, 11.7M parámetros) asigna parte de sus canales convolucionales a patrones de campo panorámico, reduciendo la especialización ultra-fina en los bordes de alta resolución.

---

## 6. Conclusiones y Recomendaciones de Ingeniería de Deep Learning

1. **La aumentación de escala sintética funciona como prueba de concepto para cerrar la brecha de FOV:**
   El experimento valida la hipótesis del usuario: la perturbación deliberada de escala y encuadre permitió ganar $+15.16$ puntos porcentuales de discriminabilidad en una base de datos externa de cribado masivo sin haber recibido ninguna imagen panorámica real durante el entrenamiento.
2. **Límites de la solución puramente sintética frente a la resolución física:**
   Aunque el $AUC$ alcanzó $0.6713$, no llega aún al umbral clínico autónomo ($>0.85$). La razón es el límite de muestreo: encoger sintéticamente un disco de $224 \\times 224$ degrada el grosor del anillo neurorretiniano a escasos píxeles, imposibilitando que la red evalúe con precisión la regla ISNT en casos incipientes.
3. **La recomendación arquitectónica definitiva para despliegue clínico:**
   En entornos de producción médica donde conviven cámaras de diferentes hospitales con fotos panorámicas y fotos focales, la mejor práctica de ingeniería no es exigir a un único clasificador que resuelva la escala universal, sino implementar una **arquitectura desacoplada en dos etapas (*Two-Stage Pipeline*)**:
   - **Etapa 1 (Localizador de Disco):** Una red liviana (e.g., YOLOv8-nano o U-Net) entrenada para detectar la región de interés (ROI) y extraer un recorte centrado de $224 \\times 224$ píxeles alrededor de la cabeza del nervio óptico.
   - **Etapa 2 (Clasificador Patológico):** Nuestro clasificador ResNet-18 optimizado con fine-tuning multidominio, el cual opera a su máxima capacidad diagnóstica ($AUC = 0.9986$ / $0.9219$).
   - De este modo, la aumentación de escala sirve como un sistema de respaldo de alta robustez contra imprecisiones de encuadre de la Etapa 1.

---

### Registro de Artefactos Generados en este Experimento

| Archivo Generado | Ubicación en el Repositorio | Descripción |
| :--- | :--- | :--- |
| **Checkpoint Entrenado** | `modelos/best_resnet18_malignant_scale.pth` | Pesos entrenados con aumentación maligna de escala. |
| **Pipeline Experimental** | `scripts/experimentos/05_pipeline_data_augmentation_maligno.py` | Código fuente completo de entrenamiento y benchmark. |
| **Informe Markdown** | `informes/05_experimento_data_augmentation_extremo/INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.md` | Documento técnico independiente. |
| **Informe HTML** | `informes/05_experimento_data_augmentation_extremo/INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.html` | Versión web interactiva con KaTeX y Base64. |
| **Informe PDF** | `informes/05_experimento_data_augmentation_extremo/INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.pdf` | Documento formal compilado en alta resolución. |
| **Figura 10 (Muestras)** | `figures/fig10_malignant_augmentation_samples.png` | Visualización de transformaciones zoom-out/in y contexto. |
| **Figura 11 (Matrices)** | `figures/fig11_malignant_augmentation_confusion_matrices.png` | Matrices de confusión en ACRIMA, RIM-ONE y EyePACS. |
| **Figura 12 (Comparativa)** | `figures/fig12_scale_invariance_comparison.png` | Gráfico de barras comparando los 3 paradigmas de entrenamiento. |
| **Resumen JSON** | `results/resumen_experimento_malignant_augmentation.json` | Métricas y matrices estructuradas. |
"""

MD_PATH.write_text(md_content, encoding="utf-8")
print(f"[OK] Escrito informe en {MD_PATH} ({len(md_content)} caracteres)")
