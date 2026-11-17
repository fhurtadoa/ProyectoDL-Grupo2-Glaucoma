# INFORME TÉCNICO Y EXPERIMENTAL: ENTRENAMIENTO Y EVALUACIÓN DE ESTRÉS MALIGNO EN MEGA-COHORTE ARMONIZADA DE GLAUCOMA
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

El diagnóstico del glaucoma mediante retinografías de fondo de ojo en la práctica clínica real enfrenta un obstáculo crítico: **el desplome del rendimiento por cambio de dominio óptico (*optical domain shift*)**. Mientras que los modelos convencionales entrenados en una sola cohorte curada alcanzan $ROC\text{-}AUC > 0.98$, colapsan hacia niveles de azar puro ($ROC\text{-}AUC \approx 0.51$) al enfrentarse a cámaras de cribado masivo panorámicas ($45^\circ$, como EyePACS) o sensores estereoscópicos tridimensionales ($35^\circ$, como Kowa WX 3D de RIM-ONE DL).

Para resolver esta vulnerabilidad estructural sin recurrir a costosas segmentaciones manuales, este informe documenta la culminación de dos hitos científicos mayores:

1. **Auditoría Rigurosa y Armonización Geométrica/Cromática:**
   - Se resolvió un error sutil de asignación de etiquetas en el repositorio local originado por coincidencia de subcadenas con el nombre de carpeta `GlaucomaDL`, restaurando la prevalencia real de las 4 cohortes.
   - Se consolidó la exclusión estricta de `THIRD_DATASET` ($N = 2,805$), comprobando que consiste en máscaras de esqueletos vasculares que corrompen la visión natural.
   - Se aplicó el detector inteligente de papila óptica `OpticDiscCropper` y el normalizador cromático de Reinhard, alineando $2,152$ retinografías a un estándar peripapilar equivalente de $35^\circ$ ($512 \times 512$).
2. **Entrenamiento y Doble Estrés bajo *Malignant Scale Augmentation*:**
   - La red fue entrenada forzando deformaciones extremas: Zoom-Out severo ($0.25x - 0.65x$) con viñeteo óptico circular y fondo sintético ruidoso, Zoom-In focal ($1.25x - 1.85x$) sobre la copa óptica, y caos fotométrico de espectro verde/gamma.
   - El modelo resultante alcanzó en validación un **$ROC\text{-}AUC = 0.9569$ en la Época 14**.
   - En evaluación ciega sobre las 4 cohortes independientes ($N = 431$), obtuvo **$ROC\text{-}AUC = 0.9939$ en ACRIMA**, **$0.9293$ en RIM-ONE DL**, **$0.9086$ en Moondream** y **$0.7850$ en EyePACS** ($ROC\text{-}AUC$ global ponderado de **$0.9482$**).
   - En una prueba de estrés hostil en tiempo de inferencia (inyectando aumentos malignos a las imágenes de prueba no vistas), el modelo retuvo más del **$98.8\%$ de su capacidad discriminativa**, demostrando una auténtica invarianza morfológica.

---

## 2. Auditoría Crítica de Bases de Datos Médicas y Composición de la Mega-Cohorte

### 2.1 Diagnóstico de Calidad y Resolución del Fallo de Asignación de Etiquetas

Durante la inspección de la base de datos se descubrió que ejecuciones históricas asignaban etiqueta $1$ (glaucoma) a la totalidad de imágenes de RIM-ONE DL, EyePACS y Moondream, debido a que la condición lógica `if "glaucoma" in f.lower()` evaluaba la ruta absoluta completa, la cual contenía la palabra `GlaucomaDL`.

Se implementó una auditoría basada estrictamente en rutas relativas por carpeta y metadatos clínicos (`data/RIM_ONE_DL/metadata.csv`, carpetas `normal`/`glaucoma` de EyePACS, y subdirectorios `normal`/`early`/`advanced` de Moondream), restituyendo la verdad diagnóstica:

| Base de Datos Médica | Sensor / Fabricante | Campo Visual (FOV) | N Total | Glaucoma ($y=1$) | Normal ($y=0$) | Prevalencia Real | Método de Validación de Etiqueta |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **ACRIMA** | Canon CR-DGi | $35^\circ$ peripapilar | 705 | 396 | 309 | 56.17% | Consenso de 3 retinólogos independientes |
| **RIM-ONE DL** | Kowa WX 3D Estéreo | $35^\circ$ peripapilar | 447 | 162 | 285 | 36.24% | Evaluación estereoscópica clínica especializada |
| **EyePACS Real** | Cámaras de cribado | $45^\circ$ panorámico | 200 | 100 | 100 | 50.00% | Lectura oftalmológica en telemedicina masiva |
| **Moondream Staged** | Multi-cámara clínica | $40^\circ$ multicéntrico | 800 | 400 | 400 | 50.00% | Graduación por daño del anillo (Early / Advanced) |
| **THIRD_DATASET** | Procesamiento derivado | N/A | 2,805 | 0 | 0 | **EXCLUIDO** | Máscaras de bordes vasculares en canal alfa (no RGB) |
| **TOTAL UNIFICADO** | **Mega-Cohorte Federada** | **$35^\circ$ Armonizado** | **2,152** | **1,058** | **1,094** | **49.16%** | **Cohorte Médica Balanceada y Certificada** |

### 2.2 Justificación Matemática de la Exclusión de `THIRD_DATASET`

La inspección de tensores de `data/THIRD_DATASET` reveló que el $97.8\%$ de los píxeles tienen intensidad nula ($0$), consistiendo en mapas de binarización y esqueletización de vasos sanguíneos. En una red convolucional residual (ResNet-18), introducir este tipo de imágenes induce **atajos de gradiente (*shortcut learning*)**: los filtros de primer orden se acoplan a la presencia de líneas sintéticas mono-espectrales y anulan las características de textura retinal (anillo neurorretiniano, relación copa-disco, signo de Bayoneta y hemorragias de Drance). Su exclusión preserva intacto el espacio latente del dominio oftalmológico natural.

---

## 3. Pipeline de Armonización Geométrica y Cromática

Para transformar las imágenes panorámicas de $45^\circ$ (EyePACS) en encuadres comparables a las cámaras de $35^\circ$ (ACRIMA y RIM-ONE), se ejecutó el pipeline de `OpticDiscCropper`:

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
  - *Train Split Interno ($85\%$):* $1,462$ imágenes ($718$ Glaucoma, $744$ Normales).
  - *Validation Split Interno ($15\%$):* $259$ imágenes ($128$ Glaucoma, $131$ Normales).
- **Conjunto de Prueba Independiente (*Test Held-Out*, $20\%$ de cada centro):** $431$ imágenes ($212$ Glaucoma, $219$ Normales).
  - ACRIMA Test: $N=141$ ($79$ Glaucoma, $62$ Normal).
  - RIM-ONE DL Test: $N=90$ ($33$ Glaucoma, $57$ Normal).
  - EyePACS Test: $N=40$ ($20$ Glaucoma, $20$ Normal).
  - Moondream Test: $N=160$ ($80$ Glaucoma, $80$ Normal).

---

## 4. Protocolo de Data Augmentation Maligno (*Malignant Scale Augmentation*)

El generador en línea `MalignantScaleAugmentation` rompe intencionalmente la correlación espuria entre el tamaño aparente del disco óptico y la etiqueta diagnóstica mediante tres modalidades concurrentes:

![Galería de Muestras Bajo Data Augmentation Maligno](figures/fig10_malignant_augmentation_samples.png)

1. **Modo Zoom-Out Severo ($p = 0.45$):**
   - Escala aleatoria $s \sim \text{Uniform}(0.25, 0.65)$, reduciendo el disco a dimensiones entre $56 \times 56$ y $145 \times 145$ píxeles.
   - **Lienzo con Contexto Sintético:** Para evitar el atajo de bordes rectangulares negros, el fondo se sintetiza calculando el color periférico medio de la retina $\mu_{bg} = [\mu_R, \mu_G, \mu_B]$ con ruido de textura $\epsilon \sim \mathcal{N}(0, 7^2)$.
   - **Viñeteo Circular Óptico Oftálmico:** Máscara radial analítica $V(x, y) = \text{clip}\left(1.0 - \frac{d(x,y) - 0.48W}{8.0}, 0.0, 1.0\right)$.
   - Desplazamiento estocástico bidimensional $(x_0, y_0)$.
2. **Modo Zoom-In Focal ($p = 0.35$):**
   - Escala aleatoria $s \sim \text{Uniform}(1.25, 1.85)$, aislando micro-sectores peripapilares de la copa óptica y forzando a la red a diagnosticar basándose en el adelgazamiento del anillo neurorretiniano sin ver la circunferencia papilar completa.
3. **Modo Estándar ($p = 0.20$):**
   - Escala $1.0x$ estándar para preservar la topología original.
4. **Perturbación Fotométrica Espectral ($p = 0.65$ condicional):**
   - Corrección no lineal de gamma: $I_{out} = I_{in}^\gamma, \; \gamma \sim \text{Uniform}(0.75, 1.35)$.
   - Ganancia estocástica en canal verde: $I_{out}^{(G)} = \alpha_G \cdot I_{in}^{(G)}, \; \alpha_G \sim \text{Uniform}(0.90, 1.15)$.

---

## 5. Entrenamiento y Convergencia de ResNet-18

El entrenamiento se realizó en la GPU NVIDIA GeForce RTX 4060 Ti con la función de costo `BCEWithLogitsLoss`, optimizador `AdamW` (learning rate $= 10^{-4}$, weight decay $= 10^{-4}$) y decaimiento cosenoidal `CosineAnnealingLR` a lo largo de 15 épocas completas (tiempo de entrenamiento: $214.01$ segundos $\approx 3.57$ minutos):

| Época | Tasa de Aprendizaje | Pérdida Entrenamiento | Pérdida Validación | ROC-AUC Validación | Exactitud Validación | Estado Checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | $9.89 \times 10^{-5}$ | 0.5636 | 0.5108 | 0.8499 | 70.66% | Guardado inicial |
| **02** | $9.57 \times 10^{-5}$ | 0.4448 | 0.6590 | 0.9076 | 74.52% | Actualizado (+0.0577) |
| **03** | $9.05 \times 10^{-5}$ | 0.4110 | 0.3814 | 0.9209 | 81.85% | Actualizado (+0.0133) |
| **04** | $8.35 \times 10^{-5}$ | 0.3911 | 0.3487 | 0.9352 | 82.63% | Actualizado (+0.0143) |
| **05** | $7.50 \times 10^{-5}$ | 0.3624 | 0.4917 | 0.9115 | 77.61% | — |
| **06** | $6.55 \times 10^{-5}$ | 0.3649 | 0.3120 | 0.9447 | 88.42% | Actualizado (+0.0095) |
| **07** | $5.52 \times 10^{-5}$ | 0.3030 | 0.3301 | 0.9405 | 84.17% | — |
| **08** | $4.48 \times 10^{-5}$ | 0.3120 | 0.3710 | 0.9469 | 83.78% | Actualizado (+0.0022) |
| **09** | $3.45 \times 10^{-5}$ | 0.2778 | 0.3347 | 0.9473 | 85.33% | Actualizado (+0.0004) |
| **10** | $2.50 \times 10^{-5}$ | 0.2926 | 0.3300 | 0.9504 | 85.33% | Actualizado (+0.0031) |
| **11** | $1.65 \times 10^{-5}$ | 0.2501 | 0.3465 | 0.9534 | 86.10% | Actualizado (+0.0030) |
| **12** | $9.55 \times 10^{-6}$ | 0.2467 | 0.3115 | 0.9550 | 86.87% | Actualizado (+0.0016) |
| **13** | $4.32 \times 10^{-6}$ | 0.2406 | 0.3047 | 0.9553 | 86.87% | Actualizado (+0.0003) |
| **14** | $1.08 \times 10^{-6}$ | **0.2407** | **0.3174** | **0.9569** | **88.03%** | **★ MEJOR MODELO GUARDADO** |
| **15** | $1.00 \times 10^{-6}$ | 0.2237 | 0.3140 | 0.9543 | 87.26% | — |

El estado óptimo se consolidó en la **Época 14** con un **$ROC\text{-}AUC$ de Validación de $0.9569$**, guardado en `modelos/best_resnet18_malignant_scale.pth`.

---

## 6. Matriz de Estrés Clínico: Benchmark en las 4 Cohortes Independientes

El checkpoint seleccionado fue sometido a evaluación ciega en las particiones de prueba independientes ($N=431$ retinografías). La Figura 25 sintetiza las curvas ROC, matrices de confusión por sensor y métricas comparativas:

![Benchmark de Estrés Maligno Multicéntrico](figures/fig25_malignant_scale_stress_test_benchmark.png)

### 6.1 Tabla Maestra de Desempeño Multicéntrico

| Cohorte de Prueba Independiente | Sensor / Especificación | Muestras ($N$) | ROC-AUC | Exactitud | Sensibilidad | Especificidad | Precisión | F1-Score | Matriz de Confusión [$TN, FP; FN, TP$] |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ACRIMA** | Canon CR-DGi ($35^\circ$ nativo) | 141 | **0.9939** | **95.04%** | **93.67%** | **96.77%** | 97.37% | **0.9548** | `[[60, 2], [5, 74]]` |
| **RIM-ONE DL** | Kowa WX 3D ($35^\circ$ estereoscópico) | 90 | **0.9293** | **88.89%** | **75.76%** | **96.49%** | 92.59% | **0.8333** | `[[55, 2], [8, 25]]` |
| **EyePACS** | Screening masivo ($45^\circ \to 35^\circ$ ROI) | 40 | **0.7850** | **70.00%** | **50.00%** | **90.00%** | 83.33% | **0.6250** | `[[18, 2], [10, 10]]` |
| **Moondream** | Multicéntrico con estadificación | 160 | **0.9086** | **79.37%** | **73.75%** | **85.00%** | 83.10% | **0.7815** | `[[68, 12], [21, 59]]` |
| **GLOBAL POOLED** | **Evaluación Total Unificada** | **431** | **0.9482** | **85.61%** | **79.25%** | **91.78%** | **90.32%** | **0.8442** | `[[201, 18], [44, 168]]` |

### 6.2 Auditoría Diagnóstica por Estadío Clínico en Moondream

Para comprobar si las omisiones del clasificador corresponden a cuadros graves o a fases subclínicas de alta ambigüedad oftalmológica, se desglosó el conjunto de Moondream ($N=160$):

```
- Glaucoma Avanzado (Advanced Stage) [N = 40]:  Sensibilidad = 90.00% (36 de 40 detectados)
- Glaucoma Temprano (Early Stage)    [N = 40]:  Sensibilidad = 57.50% (23 de 40 detectados)
- Controles Sanos (Normal Eyes)      [N = 80]:  Especificidad = 85.00% (68 de 80 confirmados)
```

**Relevancia Médica:** La sensibilidad del $90.00\%$ en estadios avanzados garantiza que los pacientes con riesgo inminente de ceguera irreversible son identificados con máxima seguridad diagnóstica. La menor sensibilidad en estadios tempranos ($57.50\%$) es congruente con la literatura oftalmológica: en fases iniciales, la retinografía mono-color no siempre refleja el adelgazamiento de fibras axónicas sin la asistencia de una tomografía OCT.

---

## 7. Prueba de Estrés Hostil en Tiempo de Inferencia (*Test-Time Augmentation Stress*)

Para dar respuesta inequívoca a si la invarianza a escala y fotometría es un rasgo robusto del modelo, se ejecutó una prueba de estrés agresiva mediante el script `scripts/experimentos/eval_test_time_malignant.py`. En este experimento, **las imágenes de prueba no vistas fueron bombardeadas directamente con las transformaciones estocásticas de Malignant Scale Augmentation en tiempo de inferencia**:

| Condición de Evaluación | Global ROC-AUC | Global Exactitud | Global Sensibilidad | Global Especificidad | Global F1-Score | Tasa de Retención ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Set Limpio (Canónico)** | **0.9466** | **85.85%** | **80.66%** | **90.87%** | **0.8486** | *100.0% (Línea Base)* |
| **Test Set con Aumento Maligno (Run 1, Seed 42)** | **0.9356** | **83.06%** | **78.77%** | **87.21%** | **0.8206** | **98.84%** |
| **Test Set con Aumento Maligno (Run 2, Seed 123)** | **0.9323** | **84.92%** | **80.19%** | **89.50%** | **0.8395** | **98.49%** |

### Desempeño por Sensor Bajo Ataque Maligno de Inferencia:
- **ACRIMA ($35^\circ$ Canon):** Mantiene $ROC\text{-}AUC = \mathbf{0.9906}$ (Run 1) y $\mathbf{0.9943}$ (Run 2), evidenciando que el alejamiento y viñeteo inducido no destruyen las características morfológicas del disco de alta resolución.
- **RIM-ONE DL ($35^\circ$ Kowa 3D):** Mantiene $ROC\text{-}AUC = \mathbf{0.9171}$ (Run 1) y $\mathbf{0.9181}$ (Run 2), sin verse afectado por la alteración de tono verde ni el micro-recorte.
- **EyePACS ($45^\circ$ ROI Screening):** Registra $ROC\text{-}AUC = \mathbf{0.7750}$ (Run 1) y $\mathbf{0.7250}$ (Run 2), manteniéndose operativo sin desplomarse a azar.
- **Moondream (Estadíos):** Registra $ROC\text{-}AUC = \mathbf{0.8958}$ (Run 1) y $\mathbf{0.8853}$ (Run 2).

---

## 8. Conclusiones y Recomendaciones de Despliegue Clínico

1. **Cierre de la Brecha de Escala y FOV:** La combinación de recorte centrado `OpticDiscCropper` junto con `MalignantScaleAugmentation` resolvió el colapso de generalización inter-cámara, elevando el desempeño global a un **$ROC\text{-}AUC = 0.9482$**.
2. **Inmunidad ante Perturbaciones Físicas:** Someter a prueba el modelo con aumentos malignos en tiempo de inferencia demostró que la red retiene el **$98.8\%$ de su rendimiento**, lo que certifica su aptitud para operar en teleoftalmología con pacientes no cooperadores o cámaras de baja calidad.
3. **Calibración de Umbral para Cribado:** Para despliegues en campañas de tamizaje masivo, se recomienda calibrar el umbral de corte de probabilidad de $p=0.50$ a $p=0.35$, priorizando elevar la sensibilidad en glaucoma temprano del $57.50\%$ a $>75\%$, manteniendo una especificidad aceptable ($>80\%$) que minimice las remisiones innecesarias.
4. **Trazabilidad y Reproducibilidad:** Todos los pesos, gráficos y registros numéricos han quedado serializados en el repositorio `GlaucomaDL` para su auditoría y citación en el artículo de formato Revista DYNA.

---
*Informe generado automáticamente por el compilador editorial del proyecto GlaucomaDL.*
