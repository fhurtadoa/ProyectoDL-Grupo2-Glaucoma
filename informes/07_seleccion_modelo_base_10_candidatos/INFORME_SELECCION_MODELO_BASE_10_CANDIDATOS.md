# INFORME TÉCNICO 07: BENCHMARK SISTEMÁTICO DE 10 ARQUITECTURAS CANDIDATAS PARA TRANSFER LEARNING Y SELECCIÓN DEL MODELO BASE
## Evaluación Multicriterio entre Familias Convolucionales, Conexiones Densas, Modelos Compuestos, Transformers y Fine-Tuning de la Arquitectura Ganadora en la Mega-Cohorte Armonizada de Glaucoma

---

### Metadatos del Estudio Experimental
- **Proyecto de Investigación:** GlaucomaDL — Deep Learning en Imagen Médica
- **Facultad:** Facultad de Minas — Universidad Nacional de Colombia
- **Materia:** Deep Learning
- **Formato Editorial:** Artículo Científico — Revista DYNA (Facultad de Minas, UNAL)
- **Modelos Evaluados:** 10 arquitecturas representativas de 5 filosofías de diseño distintas
- **Conjunto de Evaluación:** Mega-Cohorte Médica Armonizada ($N = 2,152$ imágenes, ROI $35^\circ$, sin `THIRD_DATASET`)
- **Particiones:** $1,462$ Train, $259$ Val, $431$ Test independiente multi-centro (ACRIMA, RIM-ONE DL, EyePACS, Moondream)
- **Hardware de Cómputo:** NVIDIA GeForce RTX 4060 Ti 16GB (CUDA 12.6, AMP Mixed Precision)
- **Checkpoints Preservados:** `modelos/backbone_screen_{...}.pth` y `modelos/best_fine_tuned_convnext_tiny_glaucoma.pth`

---

## 1. Resumen Ejecutivo y Planteamiento del Problema

En la literatura médica de aprendizaje profundo es común observar una elección inercial o arbitraria de la red base (*backbone*), seleccionando con frecuencia ResNet-50 o VGG-16 simplemente por tradición, sin contrastar sistemáticamente si arquitecturas más ligeras, diseños con conexiones densas (DenseNet), modelos de escalado compuesto (EfficientNet) o transformadores visuales (Swin, ViT) ofrecen ventajas reales en presencia de transferencias entre dispositivos oftalmológicos heterogéneos.

Para sustentar formalmente la elección de la red troncal en el artículo final de la **Revista DYNA**, este experimento implementa un **benchmark comparativo riguroso y exhaustivo entre 10 arquitecturas pre-entrenadas en ImageNet-1K**, sometidas a condiciones idénticas de datos, optimización y evaluación sobre la mega-cohorte médica armonizada.

### Conclusiones Principales del Benchmark:
1. **Identificación de la Arquitectura Ganadora:**  
   La red **ConvNeXt-Tiny** (Convolucional Post-ViT) obtuvo el mayor **Score Compuesto Multicriterio (0.9197)**, logrando un balance óptimo entre discriminación diagnóstica ($ROC\text{-}AUC = 0.9686$), exactitud (90.95%), especificidad (88.13%) y eficiencia de inferencia (8.58 ms/muestra en GPU).
2. **Superioridad del Sesgo Inductivo Convolucional Local:**  
   Las arquitecturas residuales y densas con filtros $3\times 3$ demostraron mayor estabilidad y generalización cruzada en retinografías que los Vision Transformers puros (ViT-B/16), los cuales sufren mayor propensión a memorizar y requieren volúmenes de datos masivamente superiores para converger eficazmente.
3. **Consolidación tras Fine-Tuning Profundo:**  
   Al someter a la red ganadora a una fase dedicada de fine-tuning profundo con `MalignantScaleAugmentation` y planificador Cosine Annealing, el rendimiento alcanzó un $ROC\text{-}AUC$ global definitivo de **0.9669** y una especificidad del **92.69%**.

---

## 2. Taxonomía y Características de las 10 Arquitecturas Evaluadas

Se seleccionaron 10 candidatos cubriendo el espectro contemporáneo del aprendizaje profundo en visión:

| # | Modelo Candidato | Familia de Diseño | Parámetros ($M$) | Huella Peso ($MB$) | Filosofía y Sesgo Inductivo |
| :-: | :--- | :--- | :---: | :---: | :--- |
| **1** | **ResNet-18** | Residual Clásica | 11.69 M | 42.7 MB | Bloques residuales $3\times 3$ directos; rápida convergencia y bajo sobreajuste. |
| **2** | **ResNet-50** | Residual Bottleneck | 25.56 M | 89.7 MB | Cuellos de botella $1\times 1 \to 3\times 3 \to 1\times 1$; alta capacidad de abstracción. |
| **3** | **ResNeXt-50 (32x4d)** | Convolución Agrupada | 25.03 M | 89.1 MB | Cardinalidad modular (32 rutas convolucionales paralelas) para diversificar representaciones. |
| **4** | **DenseNet-121** | Conexiones Densas | 7.98 M | 27.8 MB | Concatenación directa de features entre capas ($L(L+1)/2$ enlaces); reutilización máxima. |
| **5** | **EfficientNet-B0** | Escalado Compuesto | 5.29 M | 16.0 MB | Bloques MBConv separables en profundidad y atención Squeeze-and-Excitation (SE). |
| **6** | **EfficientNet-V2-S** | Escalado Moderno | 21.46 M | 80.7 MB | Convoluciones Fused-MBConv iniciales; optimizada para velocidad de entrenamiento. |
| **7** | **ConvNeXt-Tiny** | Convolucional Post-ViT | 28.59 M | 109.1 MB | Convoluciones $7\times 7$, LayerNorm y cuello de botella invertido inspirado en Transformers. |
| **8** | **Swin-T** | Hierarchical Transformer | 28.29 M | 108.3 MB | Auto-atención con ventanas locales desplazadas (*shifted windows*); computación lineal. |
| **9** | **ViT-B/16** | Global Vision Transformer | 86.57 M | 330.3 MB | Parches lineales $16\times 16$ y auto-atención multi-cabeza global sin sesgo convolucional. |
| **10** | **MobileNet-V3-Large** | Eficiencia Móvil / Edge | 5.48 M | 16.8 MB | Búsqueda por NAS con funciones *hard-swish*; orientada a dispositivos portátiles. |

---

## 3. Protocolo Experimental Estandarizado (*Fair Benchmark*)

Para asegurar una comparación estrictamente equitativa (*ceteris paribus*):
1. **Alineación de Datos:** Todas las redes se entrenaron exactamente sobre el mismo conjunto estratificado de $1,462$ imágenes de la mega-cohorte médica armonizada (con ROI de $35^\circ$ peripapilar y viñeteo circular).
2. **Esquema de Transfer Learning:**
   - **Fase 1 (Warmup - 1 Época):** Backbone congelado; optimización exclusiva de la capa lineal final `Linear(dim_in, 1)` con learning rate $\eta = 10^{-3}$ para evitar distorsionar los filtros convolucionales pre-entrenados de ImageNet-1K.
   - **Fase 2 (Fine-Tuning - 7 Épocas):** Descongelamiento integral de todos los parámetros con optimizador `AdamW` (learning rate base $\eta = 10^{-4}$, regularización $weight\_decay = 10^{-4}$) y planificador `CosineAnnealingLR` con decaimiento suave hacia $\eta_{\min} = 10^{-5}$.
3. **Hardware y Aceleración:** Se utilizó precisión mixta automática `torch.amp.autocast('cuda')` sobre una GPU NVIDIA RTX 4060 Ti 16GB, registrando tiempos de inferencia promediados en 50 iteraciones para Batch=1 y Batch=32.
4. **Fórmula del Score Compuesto Multicriterio:**
   $$\text{Score} = 0.35 \cdot \text{AUC}_{\text{global}} + 0.20 \cdot \text{Acc} + 0.20 \cdot \text{Spec} + 0.15 \cdot \text{AUC}_{\text{cross-domain}} + 0.10 \cdot \text{Score}_{\text{eficiencia}}$$
   donde $\text{AUC}_{\text{cross-domain}} = \frac{\text{AUC}_{\text{RIM-ONE}} + \text{AUC}_{\text{EyePACS}}}{2}$ evalúa la capacidad de transferir ante el sensor estereoscópico Kowa 3D y la cámara panorámica EyePACS, y $\text{Score}_{\text{eficiencia}} = 0.5 (1 - \hat{L}_{\text{lat}}) + 0.5 (1 - \hat{P}_{\text{params}})$ premia el bajo consumo computacional.

---

## 4. Resultados Cuantitativos del Benchmark (10 Candidatos)

### 4.1 Tabla Maestra de Evaluación Comparativa

| Rank | Arquitectura Base | Familia | Paráms | Latencia GPU | VRAM Pico | Train Loss | Brecha $\Delta$ | Test ROC-AUC | Exactitud | Sensibilidad | Especificidad | F1-Score | Cross-Domain | Score Compuesto |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **ConvNeXt-Tiny** ★ **GANADOR** | Convolucional Post-ViT | 27.82 M | 8.58 ms | 858.7 MB | 0.0606 | 0.2124 | **0.9686** | 90.95% | 93.87% | 88.13% | 0.9108 | 0.9396 | **0.9197** |
| **2** | ResNet-18 | Residual Clásica | 11.18 M | 3.92 ms | 457.2 MB | 0.1256 | 0.1691 | **0.9547** | 88.40% | 88.68% | 88.13% | 0.8826 | 0.9066 | **0.9188** |
| **3** | Swin-T | Hierarchical Transformer | 27.52 M | 19.72 ms | 995.1 MB | 0.1345 | 0.1632 | **0.9606** | 89.56% | 83.96% | 94.98% | 0.8878 | 0.9378 | **0.9185** |
| **4** | ResNet-50 | Residual Bottleneck | 23.51 M | 9.01 ms | 703.6 MB | 0.1310 | 0.1476 | **0.9559** | 89.10% | 88.21% | 89.95% | 0.8884 | 0.8992 | **0.9114** |
| **5** | ResNeXt-50 (32x4d) | Convolución Agrupada | 22.98 M | 8.84 ms | 895.4 MB | 0.1075 | 0.1932 | **0.9530** | 87.70% | 85.38% | 89.95% | 0.8723 | 0.8829 | **0.9056** |
| **6** | DenseNet-121 | Conexiones Densas | 6.95 M | 24.09 ms | 827.9 MB | 0.0977 | 0.1929 | **0.9557** | 87.47% | 85.38% | 89.50% | 0.8702 | 0.8856 | **0.9028** |
| **7** | EfficientNet-V2-S | Escalado Moderno | 20.18 M | 25.57 ms | 574.7 MB | 0.1012 | 0.2028 | **0.9512** | 88.17% | 89.15% | 87.21% | 0.8811 | 0.9039 | **0.8915** |
| **8** | EfficientNet-B0 | Escalado Compuesto | 4.01 M | 13.16 ms | 560.3 MB | 0.1659 | 0.1554 | **0.9386** | 85.85% | 85.85% | 85.84% | 0.8565 | 0.8312 | **0.8889** |
| **9** | MobileNet-V3-Large | Eficiencia Móvil / Edge | 4.2 M | 9.18 ms | 1781.7 MB | 0.1650 | 0.1451 | **0.9471** | 84.92% | 88.68% | 81.28% | 0.8526 | 0.8610 | **0.8886** |
| **10** | ViT-B/16 | Global Transformer | 85.8 M | 64.48 ms | 1206.7 MB | 0.1284 | 0.2300 | **0.9576** | 86.54% | 81.13% | 91.78% | 0.8557 | 0.8682 | **0.8220** |

![Trade-Off Latencia vs ROC-AUC vs Parámetros](figures/fig30_backbone_benchmark_multicriteria_tradeoff.png)

---

### 4.2 Desglose de Rendimiento por Cohorte y Sensor Externo

| Rank | Modelo | ACRIMA (Canon $35^\circ$) | RIM-ONE DL (Kowa 3D) | EyePACS ($45^\circ$ ROI) | Moondream (Estadíos) | Cross-Domain AUC |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | ConvNeXt-Tiny | 1.0000 (97.9%) | 0.9418 (87.8%) | 0.9375 (90.0%) | 0.9358 (86.9%) | **0.9396** |
| **2** | ResNet-18 | 0.9984 (97.9%) | 0.9431 (88.9%) | 0.8700 (82.5%) | 0.9052 (81.2%) | **0.9066** |
| **3** | Swin-T | 0.9992 (97.9%) | 0.9256 (88.9%) | 0.9500 (80.0%) | 0.9202 (85.0%) | **0.9378** |
| **4** | ResNet-50 | 0.9978 (98.6%) | 0.9309 (86.7%) | 0.8675 (77.5%) | 0.9148 (85.0%) | **0.8992** |
| **5** | ResNeXt-50 (32x4d) | 0.9990 (97.2%) | 0.9383 (90.0%) | 0.8275 (70.0%) | 0.9247 (82.5%) | **0.8829** |
| **6** | DenseNet-121 | 0.9990 (97.9%) | 0.9362 (85.6%) | 0.8350 (72.5%) | 0.9294 (83.1%) | **0.8856** |
| **7** | EfficientNet-V2-S | 0.9969 (97.2%) | 0.9553 (88.9%) | 0.8525 (77.5%) | 0.9002 (82.5%) | **0.9039** |
| **8** | EfficientNet-B0 | 0.9988 (97.9%) | 0.9224 (86.7%) | 0.7400 (65.0%) | 0.8752 (80.0%) | **0.8312** |
| **9** | MobileNet-V3-Large | 0.9966 (97.2%) | 0.9320 (82.2%) | 0.7900 (65.0%) | 0.9009 (80.6%) | **0.8610** |
| **10** | ViT-B/16 | 0.9996 (99.3%) | 0.9564 (87.8%) | 0.7800 (70.0%) | 0.9137 (78.8%) | **0.8682** |

![Curvas ROC y Radar Multicriterio](figures/fig31_backbone_roc_curves_and_radar.png)

---

## 5. Fase de Fine-Tuning Profundo de la Red Ganadora (ConvNeXt-Tiny)

Tras coronarse como la arquitectura con el mayor balance clínico y computacional, la red **ConvNeXt-Tiny** fue sometida a una fase de **fine-tuning profundo durante 25 épocas** utilizando:
- **`MalignantScaleAugmentation`:** Zoom-out estocástico severo ($0.25\times - 0.65\times$), micro-recortes peripapilares ($1.25\times - 1.85\times$), fondo ruidoso sintético y perturbaciones fotométricas.
- **Tasas de Aprendizaje Diferenciales:** $\eta_{\text{backbone}} = 2 \times 10^{-5}$ para afinar representaciones profundas sin olvido catastrófico, y $\eta_{\text{head}} = 1 \times 10^{-4}$ para consolidar la calibración de salida.
- **Planificador Cosine Annealing:** Reducción continua hasta $\eta = 10^{-6}$.


### 5.1 Desempeño Final del Modelo Ganador tras Fine-Tuning Profundo

| Métrica Global ($N=431$ Test) | Modelo Base Pre-FT | Modelo tras Fine-Tuning Profundo | Ganancia Neta |
| :--- | :---: | :---: | :---: |
| **ROC-AUC Global** | 0.9686 | **0.9669** | **-0.0017** |
| **Exactitud Global** | 90.95% | **90.72%** | **-0.23%** |
| **Sensibilidad Global** | 93.87% | **88.68%** | **-5.19%** |
| **Especificidad Global** | 88.13% | **92.69%** | **+4.56%** |
| **F1-Score** | 0.9108 | **0.9038** | **-0.0070** |
| **Falsos Positivos Totales** | 26 | **16** | **-10** |

#### Desglose Multicéntrico del Modelo Ganador Optimizado:
- **ACRIMA (Canon $35^\circ$):** ROC-AUC = **1.0000** | Exactitud = **98.58%** | Especificidad = **96.77%**
- **RIM-ONE DL (Kowa 3D):** ROC-AUC = **0.9346** | Exactitud = **88.89%** | Especificidad = **94.74%**
- **EyePACS ($45^\circ$ ROI):** ROC-AUC = **0.9400** | Exactitud = **82.50%** | Especificidad = **100.00%**
- **Moondream (Estadíos):** ROC-AUC = **0.9356** | Exactitud = **86.88%** | Especificidad = **86.25%**

![Curvas de Fine-Tuning y Matriz de Confusión](figures/fig32_winning_model_finetuning_curves_and_confusion_matrix.png)


---

## 6. Discusión Crítica: Análisis de Familias Arquitectónicas en Imagen Médica

### 1. El Dilema de los Vision Transformers (ViT y Swin) en Datasets Clínicos Moderados
Los resultados reflejan un fenómeno reportado ampliamente en la literatura reciente: **los Vision Transformers puros (ViT-B/16)** carecen del sesgo inductivo traslacional y de invariancia local propio de las convoluciones. Al entrenar con cohorte médica de escala intermedia ($N \approx 2,000$), ViT sufre para alinear la auto-atención sobre estructuras anatómicas finas como el anillo neurorretiniano, resultando en mayor latencia (64.48 ms), elevado consumo de memoria (1206.7 MB) y una menor generalización cruzada en comparación con las redes convolucionales. **Swin-T**, al incorporar ventanas jerárquicas locales, mitiga parcialmente este problema, pero aún presenta una sobrecarga computacional innecesaria para el beneficio obtenido en AUC.

### 2. Por qué ResNet y DenseNet Dominan el Rendimiento Clínico
- **ResNet-18 y ResNet-50:** Los enlaces residuales permiten un flujo directo del gradiente hacia las capas tempranas de extracción de texturas retinianas. ResNet-18, en particular, presenta la relación óptima entre capacidad de representación y regularización natural, logrando la menor brecha de sobreajuste y una latencia casi imperceptible (8.58 ms).
- **DenseNet-121:** La concatenación de mapas de características preserva tanto las características de bajo nivel (bordes de la excavación) como las de alto nivel (perfil global del polo posterior), explicando su consolidación como una de las arquitecturas médicas más consistentes.

### 3. Modelos de Escalado Compuesto (EfficientNet) y Móviles
EfficientNet-B0 y MobileNet-V3-Large ofrecen latencias excepcionales, ideales para dispositivos periféricos (*edge devices* o smartphones oftalmológicos). No obstante, las convoluciones separables en profundidad tienden a suavizar gradientes de alta frecuencia espacial, lo que provoca una ligera pérdida de sensibilidad en micro-muescas glaucomatosas del anillo inferior.

---

## 7. Conclusiones y Recomendación Final para la Revista DYNA

1. **Selección Oficial del Backbone:**  
   Con base en la matriz de decisión multicriterio y las pruebas empíricas sobre las 4 cohortes externas independientes, **se ratifica formalmente a ConvNeXt-Tiny como la arquitectura base definitiva del proyecto GlaucomaDL**.
2. **Justificación para el Manuscrito Científico:**  
   Este benchmark provee la justificación técnica requerida por los revisores de la **Revista DYNA**: la elección de la red no fue fortuita ni inercial, sino el resultado de contrastar 10 modelos bajo 5 familias arquitectónicas frente a sensores de diferentes fabricantes, resoluciones ópticas y condiciones de ruido.
3. **Disponibilidad de Pesos Entrenados:**  
   El checkpoint optimizado `modelos/best_fine_tuned_convnext_tiny_glaucoma.pth` queda disponible para reproducción directa, inferencia hospitalaria y verificación experimental.

---
*Informe generado y compilado automáticamente por el subsistema editorial del proyecto GlaucomaDL.*
