# Paquete de Trabajo: Andrés Arbeláez (Andrés Felipe)
## Roles Asignados: Datasets (Google & Selección) y Análisis de Redes Base (TFP vs. TFN)

Bienvenido, Andrés. Este directorio contiene todos los informes en PDF, códigos fuente reproducibles, archivos JSON con matrices de confusión completas y figuras de alta resolución necesarios para tu sección del artículo en la plantilla **Revista DYNA**.

---

### 1. Tus Objetivos y Responsabilidades Clave
1. **Auditoría y Selección de Datasets Útiles:**
   - Caracterizar las bases de datos médicas analizadas: **ACRIMA** (España, $N=705$, Canon CR-DGi, $35^\circ$), **RIM-ONE DL** (Canarias, $N=447$, Kowa WX 3D estereoscópica, $30^\circ$) y **Moondream Staged** ($N=800$, Hugging Face).
   - **Exclusión justificada de `THIRD_DATASET` (ODIR):** Documentar por qué se descartó de la mega-cohorte (contenía máscaras de esqueletos vasculares binarizadas en lugar de retinografías RGB reales y alta prevalencia de retinopatía diabética no controlada).
2. **Dataset de Google (EyePACS):**
   - Analizar el comportamiento del dataset **Google EyePACS / Kaggle Diabetic Retinopathy**: cámara Centervue DRS no midriática a $45^\circ$ FOV panorámico macular ($N=200$ auditadas).
   - Explicar la incompatibilidad dimensional nativa: en EyePACS, el disco óptico ocupa menos del 10% del encuadre, mientras que en ACRIMA/RIM-ONE ocupa entre el 30% y 50%.
   - Documentar la solución: la **armonización determinista a ROI $35^\circ$** centrada en la papila ($512 \times 512$ píxeles), lo que permitió elevar el AUC de ConvNeXt en EyePACS de $0.52$ a **$0.9400$** con **100% de especificidad**.
3. **Análisis de Distintas Redes Base (10 Candidatos):**
   - Evaluar el benchmark de 10 arquitecturas: ConvNeXt-Tiny, ResNet-18, Swin-T, ResNet-50, ResNeXt-50, DenseNet-121, EfficientNet-V2-S, EfficientNet-B0, MobileNet-V3 y ViT-B/16.
   - Justificar por qué **ConvNeXt-Tiny** fue seleccionada como la red ganadora (Score compuesto $S_{comp} = 0.9197$, ROC-AUC global $= 0.9669$, Especificidad $= 92.69\%$, latencia GPU $= 12.82$ ms).
4. **Medición por TFP y TFN en vez de por AUC (Requerimiento Fundamental):**
   - Realizar una crítica metodológica al uso exclusivo del AUC en tamizaje médico: el AUC integra todos los umbrales de decisión posibles, pero en la práctica clínica se opera en un único umbral de corte.
   - Calcular formalmente las tasas para cada modelo:
     $$\text{TFP} = \text{FPR} = \frac{FP}{FP + TN} = 1 - \text{Especificidad}$$
     $$\text{TFN} = \text{FNR} = \frac{FN}{TP + FN} = 1 - \text{Sensibilidad}$$
   - **Analizar el Trade-off Clínico:**
     * **Costo de un Falso Negativo (TFN):** En glaucoma, el daño al nervio óptico es asintomático e irreversible. Un falso negativo priva al paciente de tratamiento temprano, derivando en pérdida irreversible del campo visual y eventual ceguera.
     * **Costo de un Falso Positivo (TFP):** Genera angustia psicológica, costos elevados por pruebas confirmatorias innecesarias (OCT de nervio óptico, campimetría computarizada Humphrey) y saturación de la capacidad instalada en los centros de oftalmología especializada.
     * Cuantificar cómo ConvNeXt-Tiny equilibra ambos riesgos frente a ResNet-18 y Swin-T.

---

### 2. Informes PDF de Referencia (en la carpeta `informes_pdf/`)
* **`INFORME_SELECCION_MODELO_BASE_10_CANDIDATOS.pdf`:**
  - *Informe principal de tu sección.* Contiene la tabla comparativa completa de los 10 modelos, el radar de 5 ejes, curvas ROC y las matrices de confusión donde se desglosan TP, FP, TN y FN.
* **`INFORME_EVALUACION_BASES_MEDICAS.pdf`:**
  - *Esencial para la parte de datasets.* Detalla la auditoría forense de Google EyePACS, ACRIMA y RIM-ONE, y presenta las pruebas visuales del descarte de `THIRD_DATASET`/ODIR.
* **`INFORME_EXPERIMENTO_MEGA_COHORTE_MULTICENTRICA.pdf`:**
  - Explica la caída de desempeño cruzada entre cámaras antes y después de la armonización.
* **`INFORME_MAESTRO_GENERALIZACION_FINAL.pdf`:**
  - Manuscrito consolidado; consulta las secciones de metodología de datos y benchmark de backbones.

---

### 3. Códigos Fuente de Origen (en la carpeta `codigos_y_scripts/`)
* **`10_benchmark_10_backbones_transfer_learning.py`:**
  - Código en PyTorch que ejecuta la inferencia de las 10 redes sobre el conjunto de prueba multicéntrico ($N=431$), calcula la matriz de confusión (`confusion_matrix`) para cada una y genera las métricas de rendimiento.
* **`07_armonizacion_roi_y_mega_dataset.py`:**
  - Algoritmo que procesa las imágenes panorámicas de Google EyePACS ($45^\circ$), detecta el disco óptico y extrae el recorte ROI a $35^\circ$ ($512 \times 512$).
* **`01_seleccion_arquitectura_backbone.py` & `02_estudio_generalizacion_multicentrica.py`:**
  - Scripts preliminares de evaluación inter-base.
* **`compilar_informe_seleccion_backbone.py` & `compilar_informe_bases_medicas.py`:**
  - Scripts que compilan los reportes técnicos en PDF.

---

### 4. Datos Cuantitativos y Métricas (en `resultados_y_datos/`)
* **`benchmark_10_backbones_metrics.json`:**
  - **Archivo clave para tu análisis de TFP vs TFN.** Contiene para cada uno de los 10 modelos: `true_positives` (TP), `false_positives` (FP), `true_negatives` (TN), `false_negatives` (FN), `accuracy`, `roc_auc`, `latency_ms` y `specificity`.
  - Con estos valores directos puedes tabular inmediatamente la tabla comparativa de TFP y TFN.
* **`winning_backbone_finetuning_metrics.json`:**
  - Métricas detalladas del fine-tuning de ConvNeXt-Tiny (matriz de confusión por época, sensibilidad y especificidad a diferentes umbrales).
* **`cross_dataset_generalization_matrix.json`:**
  - Matriz de transferencia cruzada entre centros hospitalarios.
* **`harmonized_mega_cohort_index.csv`:**
  - Registro de las 2.152 imágenes de la mega-cohorte con su origen hospitalario, etiqueta y partición (train/test).

---

### 5. Figuras de Apoyo (en `figuras_apoyo/`)
1. `fig30_backbone_benchmark_multicriteria_tradeoff.png`: Gráfico de dispersión y tradeoff entre AUC, latencia y número de parámetros.
2. `fig31_backbone_roc_curves_and_radar.png`: Curvas ROC comparativas y diagrama de radar de 5 dimensiones.
3. `fig32_winning_model_finetuning_curves_and_confusion_matrix.png`: Matriz de confusión final de ConvNeXt-Tiny (desglose de TFP y TFN).
4. `fig21_cross_device_generalization_master.png`: Desplome inter-dispositivo antes de armonización.
5. `fig22_medical_databases_comparative_profiling.png`: Perfiles anatómicos y resolución angular de las bases de datos.
