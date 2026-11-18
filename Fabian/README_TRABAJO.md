# Paquete de Trabajo: Fabián
## Roles Asignados: Trinario vs. Binario, Data Augmentation & Features de Generalización

Bienvenido, Fabián. Este directorio reúne todos los informes técnicos en PDF, scripts de entrenamiento y evaluación en PyTorch, métricas en formato JSON y figuras científicas en alta resolución necesarios para tu sección del artículo en la plantilla **Revista DYNA**.

---

### 1. Tus Objetivos y Responsabilidades Clave
1. **Clasificación Trinaria (Estadificación) vs. Binaria:**
   - Comparar el paradigma clásico de detección binaria (2 clases: Sano vs. Glaucoma) frente a la estadificación clínica ordinal (3 clases: Sano $\prec$ Glaucoma Incipiente/Early $\prec$ Glaucoma Avanzado/Advanced).
   - **El Falso Confort del Modelo Binario:** Explicar cómo un modelo binario puede reportar un AUC elevado ($\sim 0.91-0.94$), pero estar diagnosticando únicamente excavaciones masivas ($CDR \ge 0.8$), ignorando la pérdida focal temprana de fibras nerviosas.
   - **Taxonomía de Errores Clínicos:**
     * **Errores Adyacentes (Leves):** Confusión entre clases contiguas ($\text{Sano} \leftrightarrow \text{Early}$ o $\text{Early} \leftrightarrow \text{Avanzado}$). Clínicamente manejables con seguimiento semestral.
     * **Errores Catastróficos (Graves):** Confusión entre extremos ($\text{Sano} \leftrightarrow \text{Avanzado}$). Un paciente con excavación terminal catalogado como sano sufre ceguera evitable.
   - **Métrica de Concordancia Ordinal:** Evaluar mediante el **Quadratic Weighted Kappa ($QWK$)**, que penaliza cuadráticamente los errores entre clases distantes:
     $$QWK = 1 - \frac{\sum_{i,j} w_{i,j} O_{i,j}}{\sum_{i,j} w_{i,j} E_{i,j}}, \quad w_{i,j} = \frac{(i - j)^2}{(K - 1)^2}$$
2. **Data Augmentation (`MalignantScaleAugmentation`):**
   - Detallar la formulación matemática y el diseño del operador estocástico de aumento de datos:
     * **Zoom-Out severo ($0.25-0.65\times$):** Incrusta la papila en un fondo sintético con propiedades estadísticas de retina y viñeteo pupilar, simulando la distancia focal lejana de sensores a $45^\circ$.
     * **Zoom-In macro ($1.25-1.85\times$):** Fuerza a la red a inspeccionar la morfología del anillo neurorretiniano y la lámina cribosa.
     * **Ganancia estocástica de canal verde ($0.90-1.15$):** Rompe la correlación entre la reflectancia espectral y la procedencia hospitalaria.
     * **Corrección gamma ($0.75-1.35$):** Simula sobreexposición o hipoiluminación de medios opacos.
   - **Análisis Factorial $2 \times 2$ (Vanilla vs. Maligno en 2 y 3 clases):**
     * Demostrar cuantitativamente que el aumento maligno redujo los errores catastróficos del **$20.0\%$ al $9.76\%$** y elevó el $QWK$ de **$0.7232$ a $0.8163$** (acuerdo casi perfecto).
3. **Análisis de Features de las Imágenes que Causan Domain Shift:**
   - **Disparidad de Escala Relativa Papila/Imagen:** El tamaño aparente del disco óptico actúa como atajo heurístico no generalizable si no se aplica aumentación o recorte homogéneo.
   - **Firma Cromática del Sensor:** La calibración espectral de Kowa WX 3D (tonos fríos azulados y contrastes duros) frente a Canon CR-DGi (tonalidades anaranjadas cálidas difusas).
   - **Calidad de Imagen y Artefactos:** Viñeteo óptico en bordes circulares, reflejos corneales y pérdida de contraste por cataratas incipientes.

---

### 2. Informes PDF de Referencia (en la carpeta `informes_pdf/`)
* **`INFORME_PREDICCION_2_VS_3_CATEGORIAS.pdf`:**
  - *Informe principal de la comparativa 2 vs. 3 clases.* Contiene las matrices de confusión 3x3 normalizadas, el desglose formal de errores adyacentes vs. catastróficos y el cálculo del $QWK$.
* **`INFORME_BENEFICIO_DATA_AUGMENTATION_2_VS_3_CLASES.pdf`:**
  - *Informe principal del impacto de Data Augmentation.* Presenta la matriz de diseño factorial $2 \times 2$ comparando los modelos Vanilla vs. Maligno.
* **`INFORME_EXPERIMENTO_DATA_AUGMENTATION_MALIGNO_ESCALA.pdf`:**
  - Documento fundacional que formula el operador `MalignantScaleAugmentation`, ecuaciones y ejemplos visuales de las muestras perturbadas.
* **`INFORME_ENTRENAMIENTO_ESTRES_MALIGNO_MEGA_COHORTE.pdf`:**
  - Pruebas de estrés y evaluación de robustez ante aberraciones fotométricas.
* **`INFORME_MAESTRO_GENERALIZACION_FINAL.pdf`:**
  - Manuscrito consolidado general (Secciones 4, 5 y 6 para la discusión de estos hallazgos).

---

### 3. Códigos Fuente de Origen (en la carpeta `codigos_y_scripts/`)
* **`11_estudio_2_vs_3_clases_y_data_augmentation.py`:**
  - Código completo en PyTorch que entrena y evalúa los 4 modelos del estudio factorial: Binario Vanilla, Binario Maligno, Multiclase Vanilla y Multiclase Maligno. Calcula las matrices de confusión, el $QWK$ y la tasa de errores catastróficos.
* **`08_entrenamiento_malignant_scale_mega_cohorte.py`:**
  - Implementación de la clase `MalignantScaleAugmentation` y del pipeline de entrenamiento en la mega-cohorte unificada ($N=2.152$).
* **`04_data_augmentation_invarianza_escala.py`:**
  - Script preliminar donde se probó la invarianza de escala frente a ventanas de recorte fijas.
* **`06_estadificacion_clinica_multietapa.py`:**
  - Script de clasificación multietapa y proyección ordinal.
* **`eval_test_time_malignant.py`:**
  - Código de evaluación en tiempo de prueba (*Test-Time Stress*) aplicando perturbaciones de escala progresivas.
* **`compilar_informe_2_vs_3_categorias.py` & `compilar_informe_beneficio_data_augmentation.py`:**
  - Compiladores que generan los informes en PDF a partir de las métricas.

---

### 4. Datos Cuantitativos y Métricas (en `resultados_y_datos/`)
* **`study_2_vs_3_classes_metrics.json`:**
  - Métricas exhaustivas de los 4 modelos (matrices de confusión $2\times2$ y $3\times3$, exactitud, macro-F1, $QWK$ y curvas de entrenamiento).
* **`data_augmentation_benefit_metrics.json`:**
  - Tabla de deltas ($\Delta$) que cuantifica la ganancia neta en porcentaje introducida por el aumento maligno.
* **`malignant_stress_test_metrics.json` & `resumen_experimento_malignant_augmentation.json`:**
  - Resultados bajo factores extremos de estrés espacial y fotométrico.

---

### 5. Figuras de Apoyo (en `figuras_apoyo/`)
1. `fig33_confusion_matrices_2_vs_3_classes.png`: Matrices de confusión comparativas entre 2 clases y 3 clases (Vanilla vs. Maligno).
2. `fig34_data_augmentation_benefit_comparison.png`: Gráfico de barras con la ganancia diagnóstica neta atribuible a Data Augmentation.
3. `fig35_error_distribution_adjacent_vs_catastrophic.png`: Distribución porcentual de errores adyacentes vs. catastróficos.
4. `fig36_external_cohorts_stage_profiling.png`: Distribución de estadios morfológicos en cohortes externas (explicación de la caída en RIM-ONE DL).
5. `fig25_malignant_scale_stress_test_benchmark.png`: Curvas de degradación diagnóstica en función de la escala aparente del disco.
6. `fig24_harmonization_pipeline_before_after.png`: Comparación visual antes y después de la armonización de ROI papilar.
