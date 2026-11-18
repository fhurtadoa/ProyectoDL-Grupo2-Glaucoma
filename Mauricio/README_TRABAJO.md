# Paquete de Trabajo: Mauricio
## Roles Asignados: Estado del Arte & Marco Teórico

Bienvenido, Mauricio. Este directorio contiene todos los informes en PDF, códigos fuente, datos de extracción, tablas de calidad y figuras científicas necesarios para redactar y fundamentar tus secciones en el artículo final bajo el formato de la **Revista DYNA** (Facultad de Minas - Universidad Nacional de Colombia).

---

### 1. Tus Objetivos y Responsabilidades Clave
1. **Estado del Arte:**
   - Consolidar la Revisión Sistemática de Literatura (RSL) siguiendo las directrices formales de **Kitchenham & Charters (2007)** y la declaración **PRISMA 2020**.
   - Responder las 5 preguntas de investigación formuladas (RQ1 a RQ5) sobre la evolución de modelos, protocolos de evaluación, magnitud de la caída por cambio de dominio ($\Delta\text{AUC}$), estrategias de mitigación (centradas en datos vs. centradas en modelo) y backbones/regímenes de optimización.
   - Contrastar los hallazgos con los datos curados en las matrices de extracción (`extraccion.csv`) y calidad (`calidad.csv`).
2. **Marco Teórico:**
   - Formalizar matemáticamente el problema de **Domain Shift** en retinografías (disparidad entre $P_{train}(X, Y)$ y $P_{test}(X, Y)$).
   - Explicar el **sesgo inductivo convolucional 2D** frente a los transformadores de visión (ViT, Swin) en conjuntos de datos médicos de escala intermedia.
   - Detallar la **física óptica del campo visual**: discrepancia angular entre sensores papilares centrados ($30^\circ-35^\circ$ como ACRIMA y RIM-ONE) frente a panorámicas maculares ($45^\circ$ como Google EyePACS).
   - Formular la **dinámica asintótica y regularización implícita**: por qué entrenar a 1.000 épocas con decaimiento cosenoidal y aumento estocástico severo induce *grokking* y previene el sobreajuste clásico.

---

### 2. Informes PDF de Referencia (en la carpeta `informes_pdf/`)
* **`INFORME_MAESTRO_GENERALIZACION_FINAL.pdf` (16.8 MB):**
  - *Lectura obligatoria.* Es el manuscrito consolidado del proyecto. Contiene el marco teórico formal completo (Sección 1 y 2), la formulación de la física óptica, la fundamentación de las arquitecturas y la síntesis global de resultados.
* **`INFORME_EXPERIMENTO_1000_EPOCAS_SOBREENTRENAMIENTO.pdf`:**
  - Fundamental para el marco teórico sobre optimización. Explica el fenómeno de *grokking*, la trayectoria de convergencia asintótica, el rol del regularizador $L_2$ (Weight Decay) y la supresión de falsos positivos en épocas tardías.
* **`INFORME_EVALUACION_BASES_MEDICAS.pdf`:**
  - Perfilado óptico de los sensores y justificación de exclusión de bases degradadas.

---

### 3. Manuscrito y Archivos del Estado del Arte (en `manuscrito_estado_del_arte/`)
* **`estado_del_arte.md`:** Manuscrito completo del Estado del Arte estructurado en 8 secciones arábigas para Revista DYNA. Ya incluye citas correlativas con DOI verificado y las 6 figuras incrustadas.
* **`referencias_dyna.md` y `referencias.bib`:** Lista de las 32 referencias bibliográficas primarias y seminales en orden de aparición con DOIs activos y en formato BibTeX.
* **`tablas/calidad.csv`:** Evaluación de calidad metodológica de los estudios primarios evaluados bajo 9 criterios (QA1 a QA9).
* **`tablas/extraccion.csv`:** Matriz de extracción con 27 variables cuantitativas por estudio (AUC intra-base, AUC fuera de base, $\Delta\text{AUC}$, FOVs, arquitecturas).
* **`PROTOCOLO.md` y `LOG_PROTOCOLO.md`:** Protocolo formal y bitácora de ejecución de la RSL.

---

### 4. Figuras Científicas (en `figuras_estado_del_arte/`)
Todas las figuras están exportadas en escala de grises y alta resolución (300 DPI) según las normas editoriales de la Revista DYNA:
1. `fig00_prisma_2020_flujo_escala_grises.png`: Diagrama de flujo PRISMA 2020 completo.
2. `fig01_publicaciones_por_ano_escala_grises.png`: Evolución temporal de publicaciones primarias (2016–2026).
3. `fig02_bases_datos_frecuencia_escala_grises.png`: Frecuencia de empleo de bases de datos públicas.
4. `fig03_protocolos_evaluacion_escala_grises.png`: Distribución de protocolos de validación (LODO, cross-dataset, intra-base).
5. `fig05_caida_delta_auc_por_estudio_escala_grises.png`: Magnitud de la caída de desempeño diagnóstica ($\Delta\text{AUC}$) frente al umbral crítico.
6. `fig04_estrategias_mitigacion_escala_grises.png`: Taxonomía y frecuencia de estrategias de mitigación reportadas.

---

### 5. Scripts de Origen y Reproducibilidad (en `scripts_rsl_y_teoria/`)
* `01_busqueda_apis.py` a `08_generar_figuras.py`: Scripts en Python que ejecutan la recuperación automática en OpenAlex, Europe PMC, PubMed y arXiv, la verificación en Crossref, el cálculo del coeficiente Kappa de Cohen ($\kappa = 0.6484$) y la diagramación de figuras.
* `09_experimento_1000_epocas_sobreentrenamiento.py`: Script del experimento longitudinal que sustenta la teoría de regularización asintótica implícita.
* `compilar_informe_maestro_generalizacion.py`: Compilador automatizado del informe maestro a PDF.
