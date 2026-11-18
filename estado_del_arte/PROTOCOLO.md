# Protocolo de Revisión Sistemática de Literatura (RSL)

**Título:** Generalización, Cambio de Dominio y Estrategias de Robustez en Redes Convolucionales Profundas para la Detección de Glaucoma en Retinografías de Fondo de Ojo: Una Revisión Sistemática de Literatura  
**Marco Metodológico:** Directrices de Kitchenham & Charters (2007) para Ingeniería de Software/Sistemas y Declaración PRISMA 2020 (*Preferred Reporting Items for Systematic Reviews and Meta-Analyses*)  
**Proyecto Asociado:** ProyectoDL-Grupo2-Glaucoma — Asignatura Deep Learning, Facultad de Minas, Universidad Nacional de Colombia  
**Fecha de Elaboración:** 17 de noviembre de 2026  
**Estado:** Versión 1.0 — Propuesta Formal para Aprobación en Punto de Control 1  

---

## 1. Introducción y Justificación de la Revisión

### 1.1 Contexto del Problema
El glaucoma es la principal causa irreversible de ceguera en el mundo. La retinografía de fondo de ojo a color (*Color Fundus Photography*, CFP) constituye la modalidad diagnóstica más accesible y económica para el tamizaje poblacional masivo. En la última década, los modelos de aprendizaje profundo (*Deep Learning*, DL), especialmente las redes neuronales convolucionales (CNN) y recientemente los modelos fundacionales basados en visión por transformadores (ViT), han demostrado una precisión diagnóstica sobresaliente dentro de conjuntos de datos individuales, alcanzando frecuentemente áreas bajo la curva ROC ($\text{ROC-AUC} > 0.98$).

No obstante, la traslación de estos algoritmos a la práctica clínica real enfrenta un obstáculo crítico: el **cambio de dominio (*domain shift*)**. Cuando un modelo entrenado en imágenes capturadas con un dispositivo específico (ej. Canon CR-DGi a $35^\circ$ de campo visual) se evalúa en imágenes de otros fabricantes (ej. Kowa WX 3D o Topcon TRC), con diferentes resoluciones, calibraciones de color, variaciones étnicas de pigmentación coroidea o distintos campos de visión ($30^\circ-35^\circ$ centrado en papila vs. $45^\circ-50^\circ$ panorámico centrado en mácula), el rendimiento diagnóstico colapsa drásticamente, cayendo en ocasiones a niveles cercanos al azar ($\text{ROC-AUC} \approx 0.52-0.58$).

### 1.2 Vacío en las Revisiones Previas (Justificación)
La auditoría exploratoria de revisiones sistemáticas y meta-análisis publicados entre 2018 y 2026 (Ting et al. 2019, Li et al. 2021, Perdomo-Charry & González-Osorio 2022, Zheng et al. 2022, Mursalin et al. 2023, Tham & Wong 2023, Hemelings et al. 2023, Ling et al. 2025) revela las siguientes carencias que justifican la presente revisión sistemática:
1. **Énfasis exclusivo en métricas agregadas intra-dominio:** La mayoría de revisiones previas agrupan resultados provenientes de particiones aleatorias (*random train/test split*) dentro de la misma base de datos, sobrestimando la eficacia clínica de los modelos y ocultando la fragilidad inter-dispositivo.
2. **Ausencia de cuantificación sistemática del $\Delta\text{AUC}$ por cambio de sensor y geometría:** No existe una síntesis que tabule formalmente la magnitud de la degradación diagnóstica ($\Delta\text{AUC} = \text{AUC}_{\text{in-domain}} - \text{AUC}_{\text{out-of-domain}}$) en función de variables ópticas específicas, como la disparidad de campo visual ($35^\circ$ vs. $45^\circ$) y las diferencias espectrales entre fabricantes.
3. **Falta de una taxonomía comparativa entre intervenciones centradas en datos vs. centradas en el modelo:** No se ha estructurado una comparación rigurosa que contraste la efectividad de estrategias centradas en datos (armonización de campo visual/ROI del disco óptico, aumentación estocástica severa de escala y fotometría, normalización de color y filtros de calidad) frente a estrategias centradas en el modelo (adaptación de dominio no supervisada - UDA, *domain generalization* - DG, *test-time adaptation* - TTA y modelos fundacionales retinianos como RETFound).
4. **Indefinición sobre dinámicas de entrenamiento prolongado y sobreajuste:** Ningún estudio secundario previo examina cómo la duración del entrenamiento (número de épocas), el uso de regularización estocástica incesante y los planificadores de tasa de aprendizaje interactúan con la capacidad de generalización fuera de dominio.

---

## 2. Preguntas de Investigación (Research Questions)

* **RQ1 (Mapeo Bibliométrico):** ¿Cómo ha evolucionado la producción científica sobre generalización y evaluación inter-dominio de modelos de aprendizaje profundo para detección de glaucoma en retinografías entre 2016 y 2026 (distribución temporal, revistas y congresos principales, países e instituciones líderes, bases de datos públicas utilizadas y co-ocurrencia temática)?  
  * *Motivación:* Caracterizar el ecosistema de investigación, el grado de madurez del campo y los recursos de datos abiertos que sustentan la experimentación internacional.
* **RQ2 (Protocolos de Evaluación y Rigor de Generalización):** ¿Qué protocolos de validación se emplean en la literatura (partición intra-base aleatoria, validación cruzada por paciente, validación *cross-dataset*, *leave-one-domain-out*, validación externa independiente multicéntrica, o ensayos prospectivos) y con qué frecuencia se somete a los algoritmos a pruebas en dominios verdaderamente no vistos?  
  * *Motivación:* Determinar si la literatura evalúa de forma fidedigna la capacidad de generalización fuera de distribución o si persiste la práctica de reportar particiones internas de cohortes agregadas como si fuesen pruebas de dominio ciego.
* **RQ3 (Magnitud y Factores Determinantes del Domain Shift):** ¿Cuál es la magnitud cuantitativa de la caída de desempeño ($\Delta\text{AUC}$, pérdida de sensibilidad a alta especificidad) cuando un modelo se despliega en un dispositivo, centro o base no observada durante el entrenamiento, y qué factores ópticos, clínicos y demográficos (disparidad de ángulo de visión $30^\circ-35^\circ$ vs $45^\circ$, tecnología de la cámara, pigmentación de la población, criterio de estándar de referencia) explican dicha degradación?  
  * *Motivación:* Identificar las fuentes de variación más nocivas para el clasificador y cuantificar la brecha real de rendimiento.
* **RQ4 (Estrategias de Mitigación y Efectividad Comparada):** ¿Qué intervenciones han demostrado ser más efectivas para mitigar el cambio de dominio y cuál es su impacto cuantitativo documentado?  
  * *(a) Estrategias centradas en datos:* Entrenamiento multifuente (agregación de cohortes), aumento de datos estocástico agresivo (escala, zoom-in/out, viñeteo, distorsión fotométrica), normalización/transferencia de color, recorte y armonización geométrica del disco óptico (ROI), y filtros de calidad de imagen.  
  * *(b) Estrategias centradas en el modelo:* Adaptación de dominio no supervisada (UDA adversarial o de discrepancia), generalización de dominio (*Domain Generalization*), adaptación en tiempo de inferencia (*Test-Time Adaptation*), aprendizaje autosupervisado y fine-tuning de modelos fundacionales oftálmicos (ej. RETFound).  
  * *Motivación:* Guiar el diseño de pipelines robustos contrastando el costo computacional frente a la recuperación de $\text{AUC}$.
* **RQ5 (Arquitecturas, Preentrenamientos y Regímenes de Entrenamiento):** ¿Qué arquitecturas convolucionales y de transformadores se utilizan preferentemente, qué impacto tiene el preentrenamiento (ImageNet-1K genérico vs. preentrenamiento autosupervisado en millones de retinas) y qué influencia tienen los regímenes de entrenamiento (parada temprana hacia épocas 15–30 vs. entrenamiento prolongado asintótico con planificadores Cosine Annealing) en la estabilidad y generalización del modelo?  
  * *Motivación:* Confrontar el dogma de la parada temprana convencional en presencia de aumento estocástico continuo y evaluar si modelos compactos (ResNet-18) ofrecen ventajas de generalización frente a redes sobreparametrizadas.

---

## 3. Criterios de Selección (Elegibilidad)

### 3.1 Criterios de Inclusión (Inclusion Criteria)
* **I1 (Tarea y Modalidad):** Estudio primario que propone o evalúa un modelo de aprendizaje profundo (redes convolucionales, transformadores de visión, ensambles o modelos fundacionales) para la clasificación, detección, tamizaje, derivación (*referral*) o estadificación de severidad de glaucoma utilizando retinografías de fondo de ojo a color (incluyendo imágenes capturadas con cámaras de mesa o adaptadores de smartphone).
* **I2 (Generalización o Robustez Explícita):** El estudio reporta explícitamente el desempeño del modelo en al menos un dominio distinto al de entrenamiento (otra base de datos pública/privada, dispositivo de otro fabricante, centro hospitalario independiente o cohorte de tamizaje externo) **O** propone e implementa un método técnico explícito concebido para mitigar el *domain shift* (adaptación de dominio, *domain generalization*, armonización geométrica/óptica, aumento de datos contra cambio de sensor, o preentrenamiento fundacional evaluado fuera de distribución).
* **I3 (Tipo de Publicación):** Publicación revisada por pares (*peer-reviewed*), aceptada y publicada en revista científica indexada o actas de conferencias de primer nivel en computación médica o visión artificial (ej. MICCAI, IEEE ISBI, CVPR, ICCV, MIDL, EMBC).
* **I4 (Idioma y Periodo):** Publicado en idioma inglés o español entre el **1 de enero de 2016 y la fecha de corte (17 de noviembre de 2026)**.

### 3.2 Criterios de Exclusión (Exclusion Criteria)
* **E1 (Modalidad No Apta):** Estudios basados exclusivamente en Tomografía de Coherencia Óptica (OCT), campos visuales perimétricos (VF), angiografía o imágenes de lámpara de hendidura sin incluir retinografía a color para la decisión diagnóstica.
* **E2 (Segmentación Aislada Sin Diagnóstico):** Estudios limitados únicamente a la segmentación del disco y copa óptica que no evalúen ni reporten una métrica de clasificación o decisión diagnóstica de glaucoma.
* **E3 (Evaluación Exclusivamente Intra-Base Sin Mitigación):** Estudios que evalúan modelos de forma estrictamente intra-base mediante partición aleatoria convencional (*train/test split* o k-fold estándar) sin incluir evaluación fuera de dominio ni formular ninguna técnica contra el cambio de dominio.
* **E4 (Tipología de Documento Gris):** Tesis de pregrado/maestría/doctorado, preprints sin versión revisada por pares aceptada (ej. arXiv no publicado), cartas al editor, editoriales, resúmenes de congreso de una página (*abstracts*), notas técnicas cortas o capítulos de libro no indexados.
* **E5 (Duplicidad o Superposición de Cohorte):** Versiones preliminares de congreso cuyos resultados hayan sido expandidos íntegramente en un artículo de revista posterior por los mismos autores (se conserva la versión más completa y madura).
* **E6 (Multi-enfermedad sin Desglose Específico):** Estudios generales de patología retiniana múltiple (ej. detección simultánea de retinopatía diabética, degeneración macular y glaucoma) que no reporten métricas discriminadas ni análisis de generalización específico para glaucoma.

> **Nota Metodológica sobre Revisiones/Surveys Previos:** Las revisiones sistemáticas, meta-análisis y estados del arte previos se excluyen de la síntesis cuantitativa primaria, pero se preservan activamente en la Fase 1 para la justificación del estudio y para procedimientos de búsqueda retrospectiva (*snowballing*).

---

## 4. Estrategia de Búsqueda y Ecuaciones Literales

La búsqueda se estructura en cuatro bloques conceptuales combinados mediante operadores booleanos:
$$ \text{Ecuación Global} = (\text{Bloque Patología}) \land (\text{Bloque IA / Deep Learning}) \land (\text{Bloque Modalidad Retinografía}) \land (\text{Bloque Generalización / Shift}) $$

### 4.1 Ecuaciones para APIs Programáticas Gratuitas (Reproducibilidad Automatizada)

#### 4.1.1 OpenAlex API
* **Endpoint:** `https://api.openalex.org/works`
* **Filtros API:** `from_publication_date:2016-01-01`, `type:article|proceedings-article`, `has_doi:true`
* **Consulta de búsqueda (`search` query string):**
```text
(glaucoma OR "glaucomatous optic neuropathy") AND ("deep learning" OR "convolutional neural network" OR CNN OR "transfer learning" OR "foundation model" OR RETFound) AND ("fundus photograph" OR "fundus image" OR "retinal photograph" OR retinography OR "color fundus") AND ("domain shift" OR "dataset shift" OR "cross-dataset" OR "domain generalization" OR "domain adaptation" OR "external validation" OR "out-of-domain" OR "multi-center" OR "harmonization")
```

#### 4.1.2 PubMed (NCBI E-utilities API)
* **Endpoint:** `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed`
* **Parámetros:** `datetype=pdat&mindate=2016/01/01&maxdate=2026/11/17`
* **Consulta Booleana Literal:**
```text
(glaucoma[Title/Abstract] OR "glaucomatous optic neuropathy"[Title/Abstract]) AND ("deep learning"[Title/Abstract] OR "convolutional neural network"[Title/Abstract] OR "CNN"[Title/Abstract] OR "transfer learning"[Title/Abstract] OR "foundation model"[Title/Abstract] OR "RETFound"[Title/Abstract]) AND ("fundus"[Title/Abstract] OR "retinal photograph"[Title/Abstract] OR "retinography"[Title/Abstract] OR "color fundus"[Title/Abstract]) AND ("domain shift"[Title/Abstract] OR "dataset shift"[Title/Abstract] OR "cross-dataset"[Title/Abstract] OR "domain generalization"[Title/Abstract] OR "domain adaptation"[Title/Abstract] OR "external validation"[Title/Abstract] OR "generalization"[Title/Abstract] OR "multi-center"[Title/Abstract] OR "harmonization"[Title/Abstract]) AND ("2016/01/01"[Date - Publication] : "2026/11/17"[Date - Publication])
```

#### 4.1.3 Europe PMC API
* **Endpoint:** `https://www.ebi.ac.uk/europepmc/webservices/rest/search`
* **Filtros:** `SRC:MED OR SRC:PMC`, `PUB_YEAR:[2016 TO 2026]`
* **Consulta Booleana Literal:**
```text
(TITLE_ABS:"glaucoma" OR TITLE_ABS:"glaucomatous optic neuropathy") AND (TITLE_ABS:"deep learning" OR TITLE_ABS:"convolutional" OR TITLE_ABS:"CNN" OR TITLE_ABS:"transfer learning" OR TITLE_ABS:"foundation model") AND (TITLE_ABS:"fundus" OR TITLE_ABS:"retinal photograph" OR TITLE_ABS:"retinography") AND (TITLE_ABS:"domain shift" OR TITLE_ABS:"cross-dataset" OR TITLE_ABS:"generalization" OR TITLE_ABS:"external validation" OR TITLE_ABS:"domain adaptation" OR TITLE_ABS:"harmonization") AND (FIRST_PDATE:[2016-01-01 TO 2026-11-17])
```

#### 4.1.4 Semantic Scholar API
* **Endpoint:** `https://api.semanticscholar.org/graph/v1/paper/search`
* **Parámetros:** `year=2016-2026`, `fields=title,authors,year,venue,externalIds,abstract,citationCount`
* **Consulta Literal:**
```text
glaucoma "deep learning" fundus ("domain shift" OR "cross-dataset" OR "generalization" OR "domain adaptation" OR "external validation")
```

#### 4.1.5 arXiv API (Uso restringido: Rastreo de Preprints hacia Publicación Revisada)
* **Endpoint:** `http://export.arxiv.org/api/query`
* **Propósito:** Identificar preprints altamente citados sobre *domain adaptation* o *foundation models* oftálmicos para verificar en Crossref si ya cuentan con versión final revisada por pares.

---

### 4.2 Ecuaciones Adaptadas para Búsqueda Manual Directa

Estas ecuaciones han sido adaptadas sintácticamente para su ejecución directa en los portales institucionales de suscripción:

#### 4.2.1 Scopus (Elsevier)
* **Sintaxis:** `TITLE-ABS-KEY`, operadores de proximidad `W/n` para vincular términos fuertemente asociados.
* **Ecuación Literal para Scopus:**
```text
TITLE-ABS-KEY ( ( glaucoma* OR "optic neuropathy" ) AND ( "deep learning" OR "convolutional neural network*" OR "CNN" OR "transfer learning" OR "foundation model*" OR "vision transformer*" OR "RETFound" ) AND ( "fundus photograph*" OR "fundus imag*" OR "retinal photograph*" OR "retinograph*" OR "color fundus" ) AND ( "domain shift" OR "dataset shift" OR "distribution shift" OR ( "domain" W/3 "generaliz*" ) OR ( "domain" W/3 "adapt*" ) OR "cross-dataset" OR "cross-domain" OR "cross-center" OR ( "external" W/2 "validat*" ) OR "out-of-domain" OR "multi-center" OR "harmoniz*" ) ) AND PUBYEAR > 2015 AND ( LIMIT-TO ( DOCTYPE , "ar" ) OR LIMIT-TO ( DOCTYPE , "cp" ) ) AND ( LIMIT-TO ( LANGUAGE , "English" ) OR LIMIT-TO ( LANGUAGE , "Spanish" ) )
```

#### 4.2.2 Web of Science (Clarivate - Core Collection)
* **Sintaxis:** `TS=` (Topic: Title, Abstract, Author Keywords, Keywords Plus), operador de proximidad `NEAR/n`.
* **Ecuación Literal para Web of Science:**
```text
TS=( ( glaucoma* OR "optic neuropathy" ) AND ( "deep learning" OR "convolutional neural network*" OR "CNN" OR "transfer learning" OR "foundation model*" OR "vision transformer*" OR "RETFound" ) AND ( "fundus photograph*" OR "fundus imag*" OR "retinal photograph*" OR "retinograph*" OR "color fundus" ) AND ( "domain shift" OR "dataset shift" OR "distribution shift" OR ( "domain" NEAR/3 generaliz* ) OR ( "domain" NEAR/3 adapt* ) OR "cross-dataset" OR "cross-domain" OR "cross-center" OR ( external NEAR/2 validat* ) OR "out-of-domain" OR "multi-center" OR "harmoniz*" ) ) AND PY=(2016-2026) AND DT=(Article OR Proceedings Paper) AND LA=(English OR Spanish)
```

#### 4.2.3 IEEE Xplore Digital Library
* **Sintaxis:** `Command Search` en Metadata, máximo 5 comodines (*wildcards*) permitidos por regla estricta de la plataforma.
* **Ecuación Literal para IEEE Xplore (Cumple restricción de comodines):**
```text
("Document Title":glaucoma OR "Abstract":glaucoma) AND ("Document Title":"deep learning" OR "Abstract":"deep learning" OR "Abstract":"convolutional neural" OR "Abstract":CNN) AND ("Abstract":fundus OR "Abstract":retinal) AND ("Abstract":"domain shift" OR "Abstract":"cross-dataset" OR "Abstract":"generalization" OR "Abstract":"domain adaptation" OR "Abstract":"external validation" OR "Abstract":harmonization)
```

---

## 5. Procedimiento de Cribado y Evaluación de Concordancia Inter-Revisor

El proceso de selección se ejecutará en dos etapas estructuradas:

```
[Búsqueda en Fuentes (APIs + Manuales)]
                 │
                 ▼
  [Deduplicación por DOI y Levenshtein ≥ 0.92]
                 │
                 ▼
     [Cribado Título y Resumen (data/processed/cribado_TA.csv)]
                 │
                 ├─ Propuesta del Agente: Incluir / Excluir / Duda
                 ├─ Decisión Final del Investigador Principal
                 └─ Re-cribado aleatorio del 10–20% de la muestra para cálculo de Kappa de Cohen (κ)
                 │
                 ▼
     [Cribado a Texto Completo (refs/pdf/ y SINAB)]
                 │
                 ├─ Snowballing hacia atrás (referencias de incluidos clave)
                 └─ Snowballing hacia adelante (citas vía OpenAlex/Semantic Scholar)
                 │
                 ▼
      [Conjunto Final de Estudios Primarios Incluidos]
```

### 5.1 Protocolo de Concordancia Inter-Revisor (Kappa de Cohen)
Para certificar el rigor metodológico exigido por Kitchenham & Charters (2007) y mitigar sesgos de selección:
1. El agente computacional propone una clasificación preliminar para cada registro (`incluir`, `excluir`, `duda`) junto con el criterio específico aplicado (ej. `E3 - Solo evaluación intra-base`) y una frase justificativa en `data/processed/cribado_TA.csv`.
2. El investigador principal (humano) revisa y emite su `decision_final`.
3. Se extrae una **muestra aleatoria estratificada del 15%** de los registros cribados para un re-cribado ciego e independiente.
4. Se calcula el estadístico Kappa de Cohen ($\kappa$):
   $$ \kappa = \frac{P_o - P_e}{1 - P_e} $$
   Donde $P_o$ es el acuerdo observado y $P_e$ es la probabilidad de acuerdo por azar.
5. Se establece como umbral metodológico $\kappa \ge 0.75$ (acuerdo sustancial a casi perfecto). En caso de desacuerdo, los casos se discutirán punto por punto hasta alcanzar consenso documentado en `LOG_PROTOCOLO.md`.

---

## 6. Evaluación de la Calidad Metodológica (QA1 a QA9)

Cada estudio primario seleccionado a texto completo será sometido a un cuestionario de evaluación de calidad de 9 preguntas críticas. Cada ítem se valorará con la siguiente escala:
* **Sí (Cumplimiento riguroso):** $1.0$ punto.
* **Parcial (Cumplimiento ambiguo o incompleto):** $0.5$ puntos.
* **No (Incumplimiento o no reportado):** $0.0$ puntos (requiere justificación obligatoria en `tablas/calidad.csv`).

### Cuestionario de Calidad:
* **QA1 (Objetivo e Hipótesis):** ¿Se formulan con claridad meridiana los objetivos, la tarea diagnóstica y las hipótesis clínicas/técnicas del estudio?
* **QA2 (Caracterización de Datos):** ¿Se describen exhaustivamente las cohortes utilizadas (nombre institucional, país, modelo de cámara y fabricante, ángulo de campo visual en grados, tamaño muestral $N$, prevalencia de glaucoma y criterios clínicos de inclusión/exclusión de imágenes)?
* **QA3 (Integridad de Partición y Fuga de Datos):** ¿La partición entre conjuntos de entrenamiento, validación y prueba se realizó de manera estricta a nivel de paciente/sujeto, descartando categóricamente la fuga de información (*data leakage*) entre ojos bilaterales del mismo individuo?
* **QA4 (Rigor de Evaluación de Dominio):** ¿El modelo fue evaluado en un dominio hospitalario o dispositivo verdaderamente no visto (*unseen domain*, cohorte externa ciega o protocolo *leave-one-domain-out*), diferenciándolo de una validación interna de una cohorte combinada?
* **QA5 (Comparabilidad con Línea Base):** ¿Se contrasta el rendimiento frente a una línea base sólida y comparable (ej. mismo backbone sin la técnica de adaptación, entrenamiento monomodal clásico o método estándar del estado del arte bajo idénticas condiciones)?
* **QA6 (Métricas e Incertidumbre Estadística):** ¿Se reportan métricas diagnósticas pertinentes ($\text{ROC-AUC}$, sensibilidad a especificidad fija $\ge 90\%-95\%$, F1-score) acompañadas de medidas de variabilidad o incertidumbre (intervalos de confianza del 95%, semillas múltiples de inicialización o validación cruzada estratificada)?
* **QA7 (Solidez del Estándar de Referencia):** ¿El diagnóstico de referencia (*ground truth*) proviene de un estándar sólido (consenso de múltiples especialistas de glaucoma, o confirmación mediante campo visual perimétrico Humphrey y análisis estructural de OCT)?
* **QA8 (Análisis de Limitaciones y Robustez ante Ruido):** ¿Los autores analizan explícitamente los casos de falla del algoritmo, el impacto de imágenes subóptimas (cataratas, artefactos de iluminación) y las amenazas a la validez de sus hallazgos?
* **QA9 (Reproducibilidad y Ciencia Abierta):** ¿El estudio ofrece garantías de reproducibilidad técnica (disponibilidad pública de código fuente en GitHub/GitLab, pesas entrenados, hiperparámetros declarados y acceso a los datos o protocolo de extracción de datos)?

> **Puntuación Total de Calidad:** Oscila entre $0.0$ y $9.0$ puntos. Siguiendo las directrices de Kitchenham, **ningún estudio será excluido de la síntesis por baja puntuación de calidad**; en su lugar, la puntuación se utilizará para realizar un **Análisis de Sensibilidad** (evaluar si las conclusiones sobre la efectividad de las estrategias y la magnitud del $\Delta\text{AUC}$ cambian al considerar únicamente estudios con calidad $\ge 6.0/9.0$ y con validación verdaderamente externa).

---

## 7. Formulario de Extracción de Datos

Para cada estudio incluido a texto completo, se registrarán las siguientes variables estructuradas en `tablas/extraccion.csv`:

1. `id_estudio`: Identificador unívoco del estudio (ej. `EST-01`).
2. `referencia_corta`: Apellido del primer autor et al. y año (ej. `Diaz-Pinto et al., 2019`).
3. `año`: Año oficial de publicación.
4. `pais`: País o países de las instituciones principales.
5. `venue_tipo`: Revista (`Journal`) o Congreso (`Conference`).
6. `venue_nombre`: Nombre de la revista o congreso.
7. `doi`: Identificador de Objeto Digital verificado en Crossref.
8. `tarea_clinica`: Clasificación binaria (glaucoma vs. sano), referencia de tamizaje (*referable glaucoma*), o estadificación en niveles de severidad (temprano, intermedio, avanzado).
9. `modalidad_imagen`: Retinografía color de mesa, portátil, smartphone, etc.
10. `bases_entrenamiento`: Nombres de las bases de entrenamiento, $N$ muestras, modelos de cámara y ángulos de visión ($\text{FOV}^\circ$).
11. `bases_prueba`: Nombres de las bases de prueba, $N$ muestras, cámaras y $\text{FOV}^\circ$.
12. `protocolo_evaluacion`: Partición intra-base aleatoria, validación cruzada por paciente, cross-dataset, leave-one-domain-out, validación externa prospectiva.
13. `dominio_verdaderamente_no_visto`: Indicador binario (`Sí` / `No`) sobre si la prueba evaluó un dominio totalmente ausente en entrenamiento.
14. `arquitectura_backbone`: Modelo de red convolucional o transformer empleado (ResNet-18, ResNet-50, DenseNet, EfficientNet, ViT, Swin, etc.).
15. `preentrenamiento`: Inicialización de pesos (ImageNet-1K, ImageNet-22K, RETFound, preentrenamiento autosupervisado retiniano, o desde cero).
16. `estrategia_mitigacion_cat`: Categoría según RQ4 (Aumento agresivo, Armonización ROI, Multifuente agregada, Normalización color, Filtro calidad, UDA adversarial, Domain Generalization, Test-Time Adaptation, Modelo Fundacional).
17. `estrategia_descripcion`: Descripción sintética de la intervención contra el cambio de dominio.
18. `regimen_entrenamiento`: Épocas, optimizador, scheduler y criterio de parada (early stopping vs asintótico).
19. `auc_intra_dominio`: $\text{ROC-AUC}$ en la evaluación interna / mismo centro.
20. `auc_fuera_dominio`: $\text{ROC-AUC}$ en la evaluación externa / otro centro.
21. `delta_auc`: Caída neta de discriminación ($\Delta\text{AUC} = \text{AUC}_{\text{in}} - \text{AUC}_{\text{out}}$).
22. `sensibilidad_especificidad`: Sensibilidad y especificidad reportadas fuera de dominio.
23. `hallazgos_severidad_calidad`: Comportamiento del modelo ante glaucoma temprano vs avanzado, o ante imágenes degradadas.
24. `codigo_disponible`: Enlace al repositorio de código o indicación de no disponible.
25. `limitaciones_declaradas`: Principales advertencias o limitaciones señaladas por los autores.
26. `rq_asociadas`: Preguntas de investigación a las que aporta evidencia (ej. `RQ2, RQ3, RQ4`).
27. `fuente_dato`: `Texto Completo` o `Solo Resumen` (regla estricta: si una cifra se extrajo solo del resumen se declara explícitamente).

---

## 8. Estrategia de Síntesis y Redacción (Formato Revista DYNA)

La síntesis de los estudios no adoptará jamás la forma de un resumen secuencial artículo por artículo. Se organizará temáticamente por categorías conceptuales respondiendo de forma comparativa y crítica a las RQs:
* **Estructura jerárquica:** Máximo 3 niveles de encabezado numerados arábigamente sin punto final (ej. `2`, `2.1`, `2.1.1`), siguiendo las normas formales de la Revista DYNA.
* **Redacción de párrafos:** Principio de "un párrafo = una idea" estructurada (primera frase como afirmación temática contundente, frases subsiguientes como sustentación con datos y citas contrastadas).
* **Voz gramatical y tiempos verbales:** Redacción en tercera persona o impersonal; tiempo presente para hechos consolidados del estado del arte y tiempo pretérito para los aportes y ensayos específicos de otros autores.
* **Formato de Citas y Ecuaciones DYNA:** Citas correlativas entre corchetes por orden de aparición en el texto (`[1]`, `[1,2]`, `[1-3]`). Ecuaciones centradas con numeración correlativa entre paréntesis a la derecha (`eq. (1)`).
* **Tablas y Figuras DYNA:** Figuras en escala de grises con resolución de 300 DPI, tipografía Times New Roman y leyenda inferior `"Fuente: Los autores."` o `"Fuente: [n]"`. Tablas con título superior, únicamente líneas horizontales de separación y fuente reducida a 8 pt.
* **Referencias DYNA:** Listado final en `refs/referencias_dyna.md` bajo el estilo exacto de DYNA:
  `Apellido, A.B., Apellido, C.D. and Apellido, E.F., Título del artículo. Nombre de la Revista, Volumen(Número), pp. páginas, Año. DOI: http://dx.doi.org/...`
  Acompañado de archivo `refs/referencias.bib` compatible con gestores bibliográficos (Zotero/Mendeley).

---

## 9. Declaración sobre el Uso de Herramientas de Inteligencia Artificial

En estricta concordancia con las recomendaciones del Comité Internacional de Editores de Revistas Médicas (*ICMJE*) y del Comité de Ética en Publicaciones (*COPE*):
* Se declara que el presente protocolo y la ejecución técnica de scripts de búsqueda, normalización de datos y diagramación contaron con la asistencia automatizada de un agente inteligente de investigación (Deep Research y APIs de lenguaje).
* Todas las afirmaciones, metadatos, cifras numéricas ($\text{AUC}$, $\Delta\text{AUC}$) y vínculos de DOIs serán sometidos a validación cruzada determinística contra la API oficial de Crossref y verificación a texto completo.
* Ningún modelo de IA figura como autor del estudio; la responsabilidad conceptual, el diseño metodológico, la supervisión de decisiones en los puntos de control y la aprobación final del protocolo corresponden exclusivamente al equipo de investigadores humanos.
