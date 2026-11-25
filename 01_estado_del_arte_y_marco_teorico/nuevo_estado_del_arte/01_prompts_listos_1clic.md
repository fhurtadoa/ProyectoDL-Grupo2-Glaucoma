# Prompts listos en 1 clic para Gemini Deep Research (R1–R6 + R7 opcional)

Cada bloque ya incluye el **Bloque de contexto común** integrado. No hay que reemplazar ni pegar nada adicional.

## Flujo rápido (6 pestañas en paralelo)
1. Abre 6 pestañas de Gemini con **Deep Research** activado.
2. Copia `R1` a `R6` (con el botón de copiar de cada bloque o abriendo los `.txt` en `deep_research/prompts_1clic/` con `Ctrl+A` → `Ctrl+C`).
3. Al revisar el plan que propone Gemini en cada pestaña, verifica que priorice artículos revisados por pares (PubMed, PMC, Nature, IEEE, Elsevier, Springer) y aprueba.
4. Cuando terminen, guarda o exporta los informes en `01_estado_del_arte_y_marco_teorico/nuevo_estado_del_arte/deep_research/` con nombres `R1.md` (o `.docx` / `.pdf` / `.txt`) a `R6.md`.

---

## R1. Validación externa y entre bases en clasificación de glaucoma
Archivo directo: `deep_research/prompts_1clic/R1_validacion_externa.txt`

```text
CONTEXTO DEL PROYECTO
Somos un equipo de estudiantes de Deep Learning (Universidad Nacional de Colombia, Medellín) escribiendo un artículo en formato de la revista DYNA titulado "Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías". Entrenamos ConvNeXt-Tiny (preentrenado en ImageNet) en el dataset Moondream Staged (Hugging Face, 3 estadios: normal, temprano, avanzado) y lo probamos sin reentrenar en ACRIMA, RIM-ONE DL y EyePACS (glaucoma binario). Comparamos cabeza binaria frente a cabeza de 3 clases proyectada a binaria, y un aumento de datos agresivo de escala y fotometría. El umbral se fija en validación interna (especificidad ≥ 0,95) y no se reajusta en las bases externas. Observamos que el AUROC cae moderadamente fuera de dominio, pero la sensibilidad colapsa en EyePACS (recortes papilares frente a campo amplio con fondo negro).

Nuestra tesis para el estado del arte: la literatura demuestra discriminación alta dentro de una base, pero rara vez separa tres fuentes de variación cuando el modelo cambia de fuente: (1) variación de la imagen (cámara, campo de visión, encuadre, color, calidad), (2) variación de la etiqueta (qué cuenta como glaucoma, sospechoso, referible o estadio, y con qué estándar de referencia) y (3) cambio del punto de decisión (umbral y calibración transportados entre dominios).

REGLAS PARA TU RESPUESTA
- Prioriza artículos revisados por pares de 2016 a 2026. Incluye preprints solo si no hay versión publicada y márcalos como preprint.
- No inventes cifras, DOI, autores, años ni nombres de revista. Si no pudiste ver el dato en la fuente, escribe "no reportado" o "no verificado". No estimes valores a partir de figuras salvo que lo indiques.
- Para cada cifra, indica dónde aparece (tabla, figura, sección o resumen) y si la tomaste del texto completo o solo del resumen.
- Distingue siempre validación interna (partición aleatoria de la misma base), validación entre bases (otra base pública), leave-one-domain-out y validación externa en un centro independiente.
- Busca en inglés. Responde en español, conservando en inglés los títulos de los artículos y los términos técnicos estándar.
- Al final incluye una sección "Afirmaciones dudosas o contradictorias" con lo que no pudiste confirmar o donde las fuentes no coinciden.
- Incluye siempre el DOI exacto en formato canónico (10.xxxx/...) en la columna correspondiente; si el trabajo no tiene DOI o no pudiste comprobarlo, escribe "sin DOI verificado".

PREGUNTA
¿Qué estudios de clasificación de glaucoma con retinografías a color mediante aprendizaje profundo evaluaron su modelo fuera de la base de entrenamiento, y cuánto cambió el desempeño?

QUÉ BUSCAR
- Estudios con prueba entre bases públicas (por ejemplo ACRIMA, RIM-ONE, Drishti-GS, ORIGA, REFUGE, LAG, G1020, PAPILA, AIROGS, SMDG-19, EyePACS, BRSET u otras), con leave-one-domain-out o con validación externa en otro hospital o país.
- Desafíos y benchmarks con conjuntos de prueba ocultos o multicéntricos (REFUGE, AIROGS y otros).
- Estudios clínicos grandes con validación externa (por ejemplo Li et al. 2018, Ting et al. 2017, Phene et al. 2019, Hemelings et al. 2023, y los que encuentres).
- Meta-análisis o revisiones sistemáticas que reporten específicamente la diferencia entre validación interna y externa.

ENTREGA
1. Una tabla con una fila por estudio y estas columnas: autor y año | revista o congreso | DOI | tarea y definición de la clase positiva | bases de entrenamiento (n, cámara, campo de visión si se reporta) | bases de prueba (n, cámara, campo de visión) | tipo de validación | ¿partición por paciente? (sí/no/no reportado) | arquitectura y preentrenamiento | AUC interna | AUC externa | sensibilidad y especificidad externas | cómo se fijó el umbral (en desarrollo o reajustado en destino) | ubicación de las cifras | texto completo o solo resumen.
2. Un párrafo sobre qué tan frecuente es cada tipo de validación en lo que encontraste (sin inventar porcentajes; cuenta solo tus filas).
3. Un párrafo sobre los casos en que la caída fue pequeña y qué tenían en común, y otro sobre los casos en que fue grande.
4. La sección de afirmaciones dudosas.

Apunta a 20 a 35 estudios. Prefiere la calidad de cada fila a la cantidad de filas.
```

## R2. Catálogo de bases públicas de retinografías para glaucoma
Archivo directo: `deep_research/prompts_1clic/R2_catalogo_bases.txt`

```text
CONTEXTO DEL PROYECTO
Somos un equipo de estudiantes de Deep Learning (Universidad Nacional de Colombia, Medellín) escribiendo un artículo en formato de la revista DYNA titulado "Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías". Entrenamos ConvNeXt-Tiny (preentrenado en ImageNet) en el dataset Moondream Staged (Hugging Face, 3 estadios: normal, temprano, avanzado) y lo probamos sin reentrenar en ACRIMA, RIM-ONE DL y EyePACS (glaucoma binario). Comparamos cabeza binaria frente a cabeza de 3 clases proyectada a binaria, y un aumento de datos agresivo de escala y fotometría. El umbral se fija en validación interna (especificidad ≥ 0,95) y no se reajusta en las bases externas. Observamos que el AUROC cae moderadamente fuera de dominio, pero la sensibilidad colapsa en EyePACS (recortes papilares frente a campo amplio con fondo negro).

Nuestra tesis para el estado del arte: la literatura demuestra discriminación alta dentro de una base, pero rara vez separa tres fuentes de variación cuando el modelo cambia de fuente: (1) variación de la imagen (cámara, campo de visión, encuadre, color, calidad), (2) variación de la etiqueta (qué cuenta como glaucoma, sospechoso, referible o estadio, y con qué estándar de referencia) y (3) cambio del punto de decisión (umbral y calibración transportados entre dominios).

REGLAS PARA TU RESPUESTA
- Prioriza artículos revisados por pares de 2016 a 2026. Incluye preprints solo si no hay versión publicada y márcalos como preprint.
- No inventes cifras, DOI, autores, años ni nombres de revista. Si no pudiste ver el dato en la fuente, escribe "no reportado" o "no verificado". No estimes valores a partir de figuras salvo que lo indiques.
- Para cada cifra, indica dónde aparece (tabla, figura, sección o resumen) y si la tomaste del texto completo o solo del resumen.
- Distingue siempre validación interna (partición aleatoria de la misma base), validación entre bases (otra base pública), leave-one-domain-out y validación externa en un centro independiente.
- Busca en inglés. Responde en español, conservando en inglés los títulos de los artículos y los términos técnicos estándar.
- Al final incluye una sección "Afirmaciones dudosas o contradictorias" con lo que no pudiste confirmar o donde las fuentes no coinciden.
- Incluye siempre el DOI exacto en formato canónico (10.xxxx/...) en la columna correspondiente; si el trabajo no tiene DOI o no pudiste comprobarlo, escribe "sin DOI verificado".

PREGUNTA
¿Cómo se construyeron las bases públicas de retinografías usadas para clasificar glaucoma, y en qué difieren en adquisición y en etiqueta?

BASES A CUBRIR (añade otras relevantes)
ACRIMA, RIM-ONE (r1, r2, r3 y RIM-ONE DL), Drishti-GS1, ORIGA, REFUGE y REFUGE2, LAG, G1020, PAPILA, AIROGS (Rotterdam EyePACS AIROGS), SMDG-19, EyePACS (aclara qué versión tiene etiquetas de glaucoma), BRSET, Harvard-GDP o FairVision si aplica, y el dataset "moondream/glaucoma-detection" de Hugging Face.

ENTREGA
1. Una tabla con una fila por base: nombre y versión | artículo de origen con DOI | país e institución | número de imágenes y de pacientes | cámara o cámaras y campo de visión | encuadre (campo completo centrado en mácula o recorte centrado en el disco) | resolución | definición de la etiqueta (glaucoma confirmado, sospechoso, referible, estadios) | quién etiquetó y con qué estándar (examen clínico, campo visual, OCT, consenso) | ¿hay un ojo por paciente o ambos? | ¿hay partición oficial? | licencia y acceso.
2. Qué bases comparten imágenes o derivan unas de otras (por ejemplo versiones de RIM-ONE, imágenes de EyePACS en AIROGS u otras compilaciones). Esto nos importa para detectar fuga de datos entre bases.
3. Qué sabes y qué no sabes del dataset de Hugging Face "moondream/glaucoma-detection": origen de las imágenes, criterio de los estadios, si reutiliza imágenes de otra base.
4. La sección de afirmaciones dudosas, sobre todo donde distintas fuentes dan números distintos para la misma base.
```

## R3. Variación de la imagen: qué factores de adquisición degradan el desempeño
Archivo directo: `deep_research/prompts_1clic/R3_variacion_imagen.txt`

```text
CONTEXTO DEL PROYECTO
Somos un equipo de estudiantes de Deep Learning (Universidad Nacional de Colombia, Medellín) escribiendo un artículo en formato de la revista DYNA titulado "Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías". Entrenamos ConvNeXt-Tiny (preentrenado en ImageNet) en el dataset Moondream Staged (Hugging Face, 3 estadios: normal, temprano, avanzado) y lo probamos sin reentrenar en ACRIMA, RIM-ONE DL y EyePACS (glaucoma binario). Comparamos cabeza binaria frente a cabeza de 3 clases proyectada a binaria, y un aumento de datos agresivo de escala y fotometría. El umbral se fija en validación interna (especificidad ≥ 0,95) y no se reajusta en las bases externas. Observamos que el AUROC cae moderadamente fuera de dominio, pero la sensibilidad colapsa en EyePACS (recortes papilares frente a campo amplio con fondo negro).

Nuestra tesis para el estado del arte: la literatura demuestra discriminación alta dentro de una base, pero rara vez separa tres fuentes de variación cuando el modelo cambia de fuente: (1) variación de la imagen (cámara, campo de visión, encuadre, color, calidad), (2) variación de la etiqueta (qué cuenta como glaucoma, sospechoso, referible o estadio, y con qué estándar de referencia) y (3) cambio del punto de decisión (umbral y calibración transportados entre dominios).

REGLAS PARA TU RESPUESTA
- Prioriza artículos revisados por pares de 2016 a 2026. Incluye preprints solo si no hay versión publicada y márcalos como preprint.
- No inventes cifras, DOI, autores, años ni nombres de revista. Si no pudiste ver el dato en la fuente, escribe "no reportado" o "no verificado". No estimes valores a partir de figuras salvo que lo indiques.
- Para cada cifra, indica dónde aparece (tabla, figura, sección o resumen) y si la tomaste del texto completo o solo del resumen.
- Distingue siempre validación interna (partición aleatoria de la misma base), validación entre bases (otra base pública), leave-one-domain-out y validación externa en un centro independiente.
- Busca en inglés. Responde en español, conservando en inglés los títulos de los artículos y los términos técnicos estándar.
- Al final incluye una sección "Afirmaciones dudosas o contradictorias" con lo que no pudiste confirmar o donde las fuentes no coinciden.
- Incluye siempre el DOI exacto en formato canónico (10.xxxx/...) en la columna correspondiente; si el trabajo no tiene DOI o no pudiste comprobarlo, escribe "sin DOI verificado".

PREGUNTA
¿Qué evidencia hay de que factores concretos de adquisición (fabricante y modelo de cámara, campo de visión, encuadre o recorte, resolución, color e iluminación, calidad de imagen, retinógrafos portátiles o de smartphone) causan la caída de desempeño de clasificadores de glaucoma o de enfermedades retinianas, y qué estudios aíslan cada factor con comparaciones controladas?

QUÉ BUSCAR
- Estudios que comparan el mismo modelo en imágenes de distintas cámaras del mismo paciente o de la misma población.
- Experimentos de ablación o perturbación controlada (recortar, cambiar escala, alterar color, degradar calidad) y su efecto en glaucoma o, si no hay, en retinopatía diabética.
- Trabajos sobre evaluación de calidad de imagen y rechazo de imágenes no gradables (AIROGS y otros).
- Evidencia de "shortcut learning" en retinografías: modelos que aprenden la fuente, el borde negro, texto incrustado o el color de la cámara en lugar de signos clínicos.

ENTREGA
1. Una tabla: autor y año | DOI | factor estudiado | diseño (controlado, observacional entre bases o perturbación sintética) | enfermedad | magnitud del efecto con su métrica | ubicación de la cifra.
2. Una síntesis que separe lo que está demostrado con comparaciones controladas de lo que solo se infiere de comparaciones entre bases donde cambian muchas cosas a la vez.
3. La sección de afirmaciones dudosas.
```

## R4. Variación de la etiqueta y estadificación
Archivo directo: `deep_research/prompts_1clic/R4_variacion_etiqueta_estadios.txt`

```text
CONTEXTO DEL PROYECTO
Somos un equipo de estudiantes de Deep Learning (Universidad Nacional de Colombia, Medellín) escribiendo un artículo en formato de la revista DYNA titulado "Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías". Entrenamos ConvNeXt-Tiny (preentrenado en ImageNet) en el dataset Moondream Staged (Hugging Face, 3 estadios: normal, temprano, avanzado) y lo probamos sin reentrenar en ACRIMA, RIM-ONE DL y EyePACS (glaucoma binario). Comparamos cabeza binaria frente a cabeza de 3 clases proyectada a binaria, y un aumento de datos agresivo de escala y fotometría. El umbral se fija en validación interna (especificidad ≥ 0,95) y no se reajusta en las bases externas. Observamos que el AUROC cae moderadamente fuera de dominio, pero la sensibilidad colapsa en EyePACS (recortes papilares frente a campo amplio con fondo negro).

Nuestra tesis para el estado del arte: la literatura demuestra discriminación alta dentro de una base, pero rara vez separa tres fuentes de variación cuando el modelo cambia de fuente: (1) variación de la imagen (cámara, campo de visión, encuadre, color, calidad), (2) variación de la etiqueta (qué cuenta como glaucoma, sospechoso, referible o estadio, y con qué estándar de referencia) y (3) cambio del punto de decisión (umbral y calibración transportados entre dominios).

REGLAS PARA TU RESPUESTA
- Prioriza artículos revisados por pares de 2016 a 2026. Incluye preprints solo si no hay versión publicada y márcalos como preprint.
- No inventes cifras, DOI, autores, años ni nombres de revista. Si no pudiste ver el dato en la fuente, escribe "no reportado" o "no verificado". No estimes valores a partir de figuras salvo que lo indiques.
- Para cada cifra, indica dónde aparece (tabla, figura, sección o resumen) y si la tomaste del texto completo o solo del resumen.
- Distingue siempre validación interna (partición aleatoria de la misma base), validación entre bases (otra base pública), leave-one-domain-out y validación externa en un centro independiente.
- Busca en inglés. Responde en español, conservando en inglés los títulos de los artículos y los términos técnicos estándar.
- Al final incluye una sección "Afirmaciones dudosas o contradictorias" con lo que no pudiste confirmar o donde las fuentes no coinciden.
- Incluye siempre el DOI exacto en formato canónico (10.xxxx/...) en la columna correspondiente; si el trabajo no tiene DOI o no pudiste comprobarlo, escribe "sin DOI verificado".

PREGUNTA
¿Cómo afectan la definición de la etiqueta, el estándar de referencia y la severidad del glaucoma al desempeño y a la generalización de clasificadores en retinografías? ¿Qué se ha hecho en clasificación por estadios u ordinal?

QUÉ BUSCAR
- Estudios que comparan definiciones de caso (glaucoma confirmado frente a sospechoso, neuropatía óptica glaucomatosa referible, etiquetas por consenso frente a un solo evaluador, etiqueta basada en fotografía frente a confirmación con campo visual u OCT).
- Variabilidad entre expertos al etiquetar glaucoma en retinografías (acuerdo interobservador).
- Desempeño estratificado por severidad (temprano frente a avanzado), por ejemplo Ahn et al. 2018 y otros.
- Clasificación en varios estadios u ordinal desde retinografías, métricas usadas (por ejemplo kappa cuadrático ponderado) y criterios de estadificación (Hodapp-Parrish-Anderson, DDLS, otros).
- Modelos que predicen medidas continuas (relación copa-disco vertical, defecto del campo visual, grosor de la capa de fibras por OCT) desde la retinografía, y su relación con el problema de la etiqueta.
- Ruido de etiqueta y su efecto en la transferencia entre bases.

ENTREGA
1. Una tabla: autor y año | DOI | definición de etiqueta o esquema de estadios | estándar de referencia | métrica por subgrupo de severidad o por estadio | ¿validación externa? | ubicación de las cifras.
2. Una síntesis sobre si la falta de concordancia de etiquetas entre bases se ha medido o solo se ha señalado.
3. La sección de afirmaciones dudosas.
```

## R5. Umbral operativo, calibración y cambio de prevalencia
Archivo directo: `deep_research/prompts_1clic/R5_umbral_calibracion_prevalencia.txt`

```text
CONTEXTO DEL PROYECTO
Somos un equipo de estudiantes de Deep Learning (Universidad Nacional de Colombia, Medellín) escribiendo un artículo en formato de la revista DYNA titulado "Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías". Entrenamos ConvNeXt-Tiny (preentrenado en ImageNet) en el dataset Moondream Staged (Hugging Face, 3 estadios: normal, temprano, avanzado) y lo probamos sin reentrenar en ACRIMA, RIM-ONE DL y EyePACS (glaucoma binario). Comparamos cabeza binaria frente a cabeza de 3 clases proyectada a binaria, y un aumento de datos agresivo de escala y fotometría. El umbral se fija en validación interna (especificidad ≥ 0,95) y no se reajusta en las bases externas. Observamos que el AUROC cae moderadamente fuera de dominio, pero la sensibilidad colapsa en EyePACS (recortes papilares frente a campo amplio con fondo negro).

Nuestra tesis para el estado del arte: la literatura demuestra discriminación alta dentro de una base, pero rara vez separa tres fuentes de variación cuando el modelo cambia de fuente: (1) variación de la imagen (cámara, campo de visión, encuadre, color, calidad), (2) variación de la etiqueta (qué cuenta como glaucoma, sospechoso, referible o estadio, y con qué estándar de referencia) y (3) cambio del punto de decisión (umbral y calibración transportados entre dominios).

REGLAS PARA TU RESPUESTA
- Prioriza artículos revisados por pares de 2016 a 2026. Incluye preprints solo si no hay versión publicada y márcalos como preprint.
- No inventes cifras, DOI, autores, años ni nombres de revista. Si no pudiste ver el dato en la fuente, escribe "no reportado" o "no verificado". No estimes valores a partir de figuras salvo que lo indiques.
- Para cada cifra, indica dónde aparece (tabla, figura, sección o resumen) y si la tomaste del texto completo o solo del resumen.
- Distingue siempre validación interna (partición aleatoria de la misma base), validación entre bases (otra base pública), leave-one-domain-out y validación externa en un centro independiente.
- Busca en inglés. Responde en español, conservando en inglés los títulos de los artículos y los términos técnicos estándar.
- Al final incluye una sección "Afirmaciones dudosas o contradictorias" con lo que no pudiste confirmar o donde las fuentes no coinciden.
- Incluye siempre el DOI exacto en formato canónico (10.xxxx/...) en la columna correspondiente; si el trabajo no tiene DOI o no pudiste comprobarlo, escribe "sin DOI verificado".

PREGUNTA
Cuando un clasificador de imágenes médicas, y en particular de glaucoma o de enfermedades retinianas, se aplica en otra fuente, ¿qué pasa con el umbral de decisión y la calibración de sus probabilidades, y cómo lo reportan y corrigen los estudios?

QUÉ BUSCAR
- Estudios de glaucoma o de retina que reportan sensibilidad y especificidad en bases externas con un umbral fijado en desarrollo, frente a los que reajustan el umbral en el destino.
- Reportes de sensibilidad a especificidad fija (por ejemplo 90 % o 95 %) y de AUC parcial en glaucoma.
- Calibración de redes profundas en imágenes médicas bajo cambio de dominio (ECE, temperature scaling, recalibración en sitio).
- Cambio de prevalencia (label shift) entre bases balanceadas de laboratorio y tamizaje poblacional, y su efecto en valores predictivos.
- Guías de reporte y validación de modelos clínicos que traten este punto (por ejemplo TRIPOD+AI, CLAIM, STARD-AI, recomendaciones de validación externa).

ENTREGA
1. Una tabla: autor y año | DOI | dominio clínico | cómo se fijó el umbral | qué pasó al cambiar de fuente (sensibilidad, especificidad, calibración) | método de corrección, si hubo | ubicación de las cifras.
2. Una síntesis sobre qué tan común es reportar el umbral y separar discriminación de punto operativo en los estudios de glaucoma.
3. La sección de afirmaciones dudosas.
```

## R6. Estrategias para mitigar el cambio de dominio y su evidencia
Archivo directo: `deep_research/prompts_1clic/R6_estrategias_mitigacion.txt`

```text
CONTEXTO DEL PROYECTO
Somos un equipo de estudiantes de Deep Learning (Universidad Nacional de Colombia, Medellín) escribiendo un artículo en formato de la revista DYNA titulado "Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías". Entrenamos ConvNeXt-Tiny (preentrenado en ImageNet) en el dataset Moondream Staged (Hugging Face, 3 estadios: normal, temprano, avanzado) y lo probamos sin reentrenar en ACRIMA, RIM-ONE DL y EyePACS (glaucoma binario). Comparamos cabeza binaria frente a cabeza de 3 clases proyectada a binaria, y un aumento de datos agresivo de escala y fotometría. El umbral se fija en validación interna (especificidad ≥ 0,95) y no se reajusta en las bases externas. Observamos que el AUROC cae moderadamente fuera de dominio, pero la sensibilidad colapsa en EyePACS (recortes papilares frente a campo amplio con fondo negro).

Nuestra tesis para el estado del arte: la literatura demuestra discriminación alta dentro de una base, pero rara vez separa tres fuentes de variación cuando el modelo cambia de fuente: (1) variación de la imagen (cámara, campo de visión, encuadre, color, calidad), (2) variación de la etiqueta (qué cuenta como glaucoma, sospechoso, referible o estadio, y con qué estándar de referencia) y (3) cambio del punto de decisión (umbral y calibración transportados entre dominios).

REGLAS PARA TU RESPUESTA
- Prioriza artículos revisados por pares de 2016 a 2026. Incluye preprints solo si no hay versión publicada y márcalos como preprint.
- No inventes cifras, DOI, autores, años ni nombres de revista. Si no pudiste ver el dato en la fuente, escribe "no reportado" o "no verificado". No estimes valores a partir de figuras salvo que lo indiques.
- Para cada cifra, indica dónde aparece (tabla, figura, sección o resumen) y si la tomaste del texto completo o solo del resumen.
- Distingue siempre validación interna (partición aleatoria de la misma base), validación entre bases (otra base pública), leave-one-domain-out y validación externa en un centro independiente.
- Busca en inglés. Responde en español, conservando en inglés los títulos de los artículos y los términos técnicos estándar.
- Al final incluye una sección "Afirmaciones dudosas o contradictorias" con lo que no pudiste confirmar o donde las fuentes no coinciden.
- Incluye siempre el DOI exacto en formato canónico (10.xxxx/...) en la columna correspondiente; si el trabajo no tiene DOI o no pudiste comprobarlo, escribe "sin DOI verificado".

PREGUNTA
¿Qué estrategias se han propuesto para mejorar la generalización entre fuentes de clasificadores de glaucoma en retinografías, qué tipo de variación ataca cada una y qué evidencia comparativa existe bajo un mismo protocolo?

ESTRATEGIAS A CUBRIR
Entrenamiento multifuente; aumento de datos geométrico y fotométrico (incluido el aumento agresivo de escala); armonización o recorte centrado en el disco óptico (ROI); normalización o estandarización de color; filtrado por calidad de imagen; generalización de dominio (por ejemplo DoFE y métodos similares); adaptación de dominio no supervisada; adaptación en tiempo de prueba; preentrenamiento autosupervisado y modelos fundacionales retinianos (RETFound, VisionFM, FLAIR, RetiZero, UrFound u otros de 2023–2026); modelos de visión y lenguaje aplicados a glaucoma.

QUÉ BUSCAR
- Estudios que comparan una estrategia frente a una línea base con la misma arquitectura y el mismo protocolo, evaluando en una base no vista.
- Benchmarks de generalización de dominio en imágenes de fondo de ojo.
- Resultados de los modelos fundacionales específicamente en glaucoma y en evaluación externa, no solo en promedio sobre muchas tareas.

ENTREGA
1. Una tabla: autor y año | DOI | estrategia | variación que ataca (imagen, etiqueta o punto de decisión) | ¿usa imágenes del dominio destino durante el desarrollo? | línea base comparada | bases de prueba no vistas | mejora reportada con su métrica | ubicación de las cifras.
2. Una síntesis que distinga adaptación de dominio (usa datos del destino) de generalización estricta (destino no visto), y que diga qué estrategias tienen evidencia controlada y cuáles solo son plausibles.
3. Una lista de los trabajos más recientes (2024–2026) que debamos conocer.
4. La sección de afirmaciones dudosas.
```

## R7 (opcional). Cifras para la introducción y el marco teórico
Archivo directo: `deep_research/prompts_1clic/R7_opcional_epidemiologia.txt`

```text
Necesito cifras verificables para la introducción de un artículo académico sobre detección automática de glaucoma en retinografías.

1. Prevalencia y número estimado de personas con glaucoma en el mundo y en América Latina, con la fuente revisada por pares más reciente (por ejemplo Tham et al. 2014 y estimaciones posteriores de GBD o del Vision Loss Expert Group). Indica año, grupo de edad y proyección.
2. Proporción de casos no diagnosticados según estudios poblacionales.
3. Recomendaciones vigentes sobre tamizaje de glaucoma (USPSTF, AAO Preferred Practice Pattern, otras), con año y nivel de recomendación.
4. Si existe, cualquier dato de Colombia sobre glaucoma.

Para cada cifra da la referencia completa con DOI y la tabla o sección donde aparece. No uses cifras de páginas web sin fuente primaria. Busca en inglés y en español; responde en español.
```
