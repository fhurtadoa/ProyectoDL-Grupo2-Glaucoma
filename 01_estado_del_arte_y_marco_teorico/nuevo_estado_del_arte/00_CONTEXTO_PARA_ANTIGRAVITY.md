# Contexto para continuar el nuevo estado del arte (Antigravity)

Documento de traspaso. Léelo completo antes de tocar cualquier archivo. Resume el proyecto, el estado real del repositorio, lo que está mal en el estado del arte anterior, las reglas de la entrega final y el plan para el nuevo estado del arte.

Preparado el 24 de noviembre de 2026 a partir del repositorio (commit `9641e5b`), de las guías del equipo y de una revisión crítica previa del estado del arte.

---

## 1. Quién, qué y para cuándo

- **Proyecto:** artículo final del curso Deep Learning (2026-2), Facultad de Minas, Universidad Nacional de Colombia, sede Medellín. Formato de la revista DYNA. No se envía a la revista; es la entrega del curso.
- **Título (ES):** Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías.
- **Título (EN):** Evaluating the generalization of deep glaucoma classifiers under data-acquisition heterogeneity in retinal fundus photographs.
- **Autores (Grupo 2):** Jair Alejandro Acevedo-Londoño (jacevedolo@unal.edu.co), Fabián Andrés Hurtado-Arango (fhurtadoa@unal.edu.co), Jose Mauricio Toscano-Aguas (jtoscanoa@unal.edu.co), Andrés Arbeláez-Morales (anarbelaezm@unal.edu.co), Andrés Felipe Díez-Ángel (adieza@unal.edu.co). Sin biografías.
- **Responsable de esta tarea:** Mauricio. Escribe Encabezado/Abstract/Resumen, 1 Introducción, 2 Marco teórico, 3 Método de revisión y estado del arte, y Referencias. También ensambla el artículo completo con las secciones de los demás.
- **Repositorio:** https://github.com/fhurtadoa/ProyectoDL-Grupo2-Glaucoma (local: `C:\Users\Mau\Documents\GitHub\ProyectoDL-Grupo2-Glaucoma`).

### Reparto del artículo (Guía maestra)

| Sección | Responsable | Páginas |
| --- | --- | --- |
| Encabezado, Abstract y Resumen | Mauricio | 0,75 |
| 1 Introducción | Mauricio | 0,25 (puede crecer) |
| 2 Marco teórico | Mauricio | 0,5 |
| 3 Método de revisión y estado del arte | Mauricio | 0,75 (~650 palabras sin figuras) |
| 4.1 Selección de datasets | Andrés Arbeláez | 1,5 |
| 4.2 Selección de modelo | Andrés Arbeláez | 1 |
| 4.3 Estadificar un dataset binario (opcional) | Andrés Arbeláez | 0,75 |
| 4.4 Análisis trinario frente a binario | Fabián | 1 |
| 4.5 Data augmentation | Fabián | 1 |
| 4.6 Conclusión sobre generalización | Fabián | ≤ 1 |
| Referencias | Mauricio (los demás confirman las suyas) | 0,5 (puede crecer) |

Jair Alejandro y Andrés Felipe no tienen sección asignada todavía; pueden encargarse de la revisión cruzada o del segundo cribado del nuevo estado del arte (ver §6).

**Tensión de extensión a resolver con el equipo:** la guía maestra fija 0,75 páginas para la sección 3, pero en la conversación previa se dijo que el trabajo no tiene límite estricto y que la mayor extensión se valora. El nuevo estado del arte se escribe primero en una versión larga y luego se condensa a la versión del artículo.

---

## 2. Qué estudia el proyecto (objeto material y formal)

- **Objeto material:** clasificadores profundos de glaucoma sobre retinografías a color.
- **Objeto formal:** cómo cambia su desempeño con la fuente de adquisición (cámara, encuadre, centro) y qué estrategias centradas en datos lo mitigan.

### Datos (base limpia en `utilidades/dataset_auditado.csv`, 2.056 imágenes)

| Colección | Imágenes | Etiquetas | Uso |
| --- | --- | --- | --- |
| Moondream Staged | 706 (504 train / 107 val / 95 test) | 3 estadios: normal, early, advanced | Entrenamiento y prueba interna |
| ACRIMA | 703 (308 normal / 395 glaucoma) | Binarias | Prueba externa |
| RIM-ONE DL | 447 (285 normal / 162 glaucoma) | Binarias | Prueba externa |
| EyePACS REAL | 200 (100 normal / 100 glaucoma) | Binarias | Prueba externa |

Depuración: MD5 y similitud coseno ResNet-18 ≥ 0,96 (Andrés, Moondream 800 → 762), más coincidencias geométricas SIFT/RANSAC y exclusión de cruces entre fuentes (Fabián, 762 → 706). Conteos por motivo en `utilidades/resumen_depuracion.csv`.

Advertencias sobre los datos:
- **Moondream** proviene de `moondream/glaucoma-detection` en Hugging Face (2024). No tiene artículo, metadatos de cámara ni EXIF; se cita como recurso en línea con fecha de consulta. Esta procedencia está marcada como NO VERIFICADA en `LOG_PROTOCOLO.md`; confírmala.
- **RIM-ONE DL**: el artículo original (Fumero Batista et al. 2020, Image Analysis & Stereology 39(3):161–167, DOI 10.5566/ias.2346) reporta 485 imágenes (313 normales, 172 glaucoma); la copia local tiene 447. Solo la v3 usó Kowa WX 3D; v1 y v2 usaron Nidek AFC-210.
- No hay máscaras verificadas de disco óptico ni datos de paciente; no se puede garantizar partición por paciente en las fuentes externas.

### Modelo y protocolo común

ConvNeXt-Tiny preentrenado en ImageNet, semilla 42, 1 época de calentamiento + 20 de ajuste fino (AdamW, CosineAnnealingLR). Época y umbral τ elegidos solo en validación de Moondream buscando especificidad ≥ 0,95 (τ = 0,81 proyección trinaria; τ = 0,90 binario). El umbral **no** se reajusta en las cohortes externas.

### Resultados verificados (leídos de los CSV del repo)

`04_experimento_trinario_vs_binario/resultados/01_metricas_binarias.csv`:

| Cohorte | n | Modelo | F1 macro | Sens | Esp | AUROC |
| --- | --- | --- | --- | --- | --- | --- |
| Moondream test | 95 | Proyección trinario | 0,916 | 0,935 | 0,898 | 0,980 |
| Moondream test | 95 | Entrenado binario | 0,852 | 0,804 | 0,898 | 0,954 |
| ACRIMA | 703 | Proyección trinario | 0,673 | 0,782 | 0,562 | 0,744 |
| ACRIMA | 703 | Entrenado binario | 0,605 | 0,714 | 0,497 | 0,635 |
| RIM-ONE DL | 447 | Proyección trinario | 0,738 | 0,704 | 0,782 | 0,816 |
| RIM-ONE DL | 447 | Entrenado binario | 0,684 | 0,691 | 0,698 | 0,783 |
| EyePACS REAL | 200 | Proyección trinario | 0,353 | 0,020 | 0,990 | 0,553 |
| EyePACS REAL | 200 | Entrenado binario | 0,333 | 0,000 | 1,000 | 0,632 |

`05_experimento_data_augmentation/resultados/02_metricas_factorial.csv` (diseño 2×2: cabeza de 2 o 3 clases × `Base` o `Malignant`). Hallazgos principales:
- Binario (2→2): `Malignant` mejora F1 en las 4 cohortes (+6,6 a +11,7 pp). En EyePACS REAL el AUROC sube de 0,546 a 0,686.
- Trinario proyectado (3→2): efecto mixto. Mejora en EyePACS REAL (AUROC 0,577 → 0,697) y empeora en RIM-ONE DL (F1 −10,1 pp, TFN +38,3 pp) porque el umbral calibrado en Moondream queda demasiado conservador.
- 3→3 en Moondream test: QWK 0,726 (`Base`) frente a 0,733 (`Malignant`).

**Lo que los experimentos sí permiten decir:** la discriminación (AUROC) y el punto operativo (τ) se degradan de forma distinta fuera de dominio. EyePACS conserva algo de ordenamiento, pero la sensibilidad colapsa con el umbral heredado. El encuadre difiere mucho entre fuentes: ACRIMA y RIM-ONE DL son recortes papilares al 100 % del marco; EyePACS REAL tiene 21,8 % de fondo negro (ver `03_experimento_caracteristicas_retinografias/`).

**Lo que no permiten decir:** que la causa de la caída sea la cámara, el FOV o el color por separado. Tampoco que haya generalización a un dominio no visto con la megacohorte: el test de 431 imágenes del material antiguo es **validación interna** sobre una cohorte agregada.

Reglas de cifras: toda cifra del artículo sale de un archivo de resultados (`04_.../resultados`, `05_.../resultados`, `03_.../datos`), nunca de los informes PDF ni del README. Nada de `archivo_historico/` es fuente final. La matriz entre fuentes de `datos_benchmark/cross_dataset_generalization_matrix.json` tiene valores escritos a mano en un script antiguo; no se cita.

---

## 3. Por qué se empieza un estado del arte nuevo

### 3.1 La revisión sistemática anterior fue simulada

Esto se comprobó leyendo `01_estado_del_arte_y_marco_teorico/scripts_rsl/`:

- **Cribado y κ:** `04_cribado_ta.py` no tiene revisores humanos. "Evaluador 1" y "Evaluador 2" son dos funciones de reglas de palabras clave sobre título y resumen. El κ = 0,6484 mide el acuerdo entre dos heurísticas, no entre personas. Además, el protocolo fijaba un umbral de κ ≥ 0,75.
- **Texto completo:** `05_texto_completo_snowballing.py` decide la inclusión con las mismas reglas de palabras clave sobre título y resumen. No leyó texto completo; 69 de los 108 artículos ni siquiera se descargaron.
- **Calidad y extracción:** `06_calidad_y_extraccion.py` contiene un diccionario fijo `DATOS_CURADOS_ESTUDIOS` con las cifras de 15 estudios escritas a mano por el agente (AUC interna, externa, ΔAUC, QA1–QA9). No vienen de los artículos. Las puntuaciones de calidad son casi perfectas (media 8,33/9).
- **Contradicción comprobada:** el script asigna a Diaz-Pinto 2019 AUC 0,996 → 0,772 (ΔAUC 0,224). La verificación del propio equipo (`marco_teorico/salidas/informe.md`) leyó el artículo y encontró AUC 0,7678–0,8575 en la prueba entre bases.
- **Consecuencia:** las figuras de bibliometría, protocolos, bases, estrategias, ΔAUC y años, así como el PRISMA con 82 + 15 estudios, salen de ese proceso simulado. No deben reutilizarse.
- **Velocidad:** el `LOG_PROTOCOLO.md` registra todo el proceso, de la búsqueda a la redacción "completada al 100 %", entre las 21:36 y las 21:57 del 17 de noviembre.

### 3.2 Problemas del PDF `estado_del_arte.pdf` (revisión previa)

- El PRISMA dice 82 + 15 estudios; las figuras suman 15 y el texto cita 9.
- Se excluyó en cribado la evaluación solo intra-base (E3), pero la Fig. 2 cuenta estudios con partición intra-base.
- La Fig. 4 pone a Hemelings 2023 con la mayor caída (ΔAUC 0,231) mientras el texto dice que su modelo conserva discriminación. Incluye estudios que no están en las referencias (Fang, Bajwa, Liu, Nakayama, Wang). El "umbral crítico" de 0,10 no tiene justificación. Comparar ΔAUC entre estudios con etiquetas y bases distintas es justo lo que el texto critica.
- La tabla de estudios afirma más que el texto: "primer benchmark sistemático" (Diaz-Pinto), "crítico subdiagnóstico del estadio temprano" (Ahn), "regresión ordinal" (Hemelings) frente a "regresión de riesgo" en el texto.
- Formato: títulos de figura duplicados ("Figura: Fig. 3"), asteriscos literales en las referencias, y un encabezado "Referencias verificadas" que debe ser solo "Referencias".

### 3.3 Qué se conserva del texto anterior

La tesis de `manuscrito/estado_del_arte_reescrito.md` es buena y encaja con los experimentos: el vacío no es otro clasificador con AUC interna alta, sino **separar tres fuentes de variación**:

1. Variación de la **imagen** (cámara, encuadre, color, calidad).
2. Variación de la **etiqueta** (qué cuenta como glaucoma, sospechoso, referible o estadio).
3. Cambio del **punto de decisión** (umbral y calibración transportados entre dominios).

Fabián retoma este vacío en 4.6. Los experimentos 04 (esquema de etiquetas), 05 (aumento de datos contra variación de imagen) y la caída de sensibilidad con τ fijo responden a cada una.

Las 9 referencias del texto reescrito tienen DOI plausibles (Gómez-Valverde 2019, Diaz-Pinto 2019, REFUGE/Orlando 2020, Li 2018, PAPILA/Kovalyk 2022, AIROGS/de Vente 2024, Hemelings 2023 npj, RETFound/Zhou 2023, Ahn 2018). Sirven como semillas, pero cada afirmación sobre ellas se vuelve a comprobar contra el texto completo.

### 3.4 Qué sí está verificado en el repo y se puede reutilizar

- `marco_teorico/salidas/estructura_y_mapa_f1.md`: esquema del marco teórico 2.1–2.6 con inventario de fuentes. Hay 36 marcadas ✅ con texto completo en `marco_teorico/refs/pdf|txt|html`.
- `marco_teorico/salidas/verificacion_refs.csv` y `respaldo_afirmaciones.csv`: verificación de metadatos y de afirmaciones del marco teórico.
- `marco_teorico/refs/crossref/*.json`: respuestas de Crossref guardadas.
- `manuscrito/PROTOCOLO.md`: las preguntas RQ1–RQ5, los criterios I/E, las ecuaciones de búsqueda y el formulario de extracción de 27 variables están bien pensados como diseño. El problema fue la ejecución, no el diseño.
- `manuscrito/referencias_dyna.md` (32 referencias) y `referencias.bib`: tómalas como candidatas, no como verificadas. La referencia [6] original de RIM-ONE DL resultó falsa (un DOI de ingeniería civil).

---

## 4. Reglas de la entrega final (plantillas en `Entrega Final/`)

Hay tres plantillas Word:
- `Plantilla_base_seccion_DYNA.docx`: hoja para que cada integrante escriba su sección.
- `Plantilla_completa_articulo_DYNA.docx`: esqueleto del artículo con marcadores por sección, autores y afiliación ya puestos, y un bloque "Declaraciones de los autores" (conflictos de interés, financiación, disponibilidad de datos, uso de herramientas de IA) antes de Referencias.
- `Plantilla_ejemplo_articulo_DYNA.docx`: el mismo esqueleto lleno de texto ficticio, con 8 tablas y 8 figuras de muestra. Incluye las subsecciones 2.1 Métricas de desempeño, 2.2 Umbral operativo y 4.1.1 Análisis descriptivo de las imágenes.

Formato que muestran las plantillas: carta, primera sección a una columna (títulos, autores, abstract, resumen) y cuerpo a dos columnas. Estilos de Word `Heading 1–3` y `Normal`. Ecuaciones numeradas a la derecha, "eq. (n)". Tablas con título arriba y "Fuente: Los autores." debajo. Figuras en escala de grises con título y fuente debajo, textos a 8 pt.

Guía maestra: Times New Roman, cuerpo 10 pt, resumen 9 pt, tablas y leyendas 8 pt, máximo 3 niveles de encabezado sin punto final, sin notas al pie. Citas [1], [1,2], [1-3]. Referencias numeradas por orden de aparición, estilo DYNA, con DOI. Abstract y Resumen de máximo 150 palabras; de 3 a 10 palabras clave separadas por punto y coma.

**Inconsistencias que hay que resolver antes de ensamblar:**
1. **Márgenes:** las tres plantillas tienen 1,69 cm en los cuatro lados y sin encuadernación. La guía maestra dice 2,2 cm arriba y abajo, 1,69 cm a los lados y 0,7 cm de encuadernación. Compruébalo contra la plantilla oficial de DYNA (Velásquez-Henao y Rada-Tobón, "Ejemplo del formato de un artículo").
2. **Formato de referencia:** la plantilla completa usa "Apellido, N., y Apellido, N., … DOI: https://doi.org/…". `referencias_dyna.md` usa "and" y "http://dx.doi.org/…". Hay que escoger uno según la plantilla oficial y aplicarlo a todas.
3. **Biografías:** la Guía de Mauricio las menciona, pero la Guía maestra y la plantilla completa no las incluyen. Se omiten.
4. **Nombre de la sección 3:** la plantilla dice "Método de revisión y estado del arte". Si el nuevo método no es una revisión sistemática completa, el texto debe decir con precisión qué tipo de revisión es (ver §5).

---

## 5. Plan del nuevo estado del arte

### 5.1 Decisión de método (Fase 0 cerrada el 2026-11-24 con Mauricio)

- **Tipo de revisión acordado:** **Revisión sistemática PRISMA 2020 estricta** (todos los conteos del diagrama PRISMA 2020 salen 100 % de archivos CSV trazables y verificables; cero heurísticas simuladas).
  - *Vía 1 PRISMA 2020 (Bases de datos y registros):* búsqueda booleana reproducible en OpenAlex, PubMed y Europe PMC con ecuaciones literales, fechas y respuestas crudas guardadas en `busqueda_prisma/`.
  - *Vía 2 PRISMA 2020 (Otros métodos):* exploración asistida por IA mediante 6 investigaciones paralelas en Gemini Deep Research (`R1`–`R6`, declarada explícitamente) + *snowballing* hacia atrás/adelante, verificando cada DOI contra Crossref antes de pasar a elegibilidad.
  - *Cribado y extracción:* cribado humano con cálculo real de $\kappa$ de Cohen sobre muestra independiente (dos revisores del equipo) y extracción manual/verificada únicamente sobre textos completos con ubicación exacta de cada cifra (tabla, figura o página).
- **Extensión acordada:** **Doble salida**: redactar primero la versión extensa verificada (`estado_del_arte_v2.md`, esquema 3.1–3.7) y luego condensarla al presupuesto de la plantilla DYNA (~0,75 páginas para la Sección 3).

*(Nota histórica: abajo se conserva la recomendación original previa a la decisión de Fase 0).*

Hacer una **revisión estructurada (de alcance) con búsqueda reproducible**, no una "revisión sistemática PRISMA" completa. Razones: el plazo del curso, las 0,75 páginas de la sección y que el equipo no puede leer 80 textos completos a tiempo. Una revisión estructurada se puede reportar con honestidad:

- Bases, ecuaciones y fecha de búsqueda (reutilizar las de `PROTOCOLO.md` §4).
- Exploración asistida por Gemini Deep Research, declarada como tal.
- Criterios de inclusión explícitos.
- Cribado de título y resumen por **dos personas del equipo** al menos sobre una muestra. Así el κ es real. Jair Alejandro y Andrés Felipe pueden ser el segundo revisor.
- Extracción solo de estudios cuyo texto completo se leyó, con la ubicación de cada cifra (tabla, figura o página).
- Un diagrama de flujo sencillo con los conteos reales. Si se usa PRISMA, solo con números que salgan de archivos.

Si el equipo prefiere mantener el nombre "revisión sistemática", cada número del PRISMA debe salir de un CSV generado por pasos reales y la extracción debe hacerse a mano sobre textos completos.

### 5.2 Fases

| Fase | Qué se hace | Salida |
| --- | --- | --- |
| 0 | Acordar el tipo de revisión y la extensión | Decisión anotada en este archivo |
| 1 | Correr las 6 investigaciones de Gemini Deep Research (`01_prompts_gemini_deep_research.md`) | Informes exportados a `deep_research/` como .md o .pdf |
| 2 | Consolidar todos los estudios citados en `02_matriz_corpus.csv`, deduplicar por DOI | Corpus candidato |
| 3 | Verificar cada DOI contra Crossref con un script que guarde la respuesta JSON; descartar lo que no resuelva o no coincida en título y autores | `verificacion_doi.csv` |
| 4 | Búsqueda reproducible complementaria (OpenAlex, PubMed, Europe PMC) con las ecuaciones de `PROTOCOLO.md` para no depender solo de Gemini | Registros con fecha |
| 5 | Cribado humano por título y resumen (dos revisores en una muestra, κ real) | `cribado.csv` |
| 6 | Conseguir los textos completos (acceso abierto o SINAB) y extraer con el formulario reducido; cada cifra con su ubicación | `extraccion_verificada.csv` |
| 7 | Redactar la versión larga, organizada por las tres fuentes de variación | `estado_del_arte_v2.md` |
| 8 | Condensar a la sección 3 del artículo y armar las referencias en estilo DYNA | Texto para la plantilla Word |
| 9 | Revisión cruzada por alguien del equipo con la lista de verificación de la Guía de Mauricio | Lista marcada |

### 5.3 Esquema propuesto (versión larga)

- **3.1 Método de la revisión.** Tipo de revisión, bases, ecuaciones, fecha, criterios, cribado, κ real, uso de IA.
- **3.2 Del rendimiento interno a la validación externa.** Transferencia de aprendizaje, benchmarks y desafíos (REFUGE, AIROGS), y la diferencia entre partición aleatoria, entre bases, LODO y validación externa prospectiva.
- **3.3 Variación de la imagen.** Cámara, FOV, encuadre, color, calidad y qué estudios aíslan cada factor con comparaciones controladas.
- **3.4 Variación de la etiqueta.** Definiciones de caso por dataset (sospechoso, referible, confirmado), estándar de referencia y estadificación ordinal.
- **3.5 Punto operativo y calibración.** Umbral fijado antes frente a reajustado en destino, sensibilidad a especificidad fija y cambio de prevalencia.
- **3.6 Estrategias de mitigación y su evidencia.** Multifuente, aumento de datos, armonización y ROI, normalización de color, adaptación y generalización de dominio, modelos fundacionales. Para cada una, qué variación ataca y con qué prueba se evaluó.
- **3.7 Vacío y posición del estudio.** Una oración de vacío que Fabián retoma en 4.6.

Tabla de estudios clave: solo con cifras extraídas por el equipo del texto completo, con autor, año, datasets, protocolo, AUC interna y externa y cómo se fijó el umbral. Figuras: como mucho una, y solo si sale de datos verificados.

### 5.4 Reglas para el agente que continúe

- No inventar cifras, DOI, autores ni páginas. Si algo no se puede leer en el texto completo, escribir `NO VERIFICADO` en lugar de estimar.
- No reutilizar `calidad.csv`, `extraccion.csv`, las figuras de `figuras/` ni los conteos del PRISMA anterior.
- Toda cita a un artículo de Gemini Deep Research se trata como pista hasta comprobarla en el artículo original. Gemini puede atribuir mal cifras entre estudios.
- No citar informes del proyecto como referencias; sus cifras se presentan como resultados propios.
- No hacer commit ni push sin que Mauricio lo pida. El repo es compartido con el equipo.
- Escribir en español con términos técnicos estándar. Evitar la jerga interna del proyecto: "data augmentation maligno" debe explicarse como aumento de escala y fotometría extremo, "megacohorte federada" no es federada, y "grokking" o "double descent" no se demostraron.
- Mantener la cautela del texto anterior, pero afirmar con firmeza donde la evidencia lo permite. Evitar repetir "esto muestra X, pero no demuestra Y" en cada párrafo.

---

## 6. Archivos de esta carpeta

| Archivo | Para qué |
| --- | --- |
| `00_CONTEXTO_PARA_ANTIGRAVITY.md` | Este documento |
| `01_prompts_gemini_deep_research.md` | Seis prompts listos para Gemini Deep Research, más instrucciones de uso |
| `02_matriz_corpus.csv` | Plantilla vacía para consolidar los estudios que salgan de Deep Research y de la búsqueda |
| `deep_research/` | Aquí se guardan los informes exportados de Gemini (crear al usarla) |

Rutas clave del repo:
- Texto anterior: `01_estado_del_arte_y_marco_teorico/manuscrito/estado_del_arte_reescrito.md`
- Protocolo (diseño reutilizable): `01_estado_del_arte_y_marco_teorico/manuscrito/PROTOCOLO.md`
- Marco teórico verificado: `01_estado_del_arte_y_marco_teorico/marco_teorico/salidas/`
- Resultados: `04_experimento_trinario_vs_binario/resultados/`, `05_experimento_data_augmentation/resultados/`, `03_experimento_caracteristicas_retinografias/datos/`
- Guías: `Entrega Final/` y `01_estado_del_arte_y_marco_teorico/Guía de Mauricio ...docx`
