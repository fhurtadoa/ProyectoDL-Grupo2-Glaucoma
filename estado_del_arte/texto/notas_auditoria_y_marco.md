# Notas de auditoría bibliográfica

**Alcance.** Esta es una revisión crítica focalizada de fuentes primarias comprobadas en las páginas de las revistas, editoriales o repositorios de autores. No se adopta la declaración de revisión sistemática del borrador. La versión de trabajo actual es [estado_del_arte_reescrito.md](estado_del_arte_reescrito.md); [estado_del_arte_revisado.md](estado_del_arte_revisado.md) conserva la primera propuesta para comparar los cambios. El archivo base permanece intacto.

## Referencias corregidas o retiradas

| Entrada del borrador | Resolución y motivo |
|---|---|
| [15] AIROGS, *Medical Image Analysis* 89, 102914 (2023), DOI 10.1016/j.media.2023.102914 | Corregida. La publicación verificada es de Vente et al., *IEEE Transactions on Medical Imaging* 43(1), 542–557 (2024), DOI [10.1109/TMI.2023.3313786](https://doi.org/10.1109/TMI.2023.3313786). |
| [14] REFUGE2, *Medical Image Analysis* 82, 102574 (2022), DOI 10.1016/j.media.2022.102574 | Retirada. Ese DOI identifica «KCB-Net: A 3D knee cartilage and bone segmentation network via sparse annotation» ([registro editorial/indexado](https://pubmed.ncbi.nlm.nih.gov/36126403/)). Hay un [manuscrito de REFUGE2 de los autores](https://arxiv.org/abs/2202.08994), pero no se le atribuye la ficha de revista errónea. |
| [16] Zhou et al., «A foundation model for generalist ophthalmic artificial intelligence» | Título corregido a «A foundation model for generalizable disease detection from retinal images», *Nature* 622, 156–163 (2023), DOI [10.1038/s41586-023-06555-x](https://doi.org/10.1038/s41586-023-06555-x). |
| [21] Li et al., autor principal «Zhen Li» | Corregido a **Zhixi Li** según el [registro de la publicación](https://pubmed.ncbi.nlm.nih.gov/29506863/). |
| [26] PAPILA, título terminado en «glaucoma evaluation» y autores incorrectos | Corregidos título y autoría: **Oleksandr Kovalyk, Juan Morales-Sánchez, Rafael Verdú-Monedero, Inmaculada Sellés-Navarro, Ana Palazón-Cabanes y José-Luis Sancho-Gómez**, según la [página de la revista](https://www.nature.com/articles/s41597-022-01388-1). |
| [17–18] DoFE y CFEA | Excluidas del texto principal porque estudian **segmentación** del disco y la copa, no validez externa de un clasificador de glaucoma. Podrían citarse solo en un marco metodológico de segmentación expresamente delimitado. |
| [32] «Moondream Staged» | Excluida como evidencia científica de estadificación: la ficha de un conjunto alojado en una plataforma no valida clínicamente la procedencia ni la graduación de sus etiquetas. |

## Afirmaciones reformuladas o retiradas

- Se retiraron los conteos de búsquedas, cribado, duplicados, estudios incluidos, concordancia kappa, puntuaciones de calidad, diagrama PRISMA y porcentajes bibliométricos. La existencia de archivos de apoyo no acredita por sí sola un proceso reproducible y una evaluación independiente. Si el equipo desea reivindicar una revisión sistemática, necesitará conservar registros íntegros de consultas, fechas, deduplicación, decisiones de cribado, extracción y revisión humana.
- Se retiraron las cifras agregadas de caída de AUC, atribuciones de causalidad al campo visual o a fabricantes, y proporciones de enfermedad temprana o avanzada entre bases. Una comparación observacional entre bases altera simultáneamente adquisición, población y etiquetas. Las cifras que permanezcan en futuras versiones deben indicar fuente, muestra, partición y umbral.
- Se sustituyó «colapso» o «superioridad» general de arquitecturas y estrategias por descripciones acotadas al protocolo de cada estudio. Se retiraron las afirmaciones sobre entrenamiento prolongado, aumento de escala específico y resultados del grupo: no son antecedentes externos comprobados.
- Se distinguieron clasificación, segmentación y regresión de riesgo. En particular, el estudio de Hemelings et al. evalúa una salida continua de riesgo frente a etiquetas de varias fuentes; no constituye una prueba aislada de que la armonización cause todo el desempeño observado.
- Se precisó que la etiqueta de neuropatía glaucomatosa remitible de Li et al. no equivale a glaucoma clínicamente confirmado, y que la validez de un umbral exige una prueba externa sin ajuste con sus etiquetas.

## Pendientes de verificación antes de enviar el manuscrito

1. Comprobar en el artículo final la norma bibliográfica exigida por la revista y completar autores cuando no admita «et al.».
2. Si se incorpora REFUGE2, localizar y verificar una versión formal publicada y su DOI; mientras tanto, citar únicamente el manuscrito de sus autores con su condición de prepublicación.
3. Si se desea una conclusión cuantitativa sobre entrenamiento multifuente, aumentos o normalización cromática, localizar ensayos de **clasificación** con comparación controlada y destino externo reservado. La evidencia citada aquí contextualiza esas estrategias, pero no demuestra una jerarquía universal.
4. Antes de usar grados de severidad de cualquier conjunto, verificar el procedimiento de adjudicación, las pruebas clínicas disponibles y la correspondencia entre grados de fuentes distintas.

# Esquema mínimo del marco teórico futuro

**Pregunta articuladora:** ¿cuándo una predicción obtenida de una retinografía conserva el mismo significado clínico al cambiar de fuente? El marco profundiza en los supuestos **oftalmológicos y de transporte** necesarios para responderla.

1. **Qué condición representa la etiqueta.** Relacionar el daño glaucomatoso con los signos visibles en la papila y distinguir sospecha, neuropatía remitible, glaucoma confirmado y severidad. Precisar cómo se adjudica el diagnóstico cuando también se dispone de exploración clínica, campo visual u otras pruebas. Esta distinción determina el significado de la clase positiva y la comparabilidad de los grados entre bases. *Sustento:* Weinreb et al. [M1], Li et al. [4], PAPILA [5] y Ahn et al. [9].
2. **Qué muestra y qué oculta la retinografía.** Delimitar la información papilar y peripapilar presente en una fotografía frente a la información clínica no observable en ella. Examinar cómo el centrado, la escala de la papila, el color, la calidad y las enfermedades coexistentes modifican la evidencia visual disponible. La comparación entre cámaras debe tratar estos factores como variables concurrentes, no como causas identificadas por una simple diferencia de rendimiento. *Sustento:* Li et al. [4], PAPILA [5], AIROGS [6] y Hemelings et al. [7].
3. **Qué significa transportar un clasificador.** Definir una fuente mediante población, adquisición y criterio de etiqueta; explicar por qué el cambio de cualquiera de ellos puede alterar el significado de la predicción. Concentrarse en validación entre bases o centros con destino no visto y en la diferencia entre conservar capacidad de discriminación y conservar el mismo punto de decisión. La literatura de cambio de distribución aporta el lenguaje conceptual; los estudios de glaucoma delimitan su aplicación. *Sustento:* Moreno-Torres et al. [M2], Diaz-Pinto et al. [2], AIROGS [6] y Hemelings et al. [7].
4. **Cómo interpretar la decisión de remisión.** Explicar qué representa una puntuación de riesgo para un paciente y por qué el umbral operativo depende del desenlace definido, la población y las consecuencias de falsos positivos y negativos. Tratar la calibración y la unidad de decisión —ojo o paciente— en cuanto afectan esa interpretación clínica. *Sustento:* Li et al. [4], PAPILA [5], Hemelings et al. [7] y Guo et al. [M3] para el concepto de calibración.

**Relación integradora:** condición clínica → signos disponibles y etiqueta adjudicada → predicción en una fuente → decisión de remisión en otra. El cambio de fuente puede alterar tanto la imagen como el significado de la etiqueta; por eso, el marco permite interpretar una evaluación externa sin atribuir de antemano el resultado a un único dispositivo o procedimiento.

**Fuentes conceptuales adicionales verificadas** (las citas [1]–[9] remiten a la bibliografía del estado del arte reescrito):

- [M1] Weinreb, R. N., Aung, T. y Medeiros, F. A. «The pathophysiology and treatment of glaucoma: a review». *JAMA* 311(18), 1901–1911 (2014). https://doi.org/10.1001/jama.2014.3192
- [M2] Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V. y Herrera, F. «A unifying view on dataset shift in classification». *Pattern Recognition* 45(1), 521–530 (2012). https://doi.org/10.1016/j.patcog.2011.06.019
- [M3] Guo, C., Pleiss, G., Sun, Y. y Weinberger, K. Q. «On Calibration of Modern Neural Networks». *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 1321–1330 (2017). https://proceedings.mlr.press/v70/guo17a.html
