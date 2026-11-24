# Nota de alineación para Andrés (Secciones 4.1, 4.2 y 4.3)

No se modificó `Comprobacion_mejor_modelo.ipynb` ni `Guía de Andrés selección de datasets y de modelo.docx`.  
Al reordenar el repositorio por experimento y actualizar los notebooks de Fabián con `ConvNeXt-Tiny`, conviene tener presentes estos 4 puntos para el artículo DYNA:

1. **Prueba con 588 imágenes de Google (al final de `Comprobacion_mejor_modelo.ipynb`):**  
   Queda **descartada** del alcance del artículo formal (luego puedes quitar esas celdas del notebook si lo ves conveniente).
2. **Sección opcional 4.3 (*Estadificar un dataset binario*):**  
   Queda pendiente de lo que decidas/logres en tu parte para definir si entra como subsección propia (0,75 p.) o si se resume como limitación/trabajo futuro dentro de la sección 4.6.
3. **Actualización sobre el modelo usado por Fabián (`Guía de Andrés`, sec. 4.2):**  
   En tu guía decía que el estudio de 2 vs. 3 clases de Fabián usó `ResNet-18`. Eso correspondía al script antiguo (ahora archivado en `archivo_historico/`). Los notebooks vigentes (`04_experimento_trinario_vs_binario/` y `05_experimento_data_augmentation/`) ya evalúan **ConvNeXt-Tiny** con exactamente el mismo backbone y receta de referencia.
4. **Diferencia de conteo en `MOONDREAM_STAGED` (`762` vs. `706`) y separación de benchmarks en 4.2:**
   - En `Comprobacion_mejor_modelo.ipynb`, Moondream queda en **762** imágenes (`532/115/115`) tras deduplicar por MD5 y similitud coseno ResNet-18 $\ge 0.96$.
   - En los experimentos `04` y `05` de Fabián (`utilidades/dataset_auditado.csv`), se añade una segunda limpieza estricta (coincidencias geométricas SIFT/RANSAC y exclusión de cruces con `ACRIMA`, `RIM-ONE DL` y `EyePACS REAL`), quedando Moondream en **706** imágenes (`504/107/95`).
   - En **4.2**, conviene distinguir claramente entre el benchmark de **10 arquitecturas** sobre la megacohorte armonizada (`datos_benchmark/benchmark_10_backbones_metrics.json`, validación interna $N=431$) y la comprobación controlada de los top-3 sobre Moondream (`532/115/115`) de tu notebook.
