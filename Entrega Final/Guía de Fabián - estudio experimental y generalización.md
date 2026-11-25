# Guía de Fabián: análisis descriptivo, clasificación 2 vs. 3 clases, aumento de datos y conclusión sobre generalización

Actualizada con la estructura reorganizada del repositorio y los resultados verificados de los notebooks con **ConvNeXt-Tiny**.

---

## Tus secciones en el artículo (Formato DYNA)

| Sección | Presupuesto | Fuente principal en el repo |
| --- | --- | --- |
| Aporte a **4.1** (análisis descriptivo de retinografías) | Integrado en las 1,5 p. de Andrés | `03_experimento_caracteristicas_retinografias/` |
| **4.4** Análisis trinario frente a binario | 1,0 p. | `04_experimento_trinario_vs_binario/` |
| **4.5** Efecto del aumento de datos (`Base` vs. `Con aumento`) | 1,0 p. | `05_experimento_data_augmentation/` |
| **4.6** Conclusión sobre generalización de los modelos | $\le$ 1,0 p. | Cruce de `03`, `04` y `05` |

> **Regla DYNA:** todas las figuras para el artículo final deben exportarse en **escala de grises, ancho de una columna, 300 dpi**, con pie `"Fuente: Los autores."`. Las tablas llevan solo líneas horizontales (8 pt).

---

## Contexto y alineación con Andrés y Mauricio

- **Mauricio** cubre Encabezado/Abstract/Resumen (`0,75 p.`), Introducción (`0,25+ p.`), Marco teórico (`0,5 p.`), Método RSL y Estado del arte (`0,75 p.`) y Referencias (`0,5+ p.`). Su estado del arte cierra dejando el vacío que tú respondes en `4.4`–`4.6`: *separar la variación física de la imagen, el esquema de etiquetas y el traslado del punto operativo ($\tau$)*.
- **Andrés** cubre `4.1 Selección de datasets` (`1,5 p.`), `4.2 Selección de modelo` (`1,0 p.`) y queda pendiente la decisión sobre la opcional `4.3 Estadificar un dataset binario` (`0,75 p.`). La prueba de 588 imágenes de Google en su notebook queda **descartada** para el artículo.
- **Reconciliación de splits en Moondream:** Andrés parte de `800` $\rightarrow$ `762` (`532/115/115`) por MD5 + ResNet-18 ($\ge 0.96$). Tus experimentos en `04` y `05` (`utilidades/dataset_auditado.csv`) aplican además el grafo de coincidencias geométricas SIFT/RANSAC y exclusión de cruces con fuentes externas, dejando **2.056 imágenes limpias**:
  - `Moondream Staged`: **706** (`504` train / `107` val / `95` test)
  - `ACRIMA`: **703** (`308` normal / `395` glaucoma)
  - `RIM-ONE DL`: **447** (`285` normal / `162` glaucoma)
  - `EyePACS REAL`: **200** (`100` normal / `100` glaucoma)

---

## Contenido y cifras verificadas por sección

### Aporte a 4.1 · Análisis descriptivo de características de retinografías
Fuentes: `03_experimento_caracteristicas_retinografias/INFORME_CARACTERISTICAS_RETINOGRAFIAS.pdf` y `datos/03_*.csv`.
- **Encuadre:** `ACRIMA` y `RIM-ONE DL` son recortes que llenan el marco (`100 %` área visible mediana, `0 %` fondo negro). En contraste, `EyePACS REAL` (`78,18 %` área visible, `21,76 %` fondo casi negro) y `EyePACS TRAIN` (`72,83 %`, `27,12 %` fondo negro) conservan campo amplio con borde circular.
- **Fotometría y nitidez (medianas):** brillo desde `77` en `EyePACS REAL` hasta `121 [65–171]` en `Moondream`; saturación de `87,82 %` en `ACRIMA` frente a `61,57 %` en `EyePACS REAL`; Laplaciano (lado 512 px) de `59,97` en `EyePACS REAL` frente a `5,11` en `ACRIMA`.
- **Efecto medido del preprocesamiento (`03_cambios_pareados.csv`):** en 199 pares verificados de `EyePACS REAL`, el recorte y ajuste de color sube la mediana de brillo en `+25` niveles, la saturación en `+19,23 pp` y el área visible en `+21,05 pp`, reduciendo la varianza del Laplaciano en `47,53` unidades.
- **Límite clave:** no hay máscaras verificadas de disco óptico ni metadatos EXIF de cámara por imagen.

### 4.4 · Análisis trinario frente a binario (1,0 p.)
Fuente: `04_experimento_trinario_vs_binario/01_convnext_trinario_vs_binario.ipynb`, `resultados/01_metricas_binarias.csv` y `resultados/01_sensibilidad_umbral.csv`.
- **Delimitación metodológica:** `Moondream Staged` es la **única fuente local con etiquetas de estadio** (`Sano`, `Etapa inicial`, `Avanzado`). Por eso ambos modelos entrenan y calibran en `Moondream` (`504 train: 264/115/125`; `107 val`), la evaluación estricta de estadificación (`3→3`) se presenta de forma concisa en `Moondream test` (`n=95`, `QWK=0.801`, `recall inicial=0.476`, `recall avanzado=0.840`), y el contraste externo se realiza mediante proyección binaria (`3→2` vs. `2→2`).
- **Protocolo común y reglas de umbral ($\tau$):** `ConvNeXt-Tiny`, semilla `42`, aumento estándar de `Comprobacion_mejor_modelo.ipynb`, 1 época warmup + 20 épocas (`AdamW`, `CosineAnnealingLR`). Umbral principal calibrado **solo en validación** buscando especificidad $\ge 0{,}95$ ($\tau=0{,}81$ para `3→2`; $\tau=0{,}90$ para `2→2`), contrastado frente a `Youden (val)` ($\tau=0{,}81$ y `0,86`) y `Estándar` ($\tau=0{,}50$).
- **Resultados (Proyección trinario vs. Entrenado binario · regla principal + IC 95% bootstrap):**
  - `Moondream · test` ($n=95$): Proyección `F1=0.916`, `Sens=0.935`, `Espec=0.898`, `AUROC=0.980` vs. Binario `F1=0.852`, `Sens=0.804`, `Espec=0.898`, `AUROC=0.954` ($\Delta\text{F1}=+6.4\text{ pp}$, $\Delta\text{Sens}=+13.0\text{ pp}$).
  - `ACRIMA` ($n=703$): Proyección `F1=0.671` (`AUROC=0.740`) vs. Binario `F1=0.598` (`AUROC=0.626`) ($\Delta\text{F1}=+7.3\text{ pp}$, IC 95% `[+4.3, +10.2]`; $\Delta\text{AUROC}=+11.4\text{ pp}$, IC `[+9.0, +14.1]`).
  - `RIM-ONE DL` ($n=447$): Proyección `F1=0.735` (`AUROC=0.815`) vs. Binario `F1=0.682` (`AUROC=0.780`) ($\Delta\text{F1}=+5.3\text{ pp}$, IC `[+1.6, +9.3]`; $\Delta\text{AUROC}=+3.4\text{ pp}$, IC `[+0.7, +6.4]`).
  - `EyePACS REAL` ($n=200$): Ambos colapsan en sensibilidad con $\tau$ conservador o Youden (`Sens=0.020` en proyección; `0.000` en binario), y con $\tau=0.50$ apenas llegan a `Sens=0.090`–`0.150` (`AUROC=0.552` y `0.628`).
- **Qué mostrar:** tabla comparativa por cohorte (con $\tau$ principal y sensibilidad a $\tau$) + resumen breve `3→3` en `Moondream test` + 1 figura compuesta en escala de grises + discusión del desbalance `2:1:1` y etiquetas binarias externas.

### 4.5 · Efecto del aumento de datos en LODO binario multi-semilla: `Base` vs. `Con aumento` (1,0 p.)
Fuente: `05_experimento_data_augmentation/02_convnext_data_augmentation.ipynb`, `resultados/02_metricas_lodo.csv`, `02_metricas_lodo_por_semilla.csv` y `02_sensibilidad_umbral_lodo.csv`.
- **Diseño LODO multi-semilla ($4\text{ folds} \times 2\text{ brazos} \times 3\text{ semillas} = 24\text{ modelos}$):** esquema puramente binario (`2.056` imágenes auditadas). En cada *fold*, una cohorte completa queda 100 % ciega mientras las 3 restantes se dividen en `85 % train / 15 % val` estratificado. Se repite con semillas `42`, `123` y `777` para separar el efecto real del ruido estocástico.
- **Regla de calibración del umbral ($\tau$) — incluir explícitamente en el informe:** tanto en `4.4` como en `4.5`, el corte principal $\tau \in [0{,}20, 0{,}99]$ se calibra **únicamente en validación** maximizando $0{,}5\cdot\text{F1 macro} + 0{,}5\cdot\text{Recall}$ sujeto a **especificidad $\ge 0{,}95$**, y luego se traslada congelado al dominio externo. Se contrasta además contra **Youden (`val`)** y **$\tau = 0{,}50$** para aislar discriminación (`AUROC`) de punto operativo.
- **Hallazgo principal LODO (`media ± DE` en 3 semillas; regla principal + sensibilidad a $\tau$):**
  - `EyePACS REAL` ($n=200$, campo amplio `21.8 %` borde negro): **`Con aumento` rescata fuertemente al modelo**: `F1` pasa de `0.343 ± 0.001` a **`0.618 ± 0.039`** ($\Delta=+27.5\pm 4.0\text{ pp}$), `Sens` de `0.010` a `0.353 ± 0.060` y **`AUROC` de `0.655 ± 0.019` a `0.809 ± 0.014`** (`+15.4 ± 2.1 pp`; ensamble IC 95% `[+10.9, +20.3 pp]`). Con umbral `Youden (val)` o `0.50`, `Con aumento` alcanza **`F1=0.715–0.721`** (`Sens=64.0–70.7 %`, `TFN=29.3–36.0 %`).
  - `Moondream Staged` ($n=706$): `F1` sube de `0.519 ± 0.031` a **`0.580 ± 0.011`** ($\Delta=+6.1\pm 3.2\text{ pp}$), `AUROC` de `0.556 ± 0.054` a **`0.605 ± 0.016`** (ensamble IC 95% `AUROC` `[+3.5, +7.1 pp]`).
  - `ACRIMA` ($n=703$): empate en `F1` (`0.625 ± 0.024` vs. `0.628 ± 0.084`, $\Delta=+0.3\pm 7.8\text{ pp}$), pero mejora significativa en ordenamiento (`AUROC` `0.686` $\rightarrow$ **`0.730`**, ensamble IC 95% `[+3.0, +6.6 pp]`; con Youden `F1` sube a `0.664` vs. `0.606`).
  - `RIM-ONE DL` ($n=447$): efecto prácticamente neutro (`F1` `0.484 ± 0.054` vs. `0.485 ± 0.044`; `AUROC` `0.671` vs. `0.677`, IC 95% `[-1.5, +5.4 pp]`).
- **Qué mostrar:** gráfica de barras horizontales con el $\Delta$ LODO (`media ± DE` en 3 semillas) + tabla LODO (`media ± DE` + contraste de reglas de $\tau$ en `EyePACS` y `ACRIMA` + IC 95% bootstrap incluyendo `AUROC`).

### 4.6 · Conclusión sobre generalización ($\le$ 1,0 p.)
Integra lo hallado en `03`, `04` y `05`:
1. **El encuadre manda en la transferencia entre sensores:** sin aumento de escala (`Base`), `ConvNeXt-Tiny` falla casi por completo al saltar hacia campo amplio (`EyePACS REAL`: `Sens 0.0–1.0 %`, `AUROC 0.655`). Introducir variaciones de escala y color (`AumentoEscalaFotometrico`) eleva el `AUROC` en `EyePACS REAL` a `0.809 ± 0.014` (`+15.4 pp`) y en `Moondream` a `0.605`, demostrando un rescate real cuando el dominio ciego difiere en geometría o fotometría.
2. **Separar discriminación (`AUROC`) de punto operativo ($\tau$):** el estudio de sensibilidad a $\tau$ prueba que exigir `especificidad >= 0.95` en validación empuja $\tau$ a `0.94–0.99`, ocultando parte de la mejora real del modelo (`TFN` pasa en `EyePACS + aumento` de `64.7 %` con $\tau$ conservador a `29.3–36.0 %` con `τ = 0.50` o `Youden`, manteniendo especificidad `73.7–79.7 %`).
3. **Complementariedad entre supervisión clínica y aumento de escala:** entrenar con estadios (`04`) mejora la separación en recortes papilares desde un origen estadificado (`ACRIMA`, `RIM-ONE DL`), mientras que el aumento de escala (`05`) es decisivo ante cambios de campo visual (`EyePACS REAL`) y aporta robustez promedio sin dañar `ACRIMA`/`RIM-ONE DL` al promediar 3 semillas.
4. **Trabajo futuro / puente con 4.3:** sin etiquetas reales de estadio en cohortes externas ni máscaras de disco/copa o mapas de saliencia (`Grad-CAM`), las salidas trinarias fuera de Moondream deben tratarse como exploratorias; la armonización geométrica del encuadre + recalibración de umbral en el sitio clínico destino son pasos necesarios antes del despliegue.

---

## Lista de verificación antes de entregar a Word
- [ ] Todas las cifras salen directamente de los `.csv` en `03_.../datos/`, `04_.../resultados/` y `05_.../resultados/`.
- [ ] Las figuras incluidas en Word están en escala de grises, 1 columna, 300 dpi y con fuente `"Los autores"`.
- [ ] Las ecuaciones (entropía cruzada, softmax, QWK, proyección binaria) se citan desde el Marco teórico de Mauricio (`eq. (n)`).
- [ ] Se envían a Mauricio las referencias nuevas citadas en `4.4`–`4.6` (p.ej. `ConvNeXt` / `QWK` si aplica) con su DOI verificado.
- [ ] Se verifica el conteo real de páginas dentro de `Entrega Final/Plantilla_base_seccion_DYNA.docx`.
