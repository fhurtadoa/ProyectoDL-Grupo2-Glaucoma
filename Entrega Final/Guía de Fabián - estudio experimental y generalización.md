# Guía de Fabián: análisis descriptivo, clasificación 2 vs. 3 clases, aumento de datos y conclusión sobre generalización

Actualizada con la estructura reorganizada del repositorio y los resultados verificados de los notebooks con **ConvNeXt-Tiny**.

---

## Tus secciones en el artículo (Formato DYNA)

| Sección | Presupuesto | Fuente principal en el repo |
| --- | --- | --- |
| Aporte a **4.1** (análisis descriptivo de retinografías) | Integrado en las 1,5 p. de Andrés | `03_experimento_caracteristicas_retinografias/` |
| **4.4** Análisis trinario frente a binario | 1,0 p. | `04_experimento_trinario_vs_binario/` |
| **4.5** Efecto del aumento de datos (`Base` vs. `Malignant`) | 1,0 p. | `05_experimento_data_augmentation/` |
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
Fuente: `04_experimento_trinario_vs_binario/01_convnext_trinario_vs_binario.ipynb` y `resultados/01_metricas_binarias.csv`.
- **Protocolo común:** `ConvNeXt-Tiny`, misma semilla (`42`), receta de Andrés (`RandomResizedCrop`, volteo horizontal, rotación $\pm 15^\circ$, `ColorJitter`), 1 época warmup + 20 épocas fine-tuning (`AdamW`, `CosineAnnealingLR`). Mejor época y umbral $\tau$ elegidos solo en validación buscando especificidad $\ge 0.95$ ($\tau = 0.81$ para proyección trinaria; $\tau = 0.90$ para entrenado binario).
- **Control de atajos:** una regresión logística solo con `(R, G, B)` medio queda cerca del azar mayoritario en `Moondream test`, mostrando que el color global por sí solo no resuelve la clasificación.
- **Resultados (Proyección trinario vs. Entrenado binario):**
  - `Moondream · test` ($n=95$): Proyección alcanza `F1=0.916`, `Sens=0.935`, `Espec=0.898`, `AUROC=0.980` vs. Binario `F1=0.852`, `Sens=0.804`, `Espec=0.898`, `AUROC=0.954` ($\Delta\text{F1}=+6.4\text{ pp}$, $\Delta\text{Sens}=+13.0\text{ pp}$).
  - `ACRIMA` ($n=703$): Proyección `F1=0.673` (`AUROC=0.744`) vs. Binario `F1=0.605` (`AUROC=0.635`) ($\Delta\text{F1}=+6.8\text{ pp}$).
  - `RIM-ONE DL` ($n=447$): Proyección `F1=0.738` (`AUROC=0.816`) vs. Binario `F1=0.684` (`AUROC=0.783`) ($\Delta\text{F1}=+5.5\text{ pp}$).
  - `EyePACS REAL` ($n=200$): Ambos colapsan en sensibilidad al trasladar el umbral rígido (`Sens=0.020` / `TFN=0.980` en proyección; `Sens=0.000` / `TFN=1.000` en binario), aunque conservan algo de separabilidad de ranking (`AUROC=0.553` y `0.632`).
- **Qué mostrar:** tabla comparativa por cohorte + 1 figura compuesta en escala de grises (matrices de confusión o curvas ROC/PR en Moondream vs. cohortes externas) + discusión de errores y limitación de etiquetas binarias en fuentes externas.

### 4.5 · Efecto del aumento de datos: `Base` vs. `Malignant` (1,0 p.)
Fuente: `05_experimento_data_augmentation/02_convnext_data_augmentation.ipynb` y `resultados/02_metricas_factorial.csv`.
- **Diseño factorial $2 \times 2$:** cabezas de `2` y `3` clases $\times$ `Base` (solo `Resize(224,224)`) vs. `Malignant` (`MalignantScaleAugmentation` + volteos solo en entrenamiento).
- **Comprobación visual/fotométrica:** `Malignant` introduce variación controlada de escala (zoom-out con fondo sintético circular y zoom-in papilar) y ganancia de gamma/verde, evitando que la red dependa de un único encuadre.
- **Hallazgo principal (efecto diferenciado por esquema):**
  - En el modelo **binario (`2→2`)**, `Malignant` mejora consistentemente en las 4 cohortes:
    - `Moondream · test`: $\Delta\text{F1}=+11.7\text{ pp}$ (`0.734` $\rightarrow$ `0.852`), $\Delta\text{TFN}=-26.1\text{ pp}$.
    - `ACRIMA`: $\Delta\text{F1}=+6.6\text{ pp}$ (`0.545` $\rightarrow$ `0.611`), $\Delta\text{TFN}=-23.8\text{ pp}$.
    - `RIM-ONE DL`: $\Delta\text{F1}=+11.3\text{ pp}$ (`0.612` $\rightarrow$ `0.725`), $\Delta\text{TFN}=-25.9\text{ pp}$.
    - `EyePACS REAL`: $\Delta\text{F1}=+10.9\text{ pp}$ (`0.333` $\rightarrow$ `0.442`), $\Delta\text{TFN}=-11.0\text{ pp}$ (`AUROC` sube de `0.546` a `0.686`).
  - En el modelo **trinario proyectado (`3→2`)**, el efecto es mixto: mejora fuerte en `EyePACS REAL` ($\Delta\text{F1}=+16.5\text{ pp}$, $\Delta\text{TFN}=-18.0\text{ pp}$, `AUROC` `0.577` $\rightarrow$ `0.697`), pero empeora en `RIM-ONE DL` ($\Delta\text{F1}=-10.1\text{ pp}$, $\Delta\text{TFN}=+38.3\text{ pp}$) y ligeramente en `ACRIMA` ($\Delta\text{F1}=-1.7\text{ pp}$) porque la calibración en validación fija una especificidad muy alta (`1.000` en Moondream test con `Malignant`) que vuelve el umbral demasiado conservador fuera de dominio.
  - En estadio **`3→3`** (Moondream test): `QWK` pasa de `0.726` (`Base`) a `0.733` (`Malignant`), manteniendo `5` errores extremos sano $\leftrightarrow$ avanzado.
- **Qué mostrar:** gráfica de barras horizontales con el $\Delta$ (`Malignant − Base` en pp) + tabla resumida con IC 95% bootstrap.

### 4.6 · Conclusión sobre generalización ($\le$ 1,0 p.)
Integra lo hallado en `03`, `04` y `05`:
1. **No hay generalización gratuita fuera de dominio:** aunque en Moondream `ConvNeXt-Tiny` supera `AUROC 0.95` (`0.980` en proyección trinaria), en `EyePACS REAL` la sensibilidad se desploma (`0 %–18 %`) por la diferencia de encuadre ($21.8\%$ de fondo negro frente a recortes papilares al $100\%$).
2. **Separar discriminación (`AUROC`) de punto operativo ($\tau$):** parte de la caída en `ACRIMA` y `RIM-ONE DL` (especialmente en `3→2` con `Malignant`) no es pérdida de capacidad de ordenamiento (`AUROC 0.818` en RIM-ONE DL), sino un umbral $\tau$ calibrado exclusivamente en Moondream que no se reajusta al dominio destino.
3. **Valor del diseño factorial y del aumento de escala:** `MalignantScaleAugmentation` rescata claramente al clasificador binario (`+6.6` a `+11.7 pp` en F1), mientras que entrenar con estadios (`3→2`) aporta mejor separación base sin aumento agresivo, pero requiere calibración de umbral por dominio.
4. **Trabajo futuro / puente con 4.3:** sin etiquetas reales de estadio en cohortes externas ni máscaras verificadas de disco/copa, las salidas trinarias fuera de Moondream deben tratarse como exploratorias; la armonización geométrica del encuadre + recalibración de umbral en el sitio clínico destino son pasos necesarios antes del despliegue.

---

## Lista de verificación antes de entregar a Word
- [ ] Todas las cifras salen directamente de los `.csv` en `03_.../datos/`, `04_.../resultados/` y `05_.../resultados/`.
- [ ] Las figuras incluidas en Word están en escala de grises, 1 columna, 300 dpi y con fuente `"Los autores"`.
- [ ] Las ecuaciones (entropía cruzada, softmax, QWK, proyección binaria) se citan desde el Marco teórico de Mauricio (`eq. (n)`).
- [ ] Se envían a Mauricio las referencias nuevas citadas en `4.4`–`4.6` (p.ej. `ConvNeXt` / `QWK` si aplica) con su DOI verificado.
- [ ] Se verifica el conteo real de páginas dentro de `Entrega Final/Plantilla_base_seccion_DYNA.docx`.
