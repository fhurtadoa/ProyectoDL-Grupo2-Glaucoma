# Características visuales de las retinografías locales

Se evaluaron las **cinco colecciones originales de retinografías RGB** del estudio (**2.352** fotografías), tras excluir `THIRD_DATASET` (`2.805` PNG RGBA que corresponden a mapas binarios de bordes vasculares con `98,14 %` de píxeles casi negros y no a retinografías de fondo de ojo). Tras depurar copias exactas (`SHA-256`) y variantes geométricas (`RootSIFT`/`FLANN`/`RANSAC`), la comparación quedó conformada por **2.149 representantes visuales** (con brillo y contraste evaluables en `2.147`).

## Inventario y depuración por fuente

| Colección local | Archivos en disco | Representantes comparados | Elegibles en Exp. `04`/`05` |
| --- | ---: | ---: | ---: |
| `Database/Images` · ACRIMA | 705 | 703 | 703 |
| `RIM_ONE_DL` | 447 | 447 | 447 |
| `EYEPACS_REAL` | 200 | 200 | 200 |
| `EYEPACS_TRAIN` · serie BEH | 200 | 93 | — (excluida por duplicados e inconsistencias internas) |
| `MOONDREAM_STAGED` | 800 | 706 | 706 (`504 / 107 / 95`) |
| `THIRD_DATASET` | 2.805 | Excluido | — (mapas binarios RGBA, no retinografías) |

- En `EyePACS TRAIN` (`BEH`), el agrupamiento redujo `200` archivos a `93` representantes únicos y mostró un grupo de archivos con etiquetas discordantes, por lo que se excluyó de entrenamiento/evaluación en los experimentos `04` y `05`, consolidando **`2.056`** imágenes auditadas en [dataset_auditado.csv](../../utilidades/dataset_auditado.csv).
- En `Moondream Staged`, la auditoría detectó `12` enlaces exactos y `31` enlaces geométricos que cruzaban `train` y `test` o repetían capturas con rotaciones y rellenos, consolidando `706` imágenes limpias.

## Diferencias observadas

| Fuente (`n`) | Ancho mediano (px) | Brillo mediano `[p25–p75]` | Saturación (%) | Fondo casi negro (%) | Área visible (%) | Nitidez (512 px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACRIMA (`703`) | 424 | 110 `[92–126]` | 87,8 | 0,0 | 100,0 | 5,1 |
| RIM-ONE DL (`447`) | 472 | 85 `[63–101]` | 77,1 | 0,0 | 100,0 | 8,1 |
| EyePACS REAL (`200`) | 512 | 77 `[52–107]` | 61,6 | 21,8 | 78,2 | 60,0 |
| EyePACS TRAIN (`93`) | 512 | 114 `[97–123]` | 85,7 | 27,1 | 72,8 | 32,0 |
| Moondream Staged (`706`) | 416 | 121 `[65–171]` | 70,9 | 0,0 | 94,1 | 14,2 |

- **Encuadre entre fuentes:** `ACRIMA` y `RIM-ONE DL` son recortes papilares cerrados (`100 %` de área útil, `RIC = 0,0`), mientras `EyePACS REAL` y `EyePACS TRAIN` conservan campo amplio con borde circular oscuro (`21,8 %` y `27,1 %`).
- **Dispersión interna y señales espurias:** `Moondream Staged` exhibe la mayor dispersión de brillo (`RIC = 106,0`) y saturación (`RIC = 41,5 pp`), además de caída de brillo por estadio (`126 → 111 → 108` entre `normal`, `early` y `advanced`) y mayor relleno negro artificial en esquinas en sanos (`53 %`) frente a avanzados (`45 %`). En `ACRIMA`, el ancho nativo mediano pasa de `346 px` en sanos a `673 px` en glaucoma.
- **Límite documental:** ninguna colección posee metadatos EXIF de cámara por archivo ni máscaras verificadas de disco óptico.

Para detalles metodológicos, consulta [METODO.md](METODO.md).
