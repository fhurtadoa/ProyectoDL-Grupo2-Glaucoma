# Características visuales de las retinografías locales

Se abrieron los **8.242 archivos de imagen** encontrados en ocho carpetas de `data/`. La comparación entre colecciones quedó en **2.149 representantes visuales**, con brillo y contraste evaluables en 2.147. ACRIMA y RIM-ONE DL contienen principalmente recortes que llenan el marco. Los dos lotes rotulados EyePACS conservan más fondo negro. Moondream presentó la mayor dispersión de brillo entre los recortes comparados.

## Inventario y conteo

| Colección local | Archivos legibles | Representantes en la comparación | Cobertura de región visible en la comparación |
| --- | ---: | ---: | ---: |
| `Database/Images` · ACRIMA | 705 | 703 | 703/703 |
| `RIM_ONE_DL` | 447 | 447 | 447/447 |
| `EYEPACS_REAL` | 200 | 200 | 199/200 |
| `EYEPACS_TRAIN` · serie BEH | 200 | 93 | 93/93 |
| `MOONDREAM_STAGED` | 800 | 706 | 705/706 |
| `HARMONIZED_MEGA_COHORT` | 2.937 | Comparación pareada | 2.935/2.937 archivos |
| `THIRD_DATASET` | 2.805 | Mapas derivados | — |
| `WEB_IN_THE_WILD` | 148 | Emulación y distractores | 80/80 retinografías emuladas |

- ACRIMA suma 705 JPEG, incluidos dos con extensión `.JPG`. RIM-ONE DL tiene 447 archivos y su `metadata.csv` documenta 162 etiquetas `glaucoma` y 285 `normal`. Las copias locales de ambas fuentes carecen de partición fija. `EYEPACS_REAL` tiene nombres RG/NRG y `EYEPACS_TRAIN` nombres BEH. El proyecto usa el primero como prueba externa y el segundo como entrenamiento. Falta un manifiesto que establezca la relación de procedencia entre los dos lotes. Moondream conserva sus carpetas `train`/`test` y etiquetas `normal`, `early` y `advanced`. Su procedencia y estadificación requieren verificación independiente. [Inventario](inventario.csv)
- Se detectaron 849 grupos de píxeles idénticos, con 875 copias por encima de un archivo por grupo. En la megacohorte, las 2.152 rutas canónicas del índice estaban presentes y 785 tenían una copia exacta bajo `glaucoma` pese a estar indexadas como `normal`. Se conservaron las dos rutas en la tabla y se usó la clasificación del índice. Se verificó la correspondencia de los 2.937 JPEG con sus archivos fuente reproduciendo el procesamiento. El análisis pareado conserva **2.056 pares** de megacohorte y **40 pares** web después de agrupar los archivos fuente parecidos. [Duplicados](duplicados.csv) · [Pares](pares.csv)
- La búsqueda geométrica encontró 270 enlaces entre archivos de píxeles distintos. En Moondream, 31 enlaces de ese tipo y 12 exactos cruzaron `train` y `test`. Se observan rotaciones, relleno y cambios de brillo entre variantes. En la serie BEH, el agrupamiento conservador redujo 197 imágenes de píxeles distintos a 93 representantes. Un grupo de cuatro archivos tenía etiquetas de carpeta discordantes y quedó fuera del desglose por clase. Estos grupos describen similitud visual entre archivos. La identidad del ojo y la del paciente no están documentadas. [Coincidencias](coincidencias_geometricas.csv) · [Mosaico de pares](mosaico_coincidencias.png)
- `THIRD_DATASET` contiene PNG RGBA de 369 × 369, con una mediana de 98,14 % de píxeles casi negros. Se midieron geometría, fondo y canales disponibles. En `WEB_IN_THE_WILD`, los 40 recortes de ACRIMA degradados se guardaron en `raw` y se recodificaron en `filtered`. Los 34 distractores dibujados se encuentran en `raw` y `rejected`. `raw` y sus salidas se contabilizan por separado. El inventario registra las fuentes mencionadas sin datos disponibles, entre ellas REFUGE, PAPILA, AIROGS completo, ODIR-5K RGB y una colección web independiente. [Inventario](inventario.csv) · [Método](METODO.md)

## Diferencias observadas

| Fuente | Ancho mediano (px) | Brillo mediano [p25–p75] | Saturación mediana (%) | Fondo casi negro (%) | Área visible (%) | Laplaciano, lado 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACRIMA | 424 | 110 [92–126] | 87,82 | 0,00 | 100,00 | 5,11 |
| RIM-ONE DL | 472 | 85 [63–101] | 77,13 | 0,00 | 100,00 | 8,05 |
| EyePACS REAL | 512 | 77 [51,5–107] | 61,57 | 21,76 | 78,18 | 59,97 |
| EyePACS TRAIN | 512 | 114 [97–123] | 85,74 | 27,12 | 72,83 | 32,01 |
| Moondream Staged | 416 | 121 [65–171] | 70,89 | 0,00 | 94,08 | 14,17 |

El fondo negro y el área visible distinguen los dos lotes EyePACS de los recortes de ACRIMA y RIM-ONE DL. El contorno circular fue evaluable en 171/200 imágenes de REAL y 46/93 de TRAIN. La caída mediana del centro hacia la periferia fue de 16 y 14 niveles de luma, respectivamente. Cinco representantes de TRAIN tienen RGB idéntico en al menos el 99,5 % del archivo. Al contar todos los archivos de píxeles distintos, la mediana de brillo de TRAIN pasa de 114 a 109 y la de Moondream de 121 a 122. [Encuadre](encuadre.png) · [Fotometría](fotometria.png) · [Sensibilidad al agrupamiento](resumen_solo_duplicados_exactos.csv)

En los 199 pares EyePACS REAL con región medible en ambas versiones, el recorte y ajuste de color de la megacohorte aumentaron la mediana de brillo en **25 niveles** (p25–p75 de −3 a +51), la saturación en **19,23 puntos porcentuales** y el área estimada en **21,05 puntos**. La varianza del Laplaciano a lado 512 bajó 47,53 unidades. En ACRIMA, RIM-ONE DL y Moondream la diferencia mediana de brillo pareada fue cero. En Moondream el Laplaciano mediano cambia de 26,27 en resolución almacenada a 14,17 con lado largo 512. La resolución, la interpolación y el contenido visible influyen en la lectura de esa métrica. [Cambios pareados](diferencias_pareadas.png) · [Resolución y nitidez](resolucion_nitidez.png) · [Resumen de pares](resumen_pares.csv)

La composición también puede influir. En ACRIMA, las medianas de brillo fueron 114,5 en `normal` y 104 en `glaucoma`. En RIM-ONE DL fueron 85 y 86. En Moondream fueron 126, 111 y 108 para `normal`, `early` y `advanced`, con sus etiquetas locales intactas. Se mantuvieron los esquemas separados por procedencia. Ningún archivo tenía fabricante o modelo de cámara en EXIF. El modelo de cámara citado a nivel de colección queda como metadato documental. Fuente, cámara y procesamiento no se pueden estimar por separado con estas copias. [Resumen por clase](resumen_por_clase.csv) · [Resumen por dataset](resumen_por_dataset.csv)

## Revisión visual y límites

El piloto abrió 113 archivos. La revisión permitió ajustar la detección de rellenos y conservar círculos que tocan ligeramente el marco. En la corrida completa, `EyePACS-TRAIN-RG-2911.png` y `moon_tr_norm_0232.jpg` ilustran cómo la oscuridad dentro de la retina puede fragmentar o achicar la máscara. Algunas versiones armonizadas de EyePACS todavía muestran fondo coloreado que el umbral puede incluir. El único candidato a texto entre las retinografías comparadas fue `EyePACS-TRAIN-RG-2925.png`. La revisión mostró reflejos periféricos sin texto legible. En controles sintéticos, el indicador devolvió candidatos en 10/10 imágenes de equipo y 12/12 de OCT simulado, y en 0/12 diagramas con letras oscuras. Estos ejemplos describen modos de fallo en una muestra dirigida.

La cobertura de área de disco óptico fue **0** en todas las colecciones. Las máscaras de `THIRD_DATASET` no tienen correspondencia verificable con una retinografía de la muestra, y el recorte de EyePACS del script 07 se centra en un máximo de brillo sin delimitar el disco. El tamaño aparente de la región visible se expresa en píxeles o como fracción del archivo. [Mosaico de revisión](mosaico_revision.png) · [Registro por imagen](por_imagen.csv)

## Hipótesis para la etapa del modelo

1. En los pares EyePACS REAL, comparar recorte y ajuste de color por separado para medir cuánto cambia cada distribución y, después, las predicciones del modelo final.
2. Medir si el fondo negro y la región visible se asocian con cambios en las predicciones al controlar la fuente, el esquema de etiquetas y la resolución de entrada.
3. Repetir el análisis de Moondream y BEH con identificadores de captura verificados y particiones por grupo para comprobar cuánto depende la distribución del brillo de las variantes repetidas.

Para reproducir desde la raíz del repositorio, consulta las dependencias, criterios y opciones en el [método](METODO.md).

```powershell
python -X utf8 Fabian/codigos_y_scripts/analizar_retinografias.py --trabajadores 6
```
