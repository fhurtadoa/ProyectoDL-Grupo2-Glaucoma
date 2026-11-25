# Método de extracción y reproducción

Desde la raíz del repositorio, instala las versiones de la corrida y ejecuta el análisis o el notebook. Se usó Python 3.14.2.

```powershell
python -m pip install -r 03_experimento_caracteristicas_retinografias/scripts/requirements_retinografias.txt
python -X utf8 03_experimento_caracteristicas_retinografias/scripts/analizar_retinografias.py --piloto 12
python -X utf8 03_experimento_caracteristicas_retinografias/scripts/analizar_retinografias.py --trabajadores 6
```

`--salida` permite usar otra carpeta y `--config` admite un JSON alternativo basado en [fuentes_retinografias.json](../scripts/fuentes_retinografias.json). En el JSON, cada fuente requiere `id`, `ruta` y `tipo`. Las fuentes se abren en modo de lectura.

## Variables por archivo

| Familia | Definición |
| --- | --- |
| Geometría | Ancho y alto en píxeles, relación ancho/alto, formato decodificado y modo original. |
| Brillo, contraste y RGB | RGB de 8 bits y luma (`Y′`) de OpenCV, aproximadamente `(0,299 R + 0,587 G + 0,114 B)`. En la región visible se calculan media, mediana y desviación estándar poblacional de `Y′` en escala `0–255`. |
| Saturación | Media del canal `S` de HSV de OpenCV dentro de la región visible, escalada a `0–100 %`. |
| Nitidez | Varianza poblacional del Laplaciano discreto de `Y′` sobre la máscara erosionada 1 px, calculada tanto a resolución nativa como normalizando el lado largo a `512 px` para hacer comparables distintas resoluciones. |
| Fondo | Porcentaje de píxeles con máximo RGB `≤ 10` en el archivo y en la banda exterior del `5 %`. |
| Región y encuadre | Máscara estimada sobre el lado largo de `512 px`: umbral (`Y′ > 12` o `R > 24`), retiro de relleno uniforme conectado a esquinas, cierre elíptico de `5 px` y relleno del mayor contorno externo. |
| Tamaño y posición | Centroide normalizado (`0–1`), área ocupada (`%`), extensión horizontal/vertical y diámetro equivalente. |

## Conteo, depuración en dos pasos y trazabilidad

1. **Deduplicación exacta:** se calcula el `SHA-256` del archivo y de los píxeles decodificados (`dimensiones + modo + bytes`). `duplicados.csv` conserva todas las rutas con coincidencia exacta de píxeles.
2. **Agrupamiento de variantes geométricas:** sobre retinografías en grises reescaladas a lado largo `≤ 320 px`, se extraen hasta `150` descriptores **RootSIFT** (umbral de contraste `0,016`), se buscan `12` vecinos con **FLANN** (*ratio test* `< 0,76`) y se verifica transformación de semejanza mediante **RANSAC** (`≥ 12` puntos consistentes, soporte `≥ 75 %`, tolerancia `2,6 px`). `coincidencias_geometricas.csv` registra cada enlace.
3. **Selección de representantes:** se conserva un único representante por componente conectado de coincidencias exactas y geométricas (`2.149` representantes visuales sobre las `2.352` imágenes de las cinco colecciones RGB reales).
4. **Resúmenes:** percentiles `p05`, `p25`, mediana, `p75`, `p95` y rango intercuartílico (`RIC = p75 − p25`).
