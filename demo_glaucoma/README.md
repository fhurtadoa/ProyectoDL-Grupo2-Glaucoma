# Demo de clasificación de glaucoma

Aplicación académica del Grupo 2 para la sustentación. Ejecuta inferencia con los modelos **ConvNeXt-Tiny** entrenados en Moondream sobre retinografías locales y compara cada predicción contra su referencia disponible. El resultado tiene alcance académico.

## Instalación y arranque

Desde la raíz del repositorio, con Python y una combinación compatible de PyTorch y TorchVision:

```powershell
python -m pip install -r demo_glaucoma/requirements.txt
python -m demo_glaucoma
```

Abre [la demo local](http://127.0.0.1:8765/). Con las dependencias, los checkpoints y el dataset local ya disponibles, el navegador y el motor se comunican por loopback. Las imágenes cargadas se mantienen en memoria.

El arranque usa CUDA automáticamente si está disponible. Usa `python -m demo_glaucoma --device cpu` para forzar CPU o añade `--port 8766` si el puerto está ocupado.

## Modelos habilitados (ConvNeXt-Tiny)

Los diez checkpoints provienen de los dos experimentos finales independientes ([04](../04_experimento_trinario_vs_binario/01_convnext_trinario_vs_binario.ipynb) y [05](../05_experimento_data_augmentation/02_convnext_data_augmentation.ipynb)) sobre el dataset auditado sin fuga de datos ([dataset_auditado.csv](../utilidades/dataset_auditado.csv)). Todos comparten la arquitectura **ConvNeXt-Tiny**, softmax y umbral operativo $\tau$ calibrado únicamente en la partición de validación buscando especificidad $\ge 0{,}95$.

En **Modo Binario**, la opción predeterminada **LODO · Comparar Sin aumento vs. Con aumento (auto)** ejecuta automáticamente ambos brazos (`Base` y `Aumento de escala`) para cada imagen usando el *fold* que excluyó su dataset, y reporta las dos tablas comparativas de métricas por dataset (`Exactitud`, `Precisión`, `Sensibilidad`, `Especificidad`, `TFP`, `TFN`, `F1 macro` y `AUROC`) junto con sus matrices de confusión pareadas:

| Modo y variante | Checkpoint en `modelos/` | Regla de decisión |
| --- | --- | --- |
| Binario · LODO sin RIM-ONE DL (Aumento escala / Base) | `convnext_tiny_lodo_sin_rim_one_dl_{aumento,base}.pth` | Softmax con $\tau = 0{,}78$ / $0{,}98$ |
| Binario · LODO sin ACRIMA (Aumento escala / Base) | `convnext_tiny_lodo_sin_acrima_{aumento,base}.pth` | Softmax con $\tau = 0{,}90$ / $0{,}99$ |
| Binario · LODO sin EyePACS REAL (Aumento escala / Base) | `convnext_tiny_lodo_sin_eyepacs_real_{aumento,base}.pth` | Softmax con $\tau = 0{,}97$ / $0{,}99$ |
| Binario · LODO sin Moondream (Aumento escala / Base) | `convnext_tiny_lodo_sin_moondream_staged_{aumento,base}.pth` | Softmax con $\tau = 0{,}36$ / $0{,}91$ |
| Binario · Exp. 04 Moondream estándar | `convnext_tiny_2c_estandar.pth` | Softmax con $\tau = 0{,}90$ |
| Trinario · Exp. 04 Moondream estándar **(predeterminado)** | `convnext_tiny_3c_estandar.pth` | Softmax; si $P(\text{Temprano})+P(\text{Avanzado}) \ge 0{,}81$, elige mayor estadio |

Las seis variantes convierten a RGB, redimensionan directamente a 224 × 224 con interpolación bilineal, convierten a tensor y normalizan con media `[0.485, 0.456, 0.406]` y desviación `[0.229, 0.224, 0.225]`, sin recortes ni reorientación EXIF adicional.

Si falta un peso o es incompatible, la interfaz indica el estado en el selector y preselecciona una variante disponible del mismo modo cuando existe.

## Ejemplos precargados por dataset

La demo incluye **24 retinografías precargadas (6 por cada cohorte)** verificadas en [metadata.csv](../data/RIM_ONE_DL/metadata.csv) y [dataset_auditado.csv](../utilidades/dataset_auditado.csv):

- **RIM-ONE DL (6):** 3 Sano, 3 Glaucoma (cohorte externa papilar recortada).
- **ACRIMA (6):** 3 Sano, 3 Glaucoma (cohorte externa papilar recortada).
- **EyePACS REAL (6):** 3 Sano, 3 Glaucoma (cohorte externa de campo amplio).
- **Moondream Staged · test (6):** 2 Sano, 2 Temprano, 2 Avanzado (conjunto de prueba interno nunca visto en entrenamiento ni validación).

Desde el selector **Lote precargado por dataset** puedes cargar los 24 casos juntos o filtrar 6 casos de un dataset específico.

Para imágenes propias, abre la fila y selecciona **Editar referencia** o importa un CSV UTF-8 con comas (`id,nivel,etiqueta` o `archivo,nivel,etiqueta`), hasta 100 archivos JPEG, PNG o WebP (máx. 18 MiB y 40 MP).

## Sustituir modelos y ejemplos

1. Guarda el checkpoint dentro de `modelos/` y regístralo en [models.json](models.json) indicando `id`, `mode`, `name`, `architecture` (`convnext_tiny` o `resnet18`), `checkpoint_key` si aplica, `weights`, `classes`, `protocol`, `decision` y `summary`.
2. La API expone `GET /api/catalog`, `GET /api/examples/<id>/image`, `POST /api/predict?model_id=…` y `POST /api/labels`. Una inferencia exitosa devuelve `status`, `model_id`, `prediction`, `scores`, `binary_projection`, `decision_summary` y `diagnostics`.

## Verificación

```powershell
python -m demo_glaucoma.verify --device cpu
python -m demo_glaucoma.verify --device cuda
```

## Guion de cinco minutos

| Tiempo | Acción y explicación |
| --- | --- |
| **0:00–0:45** | Abre la demo. Señala **Binario** y **ConvNeXt-Tiny · Con aumento de escala (ganador binario)** ($\tau = 0{,}95$). Presenta el lote RIM-ONE DL y ejecuta predicción. |
| **0:45–1:45** | En `r1_Im003.png` y `r1_Im069.png`, muestra el panel **Explicación de la predicción**: encuadre lleno (`0 %` borde negro), margen holgado frente a $\tau$ y coincidencia clara. |
| **1:45–2:45** | Abre `r1_Im008.png` (`Sano` marcado `Glaucoma` con `96,0 %`): explica en el panel que está a solo `+1,0 pp` del umbral $\tau=0{,}95$ calibrado en Moondream. |
| **2:45–4:15** | Cambia a **Trinario** (**ConvNeXt-Tiny · Receta estándar**, $\tau=0{,}81$) y ejecuta el lote. Abre `r1_Im069.png` (`Avanzado`) y señala la comparación binaria derivada. |
| **4:15–5:00** | Abre `r2_Im275.png`: en trinario estándar queda en `54,5 %` (< $\tau=0{,}81$, falso negativo), mientras que en binario con aumento supera el corte (`96,0 %`). Cierra conectando invariancia de escala y calibración de umbral fuera de dominio. |
