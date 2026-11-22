# Demo de clasificación de glaucoma

Aplicación académica del Grupo 2 para la sustentación. Ejecuta inferencia sobre retinografías en el equipo local y compara cada predicción con la referencia disponible. Interpreta las puntuaciones como salidas del modelo experimental. El resultado tiene alcance académico.

## Instalación y arranque

Desde la raíz del repositorio, con Python y una combinación compatible de PyTorch y TorchVision:

```powershell
python -m pip install -r demo_glaucoma/requirements.txt
python -m demo_glaucoma
```

Abre [la demo local](http://127.0.0.1:8765/). La instalación inicial puede requerir internet. Con las dependencias, los checkpoints y el dataset local ya disponibles, el navegador y el motor se comunican por loopback. Las imágenes cargadas se mantienen en memoria. El cierre de la pestaña limpia la selección y las etiquetas de esa sesión.

El arranque usa CUDA automáticamente si está disponible. Usa `python -m demo_glaucoma --device cpu` para forzar CPU o añade `--port 8766` si el puerto está ocupado. Para instalar una compilación de PyTorch para tu GPU o solo CPU, usa el [selector oficial de PyTorch](https://pytorch.org/get-started/locally/). Si faltan archivos por sincronizar, verifica que los pesos y las imágenes estén disponibles localmente antes de desconectarte.

## Modelos habilitados

Los cuatro checkpoints son diccionarios de pesos PyTorch de ResNet-18, con una cabeza lineal sobre 512 características. El cargador verifica la correspondencia de todas las claves y dimensiones y prueba una pasada antes de habilitar cada variante. El motor reutiliza los modelos cargados entre predicciones.

| Modo y variante | Checkpoint en `modelos/` | Orden semántico de clases | Regla del experimento |
| --- | --- | --- | --- |
| Binario Malignant Scale **predeterminado** | `model_staged_binary_malignant.pth` | Sano, Glaucoma | Un logit para Glaucoma, sigmoide ≥ 0,5 |
| Binario Vanilla | `model_staged_binary_vanilla.pth` | Sano, Glaucoma | Un logit para Glaucoma, sigmoide ≥ 0,5 |
| Trinario Malignant Scale **predeterminado** | `model_staged_multiclass_malignant.pth` | Sano, Temprano, Avanzado | Tres logits, softmax y máximo |
| Trinario Vanilla | `model_staged_multiclass_vanilla.pth` | Sano, Temprano, Avanzado | Tres logits, softmax y máximo |

Las cuatro variantes convierten a RGB, redimensionan directamente a 224 × 224 con interpolación bilineal, convierten a tensor y normalizan con media `[0.485, 0.456, 0.406]` y desviación `[0.229, 0.224, 0.225]`. El protocolo evalúa la imagen recibida sin recorte, reorientación EXIF ni aumento y usa autocast cuando ejecuta en CUDA, tal como el experimento. Las retinografías RIM-ONE DL ya llegan recortadas desde su fuente. [Código del experimento 11](../scripts/experimentos/11_estudio_2_vs_3_clases_y_data_augmentation.py) · [Resultados originales](../results/study_2_vs_3_classes_metrics.json).

Si falta un peso o es incompatible, la interfaz indica el estado en el selector y preselecciona una variante disponible del mismo modo cuando existe.

### Otros pesos del repositorio

Se inspeccionaron los 30 archivos `.pth` del equipo. El catálogo de la demo se ciñe al experimento 11 para mantener un protocolo común entre modos y variantes. Los demás archivos se agrupan aquí por el experimento al que corresponden.

| Pesos en `modelos/` | Arquitectura y clases | Protocolo o estado de verificación |
| --- | --- | --- |
| `backbone_screen_*.pth` | ResNet-18, ResNet-50, ResNeXt-50 32×4d, DenseNet-121, MobileNet V3 Large, EfficientNet B0 y V2 S, ConvNeXt Tiny, Swin T y ViT B/16. Un logit de Glaucoma en cada variante. | Diccionarios de pesos. RGB 224 × 224 bilineal, ImageNet, sigmoide ≥ 0,5 en el [experimento 10](../scripts/experimentos/10_benchmark_10_backbones_transfer_learning.py). La entrada del benchmark pasa por su cohorte armonizada. |
| `best_fine_tuned_convnext_tiny_glaucoma.pth` | ConvNeXt Tiny, un logit de Glaucoma. | Diccionario de pesos, protocolo de evaluación y regla del [experimento 10](../scripts/experimentos/10_benchmark_10_backbones_transfer_learning.py). |
| `best_resnet18_multistage.pth` | ResNet-18, tres logits en orden Sano, Temprano, Avanzado. | Diccionario de pesos. RGB 224 × 224 bilineal, ImageNet, softmax y máximo, con proyección binaria por umbral de 0,5 en el [experimento 06](../scripts/experimentos/06_estadificacion_clinica_multietapa.py). |
| `best_resnet18_megacohort.pth` | ResNet-18, un logit de Glaucoma. | Diccionario de pesos. RGB 224 × 224 bilineal, ImageNet, sigmoide ≥ 0,5 en el [experimento 05](../scripts/experimentos/05_mega_cohorte_federada_multisensor.py). |
| `best_resnet18_malignant_scale.pth` | ResNet-18, un logit de Glaucoma. | Diccionario de pesos. Los experimentos [04](../scripts/experimentos/04_data_augmentation_invarianza_escala.py) y [08](../scripts/experimentos/08_entrenamiento_malignant_scale_mega_cohorte.py) guardan con el mismo nombre y usan interpolaciones de evaluación distintas. La procedencia exacta requiere el registro de entrenamiento que generó el archivo. |
| `best_resnet18_1000ep_val_auc.pth`, `checkpoint_resnet18_epoch_*.pth`, `final_resnet18_1000ep.pth` | ResNet-18, un logit de Glaucoma. | Diccionarios de pesos. RGB 224 × 224 bicúbica, ImageNet, sigmoide ≥ 0,5 en el [experimento 09](../scripts/experimentos/09_experimento_1000_epocas_sobreentrenamiento.py). Sus entradas de evaluación dependen de la cohorte armonizada. |
| `best_resnet18_acrima.pth` | ResNet-18, un logit de Glaucoma en el [experimento 03](../scripts/experimentos/03_adquisicion_web_y_filtro_calidad.py). | Diccionario de pesos. El consumidor aplica RGB 224 × 224 bilineal e ImageNet. La procedencia del entrenamiento queda pendiente de confirmar. |
| `best_resnet18_multidomain.pth` | Tensores compatibles con una ResNet-18 y una cabeza de un logit. | Diccionario de pesos. Orden semántico, transformaciones y umbral pendientes de confirmar con el código de entrenamiento. |

## Ejemplos y referencias

La demo abre con seis archivos ya presentes en el repositorio y selecciona `r1_Im003.png` en el visor. **Cargar lote de ejemplo** restaura esa selección si cambias de lote. Las referencias se cotejan por ruta y etiqueta contra [el manifiesto local de RIM-ONE DL](../data/RIM_ONE_DL/metadata.csv). Los ejemplos se eligieron para practicar la exposición y contienen una discrepancia. No son una medida de rendimiento sobre la cohorte.

| Imagen | Referencia binaria | Predicción binaria | Predicción trinaria |
| --- | --- | --- | --- |
| `r1_Im003.png` | Sano | Sano | Sano |
| `r1_Im069.png` | Glaucoma | Glaucoma | Avanzado |
| `r1_Im008.png` | Sano | Glaucoma | Temprano |
| `r1_Im023.png` | Sano | Sano | Sano |
| `r1_Im103.png` | Glaucoma | Glaucoma | Temprano |
| `r2_Im275.png` | Glaucoma | Sano | Avanzado |

La tabla registra el comportamiento observado con las variantes predeterminadas. RIM-ONE DL autoriza el uso educativo y de investigación bajo sus condiciones. La app referencia las imágenes originales en el equipo. La ficha pública de Moondream enumera las tres clases, pero no especifica una licencia de uso para precargar imágenes en la exposición. La demo ofrece RIM-ONE DL como lote precargado. [Condiciones y referencia de RIM-ONE DL](https://github.com/miag-ull/rim-one-dl#using-the-database) · [Ficha publicada de Moondream](https://huggingface.co/datasets/moondream/glaucoma-detection).

Para imágenes propias, abre la fila y selecciona **Editar referencia** o importa un CSV UTF-8 con comas. El lote admite hasta 100 archivos JPEG, PNG o WebP, con un máximo de 18 MiB y 40 megapíxeles por imagen. Usa los IDs de la tabla, únicos dentro del lote actual.

```csv
id,nivel,etiqueta
U001,binario,Glaucoma
U002,trinario,Temprano
```

También se acepta `archivo,nivel,etiqueta`. En ese formato el nombre debe corresponder a un solo archivo del lote. El importador informa nombres ambiguos, identificadores ausentes, duplicados y etiquetas inválidas. Aplica las filas válidas que identifican archivos de forma inequívoca. Las filas sin referencia conservan el texto **Sin etiqueta de referencia**.

En modo trinario, una referencia binaria se compara con la **predicción binaria derivada**. Se calcula P(Temprano) + P(Avanzado), equivalente a 1 − P(Sano), y se aplica el umbral de 0,5. La referencia conserva su nivel binario y la severidad figura como no disponible. El resumen muestra esas coincidencias por separado. En modo binario, una referencia trinaria se agrupa a Sano o Glaucoma y la fila lo indica. Cada denominador incluye solo predicciones exitosas con referencia comparable al nivel correspondiente.

## Sustituir modelos y ejemplos

1. Guarda el checkpoint dentro del repositorio. En [models.json](models.json) configura un ID único, nombre visible, modo, arquitectura, ruta relativa, clases ordenadas, protocolo y regla. Si el diccionario de pesos viene dentro de un contenedor, indica su clave en `checkpoint_key`. Selecciona el ID deseado dentro de `defaults` y reinicia el servidor.
2. Para el adaptador actual de ResNet-18 puedes cambiar los pesos, el tamaño, la interpolación, el recorte central después del redimensionamiento, la normalización, el uso de autocast en CUDA y la regla. Se admiten una salida sigmoide para Glaucoma y salidas softmax para ambos modos. Agrega un constructor a `ADAPTERS` en [inference.py](inference.py) si cambia el backbone o la cabeza. Si cambia el orden de operaciones, se necesita recorte de ROI dependiente de la imagen, o se usa otro formato de entrada o de salida, implementa ese protocolo en [images.py](images.py) o su interpretación en [inference.py](inference.py). Valida contra la evaluación original antes de cambiar el predeterminado.
3. En [examples.json](examples.json) configura un `id`, la ruta relativa `path`, el nombre `source`, la ruta del CSV `evidence`, el nivel `level` y el código `label`. El CSV de procedencia debe incluir `filepath,label_name` y una fila única por ruta. Los códigos admitidos son `normal`, `glaucoma`, `early` y `advanced`, de acuerdo con el nivel. Agrega archivos con permiso para la exposición y coteja el CSV de origen. Reinicia para recargar el manifiesto.

El navegador usa `GET /api/catalog` y `POST /api/predict?model_id=…`. Este último recibe bytes de una imagen o un `example_id` en la consulta. Devuelve `status` y `model_id`. Una inferencia exitosa incluye `prediction`, `scores` ordenados y `binary_projection` cuando aplica. Un fallo incluye `message`. La interfaz conserva el resultado de cada fila y actualiza el progreso al terminar cada solicitud.

## Verificación

```powershell
python -m demo_glaucoma.verify --device cpu
python -m demo_glaucoma.verify --device cuda
```

El verificador usa las cuatro variantes originales y tres casos por variante del test del experimento 11. En CPU y en CUDA, las doce clases y puntuaciones coincidieron con la ejecución directa dentro de una tolerancia absoluta de 0,000002. Las doce clases también coincidieron con el JSON histórico. La diferencia máxima de puntuación frente al histórico fue de 0,00443 en CPU y de 0,00037 en CUDA. El verificador también prueba una imagen corrupta, etiquetas CSV ambiguas, inválidas y duplicadas, y la disponibilidad de variantes cuando se simulan pesos faltantes e incompatibles.

En el navegador se probó una imagen individual en modo binario y el lote de RIM-ONE DL en ambos modos. El lote mixto devolvió tres predicciones, un error de decodificación, una coincidencia trinaria, una coincidencia binaria derivada y una imagen sin comparación. La comprobación a 1280 × 720 confirmó que los controles, la retinografía y el resultado caben en la vista inicial. Desplázate hacia abajo para revisar el lote completo.

## Guion de cinco minutos

| Tiempo | Acción y explicación |
| --- | --- |
| **0:00–0:40** | Abre la demo. Señala **Binario** y **ResNet-18 · Malignant Scale**. Presenta la retinografía RIM-ONE DL y su referencia binaria. |
| **0:40–1:30** | Pulsa **Ejecutar predicción**. El primer caso sale **Sano**. Señala las dos puntuaciones, la referencia y el estado **Coincide**. Explica el umbral del 50 % para Glaucoma. |
| **1:30–2:30** | Revisa el lote precargado en **Binario**. Señala los conteos de la matriz y la fila discrepante `r1_Im008.png`. |
| **2:30–4:10** | Cambia a **Trinario** y vuelve a ejecutar el mismo lote. Abre `r1_Im069.png`. El modelo predice **Avanzado** y la referencia dice **Glaucoma** a nivel binario. Señala la comparación derivada y la severidad de referencia no disponible. |
| **4:10–5:00** | Abre `r1_Im008.png`. Muestra la predicción **Temprano**, la referencia binaria **Sano** y la discrepancia. Cierra con la distinción entre la predicción en tres clases y la comparación que permite la anotación disponible. |

La matriz de confusión de cada lote usa referencias como filas y predicciones como columnas. Se muestra una matriz 2 × 2 en modo binario o una 3 × 3 en modo trinario. La 3 × 3 admite referencias trinarias y referencias binarias Sano en la fila Sano. Las referencias binarias Glaucoma quedan fuera del conteo hasta disponer de su estadio. La matriz indica cuántas referencias binarias se incluyeron y cuántas quedaron fuera.
