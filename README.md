# Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías

**Proyecto de Investigación — Deep Learning**  
**Facultad de Minas — Universidad Nacional de Colombia**  
**Formato del documento final:** Artículo científico — Revista DYNA ([guía maestra](Entrega%20Final/Gu%C3%ADa%20maestra%20del%20art%C3%ADculo%20DYNA%20generalizaci%C3%B3n%20de%20clasificadores%20de%20glaucoma.docx) y [plantilla Word](Entrega%20Final/Plantilla_base_seccion_DYNA.docx)).

---

## 1. ¿Dónde quedó cada cosa? (Mapa rápido tras la reorganización)

Para que sea fácil encontrar todo sin buscar entre copias repetidas, el repositorio dejó de dividirse por persona y pasó a organizarse **por tema y experimento**:

| Antes estaba en... | Ahora está en... | Para qué sirve |
| --- | --- | --- |
| `Mauricio/` y `estado_del_arte/` | [`01_estado_del_arte_y_marco_teorico/`](01_estado_del_arte_y_marco_teorico/) | Revisión sistemática (PRISMA 2020), marco teórico, referencias y guía de Mauricio. |
| `Andres_Arbelaez/` (notebook, guía y benchmark) | [`02_experimento_seleccion_datasets_y_modelos/`](02_experimento_seleccion_datasets_y_modelos/) | Notebook de Andrés, su guía, JSONs del benchmark de 10 redes y nota de alineación. |
| `Fabian/` (análisis de características + NB 03) | [`03_experimento_caracteristicas_retinografias/`](03_experimento_caracteristicas_retinografias/) | Notebook 03, informe descriptivo de retinografías, figuras y tablas de características. |
| `Fabian/01_convnext_trinario_vs_binario.ipynb` | [`04_experimento_trinario_vs_binario/`](04_experimento_trinario_vs_binario/) | Experimento `ConvNeXt-Tiny` 3 clases vs. 2 clases y métricas/predicciones resultantes. |
| `Fabian/02_convnext_data_augmentation.ipynb` | [`05_experimento_data_augmentation/`](05_experimento_data_augmentation/) | Experimento factorial `Base` vs. `MalignantScaleAugmentation` y sus métricas resultantes. |
| Código compartido y dataset depurado de Fabián | [`utilidades/`](utilidades/) | Dataset limpio sin fuga de datos (`dataset_auditado.csv`), `preparar_dataset.py` y `utilidades.py`. |
| `scripts/`, `informes/`, `results/`, `figures/` (y copias viejas) | [`archivo_historico/`](archivo_historico/) | **Material base antiguo** (no generado por los notebooks finales de Andrés y Fabián). |

---

## 2. Estructura completa de carpetas

```text
ProyectoDL-Grupo2-Glaucoma/
├── utilidades/                                    # Base común y código compartido (experimentos 03–05)
│   ├── dataset_auditado.csv                       # 2.056 imágenes elegibles sin fuga entre splits/fuentes
│   ├── resumen_depuracion.csv                     # Conteo de imágenes elegibles y descartadas por motivo
│   ├── preparar_dataset.py                        # Script que genera dataset_auditado.csv
│   └── utilidades.py                              # Entrenamiento/carga ConvNeXt-Tiny, métricas, bootstrap y gráficas
│
├── 01_estado_del_arte_y_marco_teorico/            # Mauricio (Encabezado, 1. Intro, 2. Marco teórico, 3. RSL, Refs)
│   ├── Guía de Mauricio ...docx
│   ├── estado_del_arte.pdf
│   ├── manuscrito/                                # estado_del_arte.md, estado_del_arte_reescrito.md, referencias.bib, etc.
│   ├── marco_teorico/                             # Fuentes y salidas del marco teórico
│   ├── figuras/                                   # 6 figuras RSL en escala de grises (300 dpi)
│   └── scripts_rsl/                               # Scripts 01_ a 08_ de búsqueda, cribado y PRISMA
│
├── 02_experimento_seleccion_datasets_y_modelos/   # Andrés (4.1, 4.2 y opcional 4.3)
│   ├── Comprobacion_mejor_modelo.ipynb            # Notebook principal de Andrés (sin modificar)
│   ├── Guía de Andrés selección de datasets y de modelo.docx
│   ├── NOTA_ALINEACION_ANDRES.md                  # Puntos de sincronización entre Andrés y Fabián
│   └── datos_benchmark/                           # JSON/CSV del benchmark de 10 backbones y armonización
│
├── 03_experimento_caracteristicas_retinografias/  # Fabián (insumo descriptivo para 4.1)
│   ├── 03_caracteristicas_de_retinografias.ipynb  # Notebook descriptivo y verificación en disco
│   ├── INFORME_CARACTERISTICAS_RETINOGRAFIAS.pdf
│   ├── figuras/                                   # 7 figuras descriptivas (brillo, color, nitidez, encuadre, mosaicos)
│   ├── scripts/                                   # Extractores deterministas de métricas y procedencia
│   └── datos/                                     # Tablas maestras (por_imagen.csv, pares.csv) y salidas 03_*.csv
│
├── 04_experimento_trinario_vs_binario/            # Fabián (Sección 4.4)
│   ├── 01_convnext_trinario_vs_binario.ipynb
│   └── resultados/                                # 01_metricas_binarias.csv y 01_predicciones_binarias.csv
│
├── 05_experimento_data_augmentation/              # Fabián (Secciones 4.5 y 4.6)
│   ├── 02_convnext_data_augmentation.ipynb
│   └── resultados/                                # 02_metricas_factorial.csv y 02_senales_visuales_pareadas.csv
│
├── Entrega Final/                                 # Guías y plantilla oficial para redactar el artículo
│   ├── Guía maestra del artículo DYNA ...docx
│   ├── Guía de Fabián - estudio experimental y generalización.md
│   └── Plantilla_base_seccion_DYNA.docx
│
├── archivo_historico/                             # Bases iniciales / scripts y PDFs viejos ya reemplazados
│   ├── scripts_base/                              # Scripts antiguos 01–11 y compiladores de informes
│   ├── informes_pdf_base/                         # PDFs 01–10 compilados con los scripts iniciales
│   ├── resultados_base/                           # JSONs antiguos previos a los notebooks finales
│   └── figuras_base/                              # Figuras antiguas de los informes base
│
├── data/                                          # Retinografías locales (MOONDREAM_STAGED, ACRIMA, RIM_ONE_DL, EYEPACS_*, etc.)
├── modelos/                                       # Checkpoints .pth (incluye fabian_convnext_tiny_*.pth en local)
├── demo_glaucoma/                                 # Demo web local para la sustentación
└── assets/                                        # Recursos de renderizado
```

---

## 3. Quién usa qué para redactar el artículo DYNA

| Sección del artículo | Responsable | Carpeta y archivos clave que debe usar |
| --- | --- | --- |
| Encabezado, Abstract, 1. Intro, 2. Marco teórico, 3. RSL, Referencias | **Mauricio** | [`01_estado_del_arte_y_marco_teorico/`](01_estado_del_arte_y_marco_teorico/) y su guía `.docx`. |
| 4.1 Selección de datasets + análisis descriptivo | **Andrés** (con insumo de **Fabián**) | [`02_experimento_seleccion_datasets_y_modelos/`](02_experimento_seleccion_datasets_y_modelos/) + [`03_experimento_caracteristicas_retinografias/`](03_experimento_caracteristicas_retinografias/). |
| 4.2 Selección de modelo | **Andrés** | [`Comprobacion_mejor_modelo.ipynb`](02_experimento_seleccion_datasets_y_modelos/Comprobacion_mejor_modelo.ipynb) y [`datos_benchmark/`](02_experimento_seleccion_datasets_y_modelos/datos_benchmark/). |
| 4.3 Estadificar un dataset binario *(opcional)* | **Andrés** *(pendiente)* | Por definir por Andrés si se incluye como subsección o si pasa como trabajo futuro en 4.6. |
| 4.4 Análisis trinario vs. binario | **Fabián** | [`04_experimento_trinario_vs_binario/`](04_experimento_trinario_vs_binario/) y [`Guía de Fabián`](Entrega%20Final/Gu%C3%ADa%20de%20Fabi%C3%A1n%20-%20estudio%20experimental%20y%20generalizaci%C3%B3n.md). |
| 4.5 Data augmentation (`Base` vs. `Malignant`) y 4.6 Conclusión | **Fabián** | [`05_experimento_data_augmentation/`](05_experimento_data_augmentation/) + [`03_experimento_caracteristicas_retinografias/`](03_experimento_caracteristicas_retinografias/). |

---

## 4. Aclaraciones metodológicas importantes para todo el equipo

1. **Partición limpia sin fuga de datos (`utilidades/dataset_auditado.csv`):**  
   Andrés depuró inicialmente `MOONDREAM_STAGED` (`800` $\rightarrow$ `762` imágenes: `532 train / 115 val / 115 test`) mediante MD5 y similitud coseno ResNet-18 $\ge 0.96$. Los experimentos `04` y `05` aplican además una auditoría estricta de coincidencias geométricas (SIFT/RANSAC) y cruces entre fuentes, quedando **2.056 imágenes elegibles**:
   - `Moondream Staged`: **706** (`504` train / `107` val / `95` test)
   - `ACRIMA`: **703** (`308` normal / `395` glaucoma)
   - `RIM-ONE DL`: **447** (`285` normal / `162` glaucoma)
   - `EyePACS REAL`: **200** (`100` normal / `100` glaucoma)
2. **Modelo común (`ConvNeXt-Tiny`):**  
   Tanto el notebook de Andrés como los experimentos `04` y `05` de Fabián usan **ConvNeXt-Tiny** (el script viejo que usaba `ResNet-18` quedó archivado en `archivo_historico/`).
3. **Prueba con 588 imágenes de Google:**  
   La sección final del notebook de Andrés sobre imágenes web privadas de Google Drive queda **descartada** del alcance del artículo formal.
4. **Material archivado:**  
   Nada dentro de [`archivo_historico/`](archivo_historico/) debe tomarse como fuente final del artículo; son respaldos de los scripts e informes preliminares iniciales.

---

## 5. Cómo ejecutar y verificar

- **Regenerar el dataset depurado (opcional, ya viene generado en `utilidades/dataset_auditado.csv`):**
  ```bash
  python utilidades/preparar_dataset.py
  ```
- **Correr los notebooks de Fabián (`03`, `04`, `05`):**  
  Abre cualquier notebook y ejecútalo normalmente. Por defecto (`REENTRENAR = False`) cargan los pesos entrenados de `modelos/fabian_convnext_tiny_*.pth` (excluidos de Git por tamaño en `.gitignore`) y terminan en segundos; si quieres reentrenar las 20 épocas desde cero, cambia `REENTRENAR = True` en la primera celda.
- **Verificar e iniciar el demo de sustentación:**  
  ```bash
  python -m demo_glaucoma.verify
  ```
  Instrucciones completas en [`demo_glaucoma/README.md`](demo_glaucoma/README.md).
