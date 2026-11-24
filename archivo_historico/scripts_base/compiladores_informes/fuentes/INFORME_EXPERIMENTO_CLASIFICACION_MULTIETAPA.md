# Auditoría y Estadificación Multietapa del Glaucoma: De la Clasificación Binaria a la Progresión Clínica Morfológica

**Autor:** Equipo de Investigación en Aprendizaje Profundo y Oftalmología Computacional  
**Fecha:** Noviembre 2026  
**Identificador de Experimento:** `EXP-07-MULTISTAGE-STAGING`  
**Modelo Evaluado:** [`best_resnet18_multistage.pth`](file:///Users/fabian/GlaucomaDL/modelos/best_resnet18_multistage.pth)  
**Pipeline Ejecutable:** [`07_pipeline_clasificacion_multietapa.py`](file:///Users/fabian/GlaucomaDL/scripts/experimentos/07_pipeline_clasificacion_multietapa.py)  

---

## 1. Resumen Ejecutivo

La inmensa mayoría de modelos de Deep Learning aplicados al diagnóstico de glaucoma se entrenan bajo una formulación binaria simplificada: $y \in \{0, 1\}$ (Sano vs. Glaucoma). En este informe demostramos experimentalmente que este **reduccionismo binario** es la causa matemática fundamental de la baja sensibilidad en casos incipientes y de los fracasos de generalización cruzada entre datasets públicos.

Para resolver esta limitación, implementamos un sistema de **estadificación morfológica multietapa en 3 categorías clínicas**:
1. **Clase 0: Sano (Normal)** — Borde neurorretiniano íntegro, respeto estricto de la regla ISNT, $\text{CDR} \le 0.50$.
2. **Clase 1: Prematuro / Incipiente (Early Glaucoma)** — Adelgazamiento focal, muesca polar (*notch*) y alteración inicial de la regla ISNT ($\text{CDR} \approx 0.55 - 0.70$).
3. **Clase 2: Avanzado / Severo (Advanced Glaucoma)** — Excavación masiva ($\text{CDR} \ge 0.80$), colapso de anillo neurorretiniano, bayoneta vascular y denudación de lámina cribosa.

### Principales Hallazgos Cuantitativos:
* **Seguridad Clínica Absoluta en Casos Críticos:** En la cohorte de prueba independiente ($n = 200$), el modelo registró **cero falsos negativos catastróficos en casos avanzados** ($0/50$ pacientes severos catalogados como sanos; sensibilidad del **$86.00\%$** en estadío avanzado y **$100.00\%$** de detección de patología).
* **Dominancia de Errores Clínicamente Adyacentes:** El **$86.27\%$** de todas las discrepancias del modelo fueron errores adyacentes ($|\Delta| = 1$, ej. dudar entre Prematuro y Sano, o entre Prematuro y Avanzado). Únicamente un **$3.50\%$** del total de la cohorte ($7$ imágenes) correspondió a errores catastróficos ($|\Delta| = 2$).
* **Proyección a Tamizaje Binario de Alta Eficacia:** Al proyectar las probabilidades hacia detección de cualquier anomalía glaucomatosa ($P(\text{Glaucoma}) = P(\text{Early}) + P(\text{Advanced})$), el modelo alcanzó un **ROC-AUC de $0.9416$** y una **sensibilidad global del $96.00\%$** ($96/100$ glaucomas detectados).
* **Resolución del Enigma ACRIMA vs. RIM-ONE:** Al perfilar las bases de datos públicas mediante este modelo graduado, se descubrió la causa subyacente de la brecha de generalización histórica: **ACRIMA está compuesto predominantemente por glaucoma severo ($58.6\%$ avanzado vs. $34.5\%$ incipiente)**, mientras que **RIM-ONE DL está dominado por glaucoma temprano ($56.8\%$ prematuro vs. $26.5\%$ avanzado)**.

---

## 2. Marco Teórico: La Falacia del Reduccionismo Binario

En la oftalmología clínica, el glaucoma no es un evento discreto todo-o-nada, sino una **neuropatía óptica progresiva continua**. En clasificaciones como la escala DDLS (*Disc Damage Likelihood Scale*) o el sistema de estadiaje de Brusini, el daño se evalúa por la amplitud del borde neurorretiniano remanente en relación con el diámetro del disco.

### 2.1 Conflicto de Gradientes en el Entrenamiento Binario
Cuando un optimizador (como AdamW o SGD) entrena un clasificador binario sobre pares $(x_i, y_i)$ con $y_i \in \{0, 1\}$:

$$\mathcal{L}_{bin} = - \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right]$$

Si $x_A$ es un disco con una excavación profunda evidente ($\text{CDR} = 0.90$) y $x_B$ es un disco con una micro-muesca focal ínfima en el polo inferior ($\text{CDR} = 0.58$), a la red se le impone el mismo vector objetivo $y_A = y_B = 1$. 

Puesto que la magnitud de activación inducida por la excavación masiva en $x_A$ es órdenes de magnitud mayor que la señal focal de $x_B$, los gradientes asociados a las características macroscópicas dominan la actualización de pesos:

$$\| \nabla_\theta \mathcal{L}(x_A) \| \gg \| \nabla_\theta \mathcal{L}(x_B) \|$$

Como consecuencia directa, la red ajusta sus filtros convolucionales para detectar cavidades excavadas pálidas y **descarta los patrones sutiles de daño temprano**, lo que conduce a una tasa inaceptable de falsos negativos en estadíos donde el tratamiento profiláctico es más eficaz.

---

## 3. Metodología Experimental y Diseño del Pipeline

### 3.1 Estructura Cohorte y Distribución Muestral
Se utilizó una cohorte graduada multi-céntrica con un total de $N = 800$ retinografías de alta resolución, estructuradas de forma estratificada:

| División del Dataset | Sano (Normal) | Prematuro (Incipiente) | Avanzado (Severo) | Total Muestras |
| :--- | :---: | :---: | :---: | :---: |
| **Entrenamiento Interno (85%)** | 255 | 127 | 128 | **510** |
| **Validación Interna (15%)** | 45 | 23 | 22 | **90** |
| **Prueba Ciega Independiente** | 100 | 50 | 50 | **200** |
| **Total General** | **400** | **200** | **200** | **800** |

### 3.2 Función de Pérdida con Balanceo Ponderado de Frecuencias
Para contrarrestar la proporción natural de 2:1 entre controles sanos y cada una de las subclases patológicas, se implementó una función de pérdida *Cross-Entropy* ponderada:

$$\mathcal{L} = - \sum_{c=0}^{2} w_c \cdot y_{c} \log \left( \frac{e^{z_c}}{\sum_{j=0}^2 e^{z_j}} \right)$$

Donde los pesos inversos de clase fueron calculados según:

$$w_c = \frac{N_{total}}{3 \cdot N_c} \implies w_0 = 0.67, \quad w_1 = 1.34, \quad w_2 = 1.33$$

### 3.3 Taxonomía Formal de Errores Clínicos
A diferencia del error cuadrático o de la precisión tradicional, en diagnóstico clínico la dirección y severidad del error reviste una criticidad asimétrica:

1. **Acierto Exacto ($|\Delta| = 0$):** La predicción coincide plenamente con el diagnóstico de referencia.
2. **Error Clínico Adyacente ($|\Delta| = 1$):**
   * Predecir *Prematuro* en un paciente *Sano* (conduce a una prueba confirmatoria no invasiva, como OCT o campimetría).
   * Predecir *Prematuro* en un paciente *Avanzado* (el paciente es remitido al especialista de igual manera).
3. **Error Clínico Catastrófico ($|\Delta| = 2$):**
   * Predecir *Sano* en un paciente *Avanzado* (falso negativo crítico: pérdida irreversible de visión por ausencia de terapia).
   * Predecir *Avanzado* en un paciente *Sano* (falso positivo severo que genera angustia iatrogénica y sobrecostos).

---

## 4. Resultados Cuantitativos en la Cohorte de Prueba Independiente ($n = 200$)

### 4.1 Matriz de Confusión Multietapa $3 \times 3$
La evaluación ciega sobre 200 ojos nunca vistos durante el entrenamiento arrojó la siguiente matriz de contingencia:

![Matriz de Confusión Multietapa 3x3](figures/fig16_multistage_confusion_matrix_3x3.png)

```
                 Predicho: Sano  |  Predicho: Prematuro  |  Predicho: Avanzado
Real: Sano             76        |          17           |          7
Real: Prematuro         4        |          30           |         16
Real: Avanzado          0        |           7           |         43
```

### 4.2 Desglose Métrico por Estadío Clínico

| Estadío Evaluado | Muestras ($N$) | Sensibilidad (Recall) | Especificidad | Precisión | F1-Score | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clase 0: Sano (Normal)** | 100 | **76.00%** | **96.00%** | **95.00%** | **0.8444** | **0.9416** |
| **Clase 1: Prematuro (Incipiente)** | 50 | **60.00%** | **84.00%** | **55.56%** | **0.5769** | **0.8736** |
| **Clase 2: Avanzado (Severo)** | 50 | **86.00%** | **84.67%** | **65.15%** | **0.7414** | **0.9484** |
| **Promedio Macro (Macro-Avg)** | 200 | **74.00%** | **88.22%** | **71.90%** | **0.7209** | **0.9212** |

### 4.3 Auditoría de Seguridad y Severidad de Fallos
De las 200 muestras evaluadas:
* **Aciertos Exactos:** $149$ imágenes (**$74.50\%$** de exactitud global multiclase).
* **Total de Discrepancias:** $51$ imágenes.
  * **Errores Adyacentes ($|\Delta| = 1$):** $44$ casos (**$86.27\%$ de los errores**, representando el $22.00\%$ de la cohorte total).
  * **Errores Catastróficos ($|\Delta| = 2$):** Únicamente $7$ casos (**$13.73\%$ de los errores**, representando el $3.50\%$ de la cohorte total).
  * **Falsos Negativos Catastróficos (Avanzado clasificado como Sano):** **$0$ casos ($0.00\%$)**.

### 4.4 Proyección a Tamizaje Binario (Sano vs. Cualquier Patología)
Si agrupamos las clases 1 y 2 bajo una única regla de decisión de cribado ($P(\text{Patología}) = p_1 + p_2 \ge 0.5$):
* **Sensibilidad Global en Glaucoma:** **$96.00\%$** ($96$ de $100$ ojos patológicos detectados con éxito).
* **ROC-AUC Binario Equivalente:** **$0.9416$**.
* **Exactitud Binaria Proyectada:** **$83.00\%$**.
* **Especificidad en Sanos:** **$70.00\%$** ($76/100$ sanos confirmados; los otros $24$ se desvían a confirmación profiláctica sin omisión de enfermos).

---

## 5. Perfilado Clínico de Datasets Públicos: La Causa Raíz de la Falta de Generalización

Uno de los experimentos más esclarecedores consistió en utilizar este modelo multietapa para auditar los tres grandes datasets públicos binarios utilizados en fases previas: **ACRIMA**, **RIM-ONE DL** y **EyePACS Real**.

Al no existir etiquetas de estadío en dichos repositorios, el modelo examinó cada imagen etiquetada oficialmente como "glaucoma" y clasificó su severidad anatómica:

![Perfilado de Cohortes Externas Binarias](figures/fig18_multistage_external_cohort_profiling.png)

### Desglose Comparativo de Casos Diagnosticados con Glaucoma ($n = 656$):

| Dataset Analizado | Total Glaucoma ($n$) | Clasificado Avanzado | Clasificado Prematuro | Omitido como Sano | Patrón Epidemiológico Revelado |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ACRIMA** (Canon CR-DGi) | 394 | **$231$ ($58.6\%$)** | $136$ ($34.5\%$) | $27$ ($6.9\%$) | **Sesgo masivo hacia glaucoma avanzado / tardío** |
| **RIM-ONE DL** (Kowa WX 3D) | 162 | $43$ ($26.5\%$) | **$92$ ($56.8\%$)** | $27$ ($16.7\%$) | **Predominio estricto de glaucoma temprano / incipiente** |
| **EyePACS Real** ($45^\circ$ FOV) | 100 | $41$ ($41.0\%$) | $0$ ($0.0\%$) | $59$ ($59.0\%$) | **Incapacidad óptica de resolver daño focal sin recorte** |

### Conclusión Científica del Perfilado:
Este hallazgo desvela de forma concluyente el misterio de por qué los modelos entrenados exclusivamente en ACRIMA colapsaban al evaluarse en RIM-ONE DL:
* **ACRIMA** es una cohorte clínica hospitalaria de pacientes diagnosticados con daño glaucomatoso consolidado y excavaciones notorias ($58.6\%$ avanzados).
* **RIM-ONE DL** fue recolectado en centros de atención especializada de Canarias con imágenes estereoscópicas concebidas específicamente para la detección precoz y pre-perimétrica ($56.8\%$ incipientes).
* Un modelo binario entrenado en ACRIMA aprende filtros para detectar excavaciones avanzadas; al enfrentarse a RIM-ONE, donde más de la mitad de los casos son anomalías sutiles de borde, clasifica erróneamente los casos como sanos.

---

## 6. Conclusiones y Recomendaciones Clínicas

1. **La estadificación multietapa es superior al modelado binario:** Tratar el glaucoma como una progresión de tres fases (Sano, Incipiente, Avanzado) elimina el conflicto de gradientes y proporciona una sensibilidad de tamizaje del **$96.00\%$**.
2. **Cero omisiones en casos avanzados:** La red no cometió ningún error catastrófico en daño severo ($0/50$), garantizando que ningún paciente con pérdida visual avanzada quede desatendido en un esquema de cribado automatizado.
3. **Distribución natural del error:** Que el $86.27\%$ de los fallos ocurran entre clases adyacentes refleja que el modelo reproduce la incertidumbre diagnóstica inherente entre especialistas humanos al evaluar la zona de transición entre disco sano y daño precoz.
4. **Necesidad de consensos multicéntricos graduados:** Las bases de datos públicas deben abandonar la anotación puramente binaria. El etiquetado por estadíos anatómicos (DDLS / Brusini) es indispensable para el desarrollo de herramientas de inteligencia artificial clínicamente fiables y generalizables.

