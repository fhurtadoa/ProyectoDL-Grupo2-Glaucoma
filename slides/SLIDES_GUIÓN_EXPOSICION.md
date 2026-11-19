# Guión Oficial de Exposición (11 Diapositivas / 12–14 Minutos)
## Formato Híbrido Institucional — Facultad de Minas (Universidad Nacional de Colombia)

**Materia:** Deep Learning (2026-2)  
**Tiempo estimado total:** 12 a 14 minutos + 3 a 5 minutos de preguntas.  

---

### Diapositiva 1: Portada y Presentación [FONDO VERDE MINAS] (1:00 min)
* "Buenas tardes profesor y compañeros. Hoy presentamos nuestra investigación titulada: *Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías*."
* "El propósito central de este trabajo fue superar el espejismo de los laboratorios cerrados: evaluar con rigor por qué los modelos que reportan 98% de AUC en un hospital colapsan al llevarse a la práctica clínica real ante cámaras y calibraciones distintas, y cómo resolverlo mediante ingeniería centrada en los datos."

---

### Diapositiva 2: La Problemática Clínica y el Espejismo Monocéntrico [FONDO CLARO] (1:15 min)
* "El glaucoma es una neuropatía silenciosa e irreversible: cuando el paciente nota pérdida visual, ya ha perdido más del 40% de sus fibras nerviosas."
* "En la literatura abundan modelos con ROC-AUC mayores a 0.98. Sin embargo, al evaluarlos en condiciones externas reales, descubrimos un desplome crítico: un modelo entrenado en ACRIMA (Canon 35°) se derrumba a 0.528 en EyePACS, lo cual equivale a clasificar al azar."
* "Las causas físicas son determinantes: disparidad de encuadre (35° macro frente a 45° panorámico) y variaciones de temperatura de color entre fabricantes de cámaras."

---

### Diapositiva 3: Delimitación Metodológica y Enfoque Sanitario (TFP vs. TFN) [FONDO CLARO] (1:15 min)
* "Delimitamos nuestro objeto material (clasificadores profundos CNNs y Transformers sobre retinografías) y objeto formal (la capacidad de generalización inter-dispositivo)."
* "Y adoptamos un mandato clínico deontológico: en medicina, el ROC-AUC abstracto es insuficiente. Lo que manda en salud pública es el balance entre Tasa de Falsos Negativos (TFN) y Tasa de Falsos Positivos (TFP)."
* "Minimizar TFN es éticamente imperativo para no enviar a casa a un paciente enfermo que terminará ciego. Minimizar TFP es indispensable para no colapsar las consultas de oftalmología con falsas alarmas."

---

### Diapositiva 4: Estado del Arte y Vacíos en la Literatura (PRISMA 2020) [FONDO CLARO] (1:15 min)
* "Ejecutamos una Revisión Sistemática de Literatura buscando en 4 APIs científicas (OpenAlex, Europe PMC, PubMed y arXiv), validando 190 artículos en Crossref y alcanzando un acuerdo sustancial de Kappa inter-evaluador de 0.648."
* "La literatura reveló dos vacíos mayúsculos: una pérdida promedio inter-dominio de casi 10 puntos de AUC en la literatura, y el hecho de que más del 90% de los estudios tratan el glaucoma como una variable binaria trivial, omitiendo por completo los estadíos tempranos."

---

### Diapositiva 5: Ecosistema de Datos y Heterogeneidad de Sensores [FONDO CLARO] (1:15 min)
* "Consolidamos cuatro cohortes internacionales con características ópticas y epidemiológicas marcadamente distintas:"
  * "ACRIMA (España): cámara Canon digital, recorte a 35°, con un 58.6% de casos avanzados."
  * "RIM-ONE DL (Canarias): cámara Kowa 3D estereoscópica, con un 56.8% de casos tempranos incipientes."
  * "EyePACS (EE.UU.): campo abierto a 45° donde el disco ocupa menos del 6% de la imagen."
  * "Moondream: cohorte con anotación ordinal de severidad en tres estadíos."
* "Evaluamos también la recolección web en Google Images con un filtro inteligente de calidad retiniana; sin embargo, confirmamos que el ruido y la falta de metadatos clínicos de internet hacen indispensable trabajar con cohortes clínicas auditadas."

---

### Diapositiva 6: Benchmark de Backbones y Tradeoff TFP vs. TFN [FONDO CLARO] (1:15 min)
* "Evaluamos diez arquitecturas base con Transfer Learning desde ImageNet."
* "El modelo ganador fue **ConvNeXt-Tiny**, logrando un AUC de 0.9686 y la menor tasa de omisión de enfermos del benchmark (TFN = 6.13%)."
* "**ResNet-18** demostró ser la red más rápida y eficiente (5.86 milisegundos de inferencia)."
* "Un hallazgo teórico fundamental: los Vision Transformers puros como ViT-B/16 sufrieron fuertemente (TFN = 17.18%), confirmando que el sesgo inductivo 2D de las convoluciones es superior cuando se cuenta con conjuntos biomédicos de tamaño moderado."

---

### Diapositiva 7: Paradigmas de Decisión: 2 vs. 3 Clases [FONDO CLARO] (1:15 min)
* "Demostramos que el paradigma binario ofrece un falso confort: la red aprende a buscar ojos terminales con excavaciones masivas y falla en la detección temprana."
* "Implementamos una estadificación en tres clases: Sano, Incipiente y Avanzado."
* "Esto nos permitió descomponer los fallos: los errores adyacentes (Sano confundido con Incipiente) reflejan la incertidumbre humana normal; pero los errores catastróficos (un paciente avanzado catalogado como sano) son inaceptables."

---

### Diapositiva 8: Data Augmentation Maligno e Invarianza de Escala [FONDO CLARO] (1:30 min)
* "Para cerrar la brecha geométrica entre cámaras de 35° y 45°, diseñamos *Malignant Scale Augmentation*."
* "En la tira de 4 paneles pueden ver la técnica: Zoom-In focal para forzar atención en la regla ISNT, Zoom-Out severo incrustando la papila en un lienzo sintético con máscara pupilar para simular campo abierto, y perturbaciones fotométricas para emular diferencias de sensor."
* "El resultado fue contundente: **redujimos los errores catastróficos a más de la mitad (del 20% al 9.76%)** y logramos un acuerdo cuadrático casi perfecto de QWK = 0.8163."

---

### Diapositiva 9: Matriz de Generalización Inter-Cámara [FONDO CLARO] (1:15 min)
* "Al integrar el aumento de escala y la mega-cohorte federada, superamos el colapso inter-hospitalario."
* "El rendimiento en EyePACS —que inicialmente era de 0.5283 (puro azar)— escaló hasta un AUC de **0.9400 con un 100% de especificidad**."
* "Esto demuestra que una estrategia centrada en los datos resuelve la brecha de dominio sin requerir costosas arquitecturas adversarias (GANs)."

---

### Diapositiva 10: DEMOSTRACIÓN EN VIVO (LIVE DEMO) [FONDO VERDE MINAS] (2:00 - 2:30 min)
* "Como la mejor prueba de robustez es la verificación en tiempo real, pasamos ahora a la demostración en vivo de nuestro clasificador."
* *(Transición para ejecutar la inferencia interactiva en vivo frente al jurado: cargar una imagen, mostrar la predicción binaria, la estadificación en 3 niveles y la distribución de confianza Softmax).*

---

### Diapositiva 11: Conclusiones y Cierre [FONDO VERDE MINAS] (1:00 min)
* "En conclusión: 1) Los modelos binarios tradicionales son frágiles en entornos reales; 2) Las CNNs modernas con sesgo inductivo superan a los Transformers en volúmenes clínicos moderados; 3) La estadificación de 3 clases es indispensable para no omitir glaucomas incipientes; y 4) Las intervenciones centradas en los datos son la clave para un despliegue asistencial fiable."
* "Muchas gracias. Quedamos atentos a sus preguntas y comentarios."
