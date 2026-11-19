# Guión Oficial de Exposición (11 Diapositivas / 12–15 Minutos)
## Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías

**Materia:** Deep Learning — Facultad de Minas, Universidad Nacional de Colombia  
**Tiempo estimado total:** 12 a 14 minutos + 3 a 5 minutos de preguntas.  

---

### Diapositiva 1: Portada y Presentación del Proyecto (1:00 min)
* **Qué decir:**
  * "Buenas tardes profesor y compañeros. Hoy les presentamos nuestra investigación titulada: *Evaluación de la generalización de clasificadores profundos de glaucoma ante heterogeneidad de adquisición de datos en retinografías*."
  * "El objetivo central de este proyecto no fue simplemente entrenar un modelo para sacar un 98% en un dataset cerrado, sino responder a la pregunta que la literatura suele evadir: ¿qué pasa cuando ese modelo se lleva a un hospital real con otra cámara, otra calibración de color o un ángulo de visión distinto?"

---

### Diapositiva 2: La Problemática Clínica y el Espejismo Monocéntrico (1:15 min)
* **Qué decir:**
  * "El glaucoma es la principal causa de ceguera irreversible en el mundo y es una enfermedad silenciosa: cuando el paciente nota pérdida de visión, ya ha perdido más del 40% de sus fibras nerviosas."
  * "Al entrenar redes convolucionales en bases de datos públicas como ACRIMA, los modelos reportan métricas excepcionales, pero descubrimos que esto es un **espejismo monocéntrico**."
  * "Cuando evaluamos ese mismo modelo en una cohorte externa como EyePACS, el AUC colapsa a 0.528, que es literalmente tirar una moneda al aire. ¿La razón? El cambio de dominio óptico: cámaras de sobremesa con recorte focal a 35° frente a cámaras panorámicas de campo abierto a 45°."

---

### Diapositiva 3: Delimitación Metodológica y Enfoque en TFP vs. TFN (1:15 min)
* **Qué decir:**
  * "Para abordar esto con rigor científico, delimitamos nuestro objeto material y formal."
  * "Y adoptamos un mandato clínico fundamental: en salud pública, el ROC-AUC general es engañoso. Lo que verdaderamente importa es el balance entre la **Tasa de Falsos Negativos (TFN)** y la **Tasa de Falsos Positivos (TFP)**."
  * "Minimizar la TFN es prioritario para no enviar a casa a un paciente enfermo que terminará ciego. Minimizar la TFP es indispensable para no colapsar el sistema de salud con falsas alarmas. En este estudio evaluamos ambos frentes."

---

### Diapositiva 4: Estado del Arte y Revisión Sistemática PRISMA 2020 (1:15 min)
* **Qué decir:**
  * "Construimos un estado del arte formal mediante una Revisión Sistemática de Literatura bajo directrices PRISMA 2020 y Kitchenham."
  * "Buscamos de manera programática en 4 APIs científicas (OpenAlex, Europe PMC, PubMed y arXiv), validamos 190 artículos en Crossref y logramos un acuerdo sustancial de Kappa inter-evaluador de 0.648."
  * "La literatura reveló dos vacíos mayúsculos: una pérdida promedio inter-dominio de casi 10 puntos de AUC, y el hecho de que más del 90% de los estudios tratan el glaucoma como una variable binaria trivial, ignorando los estadíos tempranos."

---

### Diapositiva 5: Ecosistema de Datos y Filtro Web Inteligente (1:15 min)
* **Qué decir:**
  * "Consolidamos cuatro cohortes internacionales (ACRIMA de España, RIM-ONE DL de Canarias con fotos estereoscópicas, EyePACS de Estados Unidos y Moondream con anotación ordinal)."
  * "Además, exploramos la recolección de imágenes web in-the-wild desde Google Images. Como internet está lleno de diagramas y esquemas anatómicos no utilizables, diseñamos un discriminador inteligente de calidad de fondo de ojo basado en la firma fotométrica del canal verde y gradientes peripapilares."
  * "El filtro depuró el 47% de ruido con un 100% de precisión, demostrando cómo preprocesar datos abiertos con rigor."

---

### Diapositiva 6: Benchmark de Backbones y Tradeoff TFP vs. TFN (1:15 min)
* **Qué decir:**
  * "Evaluamos diez arquitecturas representativas con Transfer Learning desde ImageNet."
  * "El modelo ganador fue **ConvNeXt-Tiny**, logrando un AUC de 0.9686 y la menor tasa de falsos negativos del benchmark (TFN = 6.13%)."
  * "Por su parte, **ResNet-18** demostró ser la red más rápida y eficiente (menos de 6 milisegundos de inferencia)."
  * "Un hallazgo teórico clave: los Vision Transformers puros como ViT-B/16 sufrieron severamente en este tamaño de datos (AUC 0.9022 y TFN de 17%), confirmando que el sesgo inductivo 2D de las convoluciones es superior cuando no se cuenta con millones de imágenes médicas."

---

### Diapositiva 7: Paradigmas de Decisión: 2 vs. 3 Clases (1:15 min)
* **Qué decir:**
  * "Demostramos que clasificar glaucoma como 'Sano vs. Enfermo' es un error metodológico: la red aprende a buscar ojos con excavaciones gigantescas y falla en los casos incipientes."
  * "Implementamos una estadificación en tres clases: Sano, Incipiente y Avanzado."
  * "Esto nos permitió categorizar los fallos: los errores adyacentes (Sano confundido con Incipiente) representan la incertidumbre humana normal; pero los **errores catastróficos** (un paciente avanzado catalogado como sano) son inaceptables."

---

### Diapositiva 8: Data Augmentation Maligno e Invarianza de Escala (1:30 min)
* **Qué decir:**
  * "Para resolver el colapso geométrico entre cámaras de 35° y 45°, diseñamos *Malignant Scale Augmentation*."
  * "En la tira de imágenes pueden ver la intervención: combinamos Zoom-In focal para obligar a la red a mirar la regla ISNT en la copa, Zoom-Out severo incrustando la papila en un lienzo sintético con máscara pupilar para simular fotos de campo completo, y perturbaciones fotométricas para emular diferencias de sensor."
  * "El resultado fue contundente: **redujimos los errores catastróficos a más de la mitad (del 20% al 9.76%)** y logramos un acuerdo ponderado casi perfecto de QWK = 0.8163."

---

### Diapositiva 9: Matriz de Generalización y Validación Cruzada (1:15 min)
* **Qué decir:**
  * "Al integrar el aumento de escala y la mega-cohorte federada, rompimos el colapso inter-cámara."
  * "El rendimiento en EyePACS —que inicialmente era de 0.528— escaló hasta un AUC de 0.9400 con un 100% de especificidad."
  * "Esto demuestra que con una estrategia centrada en los datos no hace falta recurrir a arquitecturas adversarias complejas ni a modelos pesados para lograr una generalización robusta."

---

### Diapositiva 10: DEMOSTRACIÓN EN VIVO (2:00 - 2:30 min)
* **Qué decir y hacer:**
  * "Como la mejor prueba de robustez es la verificación en tiempo real, pasamos ahora a la demostración en vivo de nuestro clasificador."
  * *(Hacer la transición a la Live Demo: cargar una retinografía, mostrar la predicción instantánea binaria y de estadificación de 3 clases en pantalla con sus probabilidades Softmax).*

---

### Diapositiva 11: Conclusiones y Cierre (1:00 min)
* **Qué decir:**
  * "En conclusión: 1) Los modelos binarios tradicionales son frágiles en entornos reales; 2) Las CNNs modernas como ConvNeXt y ResNet siguen siendo superiores a los Transformers en volúmenes biomédicos moderados; 3) La estadificación de 3 clases mitiga los fallos críticos de tamizaje; y 4) Las intervenciones centradas en los datos son la clave para un despliegue clínico fiable."
  * "Muchas gracias. Quedamos atentos a sus preguntas y comentarios."
