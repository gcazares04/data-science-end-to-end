# Semana 11, Sesión 2: Esquemas de validación y evaluación de modelos

> Objetivo de la sesión: seleccionar un esquema de validación adecuado al contexto de los datos de tu proyecto (aleatorio, temporal, por grupo) y elegir la métrica de evaluación correcta según el costo relativo de cada tipo de error.

**Lectura previa de esta sesión:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 6 (desarrollo y evaluación de modelos offline).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn. Ten a mano tu `pipeline_proyecto.py` y el modelo final que elegiste la sesión pasada. Los laboratorios de hoy son notebooks, como los de la Semana 10 y la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la paradoja de la exactitud (accuracy paradox)? ¿Qué métricas recomienda en su lugar?**

Es cuando un modelo tiene una accuracy alta sin servir para nada, porque la clase que importa es poco frecuente. El ejemplo del artículo es una enfermedad que tiene 1 de cada 100 personas: un clasificador que dice "sano" a todos acierta el 99% y no detecta a ningún enfermo. La accuracy premia acertar la clase mayoritaria y esconde lo que pasa con la minoritaria. En su lugar recomienda mirar **precision**, **recall** y **F1**, que miden directamente cuántos casos positivos se detectan y cuántas alarmas son correctas. Es lo que pasa con la línea base del Ejercicio 2: 95% de accuracy y recall 0.

**El artículo cita la ley de Goodhart: "cuando una medida se convierte en objetivo, deja de ser una buena medida". ¿Qué ejemplo da, y qué tiene que ver con elegir un modelo por una sola métrica?**

El ejemplo es el de los LLM entrenados o ajustados para subir en un benchmark: al optimizar ese número, el modelo memoriza los patrones del conjunto de evaluación en vez de generalizar, y el puntaje sube sin que el modelo sea mejor en tareas que no están en el benchmark. Con la elección de modelos pasa lo mismo: si elijo el modelo solo por el menor error sobre un conjunto de prueba fijo, y además ajusto hiperparámetros mirándolo, termino optimizando para esos 595 viajes (o 46 awards) y no para los casos nuevos. Además, una sola métrica puede mejorar mientras empeora el error que le importa al negocio. Por eso hoy se valida con varios pliegues y con una métrica que refleje el costo real.

## Pregunta de apertura

La sesión pasada elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Lo suficiente para cambiar la conclusión. La sesión pasada, el mismo gradient boosting dio 1.2754 con la semilla 42 y 1.1867 con la semilla 7, una diferencia casi igual a la ventaja de la red sobre el gradient boosting (0.106). El registro de MLflow solo lo mostraba porque el Ejercicio 1 corrió a propósito una segunda división: cada corrida guarda un único número, sin dispersión. Hoy la validación cruzada lo mide directamente: con 5 pliegues al azar, el error del gradient boosting va de 1.16 a 1.28 (desviación 0.036).

## Ejercicio 1: tres formas de validar el mismo modelo

Notebook: [`starter/ejercicio1_validacion.ipynb`](./starter/ejercicio1_validacion.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Ya trae una función `evaluar` que corre una validación cruzada con el esquema que le pases y guarda el error de cada pliegue. Los 4 `TODO` son tres esquemas de validación y dos métricas. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (TODO 1):** con una sola división, el gradient boosting dio 1.2754 y el bosque 1.3016. Con 5 pliegues al azar, ¿esperas que los cinco errores del gradient boosting sean parecidos entre sí? ¿Cuánto esperas que varíen?

Sí, parecidos: con 2975 viajes, cada pliegue valida con unos 595 elegidos al azar de la misma distribución, así que esperaba errores alrededor de 1.25 con una variación de unos ±0.05, del orden de la diferencia entre las semillas 42 y 7. Salió eso: pliegues entre 1.16 y 1.28, promedio 1.2288, desviación 0.036. El bosque, entre 1.25 y 1.33 (1.2887). El gradient boosting le gana al bosque en promedio por 0.06, más que la desviación entre pliegues. (Con la versión de scikit-learn de `requirements.txt`, la división única del bosque da 1.2989, no 1.3016.)

**Respuesta 2 (TODO 2):** en la validación temporal, cada pliegue entrena con los viajes anteriores y valida con los siguientes. ¿Esperas un error mayor o menor que con pliegues al azar? ¿Por qué?

Mayor. Primero, porque predecir el futuro es más difícil que predecir viajes mezclados del mismo período: si cambia algo con el tiempo (tráfico, temporada, tarifas), el modelo no lo vio. Segundo, porque `TimeSeriesSplit` entrena el primer pliegue con solo la sexta parte más antigua de los viajes (unos 500) y valida con los siguientes. Salió así: promedio 1.3099 contra 1.2288 al azar, con el doble de desviación (0.069). El peor pliegue es el primero (1.40), el que entrena con menos datos. Es la estimación honesta para una app que estima tarifas de viajes que todavía no ocurrieron.

**Respuesta 3 (TODO 3):** en la validación por zona, ninguna zona de origen aparece a la vez en entrenamiento y en validación. ¿Qué pliegue crees que va a tener el peor error?

El que deja fuera **JFK Airport**. Los viajes al aeropuerto son los más largos y tienen tarifa fija (`es_tarifa_fija`), un comportamiento que ninguna otra zona de origen enseña. Si en entrenamiento no hay ningún viaje desde JFK, el one-hot de esa zona queda en ceros y el modelo los trata como viajes urbanos. Salió así de forma muy marcada: el pliegue 5 (Brooklyn y JFK Airport) tiene **6.43** de error, contra 1.19 a 1.31 de los otros cuatro (Midtown, Lower Manhattan, Queens con Upper East Side, Harlem con Upper West Side). Eso infla el promedio a 2.30 con desviación 2.06. Si la app se abriera en una zona nueva, el error de la primera semana se parecería a este y no al 1.23 del esquema al azar.

**Respuesta 4 (TODO 4):** la app muestra una tarifa estimada antes del viaje, y el pasajero reclama si la real sale más de 20% por encima. ¿Qué métrica dice mejor cuántos reclamos va a haber: el MAE, el RMSE, o el porcentaje de viajes subestimados en más de 20%?

El **porcentaje de viajes subestimados en más de 20%**: cuenta exactamente los viajes que generarían un reclamo. El MAE (1.2754 dólares) y el RMSE (1.9213) mezclan sobreestimar y subestimar, cuando sobreestimar no genera reclamos (el pasajero paga menos de lo que esperaba). Además están en dólares absolutos: un error de 2 dólares es un 40% en un viaje de 5 y un 3% en uno de 60. El RMSE es más alto porque castiga más los errores grandes (los viajes largos y los del aeropuerto), pero tampoco distingue la dirección. Con el gradient boosting, el **4.4%** de los 595 viajes de prueba sale más de 20% por encima de lo estimado, unos 26 reclamos de cada 595 viajes.

**Lo que confirmó la tabla, esquema por esquema:**

| Esquema | Modelo | Promedio | Desviación | Pliegues | Peor pliegue |
|---|---|---|---|---|---|
| Una división (Semana 10) | gradient boosting | 1.2754 | — | — | — |
| Aleatorio, 5 pliegues | gradient boosting | 1.2288 | 0.036 | 1.28, 1.24, 1.16, 1.24, 1.23 | 1.28 |
| Aleatorio, 5 pliegues | bosque | 1.2887 | 0.029 | 1.30, 1.27, 1.25, 1.33, 1.29 | 1.33 |
| Temporal, 5 pliegues | gradient boosting | 1.3099 | 0.069 | 1.40, 1.21, 1.38, 1.27, 1.29 | 1.40 (el primero, con menos historia) |
| Por zona de origen, 5 pliegues | gradient boosting | 2.2996 | 2.065 | 1.27, 1.31, 1.29, 1.19, 6.43 | 6.43 (Brooklyn y JFK Airport) |

El esquema al azar es el más optimista y el que menos varía. El temporal sube el error un 7% y duplica la variación. El de zona muestra que el modelo no sabe nada de una zona que no vio, sobre todo del aeropuerto. Métrica según el costo: MAE 1.2754, RMSE 1.9213 y 4.4% de viajes subestimados en más de 20%.

## Durante los ConcepTests

**ConcepTest 1. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

**ConcepTest 2. Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

## Ejercicio 2: el 95% que no detecta nada

Solo o en pareja. Notebook: [`starter/ejercicio2_fraude_y_umbral.ipynb`](./starter/ejercicio2_fraude_y_umbral.ipynb), sobre [`starter/pagos_taxi.csv`](./starter/pagos_taxi.csv): 6000 pagos de la app de taxis, de los que unos 300 resultaron fraudulentos. Mismo esqueleto del Ejercicio 1, con un problema de clasificación. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué accuracy tiene la línea base que nunca predice fraude? ¿Y qué recall?

Accuracy **0.9508** y recall **0**. Acierta el 95% porque el 95.1% de los pagos son legítimos (295 fraudes de 6000), y no detecta ninguno de los 295 fraudes. Su precision tampoco está definida, porque nunca marca un pago como fraude (scikit-learn reporta 0).

**Respuesta 2:** ¿cuánto mejora la accuracy del gradient boosting sobre la línea base, y cuánto mejora el recall? ¿Cuál de los dos números describe mejor lo que gana la empresa?

La accuracy sube de 0.9508 a **0.9712**, solo 2 puntos. El recall sube de 0 a **0.461**: detecta 136 de los 295 fraudes, con una precision de 0.905 (9 de cada 10 alarmas son fraudes de verdad). La logística queda en el medio: accuracy 0.9613 y recall 0.258. Lo que gana la empresa lo describe el **recall**: cada fraude detectado es un contracargo de 40 dólares evitado. La accuracy casi no se mueve porque está dominada por los 5705 pagos legítimos que cualquier modelo acierta, y no distingue un modelo que no detecta nada de uno que detecta la mitad.

**Respuesta 3:** con el umbral 0.5, ¿cuántos fraudes deja pasar el gradient boosting? ¿Qué umbral eligió el paso 5, y cuántos fraudes y falsas alarmas tiene con ese umbral?

Con 0.5 deja pasar **159 de los 295 fraudes** y da solo 14 falsas alarmas (matriz: 5691 legítimos bien, 14 legítimos marcados, 159 fraudes sin detectar, 136 detectados). Costo: 159 × 40 + 14 × 3 = 6402 dólares. El paso 5 eligió el umbral **0.1**: deja pasar **122 fraudes** y da **222 falsas alarmas**, con un costo de 122 × 40 + 222 × 3 = 5546 dólares, 856 menos que con 0.5. Como un fraude cuesta 13 veces más que una falsa alarma, conviene marcar más pagos aunque la mayoría de las alarmas nuevas sean legítimos. Bajar más el umbral (0.05) ya no conviene: detecta 22 fraudes más, pero suma 367 falsas alarmas.

**Si terminas antes:** si la revisión manual de una falsa alarma costara 20 dólares en vez de 3, ¿el umbral elegido sube o baja? Cambia la constante y comprueba.

**Sube**, hasta 0.5. Con las falsas alarmas a 20 dólares, un fraude cuesta solo el doble que una falsa alarma, y ya no conviene revisar muchos pagos legítimos para atrapar unos pocos fraudes más. Los costos (celda final del notebook) son 15780 con 0.05, 9320 con 0.1, 7820 con 0.15, 7300 con 0.2, 6780 con 0.3 y **6640 con 0.5**. El umbral óptimo depende de la razón entre los dos costos, no del modelo: el modelo es el mismo y solo cambió el precio de cada error.

## Ejercicio 3: la validación y la métrica de tu proyecto

Usa tu `pipeline_proyecto.py` y el modelo final de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `validacion_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que evalúe tu modelo final (y el segundo mejor) con validación cruzada, con dos esquemas: el que corresponde a tus datos (temporal si predices el futuro, por grupo si la misma entidad aparece muchas veces, estratificado si clasificas una clase poco frecuente) y uno al azar, para comparar. El notebook guarda una tabla con el promedio, la desviación y cada pliegue.

### Entregable 2: la métrica del negocio

Escribe qué error es más caro en tu proyecto (sobreestimar o subestimar, un falso positivo o un falso negativo) y cuánto, aunque sea aproximado. Calcula en el mismo notebook una métrica que refleje ese costo, además de la métrica que venías usando.

### Entregable 3: respuestas

**¿Qué esquema de validación corresponde a tu proyecto, y por qué?**

**Temporal.** GrantMatch estima el monto de convocatorias nuevas a partir de awards pasados, y los awards van de 1988 a 2025: en ese tiempo cambian los programas de cada funder, los topes y el valor del dinero (los montos están en dólares nominales). La división de la Semana 9 ya era por fecha (corte 2020-01-01) por esa razón. Validar al azar mezclaría awards de 2023 en el entrenamiento para predecir awards de 2010, información que GrantMatch nunca tendría. Usé `TimeSeriesSplit` de 5 pliegues sobre los 169 awards ordenados por `award_start_date`. Cada pliegue entrena con todo lo anterior y valida con los 28 siguientes; el primero entrena con solo 29 awards. Un award aparece una sola vez (se deduplica por `award_key`), así que no hace falta agrupar.

**¿Cambió la conclusión entre el esquema al azar y el tuyo? Muestra los números.**

Sí. Error absoluto medio en log10 (`validacion_proyecto_resumen.csv`):

| Modelo | Al azar: promedio (desv.) | Temporal: promedio (desv.) | Peor pliegue temporal |
|---|---|---|---|
| ridge alpha=10 (final) | 0.416 (0.070) | **0.442** (0.115) | 0.597 |
| red neuronal | **0.372** (0.084) | 0.455 (0.178) | 0.766 |
| línea base (mediana) | 0.476 (0.061) | 0.483 (0.112) | 0.580 |

Al azar, la red le gana claramente a Ridge (0.372 contra 0.416, gana en 4 de 5 pliegues): es lo que diría el esquema equivocado. En el temporal el orden se invierte: Ridge 0.442 y red 0.455, y la red tiene el peor pliegue de todos (0.766, el primero, con 29 awards de entrenamiento). Con pocos datos del pasado, la red se equivoca mucho más que Ridge. El esquema al azar también subestima el error de todos los modelos (0.42 contra 0.44 en Ridge) y reduce la variación a la mitad.

**¿La diferencia entre tu modelo final y el segundo es mayor que la variación entre pliegues? ¿Sigues eligiendo el mismo?**

No. En el temporal, Ridge le gana a la red por 0.013 en promedio. Esa ventaja es mucho menor que la variación entre pliegues (desviación 0.115 en Ridge y 0.178 en la red), y por pliegue la diferencia va de −0.069 a +0.169: la red gana en 3 de 5 pliegues, y Ridge gana en el promedio solo por el primero. Con estos datos no puedo decir que uno sea mejor en error. Aun así **sigo eligiendo Ridge**. Es más estable (menor desviación y peor pliegue de 0.597 contra 0.766), no pierde contra la red en el esquema que corresponde, entrena en milisegundos y se puede explicar. La red solo ganaba con una división única o al azar. Lo que sí tengo que reconocer: Ridge le gana a la mediana por 0.041 en promedio (en 4 de 5 pliegues), también menos que la desviación. Con 169 awards, el modelo apenas supera una regla simple.

**¿Qué métrica del negocio elegiste, y qué dice que no decía la anterior?**

Subestimar es más caro: el investigador ve una convocatoria más chica de lo que es y la descarta sin leerla. Sobreestimar le hace perder tiempo, pero se corrige al leer las bases. Lo aproximé como el doble de caro, y calculé dos métricas:

- **MAE asimétrico:** error en log10, multiplicado por 2 cuando el monto real supera al estimado.
- **% de awards subestimados a menos de la mitad:** monto real mayor que el doble del estimado; son los que se descartarían por error.

Esquema temporal:

| Modelo | MAE log | MAE asimétrico | % subestimado a menos de la mitad |
|---|---|---|---|
| ridge alpha=10 | 0.442 | 0.721 | 30.7% |
| red neuronal | 0.455 | 0.753 | 30.7% |
| mediana | 0.483 | 0.806 | 30.0% |

El MAE decía que Ridge mejora un 9% sobre la mediana. La métrica del negocio dice dos cosas que el MAE escondía:

- **Los errores van sobre todo hacia abajo.** El MAE asimétrico (0.72) está más cerca de 2 × 0.44 que de 0.44, así que la mayor parte del error es subestimación. Es lo esperable si los montos nominales suben con el tiempo y el modelo aprende de awards viejos.
- **En el error caro ningún modelo mejora a la mediana.** Cerca de **un tercio** de los awards nuevos aparecería con menos de la mitad de su monto real. Ridge (30.7%) no mejora a la mediana (30.0%) en ese error: su ventaja en MAE viene de awards que ya se estimaban bien.

Al azar ese porcentaje baja a 26%, otra señal de que el esquema al azar es optimista. Lo siguiente sería ajustar los montos por inflación o agregar el año como feature, y medir si baja ese 30%.

## Antes de la próxima sesión

**Lectura:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

**La tarifa, en buena parte.** Depende de pocas variables con valores conocidos de antemano: 8 zonas de origen, 8 de destino, 24 horas, día hábil o fin de semana y método de pago. Se puede precalcular cada noche una tabla para todas las combinaciones y que la app solo la consulte. La distancia exacta del recorrido y los pasajeros sí se conocen recién al pedir el viaje: o se aproximan con la distancia típica entre zonas, o el modelo se evalúa en el momento, que es barato (milisegundos). Como se muestra antes de confirmar el viaje, tiene que responder en menos de unos 200 ms.

**El fraude tiene que responder en el momento.** Sus features (`monto`, `intentos_fallidos`, `tarjeta_nueva`, la hora exacta) existen recién cuando se hace el pago, y la combinación de valores posibles es demasiado grande para precalcular. Además la decisión tiene que tomarse antes de aprobar el cobro, dentro del tiempo de autorización de la tarjeta: menos de un segundo, idealmente unas decenas de milisegundos. Un cálculo nocturno llegaría tarde: el contracargo ya habría ocurrido.
