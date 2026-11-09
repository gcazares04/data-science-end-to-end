# Semana 12, Sesión 1: Patrones de despliegue: ¿batch u online?

> Objetivo de la sesión: elegir entre servir el modelo de tu proyecto en modalidad batch o en modalidad online, justificando la decisión según la latencia que exige el caso de uso propuesto, y proteger con pruebas automáticas, que corren en cada cambio, el código que lo sirve.

**Lectura previa de esta sesión:** ["MLOps: Batch vs Online ML Systems"](https://medium.com/mlops-republic/mlops-batch-vs-online-ml-system-cbacee834837), MLOps Republic.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen, capítulo 7 (despliegue de modelos y servicio de predicción).

## Requisitos antes de esta sesión

Hoy se agregan tres herramientas: joblib (guardar un modelo entrenado en un archivo), pytest (pruebas automáticas) y GitHub Actions (integración continua). Las librerías nuevas ya están en `requirements.txt`, junto con FastAPI para la sesión siguiente. Sincroniza tu fork y reconstruye el contenedor ("Rebuild Container"), como en la Semana 11. Comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Debe mostrar `pytest 8.3.3` y `fastapi 0.115.4`.

Los laboratorios de hoy son scripts, no notebooks: lo que producen corre sin nadie mirando. Cada ejercicio entrena primero su modelo con un script que ya viene completo (`entrenar.py`), que lo guarda en un archivo `.joblib` dentro de la carpeta del ejercicio. Ese archivo no se sube al repositorio (`*.joblib` está en `.gitignore`).

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué distingue un sistema de predicción batch de uno online? Da un ejemplo de cada uno.**

Un sistema **batch** calcula las predicciones de forma periódica (cada hora, cada noche), para muchos casos a la vez, y las guarda en una tabla; cuando alguien las necesita, solo las lee. Un sistema **online** calcula la predicción en el momento en que llega la solicitud, con los datos de ese instante, y la devuelve de inmediato (normalmente a través de una API). Ejemplo batch: las recomendaciones de películas o productos que se recalculan cada noche para todos los usuarios. Ejemplo online: la detección de fraude de una transacción con tarjeta, que se decide mientras se autoriza el cobro.

**¿Qué limitación tiene el sistema batch que el online resuelve, y qué cuesta resolverla?**

El batch solo conoce los datos hasta la última corrida: sus predicciones se vuelven viejas y no puede responder por un caso que no estaba en el lote (un usuario nuevo, una transacción que acaba de ocurrir). Además calcula predicciones para casos que nadie va a pedir. El online resuelve eso porque predice con los datos frescos del momento, pero cuesta: un servicio siempre levantado que responda dentro de un límite de latencia, features que tienen que estar disponibles (o calcularse) en tiempo real, más infraestructura y monitoreo, y el riesgo de que una caída del servicio deje sin respuesta a la app.

## Pregunta de apertura

La app de taxis usa dos modelos: uno estima la tarifa antes de pedir el viaje, otro decide si un pago es fraude. ¿Alguno de los dos se podría calcular una vez por noche para todos los casos posibles? ¿Cuál tiene que responder en el momento, y en cuánto tiempo?

**La tarifa, en buena parte.** Depende de pocas variables con valores conocidos de antemano: 8 zonas de origen, 8 de destino, 24 horas, día hábil o fin de semana y método de pago. Se puede precalcular cada noche una tabla para todas las combinaciones y que la app solo la consulte, y los viajes ya reservados se pueden estimar uno por uno en el lote. La distancia exacta y los pasajeros se conocen recién al pedir un viaje no reservado: o se aproximan, o el modelo se evalúa en el momento, que es barato. Como se muestra antes de confirmar el viaje, tiene que responder en menos de unos 200 ms.

**El fraude tiene que responder en el momento.** Sus features (`monto`, `intentos_fallidos`, `tarjeta_nueva`, la hora exacta) existen recién cuando se hace el pago, y las combinaciones posibles son demasiadas para precalcular. La decisión se toma antes de aprobar el cobro, dentro del tiempo de autorización de la tarjeta: menos de un segundo, idealmente unas decenas de milisegundos.

## Ejercicio 1: la tarifa de los viajes programados, por lotes

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). La app permite reservar un viaje para el día siguiente. Cada noche, [`predecir_lote.py`](./starter/ejercicio1/predecir_lote.py) estima la tarifa de los 60 viajes reservados en [`viajes_programados.csv`](./starter/ejercicio1/viajes_programados.csv) con la red neuronal que elegiste en la Semana 11, y deja las estimaciones en una tabla que la app lee por la mañana. [`test_predecir_lote.py`](./starter/ejercicio1/test_predecir_lote.py) tiene sus pruebas. Los 5 `TODO`: cargar el modelo, estimar, armar la tabla de salida y dos pruebas.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python predecir_lote.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** sin resolver, `predecir_lote.py` escribe una tabla con una sola columna, y `pytest` dice `1 failed, 2 passed`. ¿Cómo pueden pasar dos pruebas si el script no estima ninguna tarifa?

Porque esas dos pruebas todavía no comprueban nada: `test_una_prediccion_por_viaje` y `test_viaje_incompleto_tambien_se_estima` llaman a `predecir_lote` pero solo tienen el comentario del `TODO`, sin ningún `assert`. pytest da por buena una prueba que termina sin lanzar una excepción, y `predecir_lote` sin resolver no falla: devuelve una tabla con `id_viaje`. La única prueba con `assert`, la de las columnas, es la que falla. Una prueba que pasa no dice que el código esté bien; dice que no se rompió nada de lo que la prueba revisa.

**Respuesta 2 (TODO 1 y 2):** 4 de los 60 viajes no tienen pasajeros y 2 no tienen método de pago. ¿Esperas que el script falle en esos viajes, que los salte, o que les estime una tarifa? ¿Por qué?

Que les **estime una tarifa**. El modelo guardado es un pipeline que incluye la preparación: `pasajeros` se imputa con la mediana y `metodo_pago` con la categoría más frecuente, igual que al entrenar, así que el modelo recibe todas sus columnas completas. Saltarlos dejaría al pasajero sin tarifa por la mañana, y fallar dejaría a los 60 sin tabla por culpa de 6. Así fue: los 60 viajes salieron con tarifa, y los 6 incompletos con valores razonables para su distancia (por ejemplo, 33.94 para 10.6 km sin pasajeros, 60.79 para 20 km sin método de pago).

**Respuesta 3 (TODO 3):** todas las filas de la tabla llevan la misma `version_modelo`. ¿Para qué guardarla en cada fila?

Para que cada estimación se pueda rastrear al modelo que la produjo. La tabla sale del lote y la lee otra aplicación, que no sabe qué archivo `.joblib` había esa noche. Si mañana se reentrena, las filas de hoy y las de mañana conviven en la misma tabla o en el historial, y sin la versión no se sabría con qué modelo se calculó cada tarifa. Con ella se puede investigar un reclamo, comparar dos versiones, o recalcular solo las filas de una versión con problemas. Repetirla en cada fila cuesta poco y hace que cada fila se explique sola, aunque se copie o se filtre.

**Respuesta 4 (TODO 4 y 5):** antes de escribir las dos pruebas, ¿qué cambio en `predecir_lote.py` haría fallar la del `TODO 4`? ¿Y la del `TODO 5`?

**TODO 4** fallaría si `predecir_lote` perdiera o reordenara viajes: por ejemplo, si hiciera `viajes.dropna()` antes de estimar (el viaje sin pasajeros desaparecería y habría 2 filas en vez de 3), si ordenara la salida por tarifa, o si armara `id_viaje` con algo distinto de `viajes["id_viaje"]` (como `range(len(viajes))`).

**TODO 5** fallaría si algún viaje quedara sin tarifa, por ejemplo si se pusiera `NaN` a los viajes con datos faltantes en vez de estimarlos, o si la `version_modelo` no fuera la que se pasa como argumento, por ejemplo si estuviera escrita a mano (`"v1"`) o se calculara dentro de la función con `huella(RUTA_MODELO)`.

**Lo que mostró el lote:**

Modelo `ef6e172e2a2f` (la huella depende del entrenamiento; cambia si se vuelve a correr `entrenar.py`). 60 viajes leídos, 60 filas escritas. Tarifa más baja: **4.67** (viaje 90014, 0.4 km dentro del Upper East Side). Más alta: **70.26** (viaje 90051, Upper East Side a JFK, 24 km). Los 4 viajes sin pasajeros recibieron una tarifa normal para su distancia: 33.94 (90023, 10.6 km), 29.88 (90024, 10.2 km), 25.80 (90025, 8.4 km) y 17.58 (90048, 4.8 km).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: la cola de revisión de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). Cada madrugada, [`puntuar_pagos.py`](./starter/ejercicio2/puntuar_pagos.py) calcula la probabilidad de fraude de los 300 pagos de ayer ([`pagos_ayer.csv`](./starter/ejercicio2/pagos_ayer.csv)) con el gradient boosting de la Semana 11, y arma la cola de revisión manual con los pagos que alcanzan el umbral de 0.10 que elegiste ese día. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO`; el paso 4 (la cola) y el paso 5 (las pruebas de [`test_puntuar_pagos.py`](./starter/ejercicio2/test_puntuar_pagos.py)) solo tienen un comentario que dice qué hacer. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** ¿cuántos pagos quedaron en la cola de revisión? ¿Cuántos fraudes esperarías encontrar en los 300 pagos, si la proporción es la de la Semana 11?

**26 pagos** quedaron en la cola con el umbral 0.10. Con la proporción de la Semana 11 (295 fraudes en 6000 pagos, 4.9%), en 300 pagos se esperarían unos **15 fraudes**. La cola tiene más pagos que fraudes esperados, que es lo buscado con un umbral bajo: como un fraude cuesta mucho más que una revisión, se aceptan falsas alarmas para atrapar más fraudes. Aun así, por lo visto en la Semana 11, algunos fraudes van a quedar fuera de la cola.

**Respuesta 2:** este modelo corre por lotes, una vez al día. ¿Qué pierde la empresa por no evaluar cada pago en el momento en que ocurre? ¿Qué tendría que pasar con la revisión para que eso importe?

Pierde la posibilidad de **frenar el fraude antes de que ocurra**. El lote de la madrugada marca pagos que ya se cobraron hace horas: el viaje ya se hizo, el dinero ya salió, y lo que queda es reclamar o perder el contracargo. Tampoco puede reaccionar en el día a un patrón nuevo (una tarjeta robada que se usa diez veces en la tarde). Para que eso importe, la revisión tendría que poder actuar sobre el pago mientras ocurre: bloquear o pedir verificación antes de aprobarlo. Si la revisión es humana y solo trabaja al día siguiente, evaluar en línea no cambia nada; importa si hay una acción automática en el momento, o un equipo que revise en minutos.

**Respuesta 3:** la prueba de la cola de revisión arma su propia tabla de cuatro pagos con probabilidades escritas a mano, en vez de usar el modelo. ¿Qué gana la prueba con eso?

Prueba solo la lógica de la cola (filtrar por el umbral, incluir el límite y ordenar), aislada del modelo. Con probabilidades escritas a mano se sabe de antemano cuál es la respuesta correcta (4, 2 y 3), así que el `assert` puede ser exacto. Con el modelo real, la respuesta cambiaría cada vez que se reentrena y no se sabría si una diferencia es un error del código o un modelo distinto. Además no necesita el archivo `.joblib`, que no está en el repositorio, así que corre en cualquier máquina y en GitHub Actions, en milisegundos. Y puede incluir a propósito el caso borde (un pago justo en 0.10) que con datos reales quizás nunca aparece.

**Si terminas antes:** sube `UMBRAL` a 0.50 y vuelve a correr `puntuar_pagos.py`. ¿Cuántos pagos quedan en la cola? ¿Siguen pasando todas las pruebas? ¿Por qué?

Con 0.50 quedan **13 pagos** en la cola (la mitad de los 26). Las pruebas **siguen pasando** (4 de 4) porque ninguna depende de `UMBRAL`: la de la cola llama a `a_revisar` con su propio umbral de 0.10 sobre su tabla escrita a mano, y las demás revisan la forma de la salida con un modelo de prueba. Las pruebas protegen que el código haga lo que dice, no la decisión de negocio. Que 0.10 sea el umbral correcto se decidió con el costo de cada error en la Semana 11; si se quisiera proteger ese valor, haría falta una prueba propia.

## Ejercicio 3: el despliegue y las pruebas de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md). Esta vez no hay plantilla.

### Entregable 1: `docs/despliegue.md`

Una página que diga si tu modelo se sirve por lotes o en línea, y por qué:

- Qué decisión habilita la predicción, quién la usa y cuánto tiempo puede esperar por ella (un número: milisegundos, horas, un día).
- Cuántos casos distintos hay que predecir, y si se conocen de antemano.
- Qué datos necesitan tus features en el momento de predecir, y si están disponibles en ese momento.
- Qué se pierde con el patrón que descartas.

### Entregable 2: las primeras pruebas de tu proyecto

Un archivo `psets/proyecto/modelado/test_pipeline_proyecto.py` con al menos tres pruebas sobre tu `pipeline_proyecto.py`, cada una con al menos un `assert`. Las pruebas arman sus propios datos (un `DataFrame` de pocas filas escrito a mano), no leen tu archivo de `data/`: en GitHub Actions ese archivo no existe. Ideas de qué probar: que la columna objetivo no está entre las que recibe el modelo; que una feature calculada da el valor que sabes calcular a mano; que una feature que mira hacia atrás en el tiempo no usa una fila del futuro; que la preparación acepta una categoría que no vio al entrenar.

Comprueba desde la raíz de tu fork que pasan, con el mismo comando que va a usar GitHub:

```
pytest psets/proyecto -v
```

### Entregable 3: la integración continua de tu fork

El archivo [`.github/workflows/pruebas.yml`](../../.github/workflows/pruebas.yml) le pide a GitHub que, en cada push, cree una máquina limpia, instale `requirements.txt` y corra `pytest psets/proyecto`. En tu fork viene apagado: en la página de tu fork, pestaña **Actions**, presiona "I understand my workflows, go ahead and enable them". Haz push de tus pruebas y espera la marca verde junto a tu commit.

**Pega el enlace a la ejecución de GitHub Actions donde pasan tus pruebas:**

_(tu respuesta)_

**¿Qué patrón elegiste para tu proyecto, y qué número de latencia lo decide?**

**Por lotes.** El monto estimado de un award se muestra junto a la recomendación para que el investigador decida si lee la convocatoria, y puede esperar **un día (24 horas)**: las convocatorias duran semanas y los datos de OpenAlex llegan con días de retraso. Todos los casos se conocen de antemano (los 3830 awards sin monto de los 4000 de la OBT), y todas las features salen del registro del award en el warehouse, sin nada que haga el usuario en el momento. El detalle está en [`psets/proyecto/docs/despliegue.md`](../../psets/proyecto/docs/despliegue.md).

**Rompe a propósito una de tus pruebas (cambia un valor esperado), haz push y mira qué pasa en GitHub. Después arréglala. ¿Qué te dijo GitHub, y dónde lo viste?**

_(tu respuesta)_

## Antes de la próxima sesión

**Lectura:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

Hoy la app lee las tarifas de una tabla calculada de noche. Pero cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un **servicio de predicción en línea**: una API HTTP (por ejemplo con FastAPI) siempre levantada, que cargue el modelo una sola vez al arrancar y exponga un endpoint como `POST /tarifa`. La app le mandaría un JSON con las features del viaje en ese momento: zona de origen y destino, distancia estimada de la ruta, pasajeros, método de pago, la hora (para `hora_sin` y `hora_cos`), si es fin de semana, hora pico o tarifa fija. El servicio validaría esos campos, aplicaría el mismo pipeline de preparación con que se entrenó y devolvería la tarifa estimada y la `version_modelo`, en menos de unos 200 ms. Haría falta también desplegarlo en algún lugar (un contenedor) y probar el endpoint como hoy se prueba el lote.
