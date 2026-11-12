# Semana 12, Sesión 2: Despliegue de modelos como servicio de predicción

> Objetivo de la sesión: exponer el modelo entrenado de tu proyecto como un servicio de predicción mediante una API, empaquetado en un contenedor, con pruebas de su contrato de entrada y salida.

**Lectura previa de esta sesión:** ["Journey to Full-Stack Data Scientist: Model Deployment"](https://towardsdatascience.com/journey-to-full-stack-data-scientist-model-deployment-f385f244ec67/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 5 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

Nada nuevo que instalar: FastAPI y uvicorn ya llegaron con el contenedor reconstruido la sesión pasada. Los laboratorios de hoy son scripts y un `Dockerfile`. El servicio escucha en el puerto 8000: cuando lo levantes, aparece en la pestaña **Ports** como "Servicio de predicción (Semana 12)", y su documentación interactiva queda en la ruta `/docs` de ese puerto.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué pasos separan un modelo entrenado en un notebook de un modelo que otra aplicación puede usar?**

Primero sacar el modelo del notebook: entrenarlo y guardarlo serializado en un archivo (pickle o joblib) junto con su preparación de datos. Después envolverlo en una API web que lo cargue al arrancar, reciba la entrada como JSON, la valide, calcule las features igual que en el entrenamiento y devuelva la predicción. Luego empaquetar esa API con sus dependencias en un contenedor de Docker, y por último desplegar ese contenedor en un servidor o en la nube para que otra aplicación lo llame por HTTP.

**¿Para qué usa el artículo un contenedor, si el servicio ya corre en la máquina de quien lo escribió?**

Para que el servicio corra igual en cualquier máquina, no solo en la de quien lo escribió. La imagen lleva la versión exacta de Python y de las librerías (scikit-learn incluida, que tiene que ser la misma con que se entrenó el modelo), el código y el modelo; así el servidor o la nube donde se despliega no depende de lo que tenga instalado, y se puede levantar, mover o replicar sin el "en mi máquina sí funciona".

## Pregunta de apertura

La sesión pasada, la app leía las tarifas de una tabla calculada de noche. Cuando un pasajero escribe un destino que no estaba reservado, no hay ninguna fila que leer. ¿Qué tendría que existir para que la app le pregunte al modelo en ese momento, y qué le mandaría?

Un servicio de predicción en línea: una API HTTP (con FastAPI) siempre levantada, que cargue el modelo una sola vez al arrancar y exponga una ruta como `POST /predecir`. La app le mandaría un JSON con lo que sabe del viaje en ese momento: fecha y hora, zona de origen y de destino, distancia, pasajeros y método de pago. El servicio validaría esos campos, calcularía el resto de las features con el mismo código del entrenamiento y devolvería la tarifa estimada en milisegundos. Para que corra en otro lado que no sea mi máquina, ese servicio va dentro de un contenedor.

## Ejercicio 1: la tarifa, como servicio

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). [`servicio.py`](./starter/ejercicio1/servicio.py) es una API de FastAPI con dos rutas: `/salud`, que dice si el servicio está vivo y tiene su modelo, y `/predecir`, que recibe un viaje como JSON y devuelve la tarifa estimada. El viaje llega como lo conoce la app (fecha y hora, zonas, distancia, pasajeros, método de pago), y [`features_taxis.py`](./starter/ejercicio1/features_taxis.py) calcula el resto de las features igual que se calcularon para entrenar. [`test_servicio.py`](./starter/ejercicio1/test_servicio.py) prueba la API sin levantarla, y el [`Dockerfile`](./starter/ejercicio1/Dockerfile) la empaqueta. Los 5 `TODO`: el esquema de entrada, la predicción, dos pruebas y el comando del contenedor.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
pytest -v
uvicorn servicio:app --port 8000
```

Con el servicio levantado, abre el puerto 8000 desde la pestaña **Ports**, agrega `/docs` a la dirección y prueba `/predecir` con **Try it out**. También desde otra terminal:

```
curl -X POST localhost:8000/predecir -H "Content-Type: application/json" -d '{"fecha_hora_inicio": "2026-11-05 08:30", "zona_origen": "Midtown", "zona_destino": "JFK Airport", "distancia_km": 26.4, "pasajeros": 2, "metodo_pago": "tarjeta"}'
```

**Respuesta 1 (antes de los TODO):** sin resolver, `/salud` dice que el modelo está cargado, `/predecir` responde 200 y `pytest` dice `3 passed`. ¿Qué de esa salida debería preocuparte?

Que todo dice que funciona y nada funciona. `/predecir` responde 200 con `{"tarifa_estimada": 0.0}` para el viaje de Midtown a JFK: el TODO devuelve un valor fijo y nunca llama al modelo. `/salud` solo comprueba que el archivo se cargó, no que las predicciones tengan sentido, y las 3 pruebas pasan porque todavía no verifican nada (los TODO de las pruebas están vacíos). Un servicio "sano" con pruebas en verde puede estar devolviendo basura.

**Respuesta 2 (TODO 1):** con el esquema completo, ¿qué esperas que responda el servicio a un viaje sin `distancia_km`? ¿Y a uno sin `zona_origen`?

Sin `distancia_km` espero un 200 con una tarifa: en el esquema la distancia es opcional (la app no siempre la conoce antes del viaje) y la preparación del modelo la imputa. En el contenedor respondió `{"tarifa_estimada": 68.56}`, contra 70.39 con la distancia. Sin `zona_origen` espero un 422: es un campo obligatorio del esquema, así que FastAPI rechaza la petición antes de llegar al modelo.

**Respuesta 3 (TODO 2):** el modelo se entrenó con `hora_sin`, `hora_cos` y `es_hora_pico`, pero la app manda solo la fecha y hora. ¿Qué pasaría si el servicio calculara `es_hora_pico` con otras horas que las del entrenamiento?

El modelo recibiría una feature que significa otra cosa que en el entrenamiento (training-serving skew): un viaje de las 8:30 entre semana, que en el entrenamiento era hora pico (horas 7–9 y 16–19 de lunes a viernes), llegaría marcado como no pico y el modelo le aplicaría el efecto equivocado. No da ningún error: responde 200 con una tarifa sesgada, y nadie se entera. Por eso el servicio importa `agregar_features` de `features_taxis.py`, el mismo archivo del entrenamiento, en vez de recalcular las horas por su cuenta.

**Respuesta 4 (TODO 3 y 4):** el código 422 del viaje sin zona de origen, ¿lo produce tu código o FastAPI? ¿La petición llega al modelo?

Lo produce FastAPI, no mi código: al validar el JSON contra el esquema `Viaje` (Pydantic), ve que falta `zona_origen`, que es obligatorio, y responde 422 con `loc: ["body", "zona_origen"]`. La petición no llega al modelo: la función `predecir` ni siquiera se ejecuta.

**Respuesta 5 (TODO 5):** construye y corre la imagen con el `CMD` sin resolver (`docker build -t servicio-tarifa .` y `docker run -p 8000:8000 servicio-tarifa`). El log dice que uvicorn está corriendo, pero `curl localhost:8000/salud` falla. ¿Por qué?

Porque el `CMD` sin resolver no le pasa `--host`, y uvicorn escucha por defecto en `127.0.0.1`, el localhost **del contenedor**. El log dice "Uvicorn running on http://127.0.0.1:8000", pero las peticiones que llegan por `-p 8000:8000` entran por la interfaz de red del contenedor, no por su localhost, así que nadie las atiende (`curl` termina con "Empty reply from server"). Con `--host 0.0.0.0` escucha en todas las interfaces y `/salud` responde.

**Lo que respondió el servicio:**

Viaje de Midtown a JFK Airport (2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta), con el servicio en el contenedor: `{"tarifa_estimada": 70.39}`.

El mismo viaje sin `zona_origen`, HTTP 422:

```
{"detail":[{"type":"missing","loc":["body","zona_origen"],"msg":"Field required","input":{"fecha_hora_inicio":"2026-11-05 08:30","zona_destino":"JFK Airport","distancia_km":26.4,"pasajeros":2,"metodo_pago":"tarjeta"}}]}
```

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, en el momento del pago

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). La sesión pasada, los pagos se revisaban al día siguiente. Ahora la empresa quiere decidir en el momento del cobro si un pago va a revisión. [`servicio_fraude.py`](./starter/ejercicio2/servicio_fraude.py) expone el mismo gradient boosting en `/evaluar_pago`, que devuelve la probabilidad de fraude y la decisión. Mismo esqueleto del Ejercicio 1: los pasos 1 a 3 tienen `TODO` (el esquema, la probabilidad y la prueba del pago válido); el paso 4 (la decisión con el umbral) y el paso 5 (la prueba del pago sin monto) solo tienen un comentario. El `Dockerfile` ya está completo. Primero `python entrenar_fraude.py`; después resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`.

**Respuesta 1:** levanta el servicio y evalúa dos pagos: uno de madrugada, por app, con tarjeta nueva, dos intentos fallidos y ningún viaje previo; otro de la tarde, con tarjeta conocida y 22 viajes previos. ¿Qué probabilidad y qué decisión recibió cada uno?

- Pago de madrugada (02:40, app, 41.30, 8.6 km, tarjeta nueva, 2 intentos fallidos, 0 viajes previos): `{"probabilidad_fraude": 0.9996, "revisar": true}`, va a revisión.
- Pago de la tarde (16:20, tarjeta, 18.50, 6.2 km, tarjeta conocida, 0 intentos fallidos, 22 viajes previos): `{"probabilidad_fraude": 0.0165, "revisar": false}`, se cobra sin revisión.

**Respuesta 2:** el servicio devuelve la decisión (`revisar`) además de la probabilidad. ¿Qué gana la app con eso? ¿Qué tendría que cambiar si la empresa decide mañana que el umbral es 0.20?

La app no tiene que conocer el umbral ni la regla de negocio: recibe la decisión y actúa (cobra o manda a revisión). Así la regla vive en un solo lugar y todas las apps (iOS, Android, web) deciden igual; la probabilidad queda para registrarla o explicarla. Si mañana el umbral es 0.20, solo cambia `UMBRAL` en `servicio_fraude.py` (o, mejor, una variable de entorno), se vuelve a construir y desplegar el servicio, y la prueba del pago válido se revisa si depende del umbral. La app no cambia y el modelo no se reentrena.

**Respuesta 3:** el mismo pago, evaluado por este servicio y por el lote de la sesión pasada, ¿da la misma probabilidad? ¿Qué tiene que ser igual en los dos para que así sea?

Sí. Lo comprobé: reentrené el modelo del lote de la sesión pasada con las mismas versiones de librerías y puntué los 300 pagos de `pagos_ayer.csv` con los dos modelos; la diferencia máxima entre probabilidades fue 0.0. Para que así sea tiene que ser igual: el modelo (mismo `entrenar_fraude.py`, mismos `pagos_taxi.csv` y misma semilla, o directamente el mismo `.joblib`), la preparación de features (`preparacion_pagos.py` es idéntico en las dos carpetas) y las versiones de scikit-learn y numpy. Lo último no es teórico: el `.joblib` del lote entrenado en otro entorno ni siquiera cargó con otra versión de numpy (`MT19937 is not a known BitGenerator module`).

**Si terminas antes:** manda a `/evaluar_pago` un pago con `"monto": "cuarenta"`. ¿Qué responde el servicio, y qué línea de `servicio_fraude.py` produce esa respuesta?

Responde 422, sin llegar al modelo:

```
{"detail":[{"type":"float_parsing","loc":["body","monto"],"msg":"Input should be a valid number, unable to parse string as a number","input":"cuarenta"}]}
```

La línea que lo produce es `monto: float` en la clase `Pago`: Pydantic intenta convertir `"cuarenta"` a número al validar el cuerpo, falla, y FastAPI devuelve el 422. `evaluar_pago` no se ejecuta.

## Ejercicio 3: el servicio de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el modelo final que registraste en MLflow (Semana 11). Esta vez no hay plantilla. Aunque tu proyecto se despliegue por lotes, el servicio también se construye: es la forma en que otro sistema (tu propio lote, un tablero, otra aplicación) le pide una predicción a tu modelo.

### Entregable 1: el servicio, en `psets/proyecto/servicio/`

- Un script que exporte tu modelo final desde MLflow a `servicio/modelo.joblib` (`mlflow.sklearn.load_model("runs:/<run_id>/modelo")` y `joblib.dump`). El archivo no se sube: se regenera con ese script.
- `servicio/api.py`, con `/salud` y una ruta de predicción. El esquema de entrada recibe los datos como los conoce quien llama, y el servicio calcula las features con las mismas funciones de tu `modelado/pipeline_proyecto.py`, importadas desde ahí (`from modelado.pipeline_proyecto import ...`), no copiadas.
- `servicio/Dockerfile` y `servicio/requirements.txt`, con la misma versión de scikit-learn con que entrenaste. La imagen se construye desde `psets/proyecto/`, porque necesita `modelado/` además de `servicio/`: `docker build -f servicio/Dockerfile -t <tu-servicio> .`

El servicio corre desde `psets/proyecto/`: `uvicorn servicio.api:app --port 8000`. El archivo [`psets/proyecto/conftest.py`](../../psets/proyecto/conftest.py) hace que las pruebas también encuentren `modelado/` desde ahí.

### Entregable 2: `servicio/test_api.py`

Al menos tres pruebas con `TestClient`: `/salud`, una predicción válida con un modelo de prueba (no tu `modelo.joblib`, que no existe en GitHub Actions) y una entrada inválida que recibe 422. El workflow de la sesión pasada ya corre todo lo que hay en `psets/proyecto`: en tu próximo push, GitHub corre también estas pruebas.

### Entregable 3: respuestas

En `proyecto/README.md`, agrega la sección **Cómo correr el servicio**: cómo se exporta el modelo, cómo se construye y se levanta el contenedor, y un ejemplo de petición con su respuesta.

**¿Qué datos recibe tu servicio, y cuáles de tus features calcula él mismo? ¿Cómo sabes que las calcula igual que en el entrenamiento?**

Recibe el award como está en el warehouse: `funder_name` (obligatorio), `funding_type`, `duration_days`, `institutions_count` y `has_description`. El servicio calcula `duracion_anios` (días / 365.25), `escala_financiamiento` (a partir de `funding_type`, con "proyecto" si no viene) y `has_description` como 0/1; la imputación, el escalado y el one-hot los hace el pipeline guardado en el modelo. Sé que las calcula igual porque `servicio/api.py` importa `agregar_features` y `FEATURES` de `modelado/pipeline_proyecto.py`, no las copia. Además lo verifiqué: corrí `agregar_features` sobre `obt_award_work.csv` y comparé con `obt_features.csv` de la Semana 9: las 210 filas coinciden en `duracion_anios`, `escala_financiamiento`, `has_description` y `log_amount_usd`. Y la respuesta del contenedor (313691.79 USD) es la misma que da `modelo.predict` directo sobre ese award. `test_api.py` comprueba que el modelo recibe exactamente las columnas `FEATURES` con los valores esperados.

**Pega una petición a tu servicio corriendo en el contenedor, y su respuesta:**

```
$ curl -X POST localhost:8013/estimar_monto -H "Content-Type: application/json" \
    -d '{"funder_name": "National Science Foundation", "funding_type": "grant", "duration_days": 1094, "institutions_count": 1, "has_description": true}'
{"monto_estimado_usd":313691.79,"log10_monto":5.4965,"version_modelo":"505f6532318444f3a3fc7a1f03785738"}
```

Un award incompleto también se estima (sin `institutions_count` ni descripción): `{"funder_name": "National Institutes of Health", "funding_type": "fellowship", "duration_days": 730}` → `{"monto_estimado_usd":218292.01,"log10_monto":5.339,"version_modelo":"505f6532318444f3a3fc7a1f03785738"}`. Sin `funder_name` responde 422 con `loc: ["body", "funder_name"]`.

## Antes de la próxima sesión

**Lectura:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

Tu servicio carga `modelo.joblib`, el archivo que exportaste hoy. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

Cada respuesta del servicio trae `version_modelo`, que es el `run_id` de MLflow del modelo cargado (`exportar_modelo.py` lo deja en `modelo_info.json` y `/salud` también lo informa). Si la app o el lote guardan esa versión junto a cada predicción (como pide `docs/despliegue.md`), ante un reclamo se busca la predicción, se ve su `run_id` y en MLflow están los parámetros, métricas, datos y artefactos de ese run. Para volver al modelo anterior no hace falta reentrenar: MLflow conserva todos los runs, así que se corre `python servicio/exportar_modelo.py <run_id_anterior>`, se reconstruye la imagen y se despliega; mejor todavía, se mueve la etiqueta `candidato=final` al run anterior para que el registro diga cuál es el vigente. Etiquetar cada imagen con el `run_id` permite además volver a la imagen anterior sin reconstruir.
