# Semana 13, Sesión 1: Despliegue en producción: versionado y registro de modelos

> Objetivo de la sesión: versionar y registrar el modelo desplegado de tu proyecto, explicando qué problema resuelve un registro de modelos frente a sobrescribir el mismo archivo cada vez que se reentrena.

**Lectura previa de esta sesión:** ["What Is Model Governance?"](https://www.ibm.com/think/topics/model-governance), IBM.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 10: Infrastructure and Tooling for MLOps.

## Requisitos antes de esta sesión

Nada nuevo que instalar. El registro de modelos es parte del mismo servidor de MLflow que usas desde la Semana 11: se levanta solo al iniciar el Codespace, en el puerto 5050 (si no responde, `bash .devcontainer/mlflow-server.sh`). Los scripts lo encuentran por la variable `MLFLOW_TRACKING_URI`. Los laboratorios de hoy son scripts.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la gobernanza de modelos, y qué componentes la forman?**

Es el proceso de punta a punta para definir, aplicar y mantener controles sobre cómo se construye, valida, usa y retira cada modelo, de modo que funcione como se espera, se pueda auditar y cumpla la regulación. Según IBM la forman: (1) desarrollo y documentación (objetivo, datos, metodología y supuestos), (2) un inventario central de modelos, (3) validación independiente y back-testing (sesgo, robustez, overfitting), (4) despliegue con monitoreo continuo (degradación, drift, calidad de datos), y (5) roles, responsabilidades y flujos claros (quién desarrolla, quién valida, quién aprueba y quién lo retira).

**El artículo pide un inventario central de modelos. ¿Qué datos tendría que tener ese inventario sobre cada modelo, y para qué los necesita alguien que no lo entrenó?**

Por cada modelo: nombre y propósito (qué decisión apoya), dueño, versión y estado en el ciclo de vida (en desarrollo, en producción, retirado), con qué datos se entrenó (fuente, fechas, huella), metodología y supuestos, métricas de validación y quién lo validó, dónde está desplegado y desde cuándo, y sus dependencias. Alguien que no lo entrenó (un auditor, otro equipo, la persona que hereda el proyecto) lo necesita para saber si puede confiar en sus salidas, reproducirlo, explicar una predicción concreta, saber a quién preguntar y decidir si hay que reentrenarlo o retirarlo, sin tener que reconstruir todo desde el código.

## Pregunta de apertura

La sesión pasada, tu servicio cargaba `modelo.joblib`. Si la semana que viene reentrenas y vuelves a exportar, el archivo se reemplaza. Si mañana un usuario reclama por una predicción de hoy, ¿cómo sabrías qué modelo la produjo? ¿Y cómo volverías al modelo anterior si el nuevo resulta peor?

No lo sabría: si el archivo se sobrescribe, la predicción de hoy no guarda nada que la ate a un modelo concreto, y el modelo que la produjo ya no existe en disco. Tampoco podría volver atrás, salvo reentrenando con los mismos datos y el mismo código, y esperando que salga igual. Para resolver las dos cosas, cada modelo tiene que quedar guardado con un identificador que no se reutilice (una versión), cada predicción tiene que llevar ese identificador, y “cuál está en producción” tiene que ser un puntero que se pueda mover a otra versión sin borrar ninguna.

## Ejercicio 1: la tarifa, con registro

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa de la sesión pasada, ahora con el modelo en el registro de MLflow en vez de un archivo. [`registrar_modelo.py`](./starter/ejercicio1/registrar_modelo.py) entrena la red neuronal con el número de neuronas que le pases y la registra como una versión nueva del modelo `tarifa-taxis`. [`promover.py`](./starter/ejercicio1/promover.py) pone el alias `produccion` en la versión que le pases. [`versiones.py`](./starter/ejercicio1/versiones.py) lista las versiones con su error y sus alias (no lo modifiques). [`servicio.py`](./starter/ejercicio1/servicio.py) carga la versión que tiene el alias `produccion`. Los 4 `TODO`: registrar, promover, cargar desde el registro y responder qué versión calculó cada tarifa.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python registrar_modelo.py 64
python versiones.py
python promover.py 1
pytest -v
```

**Respuesta 1 (antes de los TODO):** `registrar_modelo.py` dice `Version None`, `versiones.py` no imprime nada y `promover.py` termina con un error. ¿Qué dice el error, y qué te dice sobre lo que hay en el registro?

`promover.py 1` termina con `mlflow.exceptions.RestException: INVALID_PARAMETER_VALUE: Registered model alias produccion not found.` Sin el TODO 2 el script no pone ningún alias y solo intenta leerlo. Pero aunque lo pusiera, el registro está vacío: sin el TODO 1, `registrar_modelo.py` guarda la métrica en una corrida pero no registra ningún modelo (por eso imprime `Version None` y `versiones.py` no lista nada). No hay versiones, así que no hay nada a qué ponerle el alias.

**Respuesta 2 (TODO 1):** registra dos versiones, una con 64 neuronas y otra con 8. Antes de correr `versiones.py`, ¿cuántas versiones esperas ver, y qué pasó con la primera cuando registraste la segunda?

Esperaba ver dos versiones, y eso salió: `Version 1: red de 64 neuronas, MAE 1.1695` y `Version 2: red de 8 neuronas, MAE 1.1877`, las dos sin alias. A la primera no le pasó nada al registrar la segunda: sigue ahí con su corrida, su modelo y su métrica. El registro le dio a la nueva el número siguiente en vez de sobrescribirla, a diferencia de `modelo.joblib`.

**Respuesta 3 (TODO 2 y 3):** pon `produccion` en la versión 2 y levanta el servicio (`uvicorn servicio:app --port 8000`). ¿Qué versión dice `/salud`? Ahora mueve el alias a la versión 1 sin reiniciar uvicorn: ¿qué versión responde el servicio? ¿Y después de reiniciarlo?

Con `produccion` en la versión 2, `/salud` dice `"version_modelo": "2"`. Al mover el alias a la 1 sin reiniciar uvicorn, el servicio sigue respondiendo con la versión 2: el modelo se carga una sola vez, al importar `servicio.py`, y ya no vuelve a mirar el registro. Después de reiniciar responde la versión 1. Mover el alias no cambia nada hasta que el servicio arranca de nuevo, así que un rollback es mover el alias y reiniciar (o volver a desplegar).

**Respuesta 4 (TODO 4):** las pruebas usan un modelo de prueba y una versión inventada, `"7"`, sin servidor de MLflow. ¿Qué prueba falla antes del `TODO 4`, y qué comprueba que el `/salud` no comprobaba?

Antes del TODO 4 falla `test_cada_prediccion_dice_que_version_la_calculo`, porque la respuesta de `/predecir` era solo `{"tarifa_estimada": 20.0}`, sin `version_modelo`. `/salud` dice qué versión tiene cargada el servicio en este momento, pero no deja rastro en cada tarifa. Si el alias cambia y el servicio se reinicia, `/salud` ya no dice qué versión calculó una tarifa de ayer. La prueba nueva comprueba que cada predicción lleve la versión que la calculó, que es lo que hace falta para responder un reclamo.

**Lo que respondió el servicio:**

Viaje Midtown → JFK Airport, 2026-11-05 08:30, 26.4 km, 2 pasajeros, tarjeta:

- Con la versión 2 (8 neuronas): `{"tarifa_estimada": 69.98, "version_modelo": "2"}`.
- Con la versión 1 (64 neuronas), después de mover el alias y reiniciar: `{"tarifa_estimada": 70.07, "version_modelo": "1"}`.
- Con el alias ya en la 1 pero sin reiniciar, seguía respondiendo `69.98` con `"version_modelo": "2"`.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el ciclo de vida del modelo de fraude

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El equipo de fraude entrena modelos candidatos cada tanto, y quiere una regla: una versión llega a `produccion` solo si detecta más fraudes (mayor recall al umbral de 0.10) que la que está ahí. [`ciclo_fraude.py`](./starter/ejercicio2/ciclo_fraude.py) entrena tres candidatos, registra cada uno y decide si lo promueve. Los pasos 1 a 3 tienen `TODO`: registrar una versión con sus métricas, leer la métrica de una versión y averiguar qué versión está en producción. Los pasos 4 (decidir y mover el alias) y 5 (etiquetar la versión con el resultado) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python ciclo_fraude.py`.

**Respuesta 1:** ¿qué versión quedó en producción, y por qué no la última que se registró?

Quedó en producción la **versión 2** (gradient boosting, recall 0.661 y precisión 0.429). La 1 (logística, recall 0.644) entró porque no había ninguna en producción. La 2 la reemplazó porque detecta más fraudes (0.661 > 0.644). La 3, la última registrada (logística muy regularizada), tiene recall 0.102 y la regla la rechazó: llegar último no da prioridad, lo que cuenta es mejorar la métrica que la regla compara.

**Respuesta 2:** la regresión logística muy regularizada tiene la mayor precisión de las tres. ¿Por qué la regla la rechazó? ¿Qué tendría que cambiar en la regla si la empresa decide que las falsas alarmas cuestan más que los fraudes no detectados?

Porque la regla compara solo el recall al umbral de 0.10, y esa logística detecta apenas el 10% de los fraudes (recall 0.102 contra 0.661 de la versión 2). Su precisión de 0.667 es alta porque casi no da alertas: acierta en las pocas que da y deja pasar el resto. Si las falsas alarmas cuestan más que los fraudes no detectados, la regla tendría que comparar otra métrica. Una opción es la precisión, con un recall mínimo para no promover un modelo que no detecta nada. Mejor aún, un costo esperado con los dos errores pesados por lo que le cuestan a la empresa (por ejemplo, costo = c_FP·FP + c_FN·FN, con c_FP > c_FN). También habría que revisar el umbral de 0.10, que hoy está elegido para atrapar fraudes y no para evitar alarmas.

**Respuesta 3:** abre el modelo `fraude-pagos` en la interfaz de MLflow (puerto 5050, pestaña **Models**). Si dentro de un mes alguien pregunta por qué la versión 3 nunca llegó a producción, ¿dónde está la respuesta?

En el registro, en la versión 3 de `fraude-pagos`. Sus etiquetas dicen `validacion = rechazada` y `comparada_con = 2`, es decir, contra qué versión perdió. Desde la versión se llega a la corrida que la produjo (`logistica-muy-regularizada`, en el experimento `fraude-pagos-ciclo`), con su recall de 0.102 y su precisión de 0.667. Junto al recall de 0.661 de la versión 2, eso explica la decisión sin tener que preguntarle a nadie.

**Si terminas antes:** vuelve a correr `python ciclo_fraude.py`. ¿Cuántas versiones hay ahora, y cuál está en producción? ¿Es lo que esperabas?

Hay **6 versiones**, y en producción sigue la **versión 2**. Es lo esperable: los candidatos se reentrenan con los mismos datos y las mismas semillas, así que salen idénticos (versiones 4, 5 y 6 con recall 0.644, 0.661 y 0.102). La 5 empata con la 2, y la regla exige recall estrictamente mayor, así que no la reemplaza, y las tres quedan etiquetadas como `rechazada` y `comparada_con = 2`. El costo es que el registro acumula versiones duplicadas: cada corrida crea versiones aunque no cambie nada.

## Ejercicio 3: el registro de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio que construiste la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: el modelo, en el registro

- `servicio/registrar_modelo.py`: entrena tu modelo final, guarda su métrica principal en la corrida y lo registra como una versión nueva de un modelo con el nombre de tu proyecto. Cada vez que lo corres crea una versión; ninguna reemplaza a otra.
- `servicio/promover.py`: pone el alias `produccion` en la versión que le pases.
- Al menos dos versiones registradas (por ejemplo, con dos valores de un hiperparámetro), y una de ellas con el alias `produccion`.

Corre los scripts desde `psets/proyecto/` como módulos (`python -m servicio.registrar_modelo`), para que encuentren `modelado/`.

### Entregable 2: el servicio sabe qué versión sirve

- El script que exportaba el modelo (Semana 12) ahora exporta la versión con el alias `produccion`, y guarda su número junto al modelo (por ejemplo, en `servicio/modelo_version.txt`). La imagen lleva los dos archivos adentro: el contenedor no necesita llegar al servidor de MLflow.
- `/salud` y cada predicción responden `version_modelo`.
- Una prueba en `servicio/test_api.py` que compruebe que la predicción incluye la versión, con un modelo de prueba y una versión inventada.

### Entregable 3: `docs/despliegue.md`

Agrega la sección **Versionado y registro**: cómo se llama tu modelo en el registro, qué versión está en producción y con qué métrica, con qué criterio se mueve el alias, y los pasos para volver a la versión anterior.

**¿Con qué criterio decides que una versión nueva de tu modelo pasa a producción? ¿Quién lo decide en tu proyecto, y dónde queda registrada la decisión?**

Una versión nueva de `grantmatch-monto` pasa a producción si baja el MAE en log10 sobre el mismo corte por fecha (prueba desde 2020-01-01) y con la misma huella de datos, o si llegaron datos nuevos y el MAE no empeora. Además se revisa que no aumente la subestimación de los awards grandes, que es el error caro de GrantMatch. Hoy está la versión 1 (ridge `alpha=10`, MAE 0.3608), y la versión 2 (`alpha=1`, MAE 0.3832) quedó registrada sin alias. Lo decide el equipo del proyecto. La decisión queda en el registro de MLflow (qué versión tiene el alias y las etiquetas de cada versión) y el criterio está escrito en `docs/despliegue.md`.

**Si tu servicio corre en un contenedor sin acceso al servidor de MLflow, ¿cómo sabe qué versión está sirviendo?**

Porque la versión viaja con el modelo. `servicio/exportar_modelo.py` baja la versión con el alias `produccion`, la guarda en `servicio/modelo.joblib` y escribe su número en `servicio/modelo_version.txt`. El Dockerfile copia los dos archivos a la imagen, el servicio lee el número al arrancar y lo devuelve en `/salud` y en cada respuesta de `/estimar_monto`. El contenedor nunca consulta MLflow: la versión que sirve es la que se exportó al construir la imagen.

## Antes de la próxima sesión

**Lectura:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Se degrada en silencio: la latencia y `/salud` solo dicen que el servicio responde, no que las respuestas sigan siendo buenas. Si cambia la distribución de los datos que llegan (data drift: otros funders, otras duraciones) o la relación entre las features y el monto (concept drift), el modelo sigue respondiendo rápido pero con más error, y nadie lo nota hasta que alguien reclama. Para enterarme hay que monitorear la calidad, no solo la infraestructura. Cuando lleguen los montos reales, comparar las estimaciones con ellos, agrupando por `version_modelo`. Mientras tanto, vigilar la distribución de las features de entrada y de las predicciones frente a las del entrenamiento, con alertas cuando se alejen.
