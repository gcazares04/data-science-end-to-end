# Semana 13, Sesión 2: Monitoreo en producción I: métricas de rendimiento

> Objetivo de la sesión: definir las métricas de monitoreo del modelo de tu proyecto en producción, distinguiendo métricas de sistema (latencia, disponibilidad) de métricas de modelo (desempeño).

**Lectura previa de esta sesión:** ["Machine learning model monitoring: Best practices"](https://www.datadoghq.com/blog/ml-model-monitoring-in-production-best-practices/), Datadog.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Cap. 8: Data Distribution Shifts and Monitoring.

## Requisitos antes de esta sesión

Nada nuevo que instalar. Los laboratorios de hoy son scripts. El Ejercicio 1 no necesita levantar uvicorn: `simular_noviembre.py` le manda las peticiones al servicio con `TestClient`, como las pruebas.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿por qué no alcanza con vigilar la salud y el volumen de peticiones del servicio? ¿Qué hay que medir además?**

Porque un servicio puede estar sano (responde, sin errores, con volumen normal) y aun así dar predicciones malas: el modelo falla en silencio. Hay que medir además la calidad de los datos que entran, la distribución de las predicciones y, cuando llega la verdad, el desempeño del modelo con la métrica de negocio.

**El artículo dice que la verdad (ground truth) a veces llega con demora, o no llega. ¿Qué propone medir mientras tanto?**

Mientras no hay verdad, propone vigilar proxies: el drift de los datos de entrada y de las predicciones (que su distribución se aleje de la de entrenamiento), y la calidad de los datos (faltantes, tipos, rangos). No dicen que el modelo empeoró, pero avisan antes de que la verdad lo confirme.

## Pregunta de apertura

Tu servicio responde en unos milisegundos, `/salud` dice que todo está bien, y cada tarifa dice qué versión la calculó. ¿Qué le pasa a un modelo que responde rápido pero cada vez peor? ¿Cómo te enterarías?

Nadie se entera desde el servicio: la latencia, `/salud` y la versión siguen iguales. Solo se ve uniendo cada predicción con lo que de verdad pasó (la tarifa cobrada) y midiendo el error por período. Para eso hay que guardar cada predicción con un identificador que permita esa unión.

## Ejercicio 1: la tarifa, instrumentada

Carpeta: [`starter/ejercicio1/`](./starter/ejercicio1/). Es el servicio de la tarifa con la versión 1 del modelo (la red de 64 neuronas), ahora con dos agregados. Un middleware (una función que corre alrededor de cada petición) mide cuánto tarda cada predicción y con qué código responde; `/metricas` resume esas mediciones. Y cada predicción queda guardada en `predicciones.jsonl`, con un identificador que la app guarda junto al viaje. [`simular_noviembre.py`](./starter/ejercicio1/simular_noviembre.py) manda al servicio los 1156 viajes de noviembre y escribe `cobros.csv`, lo que el sistema de cobros anotó al terminar cada viaje: el identificador de la predicción y la tarifa cobrada (no lo modifiques). [`desempeno.py`](./starter/ejercicio1/desempeno.py) une las dos cosas y calcula el error por semana. Los 4 `TODO`: medir la latencia, resumir las métricas, guardar cada predicción y calcular el error por semana.

Desde la carpeta del ejercicio, corre todo una vez antes de tocar nada:

```
python entrenar.py
python simular_noviembre.py
python desempeno.py
pytest -v
```

**Respuesta 1 (antes de los TODO):** los 1156 viajes recibieron su tarifa. ¿Qué dicen `/metricas` y `desempeno.py`? ¿Qué podrías afirmar sobre el servicio en noviembre?

Antes de los TODO, `/metricas` no contaba nada (cero peticiones, latencias vacías) y el registro de predicciones quedaba vacío, así que `desempeno.py` no tenía qué unir con los cobros. Del servicio en noviembre solo se podía afirmar que respondió las 1156 tarifas; nada sobre qué tan rápido ni qué tan bien.

**Respuesta 2 (TODO 1 y 2):** antes de correr la simulación, ¿qué latencia p95 esperas, y cuántos errores de validación? ¿Esperas que la latencia cambie entre la primera y la última semana del mes?

Esperaba una p95 de pocos milisegundos (el modelo es chico y `TestClient` no pasa por la red) y unos pocos errores de validación, solo las peticiones malformadas que manda la simulación. La latencia no debería cambiar entre semanas: no depende de qué tan bien acierte el modelo. Salió p50 2.23 ms, p95 2.54 ms, 3 errores de validación y 0 de servidor.

**Respuesta 3 (TODO 3):** el servicio devuelve `id_prediccion` y lo guarda en el registro. ¿Para qué lo necesita el sistema de cobros? ¿Qué pasaría si solo se guardaran la fecha y la tarifa estimada?

El sistema de cobros guarda la tarifa cobrada de cada viaje con ese `id_prediccion`, y con él se une cada estimación con su cobro uno a uno. Con solo la fecha y la tarifa estimada no hay llave: dos viajes a la misma hora quedan ambiguos y no se puede calcular el error de cada predicción, solo comparar promedios.

**Respuesta 4 (TODO 4):** el MAE en prueba de la versión 1 fue 1.17. Antes de correr `desempeno.py`, ¿qué esperas para cada semana de noviembre?

Esperaba un MAE cercano a 1.17 en todas las semanas, quizás algo más alto por ser datos nuevos. No fue así: las dos primeras semanas quedaron cerca (1.18 y 1.34), pero desde el 16 de noviembre el error se multiplicó por casi 4.

**Lo que mostró el monitoreo:**

- Latencia p50 2.23 ms, p95 2.54 ms; 1159 peticiones, 3 errores de validación, 0 de servidor.
- MAE por semana (referencia 1.17):

| Semana | Viajes | MAE | Veces la referencia |
|---|---|---|---|
| 2026-11-02 | 291 | 1.18 | 1.01 |
| 2026-11-09 | 318 | 1.34 | 1.15 |
| 2026-11-16 | 308 | 4.29 | 3.67 |
| 2026-11-23 | 239 | 4.30 | 3.68 |

El sistema estuvo perfecto todo el mes; el modelo empeoró desde el 16.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el fraude, cuando la verdad tarda tres semanas

Solo o en pareja. Carpeta: [`starter/ejercicio2/`](./starter/ejercicio2/). El servicio de fraude guardó los 5874 pagos que evaluó en noviembre en [`predicciones_fraude.csv`](./starter/ejercicio2/predicciones_fraude.csv): la probabilidad, si mandó el pago a revisión y cuánto tardó. Un fraude se confirma cuando el titular de la tarjeta reclama el cobro, entre 2 y 21 días después; los confirmados están en [`contracargos.csv`](./starter/ejercicio2/contracargos.csv). Hoy es 14 de diciembre. [`monitoreo_fraude.py`](./starter/ejercicio2/monitoreo_fraude.py) arma el reporte del mes. Los pasos 1 a 3 tienen `TODO`: la latencia p95 por día, la tasa de alertas por semana y los fraudes confirmados hasta hoy. Los pasos 4 (el recall de las semanas cuyos fraudes ya se confirmaron) y 5 (las alertas) solo tienen un comentario. Resuelve y corre `python comprobar.py` hasta que los cinco pasos digan `OK`; después, `python monitoreo_fraude.py`.

**Respuesta 1:** el reporte da tres tipos de alerta. ¿Cuáles son de sistema y cuáles de modelo? ¿El día de la latencia alta coincide con alguna semana de recall bajo?

La de latencia (el 2026-11-10, p95 de 78.2 ms) es de sistema. La tasa de alertas baja (5.0 % y 4.6 % contra 7.6 %) es de modelo, pero indirecta, y el recall bajo (27.6 % la semana del 16 contra 66.1 %) es de modelo. No coinciden: el pico de latencia fue el 10 y esa semana el recall fue normal (73.6 %); la caída del recall es la semana del 16.

**Respuesta 2:** la tasa de alertas se puede calcular el mismo día; el recall, tres semanas después. ¿Qué semana mostró primero el problema la tasa de alertas, y cuándo lo confirmó el recall? ¿Por qué la tasa de alertas no alcanza para decir que el modelo empeoró?

La tasa de alertas lo mostró primero, la misma semana del 16 (bajó a 5.0 %). El recall lo confirmó recién cuando esa semana maduró, al 13 de diciembre (16 + 6 días + 21 de demora): 27.6 %. La tasa sola no alcanza porque puede bajar sin que el modelo empeore: si hay menos fraude, o los pagos son distintos, mandar menos a revisión es lo correcto. Solo la verdad dice si se escaparon fraudes.

**Respuesta 3:** ¿por qué el reporte no muestra el recall de la semana del 23 de noviembre, si ya hay fraudes confirmados de esa semana?

Porque esa semana todavía no está madura: sus fraudes se pueden confirmar hasta el 20 de diciembre (23 + 6 + 21) y al 14 de diciembre faltan confirmaciones. Un recall calculado con fraudes a medio confirmar sale engañoso, porque los que faltan son justamente los que aún no se conocen.

**Si terminas antes:** cambia `HOY` a `"2026-11-30"` y vuelve a correr. ¿Qué semanas tienen recall? ¿Desde qué día el reporte puede mostrar la caída del recall de la semana del 16?

Al 2026-11-30 solo la semana del 2 tiene recall (76.7 %); las demás no están maduras. La caída de la semana del 16 recién se puede mostrar desde el 13 de diciembre. Hasta entonces el único aviso es la tasa de alertas.

## Ejercicio 3: el monitoreo de tu proyecto

Usa tu proyecto en [`psets/proyecto/`](../../psets/README.md) y el servicio de las dos sesiones anteriores. Esta vez no hay plantilla.

### Entregable 1: el servicio, instrumentado

- Un middleware que mida la latencia y el código de cada predicción, y una ruta `/metricas` con, al menos, la cantidad de peticiones, los errores de validación, los errores de servidor y la latencia p50 y p95.
- Cada predicción queda en un registro (un archivo JSONL en `monitoreo/`, por ejemplo) con su identificador, el momento, la versión del modelo, la entrada y la salida. Lo que haga falta para unirla después con la verdad tiene que estar en ese registro.
- Pruebas en `servicio/test_api.py` que comprueben que una predicción queda registrada y que `/metricas` separa los errores de validación de los de servidor, con un registro temporal (`tmp_path`).

### Entregable 2: `monitoreo/desempeno.py`

Une el registro con la verdad de tu proyecto y calcula la métrica de la Semana 11 por período (día o semana) y por versión. Si en tu proyecto la verdad no llega, o llega muy tarde, define una métrica que se pueda calcular antes y explica qué te dice y qué no.

### Entregable 3: `docs/monitoreo.md`

Una tabla con cada métrica que vigilas: si es de sistema o de modelo, de dónde sale, cuándo se conoce y con qué umbral alerta.

**¿Cuánto tarda en llegar la verdad en tu proyecto, y qué métrica te avisa mientras tanto?**

Muy tarde, o nunca. El modelo estima el monto de awards que no lo informan, y la verdad solo llega si el funder lo publica después. En la simulación (`monitoreo/simular_lote.py`) supuse una publicación entre 30 y 730 días después del inicio: al 2022-12-31 solo 2 de las 12 estimaciones de 2022 tenían monto (17 %). Mientras tanto avisan tres indicadores que se calculan la misma noche, sin verdad: la mediana del log10 estimado (alerta si se aleja más de 0.30 de 5.741), la fracción de funders que el modelo no vio al entrenar (más del 20 %) y la fracción de awards sin duración (más del 10 %). Dicen que la entrada o la salida cambiaron, no que el modelo se equivoque más. Con la verdad, `monitoreo/desempeno.py` calcula el MAE en log10 y el sesgo por año y versión: al 2025-12-31 el MAE de 2023 fue 0.558 (más de 1.25 × 0.3608), casi todo por un cooperative agreement de 118 millones estimado en 1.3 millones. Detalle en `psets/proyecto/docs/monitoreo.md`.

**¿Qué falla de tu modelo no vería ninguna de tus métricas de sistema?**

Que el modelo subestime cada vez más porque los montos reales crecieron (inflación, programas nuevos más grandes) mientras los awards se ven igual que antes. La latencia, los errores y hasta los indicadores sin verdad se quedan quietos, porque la entrada no cambió. Solo lo muestra el sesgo con verdad, uno o dos años después. Tampoco se ve un award enorme subestimado por un factor 100, como el de 2023: el sistema le respondió en 2 ms.

## Antes de la próxima sesión

**Lectura:** ["Data Drift vs. Concept Drift and Why Monitoring for Them is Important"](https://medium.com/whylabs/data-drift-vs-concept-drift-and-why-monitoring-for-them-is-important-a68c08b141f0), WhyLabs.

Desde el 16 de noviembre, el modelo de la tarifa se equivoca casi cuatro veces más que en prueba. Los viajes que recibe el servicio son del mismo tipo que los de octubre: las mismas zonas, las mismas distancias, las mismas horas. ¿Qué cambió, entonces? ¿Lo hubieras detectado mirando solo los viajes que llegan?

Cambió la relación entre los viajes y la tarifa: con los mismos viajes, desde el 16 de noviembre se cobra más (una subida de tarifa de alrededor del 20 %). Es concept drift: P(tarifa | viaje) cambió y P(viaje) no. Mirando solo los viajes que llegan no lo hubiera detectado: zonas, distancias y horas tienen la misma distribución que en octubre. Hacía falta la verdad, la tarifa cobrada, unida con cada predicción.
