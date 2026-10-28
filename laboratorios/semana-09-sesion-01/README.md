# Semana 9, Sesión 1: Ingeniería de características: principios aplicados

> Objetivo de la sesión: explicar por qué una misma variable cruda puede necesitar codificaciones distintas según el modelo que la consuma, y diseñar al menos dos features derivadas para el dataset de tu proyecto.

**Lectura previa de esta sesión:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

**Referencia de esta sesión:** *Designing Machine Learning Systems*, Chip Huyen (O'Reilly, 2022), Capítulo 5: Feature Engineering.

## Requisitos antes de esta sesión

pandas y matplotlib, de la sesión pasada. numpy se instala junto con pandas; verifica con:

```
python3 -c "import numpy; print(numpy.__version__)"
```

Ten a mano tus 3 hallazgos de la exploración de la Semana 8, Sesión 2: las features de hoy salen de ahí.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué diferencia a una feature que tiene sentido de negocio de una que solo mejora una métrica?**

Una feature con sentido de negocio representa un **comportamiento o un valor que el negocio puede explicar**: el equipo que va a usar el modelo entiende qué mide y por qué influye. Una que solo mejora una métrica sale de combinar columnas a ciegas (sumar o multiplicar todo con todo), y aunque suba el puntaje, nadie puede justificarla ni confiar en ella. El artículo propone 3 formas de crear features explicables: **razones** (proporciones), **tendencias** (cambio en el tiempo) y **granularidad** (medir lo mismo con más o menos detalle).

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

Las **transacciones en línea como proporción**. El conocimiento del negocio es que el compromiso digital de un cliente depende de qué parte de su actividad hace en línea, no del número absoluto. Dos clientes con 5 transacciones en línea en 3 años parecen iguales, pero si uno hizo 100 transacciones en total (5%) y el otro solo 5 (100%), el segundo es un cliente exclusivamente digital. La columna se calcula como `transacciones_en_linea / transacciones_totales`.

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

Creando columnas que codifiquen ese conocimiento, porque el modelo solo aprende de lo que ve. Para la tarifa fija: `es_tarifa_fija`, que vale 1 si el viaje va entre JFK y una zona de Manhattan. Para el costo por hora: extraer la hora de `fecha_hora_inicio` y construir `es_hora_pico` (7 a 9 y 16 a 19, entre semana), `es_fin_de_semana` y la hora como ciclo (`hora_sin`, `hora_cos`). Todas salen de columnas que existen al pedir el taxi.

## Ejercicio 1: features de fecha y hora

Archivo: [`starter/ejercicio1_features_tiempo.py`](./starter/ejercicio1_features_tiempo.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada. El script ya quita los errores confirmados, convierte `fecha_hora_inicio` y compara la tarifa por kilómetro según la hora pico; faltan 4 `TODO`: hora, día de la semana, fin de semana, hora pico y la hora como ciclo. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-09-sesion-01/starter
python3 ejercicio1_features_tiempo.py
```

**Respuesta 1 (TODO 4), antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

Como números, la hora 23 queda a **23 unidades** de la hora 0, la mayor distancia posible. En el reloj están a **1 hora**: las 23:00 y las 00:00 son vecinas, y las 23:00 se parecen más a la medianoche que a las 12:00. Por eso se codifica como ciclo con seno y coseno: así la 23 y la 0 quedan juntas en el círculo.

**Respuesta 2 (TODO 3):** sin el TODO 3, el Paso 4 muestra un solo grupo. Con la hora pico construida, ¿esperas que la tarifa por kilómetro sea más alta o más baja en hora pico? ¿Por qué, si la tarifa depende de la distancia y de la duración?

**Más alta**, porque el taxímetro cobra la distancia y también el tiempo detenido en el tráfico. La salida del Paso 4 lo confirma. Hora pico es de 7 a 9 y de 16 a 19, de lunes a viernes. Se excluyen los viajes de tarifa fija de 70 USD para que no se mezclen:

| | Viajes | Tarifa mediana | USD por km | Min por km |
|---|---|---|---|---|
| Fuera de hora pico | 1842 | 18.69 | 3.22 | 2.79 |
| Hora pico | 812 | 22.14 | 3.52 | 3.64 |

Por la misma distancia se cobra cerca de 9% más. La correlación de `es_hora_pico` con la tarifa es baja (0.11), porque la distancia explica casi todo; la hora pico es un ajuste encima de la distancia.

**Respuesta 3:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

La parte del **tráfico**. La duración le dice al modelo que un viaje fue lento, y en hora pico los viajes son más lentos por la misma distancia: **3.64 minutos por km en hora pico** contra **2.79** fuera de ella. `es_hora_pico` trae ese efecto promedio, que sí se conoce al pedir el taxi. Lo que no trae es la variación de un viaje concreto: un accidente o un desvío solo se ven en la duración real.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Archivo: [`starter/ejercicio2_features_zona.py`](./starter/ejercicio2_features_zona.py), sobre el mismo `viajes_taxi.csv`. Completa los 4 `TODO`. Al final, el script imprime una comprobación por TODO: complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_features_zona.py
```

**Respuesta 1:** ¿cuántos viajes tienen `es_tarifa_fija` en 1, y qué tarifa mínima y máxima tienen? ¿Qué te dice eso sobre lo que un modelo lineal puede aprender con esta columna?

**300 viajes**, con tarifa mínima y máxima de **70.0**: todos pagan exactamente lo mismo, sin importar la distancia. Son los 300 viajes de la franja horizontal de la Semana 8. Los otros 73 viajes con JFK, hacia Brooklyn o Queens, no tienen tarifa fija y se cobran por taxímetro. Con esta columna, una regresión lineal aprende un ajuste constante, pero no puede "apagar" la pendiente de la distancia para esos viajes: seguiría sumando tarifa por cada km. Para capturarlo bien necesita una interacción (`distancia_km × (1 − es_tarifa_fija)`) o un modelo de árboles, que separa los viajes de tarifa fija en su propia rama y les asigna 70.

**Respuesta 2:** ¿cuántas columnas crea one-hot para la zona de origen, y cuántas la codificación ordinal? Si el modelo es una regresión lineal, ¿cuál de las dos usarías, y por qué la otra le da información falsa?

One-hot crea **8 columnas**, una por zona (`origen_Brooklyn` … `origen_Upper West Side`). La ordinal crea **1 columna**, con Brooklyn = 0 … Upper West Side = 7. Para una regresión lineal usaría **one-hot**. La ordinal le dice al modelo que las zonas tienen orden y distancia entre sí: que Upper West Side (7) es "más" que Brooklyn (0), y que JFK (2) queda a la mitad de camino entre Harlem (1) y Lower Manhattan (3). Ese orden es alfabético y no tiene ninguna relación con la tarifa, pero el modelo le asignaría una sola pendiente. La ordinal sirve para árboles, que solo cortan por umbrales, o para variables que sí tienen orden.

**Respuesta 3:** ¿qué columnas del dataset no existen al pedir el taxi? ¿Qué pasaría con el modelo si las usaras para entrenar?

No existen **`duracion_min`** (se conoce al terminar el viaje), **`propina`** y **`metodo_pago`** (se registran al pagar), y obviamente **`tarifa`**, que es el objetivo. `metodo_pago` depende del negocio: en una app se elige antes del viaje, pero en un taxi de la calle se sabe al final. `distancia_km` sí existe, como distancia estimada de la ruta. Si entrenaras con las que no existen, tendrías **fuga de información**: el modelo se vería excelente en la evaluación (la duración tiene correlación de 0.956 con la tarifa) y fallaría en producción, porque esas columnas llegarían vacías en el momento de predecir.

## Ejercicio 3: dos features para tu proyecto

Usa el mismo archivo de la exploración de la Semana 8 y la plantilla [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py).

### Paso 1: el momento de la predicción (5 minutos)

Escribe qué predice tu modelo y en qué momento: qué información existe en ese instante y cuál todavía no.

**Qué predice:** el monto en USD de un award, en `log10(amount_usd)`. Sirve para estimar el monto de los awards que no lo informan, que son el 95.9% de las filas de la OBT (hallazgo 2). La etiqueta `label` no sirve como objetivo porque vale 1 en todas las filas (hallazgo 1).

**En qué momento:** cuando el award se registra, es decir, cuando el funder lo otorga y antes de que existan publicaciones financiadas.

- **Existe en ese momento:** funder (`funder_name`, `funder_country_code`), `funding_type`, `funder_scheme`, fechas de inicio y fin (`award_start_date`, `duration_days`), título y descripción, institución líder e `institutions_count`.
- **Todavía no existe:** todo lo que viene de las publicaciones (`work_*`, `publication_year`, `cited_by_count`, `fwci`, `authors_count`, `is_retracted`) y `funded_outputs_count`, que cuenta publicaciones que salen años después. Usarlas sería fuga, como la duración del taxi.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

Plantilla: [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py). Guarda `obt_features.csv` para el pipeline de la Sesión 2.

```python
datos = datos[datos["amount_usd"] > 1].copy()          # descarta montos de 1 USD o menos (errores de captura)
datos["log_amount_usd"] = np.log10(datos["amount_usd"])
OBJETIVO = "log_amount_usd"

# Feature 1
datos["duracion_anios"] = datos["duration_days"] / 365.25

# Feature 2
ESCALA = {
    "fellowship": "individual", "career_development": "individual", "training": "individual",
    "cooperative_agreement": "centro", "program": "centro", "contract": "centro",
}
datos["escala_financiamiento"] = datos["funding_type"].map(ESCALA).fillna("proyecto")
```

La evidencia se mide con **un award por fila** (170 awards con monto, de 210 filas), porque la OBT repite el monto por cada publicación financiada.

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`duracion_anios` = `duration_days / 365.25`. Sale del dominio: un award que financia más años paga más salarios y más gastos. **Evidencia:** su correlación con `log10(amount_usd)` es **0.51**, la más alta de las columnas numéricas disponibles (`institutions_count` da −0.19). **Disponibilidad:** las fechas de inicio y fin forman parte del award desde que se otorga. Solo 6 de los 170 awards no tienen duración: se imputan con la mediana de entrenamiento. Es numérica: para un modelo lineal se escala; para uno de árboles no hace falta.

**Feature 2:**

`escala_financiamiento` agrupa `funding_type` en 3 niveles: **individual** (becas, *career development*, *training*), **centro** (acuerdos cooperativos, programas, contratos) y **proyecto** (el resto: *grant*, *research*). Sale del **hallazgo 3**, la cola larga del monto: los awards grandes no son *grants* raros, sino otro tipo de financiamiento. **Evidencia:** la mediana es de **176 413 USD** en individual, **450 000** en proyecto y **4 878 000** en centro, casi 11 veces más que proyecto. 6 de los 13 awards de centro están en el 10% de montos más altos (≥ 4.89 millones), contra 10 de 151 en proyecto. En cambio, agrupar por funder no sirvió: NSF y NIH tienen casi la misma mediana (log 5.63 y 5.60). **Disponibilidad:** el tipo de financiamiento se define al otorgar el award. **Codificación:** es categórica; para una regresión lineal uso one-hot, porque individual < proyecto < centro sí tiene orden, pero no distancias iguales (de 176 mil a 450 mil no es lo mismo que de 450 mil a casi 5 millones). Para árboles basta con la ordinal.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

Solo con los de **entrenamiento**. Si la media y la desviación se calculan con todos los viajes, la prueba influye en cómo se transforman los datos de entrenamiento, y la evaluación deja de medir el desempeño sobre datos nunca vistos: es una fuga, aunque sea pequeña. Lo correcto es dividir primero, hacer el `fit` del escalador con entrenamiento y aplicar `transform` con esa misma media y desviación a la prueba y a los viajes nuevos en producción. Con 2975 viajes el número casi no cambia, pero el procedimiento sí, y en producción solo existe lo que se aprendió en entrenamiento.
