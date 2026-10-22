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

_(tu respuesta)_

**Elige uno de los ejemplos del artículo: ¿qué conocimiento del negocio traduce a una columna, y cómo?**

_(tu respuesta)_

## Pregunta de apertura

La exploración encontró dos patrones que no existen como columnas: la tarifa fija entre JFK y Manhattan, y viajes que cuestan más por kilómetro a ciertas horas. El modelo solo ve columnas. ¿Cómo le das esa información?

_(tu respuesta)_

## Ejercicio 1: features de fecha y hora

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv), el mismo dataset de la sesión pasada, sin los errores confirmados. A partir de `fecha_hora_inicio`, construir hora, día de la semana, fin de semana, hora pico y la hora como ciclo, y comprobar si la hora pico aporta algo a la tarifa.

**Respuesta 1, antes de ver la solución:** si dejas la hora como un número del 0 al 23, ¿qué tan lejos queda la hora 23 de la hora 0 para un modelo que mide distancias entre números? ¿Y en el reloj?

_(tu respuesta)_

**Respuesta 2:** la duración del viaje no existe al pedir el taxi, pero la hora sí. ¿Qué parte de lo que la duración le diría al modelo puede traer `es_hora_pico`?

_(tu respuesta)_

**La formulación completa, tal como la resuelvo yo:**

_(anota aquí el script y lo que mostró, mientras se revisa en clase)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: zona, codificación y disponibilidad

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/features_de_zona.py`](./plantilla-ejercicio2/features_de_zona.py) y córrela desde su carpeta. Completa los 4 `TODO`.

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

**Qué predice:** el monto en USD de un award, en `log10(amount_usd)`. Sirve para estimar el monto de los awards que no lo informan, que son el 94.5% de la OBT (hallazgo 2). La etiqueta `label` no sirve como objetivo porque vale 1 en todas las filas (hallazgo 1).

**En qué momento:** cuando el award se registra, es decir, cuando el funder lo otorga y antes de que existan publicaciones financiadas.

- **Existe en ese momento:** funder (`funder_name`, `funder_country_code`), `funding_type`, `funder_scheme`, fechas de inicio y fin (`award_start_date`, `duration_days`), título y descripción, institución líder e `institutions_count`.
- **Todavía no existe:** todo lo que viene de las publicaciones (`work_*`, `publication_year`, `cited_by_count`, `fwci`, `authors_count`, `is_retracted`) y `funded_outputs_count`, que cuenta publicaciones que salen años después. Usarlas sería fuga, como la duración del taxi.

### Paso 2: construye las 2 features (15 minutos)

Completa la plantilla con 2 columnas nuevas, derivadas de las existentes. Al menos una tiene que salir de uno de tus hallazgos de la exploración.

Plantilla: [`plantilla-features/features_proyecto.py`](./plantilla-features/features_proyecto.py). Guarda `obt_features.csv` para el pipeline de la Sesión 2.

```python
datos = datos[datos["amount_usd"] > 1].copy()          # hallazgo 3: quita el award de 1 USD
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

La evidencia se mide con **un award por fila** (196 awards con monto, de 415 filas), porque la OBT repite el monto por cada publicación financiada.

### Paso 3: justifica cada feature (5 minutos)

Para cada una: de qué hallazgo o conocimiento del dominio sale, qué evidencia muestra que aporta (una correlación, una diferencia de promedios), y por qué existe en el momento de predecir. Si alguna es categórica, di qué codificación usarías según el modelo que planeas entrenar.

**Feature 1:**

`duracion_anios` = `duration_days / 365.25`. Sale del dominio: un award que financia más años paga más salarios y más gastos. **Evidencia:** su correlación con `log10(amount_usd)` es **0.30**, la más alta de las columnas numéricas disponibles (`institutions_count` da 0.09). **Disponibilidad:** las fechas de inicio y fin forman parte del award desde que se otorga. Solo 3 de los 196 awards no tienen duración: se imputan con la mediana de entrenamiento. Es numérica: para un modelo lineal se escala; para uno de árboles no hace falta.

**Feature 2:**

`escala_financiamiento` agrupa `funding_type` en 3 niveles: **individual** (becas, *career development*, *training*), **centro** (acuerdos cooperativos, programas, contratos) y **proyecto** (el resto: *grant*, *research*). Sale del **hallazgo 3**, la cola larga del monto: los awards grandes no son *grants* raros, sino otro tipo de financiamiento. **Evidencia:** la mediana es de **114 273 USD** en individual, **325 553** en proyecto y **3 064 013** en centro, casi 10 veces más que proyecto. 8 de los 11 awards de centro están en el 10% de montos más altos (≥ 1.36 millones). En cambio, agrupar por funder no sirvió: NSF y NIH tienen la misma mediana (log 5.50). **Disponibilidad:** el tipo de financiamiento se define al otorgar el award. **Codificación:** es categórica; para una regresión lineal uso one-hot, porque individual < proyecto < centro sí tiene orden, pero no distancias iguales (de 114 mil a 325 mil no es lo mismo que de 325 mil a 3 millones). Para árboles basta con la ordinal.

## Antes de la próxima sesión

**Lectura:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

Para escalar la distancia antes de entrenar, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

_(tu respuesta)_
