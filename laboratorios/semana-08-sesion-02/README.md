# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas ya lo tienes desde la Semana 4. Instala matplotlib:

```
pip install matplotlib
```

Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. También puedes correr el mismo código en un notebook de Jupyter, donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

_(tu respuesta)_

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

_(tu respuesta)_

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

_(tu respuesta)_

## Ejercicio 1: resumen, distribución y atípicos

Enunciado: [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv) tiene 3000 viajes en taxi de Nueva York: fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. Primera mirada: qué contiene, cómo se distribuye la tarifa y qué valores están fuera de lo esperable.

**Respuesta 1, antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

_(tu respuesta)_

**Respuesta 2:** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

_(tu respuesta)_

**La formulación completa, tal como la resuelvo yo:**

_(anota aquí el script y lo que encontró, mientras se revisa en clase)_

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/explorar_relaciones.py`](./plantilla-ejercicio2/explorar_relaciones.py) y córrela desde su carpeta (`cd plantilla-ejercicio2`, `python3 explorar_relaciones.py`): lee el mismo `starter/viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0.

Tras quitar los 25 errores confirmados (tarifa 999.99 y duración negativa) quedan **2975 viajes**.

**Hallazgo 1: qué muestra el gráfico de distancia contra tarifa, además de la tendencia general.**

La tendencia es lineal (correlación distancia–tarifa de **0.96**), pero hay una **franja horizontal en 70 USD** que no depende de la distancia: **300 viajes** con tarifa exacta de 70, de 0 a 45 km, y **los 300 tienen origen o destino en JFK Airport**. Es la tarifa fija del aeropuerto, no un error: por eso el IQR del Ejercicio 1 los marcaba como atípicos sin serlo. Un modelo lineal sobre la distancia los predeciría mal; necesita saber si el viaje es de JFK.

**Hallazgo 2: qué pasa con la propina según el método de pago, y si crees que describe a los pasajeros o al sistema que registra los datos.**

Los **888 viajes en efectivo tienen propina 0** (promedio 0.00), mientras que con tarjeta el promedio es **4.73 USD** y ninguno tiene propina 0. Describe al **sistema**, no a los pasajeros: la propina en efectivo se entrega en mano y el taxímetro no la registra. Además hay **60 viajes sin método de pago** (propina promedio 3.35, 16 con propina 0), que parecen una mezcla de ambos. La propina en efectivo es un dato faltante disfrazado de 0.

**Hallazgo 3: la duración tiene una correlación con la tarifa tan alta como la distancia. Si el modelo tiene que estimar la tarifa en el momento en que alguien pide el taxi, ¿puede usar la duración?**

No. La duración (correlación **0.956**) solo se conoce cuando el viaje termina; usarla sería **fuga de información** (data leakage): el modelo se vería excelente en entrenamiento y fallaría en producción. Lo mismo aplica a la propina (0.633). La distancia estimada de la ruta sí se puede calcular al pedir el taxi. Por otro lado, los pasajeros casi no influyen (**−0.051**) y tienen **119 vacíos y 19 en 0**, otro problema de calidad de captura.

## Ejercicio 3: explora el dataset de tu proyecto

Usa el archivo que tu pipeline de la Semana 7 deja listo para modelar, y la plantilla [`plantilla-eda/explorar_proyecto.py`](./plantilla-eda/explorar_proyecto.py).

### Paso 1: resumen y variable objetivo (5 minutos)

Reemplaza la ruta del archivo y el nombre de la columna que tu modelo va a predecir. Corre los pasos 1 a 3.

**Proyecto:** GrantMatch (PSet 2), un recomendador que empareja convocatorias de financiamiento (awards) con el perfil de investigadores (sus publicaciones). El archivo listo para modelar es la One Big Table `OBT.OBT_AWARD_WORK`, con una fila por par award + publicación financiada.

> **Nota sobre los datos:** los trials de Snowflake del equipo expiraron, así que no se pudo exportar la OBT. La reconstruí con [`plantilla-eda/reconstruir_obt.py`](./plantilla-eda/reconstruir_obt.py) desde la API pública de OpenAlex, que es la misma fuente que el bucket de S3 que carga Kestra. Apliqué las mismas reglas de limpieza de dbt Silver (`amount <= 0` pasa a NULL, `end_date < start_date` pasa a NULL y `amount_usd` solo se llena cuando la moneda es USD) y respeté el grain de `FACT_AWARD_WORKS`. Es una muestra aleatoria de 4000 awards (semilla 42), con hasta 3 publicaciones por award. [`plantilla-eda/exportar_obt.py`](./plantilla-eda/exportar_obt.py) exporta la OBT real cuando haya una cuenta activa.

**Objetivo:** la etiqueta real del modelo es `label`, pero vale 1 en todas las filas (ver hallazgo 1), así que no tiene distribución que explorar. Uso **`amount_usd`** (monto del award en USD) como objetivo numérico.

- **7564 filas y 41 columnas.**
- **Faltantes:** `amount_usd` 94.5%, `award_topic_name`/`field`/`domain` 95.8%, `lead_institution_*` 84%, `amount` 82.9%, `duration_days` 81.6%, `award_title` 79.8%, `currency` 80.3%, `fwci` 3.4%, `funder_country_code` 2.8% y `work_doi` 1.9%. Las columnas de la publicación están casi completas.
- **IQR:** marca **41 atípicos** de 416 montos informados. Son awards grandes de NIH, NSF, Gates y USAspending, no errores. El único error claro es un award de NSF con `amount_usd = 1.0`.

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

Agrupé `amount_usd` por `funder_name` (`figuras/monto_por_funder.png`), con mediana porque el promedio lo arrastran los atípicos. Solo 4 funders tienen al menos 10 montos, y los 4 son de EE. UU.:

| Funder | Montos | Mediana | Promedio |
|---|---|---|---|
| U.S. Department of Energy | 10 | 223 000 | 4 798 392 |
| NASA | 11 | 287 171 | 291 912 |
| NIH | 153 | 308 855 | 705 644 |
| NSF | 222 | 358 915 | 504 347 |

Las medianas son parecidas: entre 220 y 360 mil USD. La diferencia grande está en los promedios, porque DOE tiene un award que dispara el suyo. Las consultas del paso 5 muestran que **el monto existe o no según el origen del dato (`provenance`)**, no según el award: los 416 montos son de funders de EE. UU. (`funder_country_code = US`).

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1: la OBT no tiene ejemplos negativos.**

`label` vale 1 en las **7564 filas** (`Filas con label distinto de 1: 0`). La OBT solo guarda financiamientos observados, así que un clasificador entrenado tal cual aprendería a decir siempre "sí". **Qué cambia:** antes de entrenar hay que generar negativos, por ejemplo emparejando cada publicación con awards del mismo campo que no la financiaron (*negative sampling*), y validar con métricas de ranking (precision@k) en vez de accuracy.

**Hallazgo 2: el monto y el texto del award faltan según el origen del dato, no al azar.**

El **79.8%** de las filas no tiene ni título ni descripción del award. Por `provenance`, `crossref_work_funders` (4964 filas, 66% del total) y `europepmc_work_funders` (878) tienen **0%** de monto, título y descripción: son awards que solo se conocen por el agradecimiento de un paper. En cambio, `nsf_award_search` tiene **100%** de monto y título. `amount_usd` solo existe para funders de EE. UU. (416 de 416). **Qué cambia:** `award_text`, el texto que la siguiente etapa vectoriza con Cortex, está vacío para 4 de cada 5 awards. Para esos hay que construir el embedding con otra información (nombre del funder, `funder_award_id` o el texto de las publicaciones que ya financió), o excluirlos del recomendador. `amount_usd` no se puede usar como variable de entrada general ni imputar con la media, porque es MNAR, como ya decía `slv_awards_history`.

**Hallazgo 3: pocos funders dominan y el monto tiene una cola muy larga.**

National Natural Science Foundation of China concentra el **13.7%** de las filas, y los 5 primeros funders suman el **27.9%**. `amount_usd` tiene promedio de **1 547 878** contra mediana de **325 553** y asimetría de **10.9**, con máximo de 119 millones. En log10 la asimetría baja a **−1.42**, y esa cola izquierda la genera un solo award de NSF con monto de **1 USD**, que es un error de captura. **Qué cambia:** quitar el award de 1 USD (y fijar un mínimo de monto en Silver), usar `log10(amount_usd)` si el monto entra al modelo, y estratificar por funder al partir train/test para que el modelo no aprenda solo a recomendar NSFC. Fuera del objetivo hay otras filas que quitar: **8 publicaciones retractadas** y **1 publicación de 1905**, anterior al rango de `dim_date`. Además, el **26.3%** de las publicaciones no tiene abstract, así que su `work_text` es solo el título.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

_(tu respuesta)_
