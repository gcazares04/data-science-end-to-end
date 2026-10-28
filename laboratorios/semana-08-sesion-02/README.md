# Semana 8, Sesión 2: Exploración y visualización de datos

> Objetivo de la sesión: explorar el dataset de tu proyecto con estadística descriptiva y visualizaciones para identificar patrones, outliers y supuestos que romperían un modelo antes de entrenarlo.

**Lectura previa de esta sesión:** ["A Data Scientist's Essential Guide to Exploratory Data Analysis"](https://towardsdatascience.com/a-data-scientists-essential-guide-to-exploratory-data-analysis-25637eee0cf6/), Towards Data Science.

**Referencia de esta sesión:** [Machine Learning Zoomcamp, Módulo 1 (DataTalksClub)](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Requisitos antes de esta sesión

pandas y matplotlib ya vienen en el Codespace. En tu laptop, instálalos con `pip install -r requirements.txt` desde la raíz del repositorio. Verifica con:

```
python3 -c "import matplotlib; print(matplotlib.__version__)"
```

Los scripts de hoy guardan cada gráfico como imagen en una carpeta `figuras/`. Ábrelas desde el explorador de archivos de VS Code. También puedes correr el mismo código en un notebook de Jupyter (en el Codespace, crea un archivo `.ipynb` y elige el kernel de Python 3.11), donde el gráfico aparece debajo de la celda.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué preguntas responde la exploración de datos antes de entrenar cualquier modelo?**

Qué datos tengo: cuántas observaciones y features, de qué tipo, cuánto falta y cuántos duplicados hay (paso 1, resumen y estadística descriptiva). Cómo es cada variable y cómo se relacionan entre sí (paso 2). Y si los datos tienen problemas de calidad que romperían el modelo (paso 3): atípicos, desbalance, faltantes, variables casi constantes o duplicados sospechosos. El artículo lo resume en que la exploración guía todo lo que sigue, desde la preparación hasta la interpretación del modelo.

**El artículo distingue la exploración de una sola variable de la exploración de la relación entre dos. ¿Qué herramienta propone para cada una?**

Para **una variable** (univariado), estadísticas descriptivas según el tipo: media, desviación, asimetría, curtosis y cuantiles con **histograma** para las numéricas, y moda y **tabla de frecuencias con gráfico de barras** para las categóricas. Para **la relación entre dos** (multivariado), **diagramas de dispersión** para ver interacciones y **matrices de correlación** (mapa de calor) para medir la fuerza de la relación. El artículo propone `ydata-profiling`, que genera todo eso en un solo reporte.

## Pregunta de apertura

Entrenas un modelo que predice la tarifa de un viaje en taxi. Sin que lo sepas, 15 de los 3000 viajes tienen una tarifa de 999.99 por un error de captura. ¿Qué le pasa al modelo? ¿Lo notarías mirando solo su resultado final?

La regresión trata de acercarse también a esos 15 viajes: como el error se mide al cuadrado, un error de casi 1000 USD pesa muchísimo, y la recta se inclina hacia ellos. Sube la tarifa predicha para todos y los coeficientes dejan de representar la relación real entre distancia y tarifa. Mirando solo el resultado final (un MAE o un RMSE) probablemente no lo notaría: solo vería un error "algo alto", sin saber por qué. Se descubre mirando los datos antes de entrenar: `describe()` muestra un máximo de 999.99, o un histograma muestra un grupo aislado lejos del resto.

## Ejercicio 1: resumen, distribución y atípicos

Archivo: [`starter/ejercicio1_explorar.py`](./starter/ejercicio1_explorar.py), sobre [`starter/viajes_taxi.csv`](./starter/viajes_taxi.csv): 3000 viajes en taxi de Nueva York, con fecha y hora de inicio, zona de origen y de destino, distancia, duración, pasajeros, método de pago, tarifa y propina. El script ya lee el archivo, arma los gráficos y los guarda en `figuras/`; faltan 4 `TODO`. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-08-sesion-02/starter
python3 ejercicio1_explorar.py
```

**Respuesta 1 (TODO 1), antes de ver la solución:** en `describe()`, ¿qué dos valores de la tabla te harían sospechar de un error sin necesidad de graficar nada?

El **máximo de la tarifa, 999.99**, cuando la mediana es 22.60: un viaje en taxi no cuesta eso, y el número redondo parece un código de captura. Y el **mínimo de la duración, −74 minutos**: ningún viaje dura menos que 0. También llaman la atención el mínimo de 0 km en distancia y de 0 pasajeros, pero esos podrían tener otra explicación.

**Respuesta 2 (TODO 2):** abre `figuras/hist_tarifa.png` después del TODO 2. ¿Por qué casi todo el histograma queda comprimido a la izquierda?

No. Con límites de −13.85 y 60.40 USD, la regla marca **334 viajes**: 15 son el error de 999.99, pero **302 tienen tarifa de exactamente 70**, y casi todos (320 de los 334 atípicos) salen o llegan a JFK Airport. Eso no es un error: es la tarifa fija del aeropuerto. El resto son viajes largos y caros, pero posibles. Lo averiguaría agrupando los atípicos: contar valores repetidos (`value_counts()` de la tarifa), cruzarlos con la zona de origen y destino, y graficar distancia contra tarifa. Un error de captura se ve como un valor imposible o repetido sin lógica; un caso real tiene una explicación en otra columna.

**Respuesta 3 (TODO 3 y 4):** la regla del IQR marca más de 300 viajes con tarifa atípica. ¿Esperas que todos sean errores? ¿Cómo lo averiguarías?

Script de clase: [`starter/EDA.py`](./starter/EDA.py), que corre con `docker compose up --build`.

1. **Resumen:** `info()`, tipos, `isna().sum()`, `duplicated().sum()` y `describe()`. Ahí aparecen la tarifa máxima de 999.99 (15 viajes) y la duración mínima de −74 (10 viajes con duración negativa).
2. **Distribución:** histograma y boxplot de la tarifa (`hist_tarifa.png`, `boxplot_tarifa.png`). La distribución está sesgada a la derecha: mediana de 22.60 y promedio de 31.39, arrastrado por los atípicos.
3. **Atípicos con IQR:** límites −13.85 y 60.40, con 334 viajes marcados. La dispersión tarifa contra distancia (`scatter_tarifa.png`) muestra que la mayoría de esos atípicos son la franja de 70 USD de JFK, no errores. Solo los 15 de 999.99 y los 10 de duración negativa son errores confirmados, y son los que se quitan en el Ejercicio 2.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: relaciones entre variables

Solo o en pareja. Archivo: [`starter/ejercicio2_relaciones.py`](./starter/ejercicio2_relaciones.py), sobre el mismo `viajes_taxi.csv`. Quita solo los errores que el Ejercicio 1 ya confirmó y completa los 5 `TODO`: dispersión de distancia contra tarifa, propina por método de pago, correlaciones con la tarifa, y pasajeros vacíos o en 0. Al final, el script imprime una comprobación por TODO: completa los 5 hasta que todas digan `OK`.

```
python3 ejercicio2_relaciones.py
```

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

> **Nota sobre los datos:** volví a correr el pipeline completo (Kestra → Bronze → dbt Silver/Gold → Spark) en una cuenta nueva de Snowflake, con el snapshot de OpenAlex del 2026-10-14. La OBT completa tiene 261 633 filas y 203 173 awards (~155 MB), así que [`plantilla-eda/exportar_obt.py`](./plantilla-eda/exportar_obt.py) guarda una muestra aleatoria de **4000 awards** (semilla 42) con todas sus filas, y quita las columnas de texto largo y las listas de autores.

**Objetivo:** la etiqueta real del modelo es `label`, pero vale 1 en todas las filas (ver hallazgo 1), así que no tiene distribución que explorar. Uso **`amount_usd`** (monto del award en USD) como objetivo numérico.

- **5123 filas y 56 columnas** (4000 awards; uno de ellos tiene 107 publicaciones).
- **Faltantes:** `is_global_south` 100%, `award_subfield`/`field`/`domain_name` 97.2%, `amount_usd` 95.9%, `lead_institution_*` 87.4%, `funder_scheme` 84.9%, `duration_days` 84.8%, `amount` 84%, `award_start_date` y `currency` 83.4%, `award_title` y `funding_type` 83%, `first_author_*` 7.7%, `funder_country_code` 6.3% y `fwci` 5.5%. Las columnas de la publicación están casi completas.
- **IQR:** marca **42 atípicos** de 210 montos informados, todos de 10 millones o más. Son awards grandes de NSF (32), USAID (5) y DOE/Office of Science (4), no errores: el máximo, 211 millones, es un centro de bioenergía de Office of Science.

### Paso 2: relaciones (10 minutos)

Completa los pasos 4 y 5: la relación entre tu objetivo y la variable que esperas que más influya, y las consultas que confirman lo que encuentres.

Agrupé `amount_usd` por `funder_name` (`figuras/monto_por_funder.png`), con mediana porque el promedio lo arrastran los atípicos. Solo 2 funders tienen al menos 10 montos, y los 2 son de EE. UU.:

| Funder | Montos | Mediana | Promedio |
|---|---|---|---|
| NIH | 29 | 420 000 | 798 220 |
| NSF | 145 | 500 000 | 5 854 202 |

Las medianas son parecidas (420 y 500 mil USD). La diferencia grande está en los promedios: el de NSF es 12 veces su mediana porque incluye centros y programas de decenas de millones. Las consultas del paso 5 muestran que **el monto existe o no según el funder**, no según el award: los 210 montos son de funders de EE. UU. (`funder_country_code = US`).

### Paso 3: tres hallazgos que afectarían el modelado (10 minutos)

Para cada hallazgo, escribe qué encontraste, el número o el gráfico que lo confirma, y qué cambia en el modelo por su causa (una fila que quitas, una columna que no puedes usar, una variable que hay que transformar).

**Hallazgo 1: la OBT no tiene ejemplos negativos.**

`label` vale 1 en las **5123 filas** (`Filas con label distinto de 1: 0`), igual que en la OBT completa. La OBT solo guarda financiamientos observados, así que un clasificador entrenado tal cual aprendería a decir siempre "sí". **Qué cambia:** antes de entrenar hay que generar negativos, por ejemplo emparejando cada publicación con awards del mismo campo que no la financiaron (*negative sampling*), y validar con métricas de ranking (precision@k) en vez de accuracy.

**Hallazgo 2: el monto y el texto del award faltan según el funder, no al azar.**

El **83%** de las filas no tiene ni título ni descripción del award. Por funder, NSFC (712 filas), el National Cancer Institute (124) y el Army Research Office (107) tienen **0%** de monto y descripción: son awards que solo se conocen por el agradecimiento de un paper. En cambio, NSF tiene **86%** de monto, título y descripción, y la JSPS tiene 87% de título pero ningún monto en USD. `amount_usd` solo existe para funders de EE. UU. (210 de 210). **Qué cambia:** `award_text`, el texto que la siguiente etapa vectoriza con Cortex, está vacío para más de 4 de cada 5 awards. Para esos hay que construir el embedding con otra información (nombre del funder, `funder_award_id` o el texto de las publicaciones que ya financió), o excluirlos del recomendador. `amount_usd` no se puede usar como variable de entrada general ni imputar con la media, porque es MNAR, como ya decía `slv_awards_history`. `is_global_south` está 100% vacía, así que hay que revisar su seed en dbt antes de usarla.

**Hallazgo 3: pocos funders dominan y el monto tiene una cola muy larga.**

National Natural Science Foundation of China concentra el **13.9%** de las filas, y los 5 primeros funders suman el **29.1%**. `amount_usd` tiene promedio de **9 817 123** contra mediana de **561 386** y asimetría de **4.8**, con mínimo de 5 400 y máximo de 211 millones. En log10 la asimetría baja a **0.7**; en la OBT real no aparece el award de 1 USD de la reconstrucción, así que no hay montos menores a 1000. **Qué cambia:** usar `log10(amount_usd)` si el monto entra al modelo y estratificar por funder al partir train/test para que el modelo no aprenda solo a recomendar NSFC. Fuera del objetivo hay otras filas que revisar: **3 publicaciones retractadas** que habría que quitar. Además, el **18.5%** de las publicaciones no tiene abstract, así que su `work_text` es solo el título.

## Antes de la próxima sesión

**Lectura:** ["Feature Engineering that Makes Business Sense"](https://towardsdatascience.com/feature-engineering-that-makes-business-sense-419b9403236e/), Towards Data Science.

La tarifa fija de JFK y la hora del día no existen como columnas del dataset de taxis. ¿Cómo le darías esa información a un modelo?

Creándolas como columnas a partir de las que ya existen. Para JFK, una columna binaria `es_tarifa_fija` que vale 1 cuando el viaje va entre JFK y Manhattan, así el modelo puede separar esos viajes en vez de tratarlos como atípicos. Para la hora, extraer la hora de `fecha_hora_inicio` y codificarla como ciclo (seno y coseno), para que las 23:00 queden cerca de las 00:00, o con indicadores como `es_hora_pico` y `es_fin_de_semana`. Las dos se pueden calcular en el momento de pedir el taxi.
