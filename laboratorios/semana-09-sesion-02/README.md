# Semana 9, Sesión 2: Pipeline reproducible de features

> Objetivo de la sesión: construir un pipeline de transformación reproducible con scikit-learn (imputación, codificación, escalado) para el dataset de tu proyecto, que evite la fuga de información entre entrenamiento y validación.

**Lectura previa de esta sesión:** ["The Dreaded Antagonist: Data Leakage in Machine Learning"](https://towardsdatascience.com/the-dreaded-antagonist-data-leakage-in-machine-learning-5f08679852cc/), Towards Data Science.

**Referencia de esta sesión:** [Pipelines and composite estimators (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/compose.html).

## Requisitos antes de esta sesión

Instala scikit-learn:

```
pip install scikit-learn
```

Verifica con:

```
python3 -c "import sklearn; print(sklearn.__version__)"
```

Este laboratorio usa `sparse_output=False` en `OneHotEncoder`, disponible desde scikit-learn 1.2. Si tu versión es anterior, actualízala con `pip install --upgrade scikit-learn`.

Ten a mano el script de features de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es la fuga de información, y por qué un modelo con fuga puede verse excelente en la evaluación y fallar en producción?**

_(tu respuesta)_

**El artículo describe varias formas en que se produce una fuga. ¿Cuál de ellas viste la sesión pasada con la duración del viaje?**

_(tu respuesta)_

## Pregunta de apertura

Para escalar la distancia, le restas la media y la divides por la desviación estándar. ¿Con qué viajes calculas esa media: con todos, o solo con los de entrenamiento? ¿Cambia algo?

_(tu respuesta)_

## Ejercicio 1: encontrar y corregir la fuga

Enunciado: [`starter/viajes_features.csv`](./starter/viajes_features.csv) tiene los viajes de taxi con las features de la sesión pasada, sin los errores confirmados, y con la distancia 0 y los 0 pasajeros marcados como faltantes. Un compañero imputa y escala la distancia y los pasajeros con todo el dataset, después divide en entrenamiento y prueba, y entrena una regresión lineal. El código está en la diapositiva.

**Respuesta 1, antes de ver la solución:** ¿en qué línea entra el conjunto de prueba al entrenamiento?

En la línea que imputa con todo el dataset, `SimpleImputer(strategy="median").fit_transform(X[NUMERICAS])`, antes de `train_test_split`: la mediana de la distancia ya incluye los viajes de prueba. Vuelve a entrar en la línea siguiente, cuando `StandardScaler().fit_transform` calcula la media y la desviación estándar con todas las filas. En ambos casos, el problema es el `fit` sobre datos que incluyen la prueba.

**Respuesta 2:** ¿cuánto esperas que cambie el error absoluto medio en prueba al corregir la fuga? Si casi no cambia, ¿significa que no había fuga?

Muy poco: el MAE pasa de **2.5276** con fuga a **2.5274** corregido (diferencia de −0.0002). Que casi no cambie **no significa que no hubiera fuga**. La fuga está en el procedimiento: la prueba dejó de ser datos que el modelo nunca vio. Aquí es pequeña porque la mediana y la media de 2975 viajes casi no se mueven al quitar el 20% (mediana de distancia 6.93 vs 6.94; pasajeros 1 en ambas). Con pocos datos, con una estadística sensible (media, máximo) o con un *target encoding*, la misma fuga sí infla la métrica. Y en producción no hay "todo el dataset" para recalcular: el procedimiento con fuga ni siquiera se puede reproducir.

**La formulación completa, tal como la resuelvo yo:**

Script: [`ejercicio1/fuga.py`](./ejercicio1/fuga.py), que corre las dos versiones con la misma división (`random_state=42`). La corregida divide primero y después hace el `fit` del imputador y el escalador **solo con entrenamiento**. Aplica `transform` a entrenamiento y prueba:

```python
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
imputador = SimpleImputer(strategy="median").fit(X_ent[NUMERICAS])
X_ent[NUMERICAS] = imputador.transform(X_ent[NUMERICAS])
X_prueba[NUMERICAS] = imputador.transform(X_prueba[NUMERICAS])
escalador = StandardScaler().fit(X_ent[NUMERICAS])
X_ent[NUMERICAS] = escalador.transform(X_ent[NUMERICAS])
X_prueba[NUMERICAS] = escalador.transform(X_prueba[NUMERICAS])
modelo = LinearRegression().fit(X_ent, y_ent)
```

```
Faltantes: {'distancia_km': 25, 'pasajeros': 138}
Medianas con todo el dataset: {'distancia_km': 6.93, 'pasajeros': 1.0}
Medianas solo entrenamiento:  {'distancia_km': 6.94, 'pasajeros': 1.0}
MAE en prueba con fuga:   2.5276
MAE en prueba corregido:  2.5274
```

El Ejercicio 2 hace lo mismo con un `Pipeline`, que garantiza este orden sin tener que acordarse.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: el pipeline completo

Solo o en pareja. Parte de la plantilla [`plantilla-ejercicio2/pipeline_completo.py`](./plantilla-ejercicio2/pipeline_completo.py) y córrela desde su carpeta. Completa los 5 `TODO`: división, `ColumnTransformer` con tres grupos de columnas, `fit` solo con entrenamiento, verificación de las medianas, y un viaje desde una zona que el entrenamiento nunca vio.

**Respuesta 1:** ¿cuántas columnas quedan después de preparar los datos? Explica de dónde sale ese número a partir de los tres grupos de columnas.

Quedan **25 columnas** (entrenamiento 2380 × 25 y prueba 595 × 25). Salen de **2 numéricas** (`distancia_km`, `pasajeros`, imputadas y escaladas, siguen siendo 2), **18 one-hot** (8 zonas de origen + 8 zonas de destino + 2 métodos de pago: efectivo y tarjeta; los 60 vacíos se imputan con el más frecuente antes de codificar, así que no crean una columna propia) y **5 ya listas** que pasan sin cambios. Total: 2 + 18 + 5 = 25. Prueba tiene las mismas 25 porque usa las categorías aprendidas en entrenamiento.

**Respuesta 2:** ¿las medianas que aprendió el imputador coinciden con las de entrenamiento o con las de todo el dataset? ¿Qué demuestra eso?

Coinciden con las de **entrenamiento**: el imputador aprendió 6.94 km y 1 pasajero, igual que la mediana de `X_ent`. La mediana de todo el dataset es 6.93 km. Demuestra que `fit_transform(X_ent)` solo vio entrenamiento y que `transform(X_prueba)` aplica esos valores sin recalcularlos: la prueba no influye en la preparación. Es la fuga del Ejercicio 1, corregida sin tener que acordarse del orden.

**Respuesta 3:** ¿qué le pasa al viaje desde el Bronx? ¿Qué habría pasado con `pd.get_dummies` aplicado por separado a ese viaje?

**Ninguna** de las 8 columnas de `zona_origen` queda en 1 (0 de 8): con `handle_unknown="ignore"`, una categoría que no vio en entrenamiento se codifica como un vector de ceros, y el viaje se transforma igual en 25 columnas. El destino sí se marca (`zona_destino_Midtown` = 1). Con `pd.get_dummies` aplicado solo a ese viaje, saldrían **3 columnas** (`zona_origen_Bronx`, `zona_destino_Midtown`, `metodo_pago_tarjeta`) en vez de 25: otros nombres, otra cantidad y otro orden. El modelo fallaría por la forma, o peor, leería columnas equivocadas si se alinean por posición. El `OneHotEncoder` guarda las categorías de entrenamiento; `get_dummies` las recalcula con lo que recibe.

## Ejercicio 3: el pipeline de tu proyecto

Usa tu archivo con las 2 features de la sesión pasada y la plantilla [`plantilla-pipeline/pipeline_proyecto.py`](./plantilla-pipeline/pipeline_proyecto.py).

### Paso 1: clasifica tus columnas (5 minutos)

Escribe qué columnas son numéricas, cuáles categóricas y cuáles ya están listas, y cuál es el objetivo. Deja fuera toda columna que no exista al momento de predecir.

_(tu clasificación)_

### Paso 2: decide cómo dividir (5 minutos)

¿Tu modelo predice el futuro a partir de filas con fecha? Si sí, divide por fecha; si no, al azar. Justifica.

_(tu decisión)_

### Paso 3: construye y corre el pipeline (15 minutos)

Completa la plantilla y córrela. Anota la forma de los conjuntos de entrenamiento y prueba ya preparados, y la lista de columnas que produce.

_(lo que imprimió tu pipeline)_

## Antes de la próxima sesión

**Lectura:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_
