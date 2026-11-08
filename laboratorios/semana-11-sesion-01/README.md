# Semana 11, Sesión 1: Gestión y trazabilidad de experimentos: eligiendo el modelo ganador con evidencia

> Objetivo de la sesión: registrar de forma trazable los parámetros, métricas y artefactos de las corridas de entrenamiento de las dos sesiones anteriores, y elegir el modelo final de tu proyecto con esa evidencia.

**Lectura previa de esta sesión:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

**Referencia de esta sesión:** [ML Experiment Tracking (documentación oficial de MLflow)](https://mlflow.org/docs/latest/ml/tracking/).

## Requisitos antes de esta sesión

Hoy se agrega una herramienta: MLflow, que registra cada corrida de entrenamiento. Como en un equipo de datos, el registro vive en un servidor de seguimiento (tracking server), no en tu notebook: el Codespace levanta uno en el puerto 5050 cada vez que inicia, y los notebooks lo encuentran por la variable de entorno `MLFLOW_TRACKING_URI`.

Si tu Codespace se creó antes de esta sesión, sincroniza tu fork y reconstruye el contenedor (`Ctrl+Shift+P` o `Cmd+Shift+P` → "Codespaces: Rebuild Container"; en tu laptop, "Dev Containers: Rebuild Container"). Reconstruir instala MLflow y configura el servidor; tus archivos no se pierden. Al terminar, comprueba desde una terminal:

```
bash .devcontainer/verificar.sh
```

Las últimas líneas deben decir `mlflow, version 2.17.2` y `servidor de MLflow: responde en http://127.0.0.1:5050`. Si el servidor no responde (por ejemplo, después de detener y volver a abrir el Codespace sin que arranque), levántalo a mano con `bash .devcontainer/mlflow-server.sh`.

La interfaz de MLflow se abre desde la pestaña **Ports** de VS Code: el puerto 5050, "MLflow (Semana 11)", con el ícono del globo. Ahí ves cada experimento, sus corridas, y puedes compararlas lado a lado.

Los laboratorios de hoy son notebooks, como los de la Semana 10, y la preparación se importa de `preparacion_taxis.py`. El registro (la base de datos del servidor y los modelos guardados) queda en la carpeta `.mlflow/` de la raíz del repositorio, que no se sube (está en `.gitignore`): lo que entregas es el notebook y la tabla de corridas que exporta.

**Si trabajas en tu laptop sin el contenedor (Opción B del README):** instala con `pip install -r requirements.txt` y, en una terminal aparte desde la raíz de tu fork, levanta el mismo servidor con `mlflow server --backend-store-uri sqlite:///.mlflow/mlflow.db --artifacts-destination .mlflow/artefactos --port 5050`. Antes de abrir los notebooks, define la variable `MLFLOW_TRACKING_URI=http://127.0.0.1:5050`. Sin esa variable, MLflow guarda el registro en una carpeta `mlruns` al lado de cada notebook, sin servidor: los notebooks también funcionan, pero cada carpeta tiene su propio registro.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué tres cosas hay que versionar en un proyecto de machine learning, y con qué herramienta versiona el autor cada una?**

Los **datos**, el **modelo** y el **código**. El autor versiona el código con **Git**, los datos con **DVC** (Data Version Control, que corre encima de un repositorio de Git) y el modelo con **MLflow**, que registra por cada corrida los parámetros, las métricas, los artefactos y el commit de Git con el que se entrenó.

**¿Por qué no alcanza con Git para versionar los datos?**

Git está pensado para archivos de texto chicos que cambian línea por línea. Un dataset suele ser grande y a veces binario: Git guarda una copia completa por cada versión, el repositorio crece sin control, GitHub rechaza archivos de más de 100 MB y el diff de un CSV de millones de filas o de un parquet no se puede leer. DVC resuelve eso guardando los datos en un almacenamiento aparte (S3, una carpeta remota) y versionando en Git solo un archivo `.dvc` chico con el hash del dataset: el commit apunta a la versión exacta de los datos sin cargarlos en el repositorio.

## Pregunta de apertura

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

Lo mismo que respondí al cerrar la Semana 10: qué versión exacta de los datos usó cada uno (un hash del archivo), cómo dividió entrenamiento y prueba (semilla o fecha de corte, y el tamaño de la prueba), los hiperparámetros completos y el `random_state` del modelo, el commit del código de preparación y las versiones de las librerías. Si todo eso coincide y el número no, el problema es otro; si no coincide, ya sé por dónde empezar. Hoy la diferencia sale de la división: es lo que muestra el Ejercicio 1.

## Ejercicio 1: registrar cada corrida con MLflow

Notebook: [`starter/ejercicio1_registrar_corridas.ipynb`](./starter/ejercicio1_registrar_corridas.ipynb), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), con la preparación de [`starter/preparacion_taxis.py`](./starter/preparacion_taxis.py). Entrena el bosque y el gradient boosting de la Semana 10, y después la corrida de un compañero. Los 4 `TODO` son el registro. Córrelo completo una vez antes de tocar nada (**Run All**).

**Respuesta 1 (antes de los TODO):** sin resolver, el notebook entrena los tres modelos e imprime sus errores. ¿Cuál de los dos gradient boosting tiene razón? ¿Qué parte de la salida te permite saberlo, y qué parte no?

Los dos. Sin resolver, el notebook imprime `gradient boosting division 42 ... prueba 1.2754` y `gradient boosting division 7 ... prueba 1.1867`: es el mismo modelo, con los mismos hiperparámetros y el mismo CSV, medido sobre dos conjuntos de prueba distintos de 595 viajes. Ninguno se equivocó; están midiendo cosas distintas. Lo que me deja saberlo es que la salida imprime la semilla de la división en cada línea. Lo que no me deja saberlo: la salida no muestra los hiperparámetros ni qué archivo de datos usó cada uno, así que tengo que confiar en el código; y nada queda guardado: la última celda dice `Todavia no existe el experimento tarifa-taxis`, y al reiniciar el kernel esos tres números desaparecen.

**Respuesta 2 (TODO 2 y 3):** de todo lo que se puede registrar de una corrida, ¿qué crees que explica la diferencia entre los dos gradient boosting? ¿Qué parámetro esperas ver distinto en la tabla del paso 5?

La división. Los dos gradient boosting tienen el mismo `random_state=42`, los mismos hiperparámetros por defecto y leen el mismo archivo, así que espero ver distinto solo `semilla_division` (42 contra 7), con la misma `huella_datos` y el mismo `n_estimators` (100). Es lo que salió en la tabla del paso 5: la huella es `4800204d50c0` en las tres corridas y lo único que cambia entre los dos gradient boosting es la semilla de la división.

**Respuesta 3 (TODO 4):** el registro ya tiene los parámetros y las métricas. ¿Para qué guardar además el modelo entrenado?

Porque los parámetros describen cómo se entrenó, pero no son el modelo. Volver a entrenar puede no dar exactamente lo mismo (otra versión de scikit-learn, otro orden de filas, un modelo no determinista), y el artefacto es el objeto exacto que produjo el número registrado, con la preparación ya ajustada adentro (las medianas, la media y la escala, las categorías del one-hot). Con el artefacto puedo cargar el modelo y predecir sin reentrenar, comprobar que reproduce la métrica (el paso 5 del Ejercicio 2) y servir ese mismo modelo en la Semana 12. MLflow además guarda junto al modelo la versión de Python y las dependencias con las que se entrenó.

**Lo que mostró la tabla del paso 5:**

| Corrida | Semilla de la división | Huella de los datos | `n_estimators` | Error en prueba | Segundos |
|---|---|---|---|---|---|
| gradient boosting | 7 | `4800204d50c0` | 100 | 1.1867 | 0.13 |
| gradient boosting | 42 | `4800204d50c0` | 100 | 1.2754 | 0.13 |
| bosque aleatorio | 42 | `4800204d50c0` | 200 | 1.2989 | 0.72 |

La huella es la misma en las tres corridas: los datos no cambiaron. Entre los dos gradient boosting solo cambia la semilla de la división, y con ella el error en prueba se mueve 0.09 dólares. Con la semilla 42, el gradient boosting le gana al bosque por 0.02, mucho menos que esa diferencia.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: comparar con evidencia y elegir

Solo o en pareja. Notebook: [`starter/ejercicio2_comparar_y_elegir.ipynb`](./starter/ejercicio2_comparar_y_elegir.ipynb), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1, con los cinco modelos de la Semana 10 en un experimento propio. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

**Respuesta 1:** ¿qué modelo quedó primero en el registro, y por cuánto le gana al segundo?

La **red neuronal**, con 1.1695 dólares de error absoluto medio en prueba. Le gana al gradient boosting (1.2754) por 0.106 dólares, un 8%. Después vienen el bosque (1.2989), Ridge (2.5565) y la línea base de la mediana (12.53). Es el mismo orden de la Semana 10, ahora leído del registro con `search_runs(order_by=["metrics.mae_prueba ASC"])` y no de una tabla en memoria.

**Respuesta 2:** el paso 5 carga el modelo ganador desde el registro y vuelve a calcular su error. ¿Qué te garantiza que coincida con el número registrado? ¿Qué no te garantiza sobre ese modelo?

Que coincida (1.169505229635501 registrado y reproducido) garantiza que el artefacto guardado es el modelo exacto que produjo esa métrica, que se guardó y se cargó bien, y que estoy midiendo sobre los mismos 595 viajes de prueba. No garantiza que ese modelo sea el mejor de verdad ni que vaya a tener ese error con viajes nuevos: es una sola división, y el Ejercicio 1 mostró que solo cambiar la semilla mueve el error del gradient boosting 0.09, casi lo mismo que la ventaja de la red. Además elegí al ganador mirando el mismo conjunto de prueba con el que lo mido, así que el número es algo optimista. Tampoco dice nada de lo que pase en producción si cambian los viajes.

**Respuesta 3:** si mañana alguien cambia una fila de `viajes_features.csv` y vuelve a entrenar, ¿cómo lo notarías mirando solo el registro?

Por la columna `params.huella_datos`: cambiar un solo byte del CSV cambia el hash MD5, así que las corridas nuevas tendrían una huella distinta de `4800204d50c0`. Si la fila se agrega o se borra, también cambia `filas_entrenamiento`. Filtrando el registro por huella se ve de inmediato qué corridas se entrenaron con cada versión de los datos, y que sus errores no se pueden comparar directamente.

**Si terminas antes:** abre la interfaz de MLflow (pestaña **Ports**, puerto 5050), entra al experimento `tarifa-taxis-comparacion`, selecciona las cinco corridas y compáralas lado a lado con **Compare**. Además del error, ¿qué te dice el registro de la red neuronal que no te diría una tabla escrita a mano?

Todos sus hiperparámetros, incluidos los que nunca escribí: `hidden_layer_sizes=(64,)`, `max_iter=1000`, pero también `solver=adam`, `activation=relu`, `alpha=0.0001`, `learning_rate_init=0.001`, `batch_size=auto`. Una tabla a mano diría "red neuronal" y nada más. También dice que tardó 4.3 s en entrenar contra 0.12 s del gradient boosting (35 veces más), que su brecha entrenamiento-prueba es chica (1.088 contra 1.170), con qué huella de datos y cuántas filas se entrenó, cuándo y quién la corrió, y tiene el modelo guardado para cargarlo. En **Compare** se ven las cinco corridas lado a lado con las diferencias de parámetros resaltadas.

## Ejercicio 3: el registro de tu proyecto

Usa tu `pipeline_proyecto.py` y los modelos que entrenaste en las dos sesiones de la Semana 10. Esta vez no hay plantilla.

### Entregable 1: `experimentos_proyecto.ipynb`

Un notebook, al lado de tu `pipeline_proyecto.py`, que importe de ahí la preparación y la división y registre en MLflow, en un experimento con el nombre de tu proyecto, una corrida por cada modelo de la Semana 10 (línea base, modelo lineal, ensamblaje, red neuronal). Cada corrida registra:

- Los hiperparámetros del modelo.
- Cómo se dividieron los datos (la semilla, o la fecha de corte si divides por fecha) y la huella del archivo de datos.
- La métrica de tu proyecto en entrenamiento y en prueba, y los segundos de entrenamiento.
- El pipeline entrenado, como artefacto.

Al final, el notebook marca la corrida que eliges como final (`candidato` = `final`), carga su modelo desde el registro y comprueba que reproduce la métrica registrada.

### Entregable 2: `corridas_proyecto.csv`

La tabla de corridas de tu experimento, exportada con `mlflow.search_runs(...).to_csv("corridas_proyecto.csv", index=False)` desde el mismo notebook. Antes de hacer commit, borra las corridas de prueba del experimento de tu proyecto (en la interfaz de MLflow: selecciónalas y **Delete**), reinicia el kernel y corre todo: la tabla tiene que salir de una corrida completa.

### Entregable 3: la decisión

**¿Qué modelo elegiste como final? Escribe su `run_id` y los números del registro que sostienen la decisión.**

**Ridge con `alpha=10`**, `run_id` **`505f6532318444f3a3fc7a1f03785738`** en el experimento `grantmatch` (etiqueta `candidato` = `final`).

| Corrida | MAE log entrenamiento | MAE log prueba | Factor en prueba | Segundos |
|---|---|---|---|---|
| red neuronal (16, `alpha=1`, `lbfgs`) | 0.259 | **0.339** | 2.18× | 0.065 |
| **ridge alpha=10** | 0.394 | **0.361** | 2.30× | 0.004 |
| línea base (mediana) | 0.516 | 0.369 | 2.34× | 0.003 |
| gradient boosting regularizado | 0.281 | 0.374 | 2.37× | 0.015 |
| ridge alpha=1 | 0.343 | 0.383 | 2.42× | 0.004 |
| bosque `min_samples_leaf=5` | 0.363 | 0.390 | 2.45× | 0.059 |
| lineal | 0.321 | 0.397 | 2.49× | 0.004 |
| gradient boosting | 0.175 | 0.410 | 2.57× | 0.015 |
| bosque aleatorio | 0.179 | 0.423 | 2.65× | 0.065 |

El registro confirma lo que decidí en la Semana 10. La red gana en prueba por 0.022 (2.18× contra 2.30×), pero sobre 46 awards, y su configuración la elegí mirando esos mismos 46. Ridge es el mejor de los modelos explicables, el único que no sobreajusta (su error de entrenamiento es mayor que el de prueba; todos los árboles tienen brechas de 0.1 a 0.25) y entrena en 4 milisegundos. Lo que el registro deja claro, y que tengo que decir: Ridge apenas le gana a la mediana (0.361 contra 0.369). Con 123 awards de entrenamiento y estas 5 columnas, ningún modelo se separa mucho de la línea base; la Sesión 2 tiene que decir si esa ventaja es real o ruido de la división. La red pasó de 0.3386 en la Semana 10 a 0.3389 aquí por la versión de scikit-learn del entorno (1.5.2, la de `requirements.txt`); el resto de los números coincide.

**Si otra persona clona tu repositorio, ¿qué necesita para obtener exactamente el mismo número? ¿Qué de eso quedó registrado en MLflow y qué no?**

Necesita:

1. **Los mismos datos:** `obt_features.csv` con huella `7503142bffe6`. Está en el repositorio (`semana-09-sesion-01/plantilla-features/`).
2. **El mismo código:** `pipeline_proyecto.py` (deduplicación por `award_key`, preparación y corte por fecha) y el notebook.
3. **La misma división:** el corte en 2020-01-01. Es por fecha, así que no depende de ninguna semilla.
4. **Los mismos hiperparámetros y `random_state`.**
5. **Las mismas versiones:** Python 3.11 y scikit-learn 1.5.2, pandas 2.2.2 y numpy 1.26.4. El cambio de 0.3386 a 0.3389 en la red muestra que otra versión de scikit-learn ya mueve el número.

**Quedó registrado en MLflow:**

- los hiperparámetros completos;
- la fecha de corte y las filas de entrenamiento y prueba;
- la huella de los datos;
- las métricas y los segundos;
- el commit de Git como etiqueta (`commit`);
- el modelo entrenado. Con él, MLflow guarda la versión de Python y las dependencias (`requirements.txt`, `conda.yaml`).

**No quedó registrado:**

- **Los datos.** MLflow guarda la huella, no el archivo: si alguien cambia el CSV, el registro avisa que no coincide, pero no permite recuperarlo. Para eso estaría DVC.
- **El código del notebook en ese momento.** El commit registrado es el último commit del repositorio, no los cambios sin commitear con los que corrí.
- **Cómo se construyó la OBT desde las fuentes** (las semanas 7 y 8).
- **El registro mismo.** La base de MLflow vive en `.mlflow/`, que está en `.gitignore`. Quien clone el repositorio solo ve `corridas_proyecto.csv`, no puede abrir las corridas ni cargar los modelos. Tiene que correr el notebook con su propio servidor y comparar contra el CSV.

## Antes de la próxima sesión

**Lectura:** ["How Metrics (and LLMs) Can Trick You: A Field Guide to Paradoxes"](https://towardsdatascience.com/how-metrics-and-llms-can-trick-you-a-field-guide-to-paradoxes/), Towards Data Science.

Hoy elegiste el modelo con menor error sobre los 595 viajes de prueba. ¿Cuánto cambiaría ese error si la división hubiera sido otra? ¿Lo viste en algún lado del registro?

Bastante, comparado con las diferencias entre modelos. El Ejercicio 1 lo midió sin buscarlo: el mismo gradient boosting dio 1.2754 con la semilla 42 y 1.1867 con la semilla 7. Esos 0.09 dólares son casi lo mismo que la ventaja de la red sobre el gradient boosting (0.106) con la que la elegí como ganadora. En el registro solo lo vi porque el Ejercicio 1 corrió a propósito una segunda división. Cada corrida guarda un único número, sin ninguna medida de cuánto varía, así que el registro hace parecer exacta una comparación que puede ser ruido. En GrantMatch es peor: la prueba son 46 awards, no 595. Para saberlo habría que repetir la evaluación sobre varias divisiones (validación cruzada) y mirar la media y la dispersión, no un solo número.
