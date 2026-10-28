# Semana 10, Sesión 1: Entrenamiento y comparación de modelos I: línea base y ensamblajes

> Objetivo de la sesión: entrenar un modelo de línea base y un modelo de ensamblaje sobre los datos preparados de tu proyecto, y aplicar regularización cuando el modelo se sobreajusta, sin derivar la matemática del descenso de gradiente.

**Lectura previa de esta sesión:** ["Ensembles in Machine Learning"](https://towardsdatascience.com/ensembles-in-machine-learning-9128215629d1/), Towards Data Science.

**Referencia de esta sesión:** [Ensemble methods (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/ensemble.html).

## Requisitos antes de esta sesión

Nada nuevo que instalar: todo lo de hoy está en scikit-learn, que ya usaste la sesión pasada. Ten a mano el pipeline de tu proyecto de la Semana 9, Sesión 2.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿qué es un ensamblaje, y en qué se diferencian bagging y boosting en la forma de combinar sus modelos?**

_(tu respuesta)_

**¿Por qué un promedio de muchos árboles puede equivocarse menos que el mejor de esos árboles por separado?**

_(tu respuesta)_

## Pregunta de apertura

Ya tienes los datos listos para entrenar. Si entrenas dos modelos sobre el mismo conjunto y uno da un error mucho menor en entrenamiento que en prueba, ¿qué le está pasando?

_(tu respuesta)_

## Ejercicio 1: línea base, un árbol y un bosque

Archivo: [`starter/ejercicio1_linea_base_y_ensamblaje.py`](./starter/ejercicio1_linea_base_y_ensamblaje.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv), el mismo de la sesión pasada. El script ya trae la preparación de la Semana 9 y una función `evaluar` que entrena cada modelo, mide el error absoluto medio en entrenamiento y en prueba, y lo registra en una tabla. Los 4 `TODO` son los modelos. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-01/starter
python3 ejercicio1_linea_base_y_ensamblaje.py
```

**Respuesta 1 (TODO 1):** la línea base predice siempre la mediana de la tarifa. ¿Cuánto esperas que se equivoque en promedio? ¿Para qué sirve un modelo que no usa ninguna columna?

Mucho: cerca de **12 a 13 USD por viaje**. La tarifa va de unos pocos dólares a 70 en JFK, y predecir 22.52 para todos falla por lo que cada viaje se aleja de la mediana. Sirve como **piso**: cualquier modelo que use columnas tiene que equivocarse bastante menos que esto. Si no lo hace, las features no están aportando, o hay un error en la preparación.

**Respuesta 2 (TODO 2):** un árbol sin límite puede seguir partiendo los datos hasta dejar un viaje por hoja. ¿Qué error esperas en entrenamiento, y qué esperas en prueba?

En entrenamiento, **casi cero**: cada hoja tiene un solo viaje y predice exactamente su tarifa. En prueba, bastante más, porque el árbol memorizó el ruido de cada viaje (propinas de redondeo, tráfico puntual) y no la regla general. Es sobreajuste.

**Respuesta 3 (TODO 3 y 4):** ¿qué esperas que pase con la brecha entre entrenamiento y prueba al limitar la profundidad? ¿Y con 200 árboles promediados?

Al limitar la profundidad, la brecha **baja**: el árbol ya no puede llegar a un viaje por hoja, así que el error en entrenamiento sube un poco y el de prueba baja. Con 200 árboles promediados también debería bajar el error en prueba: cada árbol ve una muestra distinta, sus errores no coinciden y al promediarlos se cancelan.

**Lo que confirmó la tabla, TODO por TODO:**

| Modelo | MAE entrenamiento | MAE prueba | Brecha |
|---|---|---|---|
| Línea base (mediana) | 13.1998 | 12.5308 | −0.6690 |
| Árbol sin límite | 0.0029 | 1.7017 | 1.6988 |
| Árbol `max_depth=8` | 1.0098 | 1.2728 | 0.2630 |
| Bosque aleatorio (200) | 0.4743 | 1.2989 | 0.8245 |

- **TODO 1:** la línea base falla 12.53 USD en prueba. Es el piso a superar.
- **TODO 2:** el árbol sin límite memoriza: 0.003 en entrenamiento contra 1.70 en prueba. Es la brecha más grande de la tabla.
- **TODO 3:** con `max_depth=8` la brecha baja de 1.70 a 0.26, y el error en prueba también baja, de 1.70 a 1.27. Regularizar mejoró el modelo.
- **TODO 4:** el bosque baja la prueba de 1.70 a 1.30 respecto del árbol sin límite. Pero no le gana al árbol limitado (1.27), y su brecha (0.82) sigue alta, porque cada uno de sus árboles crece sin límite. Aquí el promedio redujo la varianza, pero no lo suficiente para superar a un solo árbol bien regularizado. Además tarda unas 100 veces más en entrenar (0.73 s contra 0.007 s).

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: regularizar una regresión que se sobreajusta

Solo o en pareja. Archivo: [`starter/ejercicio2_regularizacion.py`](./starter/ejercicio2_regularizacion.py), sobre el mismo `viajes_features.csv`. Mismo esqueleto del Ejercicio 1; `evaluar` acepta además `grado_interacciones=3`, que agrega como columnas todos los productos entre columnas hasta grado 3. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación. Complétalos hasta que todas digan `OK`.

```
python3 ejercicio2_regularizacion.py
```

| Modelo | MAE entrenamiento | MAE prueba | Brecha |
|---|---|---|---|
| Línea base (mediana) | 13.1998 | 12.5308 | −0.6690 |
| Lineal grado 3 | 0.9111 | ≈ 3.77 × 10⁹ | ≈ 3.77 × 10⁹ |
| Ridge grado 3 (`alpha=10`) | 1.0272 | 1.3209 | 0.2938 |
| Gradient boosting | 1.0705 | **1.2754** | 0.2048 |

**Respuesta 1:** ¿cuántas columnas recibe la regresión con interacciones de grado 3, y cuántas filas de entrenamiento hay? ¿Qué tiene que ver esa comparación con el error en prueba que obtuviste?

Recibe **3275 columnas** y solo hay **2380 filas** de entrenamiento: más coeficientes que ejemplos. Con eso, la regresión tiene infinitas formas de pasar casi exactamente por los datos de entrenamiento (MAE 0.91). La que elige usa coeficientes enormes que se cancelan entre sí en entrenamiento. En prueba esas cancelaciones ya no se dan y las predicciones se disparan: MAE de unos **3,768 millones** de USD. Es el sobreajuste llevado al extremo.

**Respuesta 2:** ¿qué cambió entre `lineal grado 3` y `ridge grado 3` en el error de entrenamiento, y qué cambió en el de prueba? ¿Por qué Ridge empeora un poco en entrenamiento?

En entrenamiento, Ridge empeora un poco: de 0.91 a 1.03. En prueba mejora de forma enorme: de 3,768 millones a **1.32**. Ridge le suma al error un castigo por el tamaño de los coeficientes (`alpha` por la suma de sus cuadrados). Ya no le conviene usar coeficientes gigantes para ajustar cada viaje de entrenamiento, así que se ajusta un poco peor a esos datos. A cambio aprende una relación más suave, que sí sirve con viajes nuevos.

**Respuesta 3:** ¿qué modelo quedó mejor en prueba? ¿Cuánto mejor que la línea base?

**Gradient boosting**, con MAE de **1.28** en prueba y la brecha más chica de los cuatro (0.20). Se equivoca 11.26 USD menos por viaje que la línea base (12.53), un **90% menos** de error.

**Si terminas antes:** la brecha de la línea base es negativa (se equivoca menos en prueba que en entrenamiento). ¿Cómo puede pasar eso con un modelo que no aprende nada?

Por **azar en la división**. La línea base predice 22.52 para todos los viajes, así que su error depende solo de qué tan lejos de la mediana caen las tarifas de cada conjunto. Al dividir al azar, al conjunto de prueba le tocaron menos viajes extremos que al de entrenamiento: **8.9%** de viajes de 70 USD (tarifa fija a JFK), contra **10.4%** en entrenamiento. No es que generalice mejor: es que su examen salió más fácil. Con otra división cambia: con `random_state=0` la brecha es +0.33 y con `random_state=3` es +0.59.

## Ejercicio 3: los primeros modelos de tu proyecto

Usa tu pipeline de la Semana 9, Sesión 2, con la misma división de entrenamiento y prueba. Esta vez no hay plantilla: escribe el script tú.

### Entregable 1: `modelos_proyecto.py`

Un script, en la carpeta de tu proyecto, que entrene con tu pipeline:

- Una línea base. Elige la que corresponde a tu problema y justifícala abajo: `DummyRegressor` o `DummyClassifier`, o una regla del dominio (por ejemplo, "mañana igual que hoy" si predices una serie en el tiempo).
- Un modelo lineal (`Ridge` si es regresión, `LogisticRegression` si es clasificación), con la regularización que haga falta.
- Un ensamblaje (`RandomForestRegressor`, `GradientBoostingRegressor` o sus versiones `Classifier`).

### Entregable 2: `resultados_proyecto.csv`

Una fila por modelo, con la métrica de tu proyecto en entrenamiento y en prueba, la brecha y los segundos de entrenamiento.

Script: [`plantilla-modelos/modelos_proyecto.py`](./plantilla-modelos/modelos_proyecto.py). Tabla: [`plantilla-modelos/resultados_proyecto.csv`](./plantilla-modelos/resultados_proyecto.csv).

Usa el mismo archivo, las mismas columnas y la misma división por fecha que el pipeline de la Semana 9: 123 awards de entrenamiento (1988 a 2019) y 46 de prueba (2020 a 2025). La métrica es el **MAE sobre `log10(monto)`**, que se lee como un factor: un MAE de 0.36 es equivocarse en promedio por un factor de 10^0.36 ≈ 2.3× en el monto.

| Modelo | MAE log ent. | MAE log prueba | Brecha | Factor en prueba | Segundos |
|---|---|---|---|---|---|
| Línea base (mediana) | 0.5158 | 0.3690 | −0.1468 | 2.34× | 0.004 |
| Regla: mediana por funder | 0.3821 | 0.4975 | 0.1154 | 3.14× | 0.000 |
| Lineal | 0.3206 | 0.3967 | 0.0762 | 2.49× | 0.004 |
| Ridge `alpha=1` | 0.3431 | 0.3832 | 0.0401 | 2.42× | 0.003 |
| **Ridge `alpha=10`** | 0.3936 | **0.3608** | −0.0328 | **2.30×** | 0.002 |
| Bosque aleatorio | 0.1786 | 0.4233 | 0.2447 | 2.65× | 0.065 |
| Bosque `min_samples_leaf=5` | 0.3629 | 0.3899 | 0.0270 | 2.45× | 0.057 |
| Gradient boosting | 0.1749 | 0.4101 | 0.2353 | 2.57× | 0.015 |
| Gradient boosting regularizado | 0.2810 | 0.3742 | 0.0932 | 2.37× | 0.013 |

### Entregable 3: respuestas

**¿Cuál es tu línea base, y por qué esa y no otra?**

Un `DummyRegressor` con la **mediana** de `log10(monto)` de entrenamiento. Es regresión sobre datos sin orden de serie (cada award es independiente), así que no aplica "mañana igual que hoy". Uso la mediana y no la media porque la métrica es el MAE, y el valor constante que minimiza el MAE es la mediana. Además, la uso sobre el logaritmo para que los pocos awards de decenas de millones no la arrastren.

Probé también una regla del dominio, **la mediana de cada funder**, que es lo que haría alguien que conoce los programas. Salió **peor** que la mediana global (0.50 contra 0.37 en prueba), por dos razones:
- La mediana de NSF, que pone 34 de los 46 awards de prueba, subió de 5.56 a 5.78 después de 2020. La mediana global (5.645) quedó más cerca de esos montos nuevos.
- Los funders con 1 o 2 awards en entrenamiento tienen medianas extremas que no se repiten: Department of Defense pasa de 6.64 a 5.58 y Office of Science de 6.67 a 5.55.

**¿Alguno de tus modelos se sobreajusta? Muestra la evidencia con números de tu tabla.**

Sí, los dos ensamblajes sin limitar. El **bosque aleatorio** baja a 0.18 en entrenamiento pero sube a 0.42 en prueba: brecha de 0.24, la más grande de la tabla. El **gradient boosting** con parámetros por defecto hace lo mismo: 0.17 contra 0.41, brecha de 0.24. Los dos le pueden dedicar ramas a funders que tienen un solo award y memorizar su monto. Con 123 awards y 21 columnas no hay suficientes datos para eso, y en prueba los dos quedan **peor que la línea base** (0.42 y 0.41 contra 0.37).

La regresión lineal sin regularizar se sobreajusta menos (brecha de 0.08), pero también queda peor que la línea base en prueba (0.40).

**¿Qué regularización aplicaste (o por qué no hizo falta), y qué cambió en la tabla?**

Regularicé los tres tipos de modelo:
- **Ridge** en el lineal: con `alpha=1` la prueba baja de 0.397 a 0.383, y con `alpha=10` a **0.361**, el mejor resultado de la tabla. El castigo encoge los coeficientes de los funders con un solo award, que son los que el modelo lineal sobreajustaba.
- **`min_samples_leaf=5`** en el bosque, para que ninguna hoja dependa de menos de 5 awards: la brecha baja de 0.24 a 0.03 y la prueba de 0.42 a 0.39.
- En el gradient boosting: árboles de profundidad 2, `learning_rate=0.05` y `subsample=0.8`. La brecha baja de 0.24 a 0.09 y la prueba de 0.41 a 0.37.

En los tres casos el error en entrenamiento sube y el de prueba baja, que es lo esperado al regularizar.

**¿Cuánto mejora tu mejor modelo a la línea base?**

Casi nada. **Ridge `alpha=10`** da 0.361 en prueba contra 0.369 de la línea base: 0.008 menos en log10, un **2%** menos de error. En monto, el factor promedio pasa de 2.34× a 2.30×. Con 46 awards de prueba esa diferencia está dentro del ruido, así que no puedo afirmar que el modelo le gane a la mediana.

La lectura honesta es que, con estas columnas, el monto de un award casi no se puede predecir mejor que con la mediana. Tiene sentido por tres cosas:
- Prueba está dominada por NSF (34 de 46), cuyos montos cambiaron después de 2020.
- La mayoría de los funders tienen muy pocos awards para aprender algo de ellos.
- Prueba es más homogénea que entrenamiento (desviación estándar de 0.58 contra 0.74), y eso favorece a predecir un valor constante. Por lo mismo, la brecha de la línea base sale negativa.

Para mejorar haría falta información que sí explique el monto, como el programa o la división de NSF. `escala_financiamiento` intenta separar becas individuales de centros, pero casi no discrimina: sale de `funding_type`, y 150 de los 169 awards son `grant` o `research`, así que caen todos en "proyecto".

## Antes de la próxima sesión

**Lectura:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_
