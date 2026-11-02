# Semana 10, Sesión 2: Entrenamiento y comparación de modelos II: ¿ensamblaje, red neuronal simple o un LLM?

> Objetivo de la sesión: comparar, sobre el problema de tu proyecto, un modelo de ensamblaje, una red neuronal simple entrenada con una librería de alto nivel y un LLM usado vía prompting, y decidir cuál conviene según desempeño, costo y explicabilidad.

**Lectura previa de esta sesión:** ["Machine Learning vs LLM: Differences, Applications & Impact"](https://www.neuralconcept.com/post/ml-vs-llm-key-differences-applications-engineering-impact), Neural Concept.

**Referencia de esta sesión:** [Neural network models, supervised (documentación oficial de scikit-learn)](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

## Requisitos antes de esta sesión

- Nada nuevo que instalar: la red neuronal es `MLPRegressor`, de scikit-learn.
- Acceso a un LLM de chat (ChatGPT, Claude, Gemini u otro; la versión gratuita alcanza), para el Ejercicio 2.
- La tabla de resultados de tu proyecto de la sesión pasada.

## Antes de empezar

A partir de la lectura previa:

**Según el artículo, ¿para qué tipo de datos y de problema conviene el aprendizaje automático clásico, y para cuál un LLM?**

_(tu respuesta)_

**El artículo compara las dos opciones en datos, costo y explicabilidad. ¿Cuál de esas tres diferencias pesa más para tu proyecto?**

_(tu respuesta)_

## Pregunta de apertura

Para estimar la tarifa de un viaje, podrías entrenar un ensamblaje, entrenar una red neuronal o escribirle a un LLM los datos del viaje y pedirle un número. ¿Qué tendrías que medir de cada opción, además del error, para decidir cuál usar en la app de taxis?

_(tu respuesta)_

## Ejercicio 1: una red neuronal frente al ensamblaje

Archivo: [`starter/ejercicio1_red_neuronal.py`](./starter/ejercicio1_red_neuronal.py), sobre [`starter/viajes_features.csv`](./starter/viajes_features.csv). El script ya entrena el gradient boosting de la sesión pasada con la misma preparación y la misma división. Los 3 `TODO` son la red neuronal, cuántos parámetros aprendió y qué columnas usa más el ensamblaje. Córrelo una vez antes de tocar nada:

```
cd laboratorios/semana-10-sesion-02/starter
python3 ejercicio1_red_neuronal.py
```

**Respuesta 1 (TODO 1):** sin resolver, la red sale peor que el ensamblaje y aparece una advertencia `ConvergenceWarning`. ¿Qué crees que significa? ¿Qué esperas al dejarla entrenar hasta 1000 iteraciones?

Que el optimizador llegó al tope de iteraciones (200 por defecto) sin que el error dejara de bajar: la red quedó a medio entrenar, no es que el modelo sea malo. Con 1000 iteraciones espero que converja y que el error baje bastante, quizá al nivel del ensamblaje, a cambio de tardar más en entrenar.

**Respuesta 2 (TODO 2):** una capa oculta de 64 neuronas recibe las 25 columnas preparadas y entrega una tarifa. ¿Cuántos números tiene que aprender?

25 × 64 pesos de entrada a la capa oculta, más 64 sesgos de la capa oculta, más 64 pesos de la capa oculta a la salida, más 1 sesgo de salida: 1600 + 64 + 64 + 1 = 1729.

**Respuesta 3 (TODO 3):** ¿qué columna esperas que el ensamblaje use más? Si la app le muestra al conductor por qué la tarifa estimada es 70 dólares, ¿cuál de los dos modelos te permite responder?

La distancia, porque la tarifa se cobra por kilómetro; después `es_tarifa_fija`, que explica los 70 dólares de aeropuerto. El ensamblaje permite responder: sus importancias dicen qué columnas pesan, y una tarifa de 70 sale de los árboles que separan `es_tarifa_fija = 1`. La red reparte la decisión en 1729 pesos sin significado individual; para explicarla haría falta una técnica externa como SHAP.

**Lo que confirmó la salida, TODO por TODO:**

| Modelo | Error entrenamiento | Error prueba | Tiempo |
|---|---|---|---|
| Gradient boosting | 1.07 | 1.28 | 0.12 s |
| Red neuronal, sin resolver (200 iteraciones, `ConvergenceWarning`) | 1.60 | 1.69 | 0.34 s |
| Red neuronal, 1000 iteraciones | 1.09 | 1.17 | 1.03 s |

- La advertencia desapareció con 1000 iteraciones y la red pasó de 1.69 a 1.17 de error: terminó un poco mejor que el ensamblaje en prueba, pero tardó unas 8 veces más en entrenar.
- Parámetros de la red: 1729, como en la cuenta de la Respuesta 2.
- Las 3 columnas más importantes: `numericas__distancia_km` (0.9824), `ya_listas__es_tarifa_fija` (0.0120) y `ya_listas__es_hora_pico` (0.0028). La distancia se lleva casi todo, como esperaba.

## Durante el ConcepTest

**Tu voto, antes de discutir en pareja** (A, B, C o D):

_(tu voto)_

**Tu razonamiento:**

_(tu respuesta)_

**Después de discutir en pareja, ¿cambiaste de voto? ¿Por qué?**

_(tu respuesta)_

## Ejercicio 2: tres soluciones sobre los mismos 20 viajes

Solo o en pareja. Archivo: [`starter/ejercicio2_tres_soluciones.py`](./starter/ejercicio2_tres_soluciones.py), sobre el mismo `viajes_features.csv`. Los pasos 1 a 3 tienen `TODO`; los pasos 4 y 5 solo tienen la comprobación.

1. Completa los `TODO` 1 a 3 y corre el script. Además de entrenar, escribe `prompt_llm.txt`: 30 viajes con su tarifa como ejemplos y 20 viajes sin tarifa.
2. Copia `prompt_llm.txt` completo en un LLM de chat. Guarda los 20 números que responda, uno por línea, en `respuestas_llm.txt`, en la carpeta `starter`.
3. Completa los pasos 4 y 5 y vuelve a correr el script hasta que todas las comprobaciones digan `OK`. Para el paso 5, busca el precio por millón de tokens de entrada del modelo que usaste en la página de precios de su proveedor.

```
python3 ejercicio2_tres_soluciones.py
```

**Respuesta 1:** ¿qué LLM usaste? Completa la tabla con los números de tu corrida.

Usé Claude Sonnet 5.5, con el prompt pegado tal cual en una conversación nueva (sin acceso al CSV). Precio: 2 dólares por millón de tokens de entrada.

| | Error en los 20 viajes | Costo de entrenar | Costo de 10 000 viajes al mes | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting | 0.90 dólares | 0.12 s en la laptop | prácticamente 0 (predice 10 000 filas en milisegundos) | Sí, a nivel global: la importancia de columnas dice que la distancia pesa 0.98 de todo, y se puede inspeccionar cada árbol |
| Red neuronal | 0.96 dólares | 1.0 s en la laptop | prácticamente 0 | Difícil: son 1729 pesos sin significado individual |
| LLM | 1.32 dólares | 0 (no se entrena, aprende de 30 ejemplos en el prompt) | 0.73 dólares: 500 prompts × 726 tokens × 2 $/M | Puede escribir una explicación, pero no hay garantía de que sea la razón real del número |

El gradient boosting gana en error y costo. El LLM acierta los viajes de aeropuerto con tarifa fija de 70 porque los ve en los ejemplos, pero en los viajes cortos se desvía más que los modelos entrenados con 2380 filas.

**Respuesta 2:** corre el mismo prompt otra vez, en una conversación nueva. ¿Te dio los mismos 20 números? ¿Qué problema le trae eso a una app que le cobra al pasajero?

No. Corrí el mismo prompt en una conversación nueva (guardado en `starter/respuestas_llm_segunda_corrida.txt`) y cambiaron 15 de los 20 números. Los cambios son pequeños, el más grande fue de 1 dólar (16.0 contra 15.0 en Lower Manhattan → Midtown, 3.73 km), y el error promedio casi no se movió (1.32 a 1.31); los tres viajes de tarifa fija de aeropuerto siguieron en 70. Para una app que cobra, eso significa que el mismo viaje, cotizado dos veces, puede salir con dos precios distintos: no se puede reproducir ni auditar una tarifa, y un pasajero que compara cotizaciones lo vería como arbitrario. El gradient boosting, con los mismos datos y el mismo `random_state`, devuelve siempre el mismo número.

**Respuesta 3:** busca en tus 20 respuestas la peor predicción del LLM. ¿Qué tiene de particular ese viaje? ¿Lo habría aprendido el ensamblaje con las 2380 filas de entrenamiento?

La peor fue JFK Airport → JFK Airport, 1.01 km: el LLM dijo 7.6 y la tarifa real era 11.46 (error 3.86). Es un viaje que sale y llega al aeropuerto, muy corto, y entre los 30 ejemplos no hay ninguno parecido; el LLM lo trató como un viaje corto en Manhattan (los de ~1 km cuestan 7). Los dos únicos ejemplos de JFK tienen tarifa fija de 70, así que no tenía de dónde sacar que un viaje corto dentro del aeropuerto cuesta más. El ensamblaje lo aproximó mejor (12.74, error 1.28): en todo el CSV hay 9 viajes JFK → JFK, con 1.8 km y 14.29 dólares en promedio, así que en las 2380 filas de entrenamiento vio algunos y aprendió que ese viaje corto cuesta más que uno en Manhattan. Con tan pocos casos no lo aprende bien, pero tiene más información que los 30 ejemplos del prompt.

**Si terminas antes:** el error en los 20 viajes de muestra no es el mismo que en los 595 de prueba. ¿Por qué no se mide el LLM sobre los 595, y qué tan confiable es una comparación sobre 20?

Porque hay que pegar a mano 30 prompts en el chat (o pagar y programar las llamadas a la API), y un prompt con los 595 viajes sería largo y el modelo podría perder el orden o saltarse filas. Con 20 viajes la comparación es poco confiable: un solo viaje raro cambia el promedio (el JFK → JFK aporta 0.19 de los 1.32 del LLM), y la propia muestra no se parece a la prueba completa: el gradient boosting tiene 0.90 en los 20 y 1.28 en los 595. Sirve para ver órdenes de magnitud y tipos de error, no para decidir entre modelos que estén cerca.

## Ejercicio 3: ¿qué solución conviene para tu proyecto?

Usa tu pipeline y la tabla de resultados de la sesión pasada. Esta vez no hay plantilla.

### Entregable 1: `comparar_proyecto.py`

Un script que entrene, con tu pipeline y tu división, el mejor ensamblaje de la sesión pasada y una red neuronal simple (`MLPRegressor` o `MLPClassifier`), y registre para cada uno la métrica de prueba, los segundos de entrenamiento y, para la red, la cantidad de parámetros.

### Entregable 2: la prueba del LLM

Elige 10 filas de tu conjunto de prueba y escribe un prompt con algunas filas de entrenamiento como ejemplos, como el de `prompt_llm.txt`. Pásalo a un LLM y registra su métrica sobre esas 10 filas, junto con la de tus dos modelos sobre las mismas 10. Si tu problema no se puede plantear así (por ejemplo, porque cada predicción necesita una historia larga), explica por qué; eso también es una respuesta.

### Entregable 3: la decisión

Una tabla con las tres opciones (métrica, costo, explicabilidad) y un párrafo que diga cuál usarías en tu proyecto y por qué, con los números de tu tabla.

Archivos: [`proyecto/comparar_proyecto.py`](./proyecto/comparar_proyecto.py), [`proyecto/comparacion_proyecto.csv`](./proyecto/comparacion_proyecto.csv), [`proyecto/prompt_llm_proyecto.txt`](./proyecto/prompt_llm_proyecto.txt) y [`proyecto/respuestas_llm_proyecto.txt`](./proyecto/respuestas_llm_proyecto.txt).

Misma OBT, mismas columnas y misma división por fecha que la sesión pasada (123 awards de entrenamiento, 46 de prueba). Métrica: MAE sobre `log10(monto)`. El LLM fue Claude Sonnet 5.5 en una conversación nueva, con 30 awards de entrenamiento como ejemplos (con el monto en USD) y 10 awards de prueba sin monto.

| | MAE log en prueba (46) | MAE log en los 10 awards | Costo | ¿Puedes explicar una predicción? |
|---|---|---|---|---|
| Gradient boosting regularizado | 0.374 (2.37×) | 0.244 (1.75×) | 0.03 s de entrenamiento; predecir es gratis | Sí: importancias de columnas y los árboles se pueden inspeccionar; el monto sale de grupos parecidos de entrenamiento |
| Red neuronal (16 neuronas, `alpha=1`, `lbfgs`) | 0.339 (2.18×) | 0.176 (1.50×) | 0.06 s, 633 iteraciones, 369 parámetros | Difícil: 369 pesos sin significado individual |
| LLM (Sonnet 5.5, 30 ejemplos) | no medido | 0.258 (1.81×) | ~760 tokens por cada 10 awards; hay que pagar por cada consulta | Puede justificar en texto, pero no es la razón real del número, y no es reproducible |


Usaría el **gradient boosting regularizado** (o el Ridge de la sesión pasada, 0.361), no la red ni el LLM. La red gana en los números: 0.339 contra 0.374 en prueba, un factor de 2.18× contra 2.37×. Pero esa diferencia es de 46 awards, y elegí su configuración (capa chica, `alpha` alto) justamente porque con 123 filas sin regularizar sobreajusta; con tan pocos datos no me fío de una ventaja de 0.035. En GrantMatch el monto estimado se le muestra a un investigador que evalúa una convocatoria, y poder decir "se parece a los grants de NSF de 4 a 5 años, que pagan entre 300 y 600 mil" vale más que esa diferencia. El LLM quedó entre los dos en los 10 awards (0.258), pero respondió casi el mismo número para todos (entre 350 y 500 mil, salvo uno de 800 mil): copió la mediana de los ejemplos de NSF sin distinguir duración, y falló en los grants de más de un millón. Además los 10 de la muestra son casi todos grants de NSF, los más fáciles, y el LLM cuesta tokens por cada consulta y no da el mismo número dos veces. Cuando la OBT completa tenga más awards con monto, volvería a comparar la red contra el ensamblaje.

## Antes de la próxima sesión

**Lectura:** ["Version Controlling in Practice: Data, ML Model, and Code"](https://towardsdatascience.com/version-controlling-in-practice-data-ml-model-and-code-e13c518067dc/), Towards Data Science.

Dos compañeros reportan errores distintos para "el mismo" gradient boosting sobre los mismos datos de taxis. ¿Qué tendrías que saber de cada corrida para descubrir por qué no coinciden?

_(tu respuesta)_
