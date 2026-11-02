"""Compara el mejor ensamblaje de GrantMatch (gradient boosting regularizado
de la Semana 10, Sesion 1) con una red neuronal simple, y prepara la prueba
del LLM sobre 10 awards de prueba.

    python3 comparar_proyecto.py

Escribe prompt_llm_proyecto.txt. Guarda los 10 montos en USD que responda el
LLM, uno por linea, en respuestas_llm_proyecto.txt y vuelve a correrlo.
"""

import os
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Mismos datos, columnas y division por fecha que modelos_proyecto.py.
datos = pd.read_csv("../../semana-09-sesion-01/plantilla-features/obt_features.csv")
datos = datos.drop_duplicates("award_key").copy()
datos["award_start_date"] = pd.to_datetime(datos["award_start_date"])
datos["has_description"] = datos["has_description"].astype(int)

OBJETIVO = "log_amount_usd"
NUMERICAS = ["duracion_anios", "institutions_count"]
CATEGORICAS = ["escala_financiamiento", "funder_name"]
YA_LISTAS = ["has_description"]
LEGIBLES = ["funder_name", "funding_type", "escala_financiamiento",
            "duracion_anios", "institutions_count", "has_description"]

datos = datos[datos["award_start_date"].notna()]
X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

CORTE = pd.Timestamp("2020-01-01")
es_prueba = datos["award_start_date"] >= CORTE
X_ent, X_prueba, y_ent, y_prueba = X[~es_prueba], X[es_prueba], y[~es_prueba], y[es_prueba]

# 10 awards de prueba para el LLM y 30 de entrenamiento como ejemplos.
muestra = X_prueba.sample(10, random_state=7)
y_muestra = y_prueba.loc[muestra.index]
ejemplos = X_ent.sample(30, random_state=7)
y_ejemplos = y_ent.loc[ejemplos.index]


def preparacion():
    return ColumnTransformer([
        ("numericas", Pipeline([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), NUMERICAS),
        ("categoricas", Pipeline([
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAS),
        ("ya_listas", "passthrough", YA_LISTAS),
    ])


# Metrica: MAE sobre log10(monto); 10**MAE es el factor por el que se
# equivoca en promedio.
def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    mae_muestra = mean_absolute_error(y_muestra, pipeline.predict(muestra))
    print(f"{nombre:32s} prueba {mae_prueba:.4f} ({10 ** mae_prueba:.2f}x)   "
          f"10 awards {mae_muestra:.4f}   {segundos:.2f} s")
    return pipeline, {
        "modelo": nombre,
        "mae_log_prueba": round(mae_prueba, 4),
        "factor_prueba": round(10 ** mae_prueba, 2),
        "mae_log_10_awards": round(mae_muestra, 4),
        "segundos": round(segundos, 3),
    }


ensamblaje, fila_ensamblaje = evaluar(
    "gradient boosting regularizado",
    GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                              subsample=0.8, random_state=42),
)
# Con 123 awards, una capa chica y regularizacion L2 (alpha) para no
# memorizar; lbfgs converge mejor que adam con tan pocas filas.
red, fila_red = evaluar(
    "red neuronal",
    MLPRegressor(hidden_layer_sizes=(16,), alpha=1.0, solver="lbfgs",
                 max_iter=2000, random_state=42),
)

modelo_red = red.named_steps["modelo"]
fila_red["parametros"] = sum(c.size for c in modelo_red.coefs_) + sum(i.size for i in modelo_red.intercepts_)
fila_red["iteraciones"] = modelo_red.n_iter_

filas = [fila_ensamblaje, fila_red]

# Prueba del LLM: los ejemplos van con su monto en USD (no en log) porque
# es como lo leeria una persona.
legibles = datos[LEGIBLES].copy()
legibles["duracion_anios"] = legibles["duracion_anios"].round(2)
filas_ejemplo = legibles.loc[ejemplos.index].copy()
filas_ejemplo["monto_usd"] = (10 ** y_ejemplos).round(0).astype(int).values
texto = (
    "Estimas el monto en dolares (USD) de un award de financiamiento de investigacion. "
    "Cada fila es un award: el funder que lo otorga, el tipo de financiamiento, la escala "
    "(individual = beca para una persona, proyecto, centro = financia un centro o programa), "
    "la duracion en anios, cuantas instituciones participan y si tiene descripcion (1 o 0). "
    "Abajo hay 30 awards con su monto real y 10 sin el. Responde solo con 10 numeros, "
    "uno por linea, en el mismo orden, sin texto adicional. Si un valor falta, esta vacio.\n\n"
    "Awards con monto:\n"
    + filas_ejemplo.to_csv(index=False)
    + "\nAwards sin monto:\n"
    + legibles.loc[muestra.index].to_csv(index=False)
)
with open("prompt_llm_proyecto.txt", "w") as archivo:
    archivo.write(texto)
print()
print("prompt_llm_proyecto.txt escrito:", len(texto), "caracteres, unos", len(texto) // 4, "tokens")

if os.path.exists("respuestas_llm_proyecto.txt"):
    with open("respuestas_llm_proyecto.txt") as archivo:
        montos_llm = [float(linea) for linea in archivo if linea.strip()]
    pred_llm = np.log10(montos_llm)
    mae_llm = mean_absolute_error(y_muestra, pred_llm)
    filas.append({"modelo": "LLM (30 ejemplos en el prompt)",
                  "mae_log_10_awards": round(mae_llm, 4), "segundos": 0.0})
    print(f"MAE del LLM en los 10 awards: {mae_llm:.4f} ({10 ** mae_llm:.2f}x)")

    detalle = legibles.loc[muestra.index, ["funder_name", "funding_type"]].copy()
    detalle["real_usd"] = (10 ** y_muestra).round(0)
    detalle["ensamblaje_usd"] = (10 ** ensamblaje.predict(muestra)).round(0)
    detalle["red_usd"] = (10 ** red.predict(muestra)).round(0)
    detalle["llm_usd"] = np.round(montos_llm, 0)
    print()
    print(detalle.to_string(index=False))
else:
    print("Falta respuestas_llm_proyecto.txt")

tabla = pd.DataFrame(filas)
tabla.to_csv("comparacion_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))
