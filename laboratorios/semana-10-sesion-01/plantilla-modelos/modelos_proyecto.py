import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Mismos datos, columnas y division por fecha que el pipeline de la
# Semana 9, Sesion 2: una fila por award, objetivo en log10.
datos = pd.read_csv("../../semana-09-sesion-01/plantilla-features/obt_features.csv")
datos = datos.drop_duplicates("award_key").copy()
datos["award_start_date"] = pd.to_datetime(datos["award_start_date"])
datos["has_description"] = datos["has_description"].astype(int)

OBJETIVO = "log_amount_usd"
NUMERICAS = ["duracion_anios", "institutions_count"]
CATEGORICAS = ["escala_financiamiento", "funder_name"]
YA_LISTAS = ["has_description"]

datos = datos[datos["award_start_date"].notna()]
X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

CORTE = pd.Timestamp("2020-01-01")
es_prueba = datos["award_start_date"] >= CORTE
X_ent, X_prueba, y_ent, y_prueba = X[~es_prueba], X[es_prueba], y[~es_prueba], y[es_prueba]
print(f"Entrenamiento: {len(X_ent)} awards; prueba: {len(X_prueba)} awards (desde {CORTE.date()})")


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


resultados = []


# Metrica: MAE sobre log10(monto). Un MAE de 0.3 es equivocarse en promedio
# por un factor de 10**0.3 = 2x en el monto.
def evaluar(nombre, modelo):
    pipeline = Pipeline([("preparar", preparacion()), ("modelo", modelo)])
    inicio = time.perf_counter()
    pipeline.fit(X_ent, y_ent)
    segundos = time.perf_counter() - inicio
    mae_ent = mean_absolute_error(y_ent, pipeline.predict(X_ent))
    mae_prueba = mean_absolute_error(y_prueba, pipeline.predict(X_prueba))
    resultados.append({
        "modelo": nombre,
        "mae_log_entrenamiento": round(mae_ent, 4),
        "mae_log_prueba": round(mae_prueba, 4),
        "brecha": round(mae_prueba - mae_ent, 4),
        "factor_prueba": round(10 ** mae_prueba, 2),
        "segundos": round(segundos, 3),
    })
    print(f"{nombre:30s} entrenamiento {mae_ent:.4f}   prueba {mae_prueba:.4f}   (factor {10 ** mae_prueba:.2f}x)")
    return pipeline


# Linea base: la mediana del monto de entrenamiento. Es lo que haria alguien
# sin modelo; la mediana y no la media porque el objetivo esta en log10 y
# el MAE se minimiza con la mediana.
evaluar("linea base (mediana)", DummyRegressor(strategy="median"))

# Linea base del dominio: la mediana del monto de cada funder en
# entrenamiento, la regla que usaria alguien que conoce los programas.
mediana_por_funder = y_ent.groupby(X_ent["funder_name"]).median()
pred_ent = X_ent["funder_name"].map(mediana_por_funder).fillna(y_ent.median())
pred_prueba = X_prueba["funder_name"].map(mediana_por_funder).fillna(y_ent.median())
mae_ent, mae_prueba = mean_absolute_error(y_ent, pred_ent), mean_absolute_error(y_prueba, pred_prueba)
resultados.append({
    "modelo": "regla: mediana por funder",
    "mae_log_entrenamiento": round(mae_ent, 4),
    "mae_log_prueba": round(mae_prueba, 4),
    "brecha": round(mae_prueba - mae_ent, 4),
    "factor_prueba": round(10 ** mae_prueba, 2),
    "segundos": 0.0,
})
print(f"{'regla: mediana por funder':30s} entrenamiento {mae_ent:.4f}   prueba {mae_prueba:.4f}   (factor {10 ** mae_prueba:.2f}x)")

# Modelo lineal: sin regularizar y con Ridge, para ver cuanto aporta el
# castigo con 123 awards y 21 columnas.
evaluar("lineal", LinearRegression())
for alpha in [1.0, 10.0]:
    evaluar(f"ridge alpha={alpha:g}", Ridge(alpha=alpha))

# Ensamblajes: un bosque sin limite y regularizado, y gradient boosting.
evaluar("bosque aleatorio", RandomForestRegressor(n_estimators=200, random_state=42))
evaluar("bosque min_samples_leaf=5", RandomForestRegressor(n_estimators=200, min_samples_leaf=5, random_state=42))
evaluar("gradient boosting", GradientBoostingRegressor(random_state=42))
evaluar("gradient boosting regularizado",
        GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42))

tabla = pd.DataFrame(resultados)
tabla.to_csv("resultados_proyecto.csv", index=False)
print()
print(tabla.to_string(index=False))
