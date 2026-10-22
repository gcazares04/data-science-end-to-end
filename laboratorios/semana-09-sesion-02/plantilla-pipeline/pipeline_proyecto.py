import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Archivo de la Sesion 1 con las 2 features (duracion_anios y
# escala_financiamiento) y el objetivo en log10 ya agregados.
datos = pd.read_csv("../../semana-09-sesion-01/plantilla-features/obt_features.csv")

# La OBT repite el award por cada publicacion financiada; el modelo predice
# el monto de un award, asi que queda una fila por award.
datos = datos.drop_duplicates("award_key").copy()
datos["award_start_date"] = pd.to_datetime(datos["award_start_date"])
datos["has_description"] = datos["has_description"].astype(int)

OBJETIVO = "log_amount_usd"
NUMERICAS = ["duracion_anios", "institutions_count"]
CATEGORICAS = ["escala_financiamiento", "funder_name"]
YA_LISTAS = ["has_description"]

X = datos[NUMERICAS + CATEGORICAS + YA_LISTAS]
y = datos[OBJETIVO]

# El modelo estima el monto de awards nuevos: se divide por fecha. Prueba
# son los awards que empiezan desde 2020 (cerca del 20% mas reciente).
CORTE = pd.Timestamp("2020-01-01")
datos = datos[datos["award_start_date"].notna()]
X, y = X.loc[datos.index], y.loc[datos.index]
es_prueba = datos["award_start_date"] >= CORTE
X_ent, X_prueba, y_ent, y_prueba = X[~es_prueba], X[es_prueba], y[~es_prueba], y[es_prueba]
print(f"Corte: {CORTE.date()}. Entrenamiento: {datos.loc[~es_prueba, 'award_start_date'].min().date()} a "
      f"{datos.loc[~es_prueba, 'award_start_date'].max().date()}; prueba: "
      f"{datos.loc[es_prueba, 'award_start_date'].min().date()} a {datos.loc[es_prueba, 'award_start_date'].max().date()}")

preparacion = ColumnTransformer([
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

X_ent_listo = preparacion.fit_transform(X_ent)
X_prueba_listo = preparacion.transform(X_prueba)
print("Entrenamiento:", X_ent_listo.shape, "Prueba:", X_prueba_listo.shape)
print(list(preparacion.get_feature_names_out()))
