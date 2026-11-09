"""Pipeline del modelo de monto de awards.

Recoge en funciones lo que en la Semana 9 se hacia en los notebooks
(features_proyecto.py y pipeline_proyecto.py), para que el entrenamiento, el
lote nocturno y las pruebas usen exactamente el mismo codigo. Ninguna funcion
lee archivos: reciben y devuelven DataFrames.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OBJETIVO = "log_amount_usd"
NUMERICAS = ["duracion_anios", "institutions_count"]
CATEGORICAS = ["escala_financiamiento", "funder_name"]
YA_LISTAS = ["has_description"]
FEATURES = NUMERICAS + CATEGORICAS + YA_LISTAS

# Prueba son los awards que empiezan desde 2020 (cerca del 20% mas reciente).
CORTE = pd.Timestamp("2020-01-01")

# Las becas pagan a una persona; los acuerdos cooperativos, programas y
# contratos financian centros enteros. El resto son proyectos.
ESCALA = {
    "fellowship": "individual",
    "career_development": "individual",
    "training": "individual",
    "cooperative_agreement": "centro",
    "program": "centro",
    "contract": "centro",
}


def agregar_features(datos):
    """Agrega duracion_anios, escala_financiamiento y has_description como
    0/1. Si viene amount_usd, agrega tambien el objetivo en log10."""
    datos = datos.copy()
    datos["duracion_anios"] = datos["duration_days"] / 365.25
    datos["escala_financiamiento"] = datos["funding_type"].map(ESCALA).fillna("proyecto")
    datos["has_description"] = datos["has_description"].eq(True).astype(int)
    if "amount_usd" in datos.columns:
        datos[OBJETIVO] = np.log10(datos["amount_usd"].where(datos["amount_usd"] > 1))
    return datos


def una_fila_por_award(datos):
    """La OBT repite el award por cada publicacion financiada: deja una fila
    por award_key."""
    return datos.drop_duplicates("award_key").reset_index(drop=True)


def dividir_por_fecha(datos, corte=CORTE):
    """Entrenamiento son los awards que empiezan antes del corte; prueba, los
    que empiezan desde el corte. Los awards sin fecha de inicio se descartan."""
    fechas = pd.to_datetime(datos["award_start_date"])
    con_fecha = fechas.notna()
    datos, fechas = datos[con_fecha], fechas[con_fecha]
    return datos[fechas < corte], datos[fechas >= corte]


def construir_preparacion():
    """Imputa y escala las numericas, imputa y codifica las categoricas, y deja
    pasar has_description. Una categoria nueva se codifica como ceros."""
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


def construir_modelo(alpha=10.0):
    """El candidato final de la Semana 11: ridge con alpha=10."""
    return Pipeline([
        ("preparar", construir_preparacion()),
        ("modelo", Ridge(alpha=alpha)),
    ])
