"""Entrena el modelo de monto, mide su error y lo registra como una version
nueva de "grantmatch-monto" en el registro de modelos de MLflow.

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python -m servicio.registrar_modelo        # ridge alpha=10, el candidato final
    python -m servicio.registrar_modelo 1      # otro alpha

Cada vez que lo corres se crea una version nueva, con su corrida y su
metrica: ninguna reemplaza a otra. Para ponerla en produccion, usa
servicio/promover.py.
"""

import hashlib
import logging
import os
import sys
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error

from modelado.pipeline_proyecto import (
    CORTE,
    FEATURES,
    OBJETIVO,
    agregar_features,
    construir_modelo,
    dividir_por_fecha,
    una_fila_por_award,
)

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "grantmatch-monto"
EXPERIMENTO = "grantmatch-registro"
RUTA_DATOS = (Path(__file__).resolve().parents[3]
              / "laboratorios/semana-09-sesion-01/plantilla-features/obt_features.csv")


def huella(ruta):
    with open(ruta, "rb") as archivo:
        return hashlib.md5(archivo.read()).hexdigest()[:12]


def cargar_y_dividir():
    """Una fila por award con monto valido, dividida por fecha de inicio."""
    datos = agregar_features(una_fila_por_award(pd.read_csv(RUTA_DATOS)))
    datos = datos[datos[OBJETIVO].notna()]
    entrenamiento, prueba = dividir_por_fecha(datos)
    return entrenamiento[FEATURES], prueba[FEATURES], entrenamiento[OBJETIVO], prueba[OBJETIVO]


if __name__ == "__main__":
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    mlflow.set_experiment(EXPERIMENTO)

    X_ent, X_prueba, y_ent, y_prueba = cargar_y_dividir()
    modelo = construir_modelo(alpha=alpha)

    with mlflow.start_run(run_name=f"ridge alpha={alpha:g}",
                          tags={"mlflow.source.name": "servicio/registrar_modelo.py"}):
        modelo.fit(X_ent, y_ent)
        mae_prueba = mean_absolute_error(y_prueba, modelo.predict(X_prueba))
        mlflow.log_params({
            "alpha": alpha,
            "fecha_corte": str(CORTE.date()),
            "huella_datos": huella(RUTA_DATOS),
            "filas_entrenamiento": len(X_ent),
            "filas_prueba": len(X_prueba),
        })
        mlflow.log_metrics({
            "mae_log_prueba": mae_prueba,
            "mae_log_entrenamiento": mean_absolute_error(y_ent, modelo.predict(X_ent)),
            "factor_prueba": 10 ** mae_prueba,
        })
        info = mlflow.sklearn.log_model(modelo, "modelo", registered_model_name=NOMBRE)

    print(f"Version {info.registered_model_version} de {NOMBRE}: ridge alpha={alpha:g}, "
          f"MAE log de prueba {mae_prueba:.4f} (factor {10 ** mae_prueba:.2f}x)")
