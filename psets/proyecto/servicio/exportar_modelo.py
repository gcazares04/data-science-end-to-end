"""Exporta la version en produccion de "grantmatch-monto" a servicio/modelo.joblib.

Se corre desde psets/proyecto/, con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py

Toma la version con el alias "produccion" del registro y deja, junto al
modelo, servicio/modelo_version.txt con su numero: el contenedor lleva los
dos archivos y no necesita llegar al servidor de MLflow. Ninguno se sube.
"""

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

CARPETA = Path(__file__).parent
NOMBRE = "grantmatch-monto"
ALIAS = "produccion"


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    version = MlflowClient().get_model_version_by_alias(NOMBRE, ALIAS).version
    modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE}/{version}")
    joblib.dump(modelo, CARPETA / "modelo.joblib")
    (CARPETA / "modelo_version.txt").write_text(f"{version}\n")
    print(f"Version {version} de {NOMBRE} (@{ALIAS}) exportada a servicio/modelo.joblib")


if __name__ == "__main__":
    main()
