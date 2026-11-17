"""Pone el alias "produccion" en una version de "grantmatch-monto".

Desde psets/proyecto/, con el servidor de MLflow levantado:

    python -m servicio.promover 2

Mover el alias es todo lo que hace falta para cambiar de modelo o volver al
anterior. El contenedor no lee el registro: despues de promover hay que
volver a exportar (servicio/exportar_modelo.py) y reconstruir la imagen.
"""

import logging
import os
import sys

import mlflow
from mlflow import MlflowClient

logging.getLogger("mlflow").setLevel(logging.ERROR)

NOMBRE = "grantmatch-monto"
ALIAS = "produccion"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Uso: python -m servicio.promover <version>")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    cliente = MlflowClient()
    cliente.set_registered_model_alias(NOMBRE, ALIAS, sys.argv[1])
    actual = cliente.get_model_version_by_alias(NOMBRE, ALIAS)
    mae = cliente.get_run(actual.run_id).data.metrics.get("mae_log_prueba")
    print(f"{NOMBRE}@{ALIAS} es ahora la version {actual.version} "
          f"(corrida {actual.run_id[:12]}, MAE log de prueba {mae:.4f})")
