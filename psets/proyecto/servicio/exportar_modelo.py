"""Exporta el modelo final desde MLflow a servicio/modelo.joblib.

Se corre desde psets/proyecto/, con el servidor de MLflow levantado:

    python servicio/exportar_modelo.py            # el run con candidato=final
    python servicio/exportar_modelo.py <run_id>   # un run concreto (para volver atras)

Deja tambien servicio/modelo_info.json con el run_id, que el servicio devuelve
en cada respuesta como version_modelo. Ninguno de los dos archivos se sube.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

CARPETA = Path(__file__).parent
EXPERIMENTO = "grantmatch"


def run_final():
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENTO],
        filter_string="tags.candidato = 'final'",
        order_by=["start_time DESC"],
    )
    if runs.empty:
        sys.exit(f"No hay ningun run con candidato=final en el experimento {EXPERIMENTO}.")
    return runs.iloc[0]["run_id"]


def main():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5050"))
    run_id = sys.argv[1] if len(sys.argv) > 1 else run_final()
    modelo = mlflow.sklearn.load_model(f"runs:/{run_id}/modelo")
    joblib.dump(modelo, CARPETA / "modelo.joblib")
    info = {
        "run_id": run_id,
        "experimento": EXPERIMENTO,
        "exportado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (CARPETA / "modelo_info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"Modelo del run {run_id} exportado a servicio/modelo.joblib")


if __name__ == "__main__":
    main()
