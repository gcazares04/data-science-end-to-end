"""Servicio de monto estimado de GrantMatch.

Recibe un award como lo guarda el warehouse y devuelve el monto estimado en
USD. Las features se calculan con agregar_features de modelado/, la misma
funcion con la que se entreno el modelo. Se levanta desde psets/proyecto/:

    uvicorn servicio.api:app --port 8000
"""

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import FEATURES, agregar_features

CARPETA = Path(__file__).parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_VERSION = RUTA_MODELO.with_name("modelo_version.txt")

# Sin modelo.joblib (en GitHub Actions, o antes de exportar) el servicio
# arranca igual: /salud lo reporta y /estimar_monto responde 503.
# version_modelo es la version de "grantmatch-monto" en el registro de MLflow.
modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None
version_modelo = RUTA_VERSION.read_text().strip() if RUTA_VERSION.exists() else None

app = FastAPI(title="GrantMatch: monto estimado de un award")


class Award(BaseModel):
    funder_name: str
    funding_type: str | None = None
    duration_days: float | None = Field(default=None, ge=0)
    institutions_count: int | None = Field(default=None, ge=0)
    has_description: bool = False


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.post("/estimar_monto")
def estimar_monto(award: Award):
    if modelo is None:
        raise HTTPException(status_code=503, detail="El servicio no tiene modelo cargado.")
    datos = pd.DataFrame([award.model_dump()]).astype({"duration_days": float, "institutions_count": float})
    log_monto = float(modelo.predict(agregar_features(datos)[FEATURES])[0])
    return {
        "monto_estimado_usd": round(10 ** log_monto, 2),
        "log10_monto": round(log_monto, 4),
        "version_modelo": version_modelo,
    }
