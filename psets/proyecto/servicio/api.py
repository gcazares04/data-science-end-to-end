"""Servicio de monto estimado de GrantMatch.

Recibe un award como lo guarda el warehouse y devuelve el monto estimado en
USD. Las features se calculan con agregar_features de modelado/, la misma
funcion con la que se entreno el modelo. Se levanta desde psets/proyecto/:

    uvicorn servicio.api:app --port 8000

Esta instrumentado: mide la latencia y el codigo de cada estimacion (en
/metricas) y guarda cada estimacion en monitoreo/predicciones.jsonl, con el
award_key para unirla despues con el monto real (monitoreo/desempeno.py).
"""

import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from modelado.pipeline_proyecto import CATEGORICAS, FEATURES, agregar_features

CARPETA = Path(__file__).parent
RUTA_MODELO = Path(os.environ.get("RUTA_MODELO", CARPETA / "modelo.joblib"))
RUTA_VERSION = RUTA_MODELO.with_name("modelo_version.txt")
RUTA_REGISTRO = Path(os.environ.get("RUTA_REGISTRO", CARPETA.parent / "monitoreo" / "predicciones.jsonl"))

# Sin modelo.joblib (en GitHub Actions, o antes de exportar) el servicio
# arranca igual: /salud lo reporta y /estimar_monto responde 503.
# version_modelo es la version de "grantmatch-monto" en el registro de MLflow.
modelo = joblib.load(RUTA_MODELO) if RUTA_MODELO.exists() else None
version_modelo = RUTA_VERSION.read_text().strip() if RUTA_VERSION.exists() else None

# El momento que queda en el registro. monitoreo/simular_lote.py la cambia
# por la noche simulada del lote.
ahora = datetime.now


class Metricas:
    """Las metricas del sistema, en memoria: una latencia y un codigo por
    cada peticion a /estimar_monto."""

    def __init__(self):
        self.latencias_ms = []
        self.codigos = []

    def registrar(self, latencia_ms, codigo):
        self.latencias_ms.append(latencia_ms)
        self.codigos.append(codigo)

    def resumen(self):
        hay = len(self.latencias_ms) > 0
        return {
            "peticiones": len(self.codigos),
            "errores_validacion": sum(codigo == 422 for codigo in self.codigos),
            "errores_servidor": sum(codigo >= 500 for codigo in self.codigos),
            "latencia_p50_ms": round(float(np.percentile(self.latencias_ms, 50)), 2) if hay else None,
            "latencia_p95_ms": round(float(np.percentile(self.latencias_ms, 95)), 2) if hay else None,
        }


metricas = Metricas()
app = FastAPI(title="GrantMatch: monto estimado de un award")


@app.middleware("http")
async def medir(request: Request, call_next):
    if request.url.path != "/estimar_monto":
        return await call_next(request)
    inicio = time.perf_counter()
    respuesta = await call_next(request)
    metricas.registrar((time.perf_counter() - inicio) * 1000, respuesta.status_code)
    return respuesta


class Award(BaseModel):
    award_key: str | None = None  # la llave del warehouse, para unir con el monto real
    funder_name: str
    funding_type: str | None = None
    duration_days: float | None = Field(default=None, ge=0)
    institutions_count: int | None = Field(default=None, ge=0)
    has_description: bool = False


def funder_conocido(funder):
    """Si el modelo vio este funder al entrenar. Uno nuevo se codifica como
    ceros y se estima solo con el resto de las features. None si el modelo no
    es el pipeline de modelado/ (en las pruebas)."""
    try:
        codificar = modelo.named_steps["preparar"].named_transformers_["categoricas"].named_steps["codificar"]
    except (AttributeError, KeyError):
        return None
    return bool(funder in codificar.categories_[CATEGORICAS.index("funder_name")])


@app.get("/salud")
def salud():
    return {"estado": "ok", "modelo_cargado": modelo is not None, "version_modelo": version_modelo}


@app.get("/metricas")
def ver_metricas():
    return metricas.resumen()


@app.post("/estimar_monto")
def estimar_monto(award: Award):
    if modelo is None:
        raise HTTPException(status_code=503, detail="El servicio no tiene modelo cargado.")
    datos = pd.DataFrame([award.model_dump()]).astype({"duration_days": float, "institutions_count": float})
    log_monto = float(modelo.predict(agregar_features(datos)[FEATURES])[0])
    id_prediccion = uuid.uuid4().hex
    salida = {"monto_estimado_usd": round(10 ** log_monto, 2), "log10_monto": round(log_monto, 4)}

    RUTA_REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    with open(RUTA_REGISTRO, "a") as archivo:
        archivo.write(json.dumps({
            "id_prediccion": id_prediccion,
            "momento": ahora().isoformat(),
            "version_modelo": version_modelo,
            **award.model_dump(mode="json"),
            "funder_conocido": funder_conocido(award.funder_name),
            **salida,
        }) + "\n")

    return {"id_prediccion": id_prediccion, **salida, "version_modelo": version_modelo}
