"""Simula el lote nocturno sobre los awards de prueba, y la publicacion
tardia de sus montos. Desde psets/proyecto/, con servicio/modelo.joblib
exportado:

    python -m monitoreo.simular_lote

En produccion el lote estima los awards que no informan monto, y su verdad
solo llega si el funder publica el monto despues. Para tener esa verdad, la
simulacion toma los 46 awards de prueba (desde 2020), que si tienen monto:
cada uno entra al catalogo el dia que empieza, el lote lo estima esa noche
sin mirar el monto, y el monto "se publica" entre 30 y 730 dias despues
(una demora inventada, con semilla fija).

Deja monitoreo/predicciones.jsonl (lo escribe el servicio) y
monitoreo/montos_publicados.csv (award_key, monto real y cuando se publico).
Los dos se borran antes de empezar.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from modelado.pipeline_proyecto import OBJETIVO, agregar_features, dividir_por_fecha, una_fila_por_award

CARPETA = Path(__file__).parent
RUTA_DATOS = (CARPETA.resolve().parents[2]
              / "laboratorios/semana-09-sesion-01/plantilla-features/obt_features.csv")
RUTA_MONTOS = CARPETA / "montos_publicados.csv"
CAMPOS = ["award_key", "funder_name", "funding_type", "duration_days", "institutions_count", "has_description"]


def peticion(award):
    """El award como lo manda el lote: sin los campos vacios."""
    cuerpo = {campo: award[campo] for campo in CAMPOS if not pd.isna(award[campo])}
    if "institutions_count" in cuerpo:
        cuerpo["institutions_count"] = int(cuerpo["institutions_count"])
    cuerpo["has_description"] = bool(cuerpo.get("has_description", False))
    return cuerpo


if __name__ == "__main__":
    import servicio.api as api

    for ruta in [api.RUTA_REGISTRO, RUTA_MONTOS]:
        ruta.unlink(missing_ok=True)

    datos = agregar_features(una_fila_por_award(pd.read_csv(RUTA_DATOS)))
    _, prueba = dividir_por_fecha(datos[datos[OBJETIVO].notna()])
    prueba = prueba.assign(inicio=pd.to_datetime(prueba["award_start_date"])).sort_values("inicio")

    cliente = TestClient(api.app)
    for award in prueba.to_dict("records"):
        noche = award["inicio"].to_pydatetime() + timedelta(hours=2)
        api.ahora = lambda noche=noche: noche
        cliente.post("/estimar_monto", json=peticion(award)).raise_for_status()

    # Dos filas que llegan sin funder, como las de una ingesta rota.
    for _ in range(2):
        cliente.post("/estimar_monto", json={"funding_type": "grant", "duration_days": 365})

    demora = np.random.default_rng(13).integers(30, 731, len(prueba))
    pd.DataFrame({
        "award_key": prueba["award_key"].to_numpy(),
        "amount_usd": prueba["amount_usd"].to_numpy(),
        "fecha_publicacion": (prueba["inicio"] + pd.to_timedelta(demora, unit="D")).dt.date.to_numpy(),
    }).to_csv(RUTA_MONTOS, index=False)

    print(f"{len(prueba)} awards estimados, del {prueba['inicio'].min():%Y-%m-%d} al {prueba['inicio'].max():%Y-%m-%d}")
    print("Metricas del sistema:", cliente.get("/metricas").json())
