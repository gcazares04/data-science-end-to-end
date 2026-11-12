"""Pruebas del contrato de servicio/api.py. No usan modelo.joblib, que no
existe en GitHub Actions: cada prueba le pone al servicio un modelo de prueba."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import servicio.api as api
from modelado.pipeline_proyecto import FEATURES

AWARD = {
    "funder_name": "National Science Foundation",
    "funding_type": "grant",
    "duration_days": 730.5,
    "institutions_count": 2,
    "has_description": True,
}


class ModeloDePrueba:
    """Responde siempre log10(monto) = 5, y guarda lo que recibio."""

    def predict(self, X):
        self.recibido = X
        return np.full(len(X), 5.0)


@pytest.fixture
def cliente(monkeypatch):
    modelo = ModeloDePrueba()
    monkeypatch.setattr(api, "modelo", modelo)
    monkeypatch.setattr(api, "version_modelo", "run-de-prueba")
    return TestClient(api.app), modelo


def test_salud(cliente):
    cliente, _ = cliente
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "run-de-prueba"}


def test_estimacion_valida(cliente):
    cliente, modelo = cliente
    respuesta = cliente.post("/estimar_monto", json=AWARD)
    assert respuesta.status_code == 200
    assert respuesta.json() == {"monto_estimado_usd": 100000.0, "log10_monto": 5.0, "version_modelo": "run-de-prueba"}
    assert list(modelo.recibido.columns) == FEATURES
    fila = modelo.recibido.iloc[0]
    assert fila["duracion_anios"] == pytest.approx(2.0)
    assert fila["escala_financiamiento"] == "proyecto"
    assert fila["has_description"] == 1


def test_award_incompleto_se_estima(cliente):
    cliente, modelo = cliente
    respuesta = cliente.post("/estimar_monto", json={"funder_name": "Wellcome Trust", "funding_type": "fellowship"})
    assert respuesta.status_code == 200
    fila = modelo.recibido.iloc[0]
    assert np.isnan(fila["duracion_anios"])
    assert fila["escala_financiamiento"] == "individual"
    assert fila["has_description"] == 0


def test_sin_funder_es_422(cliente):
    cliente, _ = cliente
    sin_funder = {k: v for k, v in AWARD.items() if k != "funder_name"}
    respuesta = cliente.post("/estimar_monto", json=sin_funder)
    assert respuesta.status_code == 422
    assert respuesta.json()["detail"][0]["loc"] == ["body", "funder_name"]


def test_duracion_negativa_es_422(cliente):
    cliente, _ = cliente
    respuesta = cliente.post("/estimar_monto", json={**AWARD, "duration_days": -10})
    assert respuesta.status_code == 422


def test_sin_modelo_es_503(monkeypatch):
    monkeypatch.setattr(api, "modelo", None)
    respuesta = TestClient(api.app).post("/estimar_monto", json=AWARD)
    assert respuesta.status_code == 503
