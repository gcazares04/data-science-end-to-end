"""Pruebas del contrato de servicio/api.py. No usan modelo.joblib, que no
existe en GitHub Actions: cada prueba le pone al servicio un modelo de prueba,
y un registro de predicciones temporal (tmp_path)."""

import json

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


@pytest.fixture(autouse=True)
def registro_temporal(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "RUTA_REGISTRO", tmp_path / "predicciones.jsonl")
    monkeypatch.setattr(api, "metricas", api.Metricas())


def leer_registro():
    with open(api.RUTA_REGISTRO) as archivo:
        return [json.loads(linea) for linea in archivo]


@pytest.fixture
def cliente(monkeypatch):
    modelo = ModeloDePrueba()
    monkeypatch.setattr(api, "modelo", modelo)
    monkeypatch.setattr(api, "version_modelo", "7")
    return TestClient(api.app), modelo


def test_salud(cliente):
    cliente, _ = cliente
    respuesta = cliente.get("/salud")
    assert respuesta.status_code == 200
    assert respuesta.json() == {"estado": "ok", "modelo_cargado": True, "version_modelo": "7"}


def test_estimacion_valida(cliente):
    cliente, modelo = cliente
    respuesta = cliente.post("/estimar_monto", json=AWARD)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert len(cuerpo.pop("id_prediccion")) == 32
    assert cuerpo == {"monto_estimado_usd": 100000.0, "log10_monto": 5.0, "version_modelo": "7"}
    assert list(modelo.recibido.columns) == FEATURES
    fila = modelo.recibido.iloc[0]
    assert fila["duracion_anios"] == pytest.approx(2.0)
    assert fila["escala_financiamiento"] == "proyecto"
    assert fila["has_description"] == 1


def test_cada_estimacion_dice_que_version_la_calculo(cliente):
    cliente, _ = cliente
    respuesta = cliente.post("/estimar_monto", json=AWARD)
    assert respuesta.status_code == 200
    assert respuesta.json()["version_modelo"] == "7"


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


def test_cada_estimacion_queda_registrada(cliente):
    cliente, _ = cliente
    respuesta = cliente.post("/estimar_monto", json={**AWARD, "award_key": "G123"}).json()
    lineas = leer_registro()
    assert len(lineas) == 1
    linea = lineas[0]
    assert linea["id_prediccion"] == respuesta["id_prediccion"]
    assert linea["award_key"] == "G123"
    assert linea["version_modelo"] == "7"
    assert linea["funder_name"] == "National Science Foundation"
    assert linea["duration_days"] == 730.5
    assert linea["log10_monto"] == 5.0
    assert linea["monto_estimado_usd"] == 100000.0
    assert "momento" in linea


def test_las_peticiones_rechazadas_no_se_registran(cliente):
    cliente, _ = cliente
    cliente.post("/estimar_monto", json={**AWARD, "duration_days": -10})
    assert not api.RUTA_REGISTRO.exists()


def test_metricas_separan_validacion_de_servidor(cliente, monkeypatch):
    cliente, _ = cliente
    cliente.post("/estimar_monto", json=AWARD)
    cliente.post("/estimar_monto", json={"funding_type": "grant"})
    monkeypatch.setattr(api, "modelo", None)
    cliente.post("/estimar_monto", json=AWARD)
    cliente.get("/salud")
    resumen = cliente.get("/metricas").json()
    assert resumen["peticiones"] == 3
    assert resumen["errores_validacion"] == 1
    assert resumen["errores_servidor"] == 1
    assert resumen["latencia_p50_ms"] > 0
    assert resumen["latencia_p95_ms"] >= resumen["latencia_p50_ms"]


def test_metricas_sin_peticiones(cliente):
    cliente, _ = cliente
    assert cliente.get("/metricas").json() == {
        "peticiones": 0, "errores_validacion": 0, "errores_servidor": 0,
        "latencia_p50_ms": None, "latencia_p95_ms": None,
    }
