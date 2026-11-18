"""Pruebas de monitoreo/desempeno.py. Desde psets/proyecto/:

    pytest monitoreo/test_desempeno.py
"""

import numpy as np
import pandas as pd

from monitoreo.desempeno import MAE_REFERENCIA, MEDIANA_LOG_REFERENCIA, alertas, por_periodo, unir


def predicciones(filas):
    base = {"version_modelo": "1", "funder_conocido": True, "duration_days": 365.0}
    return pd.DataFrame([{**base, **fila} for fila in filas])


def test_unir_solo_usa_montos_publicados_hasta_hoy():
    p = predicciones([
        {"award_key": "a", "momento": "2021-03-01T02:00", "log10_monto": 5.0},
        {"award_key": "b", "momento": "2021-03-02T02:00", "log10_monto": 5.0},
        {"award_key": "c", "momento": "2022-06-01T02:00", "log10_monto": 5.0},
    ])
    montos = pd.DataFrame({
        "award_key": ["a", "b", "c"],
        "amount_usd": [1e6, 1e5, 1e5],
        "fecha_publicacion": ["2021-06-01", "2022-06-01", "2022-08-01"],
    })
    unidas = unir(p, montos, "2021-12-31")
    assert list(unidas["award_key"]) == ["a", "b"]  # "c" todavia no se estimaba
    assert list(unidas["con_verdad"]) == [True, False]
    assert unidas.loc[0, "error"] == 1.0
    assert unidas.loc[0, "sesgo"] == 1.0  # real mayor que el estimado: subestimo


def test_por_periodo_separa_versiones_y_cuenta_cobertura():
    p = predicciones([
        {"award_key": "a", "momento": "2021-01-01", "log10_monto": 5.0, "version_modelo": "1"},
        {"award_key": "b", "momento": "2021-02-01", "log10_monto": 6.0, "version_modelo": "1",
         "funder_conocido": False, "duration_days": np.nan},
        {"award_key": "c", "momento": "2021-03-01", "log10_monto": 6.0, "version_modelo": "2"},
    ])
    montos = pd.DataFrame({"award_key": ["a"], "amount_usd": [1e5], "fecha_publicacion": ["2021-05-01"]})
    tabla = por_periodo(unir(p, montos, "2021-12-31"))
    v1 = tabla.loc[(2021, "1")]
    assert v1["estimaciones"] == 2
    assert v1["cobertura"] == 0.5
    assert v1["mae_log"] == 0.0
    assert v1["funder_desconocido"] == 0.5
    assert v1["sin_duracion"] == 0.5
    assert tabla.loc[(2021, "2"), "con_verdad"] == 0


def fila(**cambios):
    base = {"estimaciones": 10, "con_verdad": 10, "cobertura": 1.0, "mae_log": MAE_REFERENCIA,
            "sesgo_log": 0.0, "mediana_log": MEDIANA_LOG_REFERENCIA, "funder_desconocido": 0.0,
            "sin_duracion": 0.0}
    return {**base, **cambios}


def tabla(*filas):
    indice = pd.MultiIndex.from_tuples([(2020 + i, "1") for i in range(len(filas))],
                                       names=["periodo", "version_modelo"])
    return pd.DataFrame(list(filas), index=indice)


def test_sin_problemas_no_hay_alertas():
    assert alertas(tabla(fila())) == []


def test_alertas_sin_verdad():
    mensajes = alertas(tabla(fila(mediana_log=6.5, funder_desconocido=0.5, sin_duracion=0.3)))
    assert len(mensajes) == 3
    assert all("2020, version 1" in m for m in mensajes)


def test_alerta_de_mae_solo_con_suficiente_verdad():
    assert len(alertas(tabla(fila(mae_log=1.0)))) == 1
    assert alertas(tabla(fila(mae_log=1.0, con_verdad=2))) == []


def test_pocas_estimaciones_no_disparan_alertas_sin_verdad():
    assert alertas(tabla(fila(estimaciones=2, con_verdad=0, mediana_log=7.0))) == []
