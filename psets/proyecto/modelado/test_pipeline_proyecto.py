"""Pruebas de pipeline_proyecto.py. Se corren desde la raiz del fork con:

    pytest psets/proyecto -v

No leen data/: cada prueba arma sus propios awards, asi corren igual en
GitHub Actions, donde ese archivo no existe.
"""

import numpy as np
import pandas as pd
import pytest

from modelado.pipeline_proyecto import (
    FEATURES,
    OBJETIVO,
    agregar_features,
    construir_modelo,
    construir_preparacion,
    dividir_por_fecha,
    una_fila_por_award,
)


def awards_de_prueba():
    """Cuatro awards escritos a mano; el A1 aparece dos veces porque financio
    dos publicaciones, como en la OBT."""
    return pd.DataFrame({
        "award_key": ["A1", "A1", "A2", "A3", "A4"],
        "funder_name": ["NIH", "NIH", "NSF", "NIH", "NSF"],
        "funding_type": ["fellowship", "fellowship", "program", "grant", None],
        "duration_days": [730.5, 730.5, 1826.25, 365.25, np.nan],
        "institutions_count": [1, 1, 4, 2, np.nan],
        "has_description": [True, True, False, True, None],
        "award_start_date": ["2018-03-01", "2018-03-01", "2019-12-31", "2020-01-01", "2021-06-15"],
        "amount_usd": [50_000, 50_000, 2_000_000, 300_000, 1_000],
    })


def test_features_calculadas_a_mano():
    datos = agregar_features(awards_de_prueba())
    assert datos["duracion_anios"].iloc[0] == pytest.approx(2.0)
    assert datos["duracion_anios"].iloc[2] == pytest.approx(5.0)
    assert list(datos["escala_financiamiento"]) == ["individual", "individual", "centro", "proyecto", "proyecto"]
    assert datos[OBJETIVO].iloc[2] == pytest.approx(np.log10(2_000_000))


def test_objetivo_no_entra_al_modelo():
    assert OBJETIVO not in FEATURES
    assert "amount_usd" not in FEATURES


def test_una_fila_por_award():
    awards = una_fila_por_award(awards_de_prueba())
    assert len(awards) == 4
    assert awards["award_key"].is_unique


def test_division_por_fecha_no_mira_al_futuro():
    entrenamiento, prueba = dividir_por_fecha(una_fila_por_award(awards_de_prueba()))
    assert list(entrenamiento["award_key"]) == ["A1", "A2"]
    assert list(prueba["award_key"]) == ["A3", "A4"]
    assert pd.to_datetime(entrenamiento["award_start_date"]).max() < pd.to_datetime(prueba["award_start_date"]).min()


def test_preparacion_acepta_funder_nuevo():
    datos = agregar_features(una_fila_por_award(awards_de_prueba()))
    preparacion = construir_preparacion().fit(datos[FEATURES])
    nuevo = datos[FEATURES].head(1).assign(funder_name="Wellcome Trust")
    transformado = preparacion.transform(nuevo)
    assert transformado.shape == (1, preparacion.transform(datos[FEATURES].head(1)).shape[1])
    assert not np.isnan(transformado).any()


def test_modelo_estima_awards_incompletos():
    datos = agregar_features(una_fila_por_award(awards_de_prueba()))
    modelo = construir_modelo().fit(datos[FEATURES], datos[OBJETIVO])
    estimadas = modelo.predict(datos[FEATURES])
    assert len(estimadas) == len(datos)
    assert np.isfinite(estimadas).all()
