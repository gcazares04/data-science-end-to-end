"""El desempeno del modelo de monto en produccion. Desde psets/proyecto/,
despues de python -m monitoreo.simular_lote:

    python -m monitoreo.desempeno               # hoy = 2025-12-31
    python -m monitoreo.desempeno 2024-06-30    # otro "hoy"

Une cada estimacion del registro con el monto real, si ya se publico hasta
hoy, y resume por anio y por version del modelo:

- las metricas con verdad: cuantas estimaciones ya tienen monto (cobertura),
  el MAE en log10 (la metrica de la Semana 11) y el sesgo (real - estimado:
  positivo es subestimar, el error caro de GrantMatch);
- los indicadores que se calculan la misma noche, sin verdad: la mediana del
  log10 estimado, la fraccion de funders que el modelo no vio al entrenar y
  la fraccion de awards sin duracion.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CARPETA = Path(__file__).parent
HOY = "2025-12-31"

# Referencias de la version 1 (ridge alpha=10), medidas al entrenarla.
MAE_REFERENCIA = 0.3608            # MAE log10 en prueba
MEDIANA_LOG_REFERENCIA = 5.741     # mediana del log10 estimado en entrenamiento (~550 mil USD)
SIN_DURACION_REFERENCIA = 0.033    # awards sin duracion en entrenamiento

# Umbrales de alerta.
MIN_ESTIMACIONES = 5               # con menos, la mediana y las fracciones son ruido
MIN_CON_VERDAD = 5                 # con menos montos publicados, el MAE es ruido
MAE_MAXIMO = 1.25 * MAE_REFERENCIA
DESVIO_MEDIANA_MAXIMO = 0.30       # en log10: las estimaciones se movieron un factor 2
FUNDER_DESCONOCIDO_MAXIMO = 0.20
SIN_DURACION_MAXIMO = 0.10


def unir(predicciones, montos, hoy):
    """Las estimaciones hechas hasta hoy, cada una con su monto real si se
    publico hasta hoy. Agrega "periodo" (el anio de la estimacion),
    "con_verdad" y, cuando la hay, "error" y "sesgo" en log10."""
    predicciones = predicciones[pd.to_datetime(predicciones["momento"]) <= pd.Timestamp(hoy) + pd.Timedelta(days=1)]
    publicados = montos[pd.to_datetime(montos["fecha_publicacion"]) <= pd.Timestamp(hoy)]
    unidas = predicciones.merge(publicados[["award_key", "amount_usd"]], on="award_key", how="left")
    unidas["periodo"] = pd.to_datetime(unidas["momento"]).dt.year
    unidas["con_verdad"] = unidas["amount_usd"].notna()
    log_real = np.log10(unidas["amount_usd"])
    unidas["error"] = (log_real - unidas["log10_monto"]).abs()
    unidas["sesgo"] = log_real - unidas["log10_monto"]
    return unidas


def por_periodo(unidas):
    """Un DataFrame indexado por (periodo, version_modelo)."""
    unidas = unidas.assign(
        version_modelo=unidas["version_modelo"].astype(str),
        funder_desconocido=unidas["funder_conocido"].eq(False),
        sin_duracion=unidas["duration_days"].isna(),
    )
    tabla = unidas.groupby(["periodo", "version_modelo"]).agg(
        estimaciones=("log10_monto", "size"),
        con_verdad=("con_verdad", "sum"),
        mae_log=("error", "mean"),
        sesgo_log=("sesgo", "mean"),
        mediana_log=("log10_monto", "median"),
        funder_desconocido=("funder_desconocido", "mean"),
        sin_duracion=("sin_duracion", "mean"),
    )
    tabla.insert(2, "cobertura", tabla["con_verdad"] / tabla["estimaciones"])
    return tabla.round(4)


def alertas(tabla):
    """Un mensaje por cada problema, primero los que no necesitan verdad."""
    mensajes = []
    for (periodo, version), fila in tabla.iterrows():
        donde = f"{periodo}, version {version}"
        if fila["estimaciones"] >= MIN_ESTIMACIONES:
            if abs(fila["mediana_log"] - MEDIANA_LOG_REFERENCIA) > DESVIO_MEDIANA_MAXIMO:
                mensajes.append(f"modelo (sin verdad): en {donde} la mediana del log10 estimado fue {fila['mediana_log']:.2f}")
            if fila["funder_desconocido"] > FUNDER_DESCONOCIDO_MAXIMO:
                mensajes.append(f"modelo (sin verdad): en {donde} el {fila['funder_desconocido']:.0%} de los funders no estaba en el entrenamiento")
            if fila["sin_duracion"] > SIN_DURACION_MAXIMO:
                mensajes.append(f"datos: en {donde} el {fila['sin_duracion']:.0%} de los awards llego sin duracion")
        if fila["con_verdad"] >= MIN_CON_VERDAD and fila["mae_log"] > MAE_MAXIMO:
            mensajes.append(f"modelo: en {donde} el MAE log10 fue {fila['mae_log']:.3f} con {int(fila['con_verdad'])} montos publicados")
    return mensajes


if __name__ == "__main__":
    hoy = sys.argv[1] if len(sys.argv) > 1 else HOY
    ruta_registro = CARPETA / "predicciones.jsonl"
    if not ruta_registro.exists():
        sys.exit("No hay monitoreo/predicciones.jsonl: corre antes python -m monitoreo.simular_lote.")
    predicciones = pd.read_json(ruta_registro, lines=True, dtype={"version_modelo": str})
    montos = pd.read_csv(CARPETA / "montos_publicados.csv")

    unidas = unir(predicciones, montos, hoy)
    tabla = por_periodo(unidas)
    con_verdad = unidas[unidas["con_verdad"]]
    print(f"Al {hoy}: {len(unidas)} estimaciones, {len(con_verdad)} con monto publicado")
    print(f"MAE log10 de todas las que tienen verdad: {con_verdad['error'].mean():.4f} (referencia {MAE_REFERENCIA})\n")
    with pd.option_context("display.width", 200):
        print(tabla.to_string())
    print("\nAlertas:")
    for mensaje in alertas(tabla) or ["ninguna"]:
        print("-", mensaje)
