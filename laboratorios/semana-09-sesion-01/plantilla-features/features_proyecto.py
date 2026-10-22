import numpy as np
import pandas as pd

datos = pd.read_csv("../../semana-08-sesion-02/plantilla-eda/obt_award_work.csv")

# El modelo estima el monto de un award (para los funders que no lo
# informan). Por el hallazgo 3 se predice en log10 y se quita el award de
# 1 USD, que es un error de captura.
datos = datos[datos["amount_usd"] > 1].copy()
datos["log_amount_usd"] = np.log10(datos["amount_usd"])
OBJETIVO = "log_amount_usd"

# Feature 1: duracion del award en anios. Las fechas de inicio y fin se
# conocen al otorgar el award.
datos["duracion_anios"] = datos["duration_days"] / 365.25

# Feature 2: escala del financiamiento segun funding_type. Las becas pagan
# a una persona; los acuerdos cooperativos, programas y contratos financian
# centros enteros (la cola larga del hallazgo 3).
ESCALA = {
    "fellowship": "individual",
    "career_development": "individual",
    "training": "individual",
    "cooperative_agreement": "centro",
    "program": "centro",
    "contract": "centro",
}
datos["escala_financiamiento"] = datos["funding_type"].map(ESCALA).fillna("proyecto")

# La OBT tiene una fila por award + publicacion: la evidencia se mide con
# un award por fila, para no contar el mismo monto varias veces.
awards = datos.drop_duplicates("award_key")
print(f"Awards con monto: {len(awards)} (de {len(datos)} filas)")

print("\nCorrelacion de duracion_anios con el objetivo:")
print(awards[["duracion_anios", OBJETIVO]].corr())
print(f"Awards sin duracion: {awards['duracion_anios'].isna().sum()}")

print("\nObjetivo por escala_financiamiento:")
resumen = awards.groupby("escala_financiamiento")[OBJETIVO].agg(["count", "mean", "median"])
resumen["mediana_usd"] = (10 ** resumen["median"]).round(0)
print(resumen)

umbral = awards["amount_usd"].quantile(0.9)
print(f"\nAwards en el 10% mas alto (>= {umbral:,.0f} USD) por escala:")
print(awards[awards["amount_usd"] >= umbral]["escala_financiamiento"].value_counts())

datos.to_csv("obt_features.csv", index=False)
print(f"\nobt_features.csv: {datos.shape[0]} filas, {datos.shape[1]} columnas")
