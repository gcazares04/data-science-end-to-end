import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

os.makedirs("figuras", exist_ok=True)

# OBT de GrantMatch (PSet 2) exportada de Snowflake con exportar_obt.py
# (muestra de 4000 awards con todas sus filas).
datos = pd.read_csv("obt_award_work.csv")

# La etiqueta real del recomendador es label, pero vale 1 en todas las filas
# (ver hallazgo 1). Se explora el monto del award en USD.
OBJETIVO = "amount_usd"

# Paso 1: resumen.
print("Filas y columnas:", datos.shape)
print(datos.dtypes)
print(datos.isna().sum())
print(datos.describe().round(2))
print("Valores de label:", datos["label"].value_counts().to_dict())

# Paso 2: distribucion de la variable objetivo (solo los montos informados).
plt.figure()
plt.hist(datos[OBJETIVO].dropna(), bins=40)
plt.title(OBJETIVO)
plt.savefig("figuras/distribucion_objetivo.png")
plt.close()

plt.figure()
plt.hist(np.log10(datos[OBJETIVO].dropna()), bins=40)
plt.title(f"log10({OBJETIVO})")
plt.savefig("figuras/distribucion_objetivo_log.png")
plt.close()

# Paso 3: atipicos de la variable objetivo con la regla del IQR.
q1 = datos[OBJETIVO].quantile(0.25)
q3 = datos[OBJETIVO].quantile(0.75)
iqr = q3 - q1
fuera = datos[(datos[OBJETIVO] < q1 - 1.5 * iqr) | (datos[OBJETIVO] > q3 + 1.5 * iqr)]
print(f"Atipicos de {OBJETIVO} segun IQR: {len(fuera)}")
print(fuera.head(10))

# Paso 4: el funder es la variable que mas deberia influir en el monto.
# Promedio y mediana del objetivo por funder (con al menos 10 montos).
por_funder = (
    datos.dropna(subset=[OBJETIVO])
    .groupby("funder_name")[OBJETIVO]
    .agg(montos="size", mediana="median", promedio="mean")
    .query("montos >= 10")
    .sort_values("mediana")
)
print(por_funder.round(0))

plt.figure(figsize=(8, 4))
plt.barh(por_funder.index, por_funder["mediana"])
plt.xlabel("Mediana de amount_usd")
plt.title("Monto del award por funder")
plt.tight_layout()
plt.savefig("figuras/monto_por_funder.png")
plt.close()

# Paso 5: consultas que confirman los hallazgos.

# Hallazgo 1: label no tiene negativos.
print("Filas con label distinto de 1:", (datos["label"] != 1).sum())

# Hallazgo 2: el monto y los textos del award faltan segun el funder.
por_origen = datos.groupby("funder_name").agg(
    filas=("award_key", "size"),
    con_monto_usd=(OBJETIVO, lambda m: m.notna().mean()),
    con_titulo=("award_title", lambda t: t.notna().mean()),
    con_descripcion=("has_description", "mean"),
).sort_values("filas", ascending=False)
print(por_origen.head(8).round(3))
print("Awards sin titulo ni descripcion:",
      (datos["award_title"].isna() & ~datos["has_description"]).mean().round(3))

# Hallazgo 3: concentracion en pocos funders y cola larga del monto.
print((datos["funder_name"].value_counts(normalize=True).head(5) * 100).round(1))
print(f"Promedio {datos[OBJETIVO].mean():,.0f} vs mediana {datos[OBJETIVO].median():,.0f}")
print("Asimetria:", datos[OBJETIVO].skew().round(1), "| en log10:",
      np.log10(datos[OBJETIVO].dropna()).skew().round(2))
print("Montos USD menores a 1000:", (datos[OBJETIVO] < 1000).sum())
print("Pais del funder en filas con monto USD:",
      datos.dropna(subset=[OBJETIVO])["funder_country_code"].value_counts().to_dict())

# Otros problemas de calidad para el modelado.
print("Publicaciones retractadas:", datos["is_retracted"].sum())
print("Publicaciones anteriores a 1950:", (datos["publication_year"] < 1950).sum())
print("Publicaciones sin abstract:", (~datos["has_abstract"]).mean().round(3))
