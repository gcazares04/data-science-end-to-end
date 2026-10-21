import os

import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figuras", exist_ok=True)
viajes = pd.read_csv("../starter/viajes_taxi.csv")

# TODO 1: quita solo los errores que el Ejercicio 1 ya confirmo (tarifa
# 999.99 y duracion negativa). No quites los demas atipicos. Imprime
# cuantas filas quedan.
limpios = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)]
print(f"Filas originales: {len(viajes)}, filas limpias: {len(limpios)}")

# TODO 2: grafico de dispersion de distancia_km (eje x) contra tarifa
# (eje y), guardado en figuras/distancia_vs_tarifa.png. Mira el grafico
# antes de seguir: busca puntos que no sigan la tendencia general y
# cuentalos con un filtro.
plt.figure()
plt.scatter(limpios["distancia_km"], limpios["tarifa"], s=8, alpha=0.5)
plt.title("Distancia vs tarifa")
plt.xlabel("Distancia (km)")
plt.ylabel("Tarifa (USD)")
plt.savefig("figuras/distancia_vs_tarifa.png")
plt.close()

# Puntos fuera de la tendencia: una franja horizontal en 70 USD sin
# importar la distancia (tarifa fija), revisada por zona.
fuera_tendencia = limpios[limpios["tarifa"] == 70]
print(f"Viajes con tarifa exacta de 70: {len(fuera_tendencia)}")
print(fuera_tendencia["distancia_km"].describe().round(2))
es_jfk = (fuera_tendencia["zona_origen"] == "JFK Airport") | (fuera_tendencia["zona_destino"] == "JFK Airport")
print(f"De esos, con origen o destino JFK: {es_jfk.sum()}")

# TODO 3: propina promedio y cantidad de viajes con propina 0, por
# metodo de pago (groupby + agg).
propinas = limpios.groupby("metodo_pago", dropna=False).agg(
    viajes=("propina", "size"),
    propina_promedio=("propina", "mean"),
    propina_cero=("propina", lambda p: (p == 0).sum()),
)
print(propinas.round(2))

# TODO 4: correlacion de distancia_km, duracion_min, pasajeros y propina
# con la tarifa.
columnas = ["distancia_km", "duracion_min", "pasajeros", "propina"]
print(limpios[columnas].corrwith(limpios["tarifa"]).round(3))

# TODO 5: cuantos viajes tienen pasajeros vacio y cuantos tienen 0.
print(f"Pasajeros vacio: {limpios['pasajeros'].isna().sum()}")
print(f"Pasajeros en 0: {(limpios['pasajeros'] == 0).sum()}")
