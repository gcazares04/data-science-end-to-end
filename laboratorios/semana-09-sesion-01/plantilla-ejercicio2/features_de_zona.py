import pandas as pd

viajes = pd.read_csv("../starter/viajes_taxi.csv")
viajes = viajes[(viajes["tarifa"] != 999.99) & (viajes["duracion_min"] >= 0)].copy()

ZONAS_MANHATTAN = ["Midtown", "Upper East Side", "Upper West Side", "Lower Manhattan", "Harlem"]

# TODO 1: crea la columna es_tarifa_fija (1 o 0): vale 1 cuando el viaje
# va de JFK Airport a una zona de Manhattan, o de una zona de Manhattan a
# JFK Airport. Imprime cuantos viajes la tienen en 1, y la tarifa minima
# y maxima de esos viajes.
desde_jfk = (viajes["zona_origen"] == "JFK Airport") & viajes["zona_destino"].isin(ZONAS_MANHATTAN)
hacia_jfk = viajes["zona_origen"].isin(ZONAS_MANHATTAN) & (viajes["zona_destino"] == "JFK Airport")
viajes["es_tarifa_fija"] = (desde_jfk | hacia_jfk).astype(int)
fijos = viajes[viajes["es_tarifa_fija"] == 1]
print(f"Viajes con es_tarifa_fija = 1: {len(fijos)}")
print(f"Tarifa minima: {fijos['tarifa'].min()}, maxima: {fijos['tarifa'].max()}")

# TODO 2: codifica zona_origen con one-hot (pd.get_dummies), con el
# prefijo "origen" y dtype=int. Imprime los nombres de las columnas
# nuevas.
one_hot = pd.get_dummies(viajes["zona_origen"], prefix="origen", dtype=int)
viajes = pd.concat([viajes, one_hot], axis=1)
print(f"Columnas one-hot ({len(one_hot.columns)}):", list(one_hot.columns))

# TODO 3: codifica zona_origen de forma ordinal: un numero entero por
# zona, en orden alfabetico, en una columna origen_ordinal.
zonas = sorted(viajes["zona_origen"].unique())
viajes["origen_ordinal"] = viajes["zona_origen"].map({zona: i for i, zona in enumerate(zonas)})
print("Codificacion ordinal (1 columna):", {zona: i for i, zona in enumerate(zonas)})

# TODO 4: para cada columna original del dataset, decide si existe en el
# momento en que alguien pide el taxi (antes del viaje) e imprime la
# decision.
EXISTE_AL_PEDIR = {
    "id_viaje": "si, pero es un identificador: no es feature",
    "fecha_hora_inicio": "si",
    "zona_origen": "si",
    "zona_destino": "si",
    "distancia_km": "si, estimada por la ruta (la real se mide al final)",
    "duracion_min": "no, se conoce al terminar el viaje",
    "pasajeros": "si",
    "metodo_pago": "no, se registra al pagar",
    "tarifa": "no, es el objetivo",
    "propina": "no, se registra al pagar",
}
for columna, decision in EXISTE_AL_PEDIR.items():
    print(f"{columna}: {decision}")
