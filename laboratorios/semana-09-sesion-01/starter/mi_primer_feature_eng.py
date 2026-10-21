import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder

viajes = pd.read_csv('./starter/viajes_taxi.csv')

viajes_clean = viajes.query('tarifa < 999 and duracion_min > 0').copy()

print(viajes_clean.head())
print(viajes_clean.info())

fecha_inicio = pd.to_datetime(viajes_clean['fecha_hora_inicio'])

viajes_clean['fecha_hora_inicio'] = fecha_inicio
viajes_clean['hora'] = fecha_inicio.dt.hour
viajes_clean['dia_semana'] = fecha_inicio.dt.dayofweek

print(viajes_clean.head())

viajes_clean['hora_sin'] = np.sin(2 * np.pi * viajes_clean['hora'] / 24)
viajes_clean['hora_cos'] = np.cos(2 * np.pi * viajes_clean['hora'] / 24)

print(viajes_clean[['fecha_hora_inicio', 'hora', 'hora_sin', 'hora_cos']])

viajes_codificados = pd.get_dummies(viajes_clean, columns=['zona_origen', 'zona_destino'])
print(viajes_codificados.head())

encoder = OrdinalEncoder()
viajes_codificados['metodo_pago_ordinal'] = encoder.fit_transform(viajes_codificados[['metodo_pago']])

print(viajes_codificados.head())
