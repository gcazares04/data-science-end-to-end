import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

viajes = pd.read_csv("../starter/viajes_features.csv")

NUMERICAS = ["distancia_km", "pasajeros"]
YA_LISTAS = ["hora_sin", "hora_cos", "es_fin_de_semana", "es_hora_pico", "es_tarifa_fija"]

X = viajes[NUMERICAS + YA_LISTAS].copy()
y = viajes["tarifa"]
print("Faltantes:", X[NUMERICAS].isna().sum().to_dict())

# Version del companero: imputa y escala con TODO el dataset, despues divide.
X_fuga = X.copy()
X_fuga[NUMERICAS] = SimpleImputer(strategy="median").fit_transform(X_fuga[NUMERICAS])  # aqui entra la prueba
X_fuga[NUMERICAS] = StandardScaler().fit_transform(X_fuga[NUMERICAS])                  # y aqui otra vez
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X_fuga, y, test_size=0.2, random_state=42)
modelo = LinearRegression().fit(X_ent, y_ent)
mae_fuga = mean_absolute_error(y_prueba, modelo.predict(X_prueba))

# Version corregida: divide primero; imputador y escalador aprenden solo
# con entrenamiento y se aplican a prueba.
X_ent, X_prueba, y_ent, y_prueba = train_test_split(X, y, test_size=0.2, random_state=42)
X_ent, X_prueba = X_ent.copy(), X_prueba.copy()
imputador = SimpleImputer(strategy="median").fit(X_ent[NUMERICAS])
X_ent[NUMERICAS] = imputador.transform(X_ent[NUMERICAS])
X_prueba[NUMERICAS] = imputador.transform(X_prueba[NUMERICAS])
escalador = StandardScaler().fit(X_ent[NUMERICAS])
X_ent[NUMERICAS] = escalador.transform(X_ent[NUMERICAS])
X_prueba[NUMERICAS] = escalador.transform(X_prueba[NUMERICAS])
modelo = LinearRegression().fit(X_ent, y_ent)
mae_corregido = mean_absolute_error(y_prueba, modelo.predict(X_prueba))

print(f"Medianas con todo el dataset: {X[NUMERICAS].median().to_dict()}")
print(f"Medianas solo entrenamiento:  {dict(zip(NUMERICAS, imputador.statistics_.tolist()))}")
print(f"MAE en prueba con fuga:   {mae_fuga:.4f}")
print(f"MAE en prueba corregido:  {mae_corregido:.4f}")
print(f"Diferencia:               {mae_corregido - mae_fuga:+.4f}")
