import pandas as pd
import os 
import matplotlib.pyplot as plt

os.makedirs("./starter/figuras", exist_ok=True)

data = pd.read_csv("./starter/viajes_taxi.csv")

print("Analisis exploratorio descriptivo")

data.info()

print(data.dtypes)
print(data.isna().sum())
print(data.duplicated().sum())

print(data.describe())

plt.figure()
plt.hist(data["tarifa"], bins=50)
plt.xlabel("USD")
plt.ylabel("Número de viajes")
plt.title("Distribución de tarifas")
plt.savefig("./starter/figuras/hist_tarifa.png")
plt.close()

plt.figure()
plt.boxplot(data["tarifa"])
plt.title("Boxplot de tarifas")
plt.savefig("./starter/figuras/boxplot_tarifa.png")
plt.close()

# Extración de valores atípicos

q1 = data["tarifa"].quantile(0.25)
q3 = data["tarifa"].quantile(0.75)
iqr = q3 - q1
limite_superior = q3 + 1.5 * iqr
limite_inferior = q1 - 1.5 * iqr

atípicos = data.query(f"tarifa < {limite_inferior} or tarifa > {limite_superior}")
sin_atípicos = data.query(f"tarifa >= {limite_inferior} and tarifa <= {limite_superior}")
print(atípicos)
print(sin_atípicos)

plt.figure()
plt.scatter(data["tarifa"], data["distancia_km"])
plt.title("Disperción de tarifa vs distancia_km")
plt.xlabel("Tarifa (USD)")
plt.ylabel("Distancia (km)")
plt.savefig("./starter/figuras/scatter_tarifa.png")
plt.close()

#print(data.corr())

plt.figure()
plt.scatter(sin_atípicos["tarifa"], sin_atípicos["distancia_km"])
plt.title("Disperción de tarifa vs distancia_km")
plt.xlabel("Tarifa (USD)")
plt.ylabel("Distancia (km)")
plt.savefig("./starter/figuras/scatter_tarifa.png")
plt.close()
