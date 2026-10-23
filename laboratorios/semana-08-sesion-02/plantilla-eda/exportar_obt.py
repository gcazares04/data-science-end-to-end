import os
import sys

import numpy as np
import snowflake.connector

# Exporta la OBT de GrantMatch (PSet 2) a CSV para el EDA. Las credenciales
# salen del .env del proyecto; nunca se imprimen.
ENV = os.path.expanduser(os.environ.get("OBT_ENV", "~/Documents/DataMining/GrantMatch/pset_2/.env"))
SALIDA = "obt_award_work.csv"

# La OBT completa pesa ~155 MB; se guarda una muestra de awards al azar con
# todas sus filas (una por work financiado).
MUESTRA_AWARDS = 4000
SEMILLA = 42

# Columnas de texto largo y listas: no aportan al EDA tabular y hacen
# pesado el CSV.
EXCLUIR = {
    "AWARD_DESCRIPTION", "ABSTRACT_TEXT", "AWARD_TEXT", "WORK_TEXT",
    "AUTHOR_KEYS", "CORRESPONDING_AUTHOR_KEYS",
}

env = {}
with open(ENV) as f:
    for linea in f:
        linea = linea.strip()
        if linea and not linea.startswith("#") and "=" in linea:
            clave, valor = linea.split("=", 1)
            env[clave.strip()] = valor.strip().strip('"').strip("'")

conexion = snowflake.connector.connect(
    account=env["SNOWFLAKE_ACCOUNT"],
    user=env["SNOWFLAKE_USER"],
    password=env["SNOWFLAKE_PASSWORD"],
    role=env["SNOWFLAKE_ROLE"],
    warehouse=env["SNOWFLAKE_WAREHOUSE"],
    database=env["SNOWFLAKE_DATABASE"],
    schema=env["SNOWFLAKE_SCHEMA_OBT"],
)
cursor = conexion.cursor()

if len(sys.argv) > 1 and sys.argv[1] == "--contar":
    print("Filas en OBT_AWARD_WORK:", cursor.execute("SELECT COUNT(*) FROM OBT_AWARD_WORK").fetchone()[0])
    sys.exit()

columnas = [fila[0] for fila in cursor.execute("DESCRIBE TABLE OBT_AWARD_WORK").fetchall()]
seleccion = ", ".join(c for c in columnas if c not in EXCLUIR)
datos = cursor.execute(f"SELECT {seleccion} FROM OBT_AWARD_WORK").fetch_pandas_all()
datos.columns = datos.columns.str.lower()
print(f"OBT completa: {datos.shape[0]} filas, {datos['award_key'].nunique()} awards")

awards = np.sort(datos["award_key"].unique())
elegidos = np.random.default_rng(SEMILLA).choice(awards, size=MUESTRA_AWARDS, replace=False)
datos = datos[datos["award_key"].isin(elegidos)].sort_values(["award_key", "work_key"])
datos.to_csv(SALIDA, index=False)
print(f"{SALIDA}: {datos.shape[0]} filas, {datos.shape[1]} columnas")
