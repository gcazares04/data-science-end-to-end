# [Nombre de tu proyecto]

> Este es el README de tu proyecto: reemplaza cada sección entre corchetes a medida que avanzas. Escríbelo para alguien que clona tu fork sin haber estado en clase y necesita entender qué hace el proyecto y correrlo, no como una nota para ti.

[Un párrafo: qué problema de negocio resuelve, para quién, y qué predice el modelo. Sale de tu propuesta del PSet 1, que está completa en `docs/`.]

**Métrica de éxito:** [la métrica de negocio y el umbral de decisión del PSet 1]

## Datos

| Fuente | Qué contiene | Cómo se obtiene | Dónde queda |
|---|---|---|---|
| [nombre de la fuente] | [qué filas y columnas trae, desde qué fecha] | [la URL, la API o el procedimiento exacto] | `data/raw/[archivo]` |

Los datos no están en el repositorio. Para regenerarlos desde cero: [los comandos, en orden]. Se completa con el PSet 1.

## Cómo correr el pipeline de datos

[Los comandos, en orden y desde qué carpeta, para ir de la fuente al warehouse: levantar la infraestructura, construir la imagen, subir el flow a Kestra, limpiar, cargar y correr dbt. Cada paso dice qué archivo deja y dónde. Se completa con el PSet 2.]

## Cómo entrenar y evaluar el modelo

[Qué notebook corre cada etapa del modelado, en qué orden, y qué necesita levantado (MLflow). Se completa con el PSet 3.]

## Modelo final

[El modelo elegido, su `run_id` en MLflow y la métrica que lo sostiene. El razonamiento completo está en `docs/decision_modelo.md`. Se completa con el PSet 3.]

## Cómo correr el servicio

El modelo de monto se despliega por lotes (`docs/despliegue.md`), pero también se sirve como API para que otro sistema (el propio lote, un tablero) le pida una estimación. Todo se corre desde `psets/proyecto/`.

**1. Registrar, promover y exportar el modelo.** Con el servidor de MLflow levantado (`bash .devcontainer/mlflow-server.sh`, en `127.0.0.1:5050`):

```
python -m servicio.registrar_modelo        # version nueva de grantmatch-monto (ridge alpha=10)
python -m servicio.registrar_modelo 1      # otra version, con alpha=1
python -m servicio.promover 1              # alias produccion en la version 1
python servicio/exportar_modelo.py         # baja la version @produccion
```

El último deja `servicio/modelo.joblib` (el pipeline completo: preparación y ridge) y `servicio/modelo_version.txt` con el número de la versión. Hoy `produccion` es la versión 1 (ridge `alpha=10`, MAE log 0.3608). Ninguno de los dos archivos se sube al repositorio: se regeneran con el script. El criterio para mover el alias y cómo volver atrás están en `docs/despliegue.md`, sección **Versionado y registro**.

**2. Construir y levantar el contenedor.**

```
docker build -f servicio/Dockerfile -t grantmatch-monto .
docker run --rm -p 8000:8000 grantmatch-monto
```

Sin Docker: `uvicorn servicio.api:app --port 8000`. La documentación interactiva queda en `/docs`.

**3. Rutas.**

- `GET /salud`: `{"estado": "ok", "modelo_cargado": true, "version_modelo": "1"}`.
- `POST /estimar_monto`: recibe el award como está en el warehouse (`funder_name` obligatorio; `funding_type`, `duration_days`, `institutions_count` y `has_description` opcionales) y calcula las features con `agregar_features` de `modelado/pipeline_proyecto.py`. Acepta también `award_key` (la llave del warehouse), que el lote manda para poder unir la estimación con el monto real. Devuelve el `id_prediccion`, el monto estimado en USD, su log10 y la versión de `grantmatch-monto` que lo estimó. Si falta `funder_name` o la duración es negativa, responde 422; si no hay modelo cargado, 503. Cada estimación queda en `monitoreo/predicciones.jsonl`.
- `GET /metricas`: peticiones a `/estimar_monto`, errores de validación (422), errores del servidor (≥500) y latencia p50 y p95 en milisegundos.

Ejemplo:

```
$ curl -X POST localhost:8000/estimar_monto -H "Content-Type: application/json" \
    -d '{"funder_name": "National Science Foundation", "funding_type": "grant", "duration_days": 1094, "institutions_count": 1, "has_description": true}'
{"id_prediccion":"5f0c...","monto_estimado_usd":313691.79,"log10_monto":5.4965,"version_modelo":"1"}
```

**4. Pruebas.** Desde la raíz del fork, `pytest psets/proyecto -v` corre también `servicio/test_api.py`, con un modelo de prueba en vez de `modelo.joblib`, y `monitoreo/test_desempeno.py`.

## Monitoreo

El monto real llega tarde (si el funder lo publica) o nunca, así que se vigilan tres capas: el sistema (`/metricas`), indicadores de la misma noche sin verdad (mediana del monto estimado, funders desconocidos, awards sin duración) y el MAE y el sesgo cuando el monto se publica, por año y por versión. Con el modelo exportado, desde `psets/proyecto/`:

```
python -m monitoreo.simular_lote           # el lote sobre los awards de prueba, con montos publicados 30 a 730 dias despues
python -m monitoreo.desempeno              # al 2025-12-31
python -m monitoreo.desempeno 2022-12-31   # con menos verdad
```

Las métricas, sus umbrales y los resultados de la simulación están en `docs/monitoreo.md`.
