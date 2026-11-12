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

**1. Exportar el modelo desde MLflow.** Con el servidor de MLflow levantado (`bash .devcontainer/mlflow-server.sh`, en `127.0.0.1:5050`):

```
python servicio/exportar_modelo.py            # busca el run con la etiqueta candidato=final
python servicio/exportar_modelo.py <run_id>   # o exporta un run concreto
```

Deja `servicio/modelo.joblib` (el pipeline completo: preparación y ridge `alpha=10`) y `servicio/modelo_info.json` con el `run_id`. Hoy carga el run `505f6532318444f3a3fc7a1f03785738` del experimento `grantmatch`. Ninguno de los dos archivos se sube al repositorio: se regeneran con este script.

**2. Construir y levantar el contenedor.**

```
docker build -f servicio/Dockerfile -t grantmatch-monto .
docker run --rm -p 8000:8000 grantmatch-monto
```

Sin Docker: `uvicorn servicio.api:app --port 8000`. La documentación interactiva queda en `/docs`.

**3. Rutas.**

- `GET /salud`: `{"estado": "ok", "modelo_cargado": true, "version_modelo": "505f6532318444f3a3fc7a1f03785738"}`.
- `POST /estimar_monto`: recibe el award como está en el warehouse (`funder_name` obligatorio; `funding_type`, `duration_days`, `institutions_count` y `has_description` opcionales) y calcula las features con `agregar_features` de `modelado/pipeline_proyecto.py`. Devuelve el monto estimado en USD, su log10 y el `run_id` del modelo que lo estimó. Si falta `funder_name` o la duración es negativa, responde 422; si no hay modelo cargado, 503.

Ejemplo:

```
$ curl -X POST localhost:8000/estimar_monto -H "Content-Type: application/json" \
    -d '{"funder_name": "National Science Foundation", "funding_type": "grant", "duration_days": 1094, "institutions_count": 1, "has_description": true}'
{"monto_estimado_usd":313691.79,"log10_monto":5.4965,"version_modelo":"505f6532318444f3a3fc7a1f03785738"}
```

**4. Pruebas.** Desde la raíz del fork, `pytest psets/proyecto -v` corre también `servicio/test_api.py`, con un modelo de prueba en vez de `modelo.joblib`.
