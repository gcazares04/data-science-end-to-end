# Despliegue del modelo de monto: por lotes

**Decisión:** el modelo de monto de GrantMatch se sirve **por lotes**. Cada noche, después de la ingesta y de dbt, un job estima el monto de todos los awards que no lo informan y escribe la estimación en una tabla del warehouse. El recomendador y la interfaz leen esa tabla; nadie espera a que el modelo responda.

## Qué decisión habilita y cuánto puede esperar

- **Decisión:** el investigador decide si una convocatoria vale la pena leerla y postular. El monto estimado aparece junto a cada recomendación para que no descarte una convocatoria grande creyendo que es chica (el error caro de la Semana 11).
- **Quién la usa:** el investigador, a través de las recomendaciones de GrantMatch; también el propio recomendador, que puede filtrar u ordenar por monto.
- **Cuánto puede esperar:** **un día (24 horas)**. Una convocatoria nueva tarda semanas en cerrar, y los datos de OpenAlex se actualizan con días de retraso. Que el monto aparezca al día siguiente de que entra el award no cambia ninguna decisión. Al momento de mostrarlo, leer una fila ya calculada toma milisegundos.

## Cuántos casos hay y si se conocen de antemano

- Se conocen todos de antemano: son los awards que ya están en `OBT.OBT_AWARD_WORK`. No hay ningún caso que aparezca recién cuando el usuario pide la recomendación.
- En la muestra actual hay **4000 awards**, de los que **3830 no informan monto**: esos son los que se estiman. Aunque el catálogo crezca a cientos de miles, la ridge estima miles de filas por segundo, así que el lote completo cabe holgado en la ventana nocturna.

## Qué datos necesitan las features y si están disponibles

| Feature | De dónde sale | ¿Disponible al predecir? |
|---|---|---|
| `duracion_anios` | `duration_days` (fechas de inicio y fin del award) | Sí, se conocen al otorgar el award |
| `escala_financiamiento` | `funding_type` | Sí, viene en el award |
| `funder_name` | el funder | Sí |
| `institutions_count` | instituciones del award | Sí, aunque puede faltar; la preparación lo imputa con la mediana |
| `has_description` | si el award trae descripción | Sí |

Todas salen del registro del award en el warehouse, que el job lee completo. Ninguna depende de algo que haga el usuario en el momento (lo que busca, su perfil), así que no hay nada que obligue a predecir en línea.

## Qué se pierde con el patrón descartado (en línea)

- **Por usar lotes se pierde frescura:** un award que entra hoy no tiene monto estimado hasta la corrida de mañana. Para convocatorias que duran semanas, ese día no importa. Si importara, el job se puede correr después de cada ingesta en vez de una vez por noche.
- **Lo que costaría servirlo en línea:** un servicio HTTP siempre levantado (FastAPI más un contenedor), con monitoreo de latencia y caídas, y el riesgo de que la recomendación falle si el servicio no responde. Además se calcularía muchas veces la misma estimación para el mismo award, que no cambia mientras el modelo no cambie.
- **Cuándo cambiaría la decisión:** si GrantMatch dejara que un usuario cargue su propia convocatoria (un award que no está en el warehouse) y quisiera ver el monto estimado en ese momento. Ahí haría falta el patrón en línea, con una respuesta en menos de unos 500 ms.

Cada fila de la tabla guarda la `version_modelo` (la huella del `.joblib` o el `run_id` de MLflow del candidato ridge `alpha=10`), para saber con qué modelo se estimó cada monto y recalcular solo lo que cambió cuando se reentrene.

## Versionado y registro

- **Nombre en el registro:** `grantmatch-monto`, en el registro de modelos del servidor de MLflow (`127.0.0.1:5050`, pestaña **Models**). Cada corrida de `python -m servicio.registrar_modelo [alpha]` entrena la ridge, guarda sus métricas en el experimento `grantmatch-registro` y crea una versión nueva; ninguna reemplaza a otra.
- **En producción:** la **versión 1**, ridge `alpha=10`, con **MAE de 0.3608 en log10** sobre los awards desde 2020 (el monto estimado se equivoca en un factor de 2.30x en promedio). Es el mismo candidato final de la Semana 11: reproduce el MAE de su run `505f6532318444f3a3fc7a1f03785738`. La versión 2 (ridge `alpha=1`, MAE 0.3832) quedó registrada y sin alias.
- **Criterio para mover el alias:** una versión nueva pasa a `produccion` solo si baja el MAE en log10 sobre el mismo corte por fecha (prueba desde 2020-01-01) y con la misma huella de datos, o si la huella cambió porque llegaron datos nuevos y el MAE no empeora. Como el error caro es subestimar una convocatoria grande, también se revisa que no aumente la subestimación en los awards de más de 1 M USD. Lo decide el equipo del proyecto; la decisión queda en el registro (qué versión tiene el alias) y en la descripción y las etiquetas de la versión en MLflow.
- **Cómo sabe el contenedor qué versión sirve:** `servicio/exportar_modelo.py` baja la versión con el alias `produccion` a `servicio/modelo.joblib` y escribe su número en `servicio/modelo_version.txt`. La imagen copia los dos archivos, así que el contenedor no necesita llegar a MLflow. `/salud` y cada respuesta de `/estimar_monto` devuelven `version_modelo`.

**Volver a la versión anterior** (desde `psets/proyecto/`, con MLflow levantado):

```
python -m servicio.promover <version_anterior>        # mueve el alias, no borra nada
python servicio/exportar_modelo.py                    # baja esa version y su numero
docker build -f servicio/Dockerfile -t grantmatch-monto .
docker run --rm -p 8000:8000 grantmatch-monto
curl localhost:8000/salud                             # version_modelo debe ser la anterior
```

Las estimaciones del lote nocturno guardan `version_modelo`, así que después de volver atrás se pueden recalcular solo las filas que estimó la versión descartada.
