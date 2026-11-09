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
