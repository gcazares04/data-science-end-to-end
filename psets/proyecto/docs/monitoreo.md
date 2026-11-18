# Monitoreo del modelo de monto

El monto real de un award que no lo informa llega tarde o no llega nunca: solo se conoce si el funder lo publica después, meses o años más tarde. Por eso el monitoreo de GrantMatch tiene tres capas: lo que el servicio mide en cada petición, lo que se ve la misma noche sin verdad y lo que solo se sabe cuando el monto se publica.

## Qué se registra

Cada estimación de `POST /estimar_monto` deja una línea en `monitoreo/predicciones.jsonl`:

| Campo | Para qué |
|---|---|
| `id_prediccion` | identifica la estimación (también va en la respuesta) |
| `momento` | cuándo se estimó: define el período |
| `version_modelo` | qué versión de `grantmatch-monto` la hizo: separa las métricas por versión |
| `award_key`, `funder_name`, `funding_type`, `duration_days`, `institutions_count`, `has_description` | la entrada tal como llegó |
| `funder_conocido` | si el modelo vio ese funder al entrenar (uno nuevo se codifica como ceros) |
| `monto_estimado_usd`, `log10_monto` | la salida |

`award_key` es la llave del warehouse: con ella se une la estimación con el monto cuando se publica. Es opcional en la API para no romper a quien ya la llamaba, pero el lote nocturno siempre la manda.

## Las métricas

| Métrica | Tipo | De dónde sale | Cuándo se conoce | Umbral de alerta |
|---|---|---|---|---|
| Latencia p50 y p95 | sistema | `GET /metricas` (middleware) | en cada petición | — (el lote no espera respuestas en línea) |
| Errores de validación (422) | sistema / datos | `GET /metricas` | en cada petición | revisar la ingesta si aparecen: el lote solo manda filas del warehouse |
| Errores del servidor (≥500) | sistema | `GET /metricas` | en cada petición | cualquiera |
| Mediana del log10 estimado | modelo, sin verdad | registro | la misma noche | se aleja más de 0.30 de 5.741 (un factor 2) |
| Funders desconocidos | modelo, sin verdad | registro (`funder_conocido`) | la misma noche | más del 20 % |
| Awards sin duración | datos | registro | la misma noche | más del 10 % (en entrenamiento: 3.3 %) |
| Cobertura | modelo | registro + montos publicados | crece con los meses | — (dice cuánto creerle al MAE) |
| MAE en log10 | modelo, con verdad | registro + montos publicados | meses o años después | más de 1.25 × 0.3608 = 0.451, con al menos 5 montos |
| Sesgo en log10 (real − estimado) | modelo, con verdad | registro + montos publicados | meses o años después | — (positivo es subestimar, el error caro) |

Los indicadores sin verdad solo se evalúan con al menos 5 estimaciones en el período: con menos, una mediana o una fracción son ruido. Las métricas con verdad se separan por año de la estimación y por versión, para no mezclar el desempeño de dos modelos.

## La simulación

`python -m monitoreo.simular_lote` toma los 46 awards de prueba (de 2020 en adelante, con monto), los manda al servicio en orden, cada uno la noche del día en que empieza, y simula que su monto se publica entre 30 y 730 días después. Además manda dos filas sin funder, como las de una ingesta rota. `python -m monitoreo.desempeno [hoy]` une el registro con los montos publicados hasta `hoy` y resume.

Métricas del sistema después del lote: 48 peticiones, 2 errores de validación (las filas sin funder), 0 del servidor, latencia p50 2.1 ms y p95 2.5 ms.

Al 2025-12-31 (46 estimaciones, 43 con monto publicado, MAE log10 total 0.354 contra 0.3608 de referencia):

| Año | Estimaciones | Con verdad | Cobertura | MAE log10 | Sesgo log10 | Mediana log10 |
|---|---|---|---|---|---|---|
| 2020 | 10 | 10 | 100 % | 0.310 | −0.062 | 5.67 |
| 2021 | 11 | 11 | 100 % | 0.299 | −0.033 | 5.75 |
| 2022 | 12 | 12 | 100 % | 0.313 | +0.199 | 5.75 |
| 2023 | 8 | 8 | 100 % | **0.558** | +0.057 | 5.50 |
| 2024 | 3 | 2 | 67 % | 0.313 | −0.313 | 5.75 |
| 2025 | 2 | 0 | 0 % | — | — | 5.39 |

Alerta: el MAE de 2023 (0.558) pasa el umbral. Casi todo viene de un solo award: un `cooperative_agreement` de la NSF de 118 millones que el modelo estimó en 1.3 millones (error de 1.97 en log10, casi un factor 100). Sin él, el MAE de 2023 queda cerca de 0.36. Es el error caro: subestimar un award enorme.

Lo que se ve con un "hoy" anterior: al 2022-12-31 solo 2 de las 12 estimaciones de 2022 tenían monto publicado (17 %) y no había ninguna alerta. Un problema del modelo en 2022 recién se habría podido medir en 2023 o 2024.

Los indicadores sin verdad no se movieron: ningún funder desconocido (los awards de prueba son de funders que el modelo ya vio), una sola fila sin duración (en 2020) y medianas entre 5.4 y 5.75. En 2024 y 2025 hay menos de 5 estimaciones, así que no se evalúan.

## Qué no se ve sin verdad

Que el modelo subestime cada vez más porque los montos reales crecieron (inflación, programas nuevos más grandes) mientras los awards se ven igual que antes. La entrada no cambia, así que la mediana estimada, los funders y la duración se quedan quietos, y el sistema sigue respondiendo en 2 ms sin errores. Solo el sesgo con verdad lo muestra, y con la demora de publicación llega uno o dos años tarde.
