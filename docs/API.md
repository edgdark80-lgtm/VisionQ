# Referencia de la API de VisionQ (v1)

Base local: `http://localhost:8000`. Contrato completo y ejecutable en `/docs` (Swagger UI), `/redoc` y `/openapi.json`.

Todas las respuestas incluyen los encabezados `X-Request-ID` (trazabilidad; se puede enviar uno propio) y `X-Process-Time-ms`. Si se define `VISIONQ_API_KEY`, los endpoints `POST` exigen el encabezado `X-API-Key`.

| Método | Ruta | Propósito | Códigos |
|---|---|---|---|
| GET | `/api/v1/health` | Estado del servicio y de los modelos (sonda de salud) | 200 |
| GET | `/api/v1/models` | Metadatos, versión, métricas y umbrales de los modelos | 200 |
| GET | `/api/v1/metrics` | Contadores de uso, errores, latencia promedio y decisiones | 200 |
| POST | `/api/v1/inspect` | Clasifica el defecto de una imagen y emite la disposición | 200, 400, 401, 413, 415, 422, 503 |
| POST | `/api/v1/inspect/batch` | Hasta 16 imágenes; los errores por archivo no detienen el lote | 200, 401, 413, 422 |
| POST | `/api/v1/failure/predict` | Probabilidad de falla de máquina a partir de una lectura | 200, 401, 422, 503 |
| POST | `/api/v1/failure/predict/batch` | Hasta 1000 lecturas | 200, 401, 413, 422 |
| GET | `/ui/` | Frontend web (consume la API anterior) | 200 |

## Inspección visual

```bash
curl -F "file=@samples/scratches_195.jpg;type=image/jpeg" http://localhost:8000/api/v1/inspect
```

Respuesta (resumida):

```json
{
  "filename": "scratches_195.jpg",
  "defect_class": "scratches",
  "defect_name_es": "Rayaduras",
  "confidence": 0.93,
  "top_k": [{"label": "scratches", "probability": 0.93}, "..."],
  "decision": "RETRABAJO",
  "decision_reason": "Defecto 'scratches' con confianza 0.93; política: RETRABAJO",
  "out_of_distribution": false,
  "quality": {"width": 200, "height": 200, "flags": []},
  "latency_ms": 37.1
}
```

Lógica de decisión (en orden): alertas de calidad de captura → `REVISION_MANUAL`; imagen fuera de distribución → `REVISION_MANUAL`; confianza < `VISIONQ_CONFIDENCE_THRESHOLD` (0.60) → `REVISION_MANUAL`; en otro caso se aplica la política ilustrativa por clase (`RECHAZO` o `RETRABAJO`).

## Predicción de fallas

```bash
curl -X POST http://localhost:8000/api/v1/failure/predict \
  -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature_k":302.0,"process_temperature_k":310.5,
       "rotational_speed_rpm":1300,"torque_nm":60,"tool_wear_min":210}'
```

| Campo | Tipo | Rango aceptado |
|---|---|---|
| `type` | `L` \| `M` \| `H` | — |
| `air_temperature_k` | número | 250 – 350 |
| `process_temperature_k` | número | 250 – 400, y ≥ temperatura del aire |
| `rotational_speed_rpm` | número | 0 – 5000 |
| `torque_nm` | número | 0 – 150 |
| `tool_wear_min` | número | 0 – 400 |

La respuesta incluye `failure_probability`, `threshold`, `prediction`, `risk_level` (BAJO/MEDIO/ALTO), `recommended_action`, `risk_indicators` (reglas físicas HDF, PWF, OSF, TWF documentadas en AI4I 2020) y `out_of_training_range` (variables válidas pero fuera del rango visto en entrenamiento, es decir, extrapolación).

## Errores

Formato uniforme:

```json
{"error": "Datos de entrada inválidos",
 "detail": [{"campo": "body.type", "mensaje": "Input should be 'L', 'M' or 'H'"}],
 "request_id": "e2ba67903639"}
```
