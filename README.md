# VisionQ — Inspección visual de defectos y predicción de fallas

[![CI](https://github.com/edgdark80-lgtm/VisionQ/actions/workflows/ci.yml/badge.svg)](https://github.com/edgdark80-lgtm/VisionQ/actions/workflows/ci.yml)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/edgdark80-lgtm/VisionQ/blob/main/notebooks/VisionQ_Colab.ipynb)

**Proyecto académico — Gestión de proyectos de inteligencia artificial · Fase II: Ejecución.**
VisionQ es una solución desplegable para una línea de manufactura (contexto automotriz simulado) que integra dos modelos de IA detrás de una API REST y un frontend web, empaquetados en un único contenedor Docker.

| Módulo | Dataset | Modelo | Métrica en prueba |
|---|---|---|---|
| Inspección visual de defectos superficiales (6 clases) | NEU-DET (Song & Yan, 2013), 1 770 imágenes | Descriptores de textura (LBP + GLCM + HOG + estadísticos, 374 rasgos) → SVM RBF, entrenado con aumento de datos | Exactitud 97.7 %, F1-macro 0.977 |
| Predicción de falla de máquina | AI4I 2020 (Matzka, 2020), 10 000 registros | Random Forest balanceado, umbral 0.36 elegido en validación | ROC-AUC 0.984, PR-AUC 0.875, F1 0.857 |

---

## Arquitectura

```
 Navegador ──HTTP──▶  /ui/  (frontend HTML/CSS/JS, servido por FastAPI)
     │                        │ fetch() JSON / multipart
     │                        ▼
     └──────────────▶  /api/v1/*  FastAPI  ── VisionService  ── models/vision_model.joblib
                         │  validación Pydantic, X-Request-ID, logs JSON, métricas
                         └─ FailureService ── models/failure_model.joblib
                         /docs  /redoc  /openapi.json  (contrato OpenAPI 3.1)

 Contenedor Docker (python:3.11-slim, multi-etapa, usuario no root, sistema de archivos de solo lectura)
```

- **Backend** (`app/`): FastAPI + Uvicorn. Servicios desacoplados (`app/services/`), esquemas Pydantic (`app/schemas.py`), configuración 12-factor por variables `VISIONQ_*` (`app/config.py`).
- **Frontend** (`frontend/`): HTML/CSS/JS sin dependencias. Consume la API con `fetch` (`frontend/config.js` define la URL base, de modo que puede alojarse por separado si se desea).
- **Política de decisión**: el clasificador emite `ACEPTAR / RETRABAJO / RECHAZO`; si la confianza es < 0.60, la imagen está fuera de distribución o la captura es deficiente, la respuesta es `REVISION_MANUAL` (humano en el bucle).

## Estructura del repositorio

```
app/                 Código del backend (API, servicios, extracción de rasgos)
frontend/            Interfaz web que consume la API
models/              Modelos entrenados (.joblib) y sus métricas (.json)
training/            Scripts de entrenamiento y descarga de datos
tests/               46 pruebas pytest: funcionales, casos extremos y calidad del modelo
scripts/             Prueba de humo E2E, prueba de carga, evaluación de robustez, capturas
deploy/              Despliegue en la nube: Cloud Run, Render, Azure Container Apps
notebooks/           VisionQ_Colab.ipynb — ejecución completa en Google Colab
docs/                Referencia de API, manual de despliegue, documento de pruebas y evidencias
.github/workflows/   CI (lint + pytest + build + smoke) y release (GHCR / Cloud Run)
Dockerfile           Imagen de producción multi-etapa
docker-compose.yml   Ejecución local reproducible
```

## Inicio rápido con Docker (local)

Requisitos: Docker 24+ (Docker Desktop en macOS/Windows; funciona en Apple Silicon y x86_64).

```bash
git clone https://github.com/edgdark80-lgtm/VisionQ.git
cd VisionQ
docker build -t visionq:2.0.0 .
docker run -d --name visionq -p 8000:8000 --read-only --tmpfs /tmp visionq:2.0.0
docker ps                      # STATUS debe mostrar (healthy) tras ~10 s
bash scripts/smoke_test.sh     # 11 verificaciones E2E contra el contenedor
```

O bien: `docker compose up --build -d`.

Abrir:
- Frontend: <http://localhost:8000/ui/>
- Swagger UI: <http://localhost:8000/docs> · ReDoc: <http://localhost:8000/redoc>

Detener: `docker rm -f visionq`.

## Ejemplos de uso de la API

```bash
# Salud
curl http://localhost:8000/api/v1/health

# Inspección de una imagen
curl -F "file=@samples/scratches_195.jpg;type=image/jpeg" http://localhost:8000/api/v1/inspect

# Predicción de falla
curl -X POST http://localhost:8000/api/v1/failure/predict -H "Content-Type: application/json" \
  -d '{"type":"L","air_temperature_k":302.0,"process_temperature_k":310.5,"rotational_speed_rpm":1300,"torque_nm":60.0,"tool_wear_min":210}'
```

Referencia completa de endpoints: [`docs/API.md`](docs/API.md).

## Ejecución sin Docker (desarrollo)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload       # http://localhost:8000/ui/
pytest -q                           # 46 pruebas
```

## Google Colab

El cuaderno [`notebooks/VisionQ_Colab.ipynb`](notebooks/VisionQ_Colab.ipynb) clona este repositorio, instala dependencias en un entorno aislado, ejecuta las 46 pruebas, levanta la API, la prueba con imágenes y lecturas reales, abre el frontend dentro de Colab y reproduce la evaluación de robustez. Colab no permite ejecutar el daemon de Docker, por lo que el cuaderno ejecuta la misma aplicación con Uvicorn; el contenedor se valida localmente o en la nube.

## Despliegue en la nube

| Opción | Tipo | Archivo |
|---|---|---|
| Google Cloud Run (recomendada) | Contenedor serverless | `deploy/gcp-cloud-run.sh` |
| Render | PaaS con Dockerfile | `render.yaml` |
| Azure Container Apps | Contenedor gestionado | `deploy/azure-container-apps.sh` |
| GitHub Container Registry | Registro de imágenes | `.github/workflows/release.yml` |

Paso a paso, requisitos y estrategia: [`docs/Manual_Despliegue_VisionQ.docx`](docs/Manual_Despliegue_VisionQ.docx).

## Pruebas y validación

```bash
pytest -q                                   # unitarias, integración y casos extremos
bash scripts/smoke_test.sh http://localhost:8000
python scripts/load_test.py --base http://localhost:8000
python scripts/evaluate_models.py           # requiere bash training/download_data.sh
```

Resultados, casos extremos y conclusiones: [`docs/Validacion_Pruebas_VisionQ.docx`](docs/Validacion_Pruebas_VisionQ.docx). Evidencias crudas (logs, JSON, capturas): [`docs/evidence/`](docs/evidence/).

## Configuración

Variables de entorno con prefijo `VISIONQ_` (ver `.env.example`): `CONFIDENCE_THRESHOLD`, `MAX_UPLOAD_MB`, `MAX_BATCH_IMAGES`, `FAILURE_THRESHOLD`, `CORS_ORIGINS`, `API_KEY` (si se define, los `POST` exigen `X-API-Key`), `LOG_LEVEL`. `PORT` lo inyecta la plataforma (Cloud Run, Render, etc.).

## Reentrenamiento

```bash
bash training/download_data.sh      # NEU-DET (~30 MB); AI4I ya está en data/
python training/train_failure.py
python training/train_vision.py     # ~3–6 min en CPU
```

## Referencias

- Matzka, S. (2020). *AI4I 2020 predictive maintenance dataset* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- Song, K., & Yan, Y. (2013). A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects. *Applied Surface Science, 285*, 858–864. https://doi.org/10.1016/j.apsusc.2013.09.002

Proyecto académico con cifras simuladas de contexto; no utiliza datos internos de ninguna empresa. La política de disposición es ilustrativa.
