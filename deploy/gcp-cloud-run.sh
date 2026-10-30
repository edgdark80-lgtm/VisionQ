#!/usr/bin/env bash
# Despliegue de VisionQ en Google Cloud Run (contenedor serverless, escala a cero).
# Requisitos: gcloud CLI autenticado (gcloud auth login) y un proyecto con facturación.
# Uso: PROJECT_ID=mi-proyecto REGION=us-central1 bash deploy/gcp-cloud-run.sh
set -euo pipefail
: "${PROJECT_ID:?Define PROJECT_ID}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-visionq}"
SERVICE="${SERVICE:-visionq-api}"
TAG="${TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo 2.0.0)}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/visionq:${TAG}"

gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

# Construcción remota en Cloud Build (no requiere Docker local; evita problemas arm64/amd64)
gcloud builds submit --tag "$IMAGE" .

gcloud run deploy "$SERVICE" \
  --image "$IMAGE" --region "$REGION" --platform managed \
  --allow-unauthenticated --port 8000 \
  --cpu 1 --memory 1Gi --concurrency 8 --min-instances 0 --max-instances 3 \
  --set-env-vars VISIONQ_ENVIRONMENT=production,VISIONQ_LOG_LEVEL=INFO

URL=$(gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)')
echo "Servicio publicado en: $URL"
bash scripts/smoke_test.sh "$URL"
