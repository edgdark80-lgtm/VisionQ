#!/usr/bin/env bash
# Despliegue alternativo en Azure Container Apps.
# Requisitos: Azure CLI (az login) y la extensión containerapp.
set -euo pipefail
RG="${RG:-rg-visionq}"; LOC="${LOC:-eastus}"; ACR="${ACR:-visionqacr$RANDOM}"; APP="${APP:-visionq-api}"; ENVNAME="${ENVNAME:-visionq-env}"
az group create -n "$RG" -l "$LOC"
az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true
az acr build -r "$ACR" -t visionq:2.0.0 .
az containerapp env create -g "$RG" -n "$ENVNAME" -l "$LOC"
az containerapp create -g "$RG" -n "$APP" --environment "$ENVNAME" \
  --image "$ACR.azurecr.io/visionq:2.0.0" --registry-server "$ACR.azurecr.io" \
  --target-port 8000 --ingress external --cpu 1.0 --memory 2.0Gi \
  --min-replicas 0 --max-replicas 3 --env-vars VISIONQ_ENVIRONMENT=production
URL="https://$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)"
echo "Servicio publicado en: $URL"; bash scripts/smoke_test.sh "$URL"
