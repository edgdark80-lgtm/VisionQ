#!/usr/bin/env bash
# Publica este repositorio en https://github.com/edgdark80-lgtm/VisionQ
# Uso (desde la carpeta VisionQ descomprimida): bash scripts/publish_github.sh
# Requiere haber iniciado sesión en GitHub (gh auth login, o credenciales de git/llavero de macOS).
set -euo pipefail
cd "$(dirname "$0")/.."
git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/edgdark80-lgtm/VisionQ.git
git branch -M main
git push -u origin main
echo "Listo: https://github.com/edgdark80-lgtm/VisionQ"
