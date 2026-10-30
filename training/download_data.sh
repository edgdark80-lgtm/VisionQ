#!/usr/bin/env bash
# Descarga el dataset NEU-DET (Song & Yan, 2013) desde una copia pública en GitHub
# y lo deja en data/neu_det/{IMAGES,Validation_Images}. Solo se necesita para
# reentrenar; la imagen Docker ya incluye los modelos entrenados.
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
git clone --depth 1 https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection "$TMP/neu"
mkdir -p data/neu_det
cp -r "$TMP/neu/IMAGES" "$TMP/neu/Validation_Images" data/neu_det/
rm -rf "$TMP"
echo "Listo: $(ls data/neu_det/IMAGES | wc -l) imágenes de entrenamiento, $(ls data/neu_det/Validation_Images | wc -l) de validación externa"
# AI4I 2020 ya viene en data/ai4i2020.csv (CC BY 4.0, UCI Machine Learning Repository).
