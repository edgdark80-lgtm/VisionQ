"""Extracción de características para el módulo de inspección visual.

Este módulo es la ÚNICA fuente de verdad para transformar una imagen en un
vector de características. Lo usan tanto el entrenamiento
(training/train_vision.py) como la inferencia (app/services/vision.py), lo que
evita el "training/serving skew" (diferencias entre cómo se entrenó y cómo se
sirve el modelo).

Descriptores (textura + forma + intensidad), elegidos por ser ligeros y
robustos para superficies metálicas:
  * LBP uniforme multi-escala (P=8,R=1 y P=16,R=2)
  * GLCM (contraste, disimilitud, homogeneidad, energía, correlación, ASM)
  * HOG sobre una versión reducida de 64x64
  * Estadísticos de intensidad y densidad de bordes (Sobel)
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
from skimage.filters import sobel
from skimage.transform import resize

IMG_SIZE = 200  # tamaño nativo de NEU-DET
FEATURE_VERSION = "1.0.0"


def load_image_bytes(data: bytes) -> np.ndarray:
    """Decodifica bytes de imagen a un arreglo uint8 en escala de grises."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        gray = im.convert("L")
        return np.asarray(gray, dtype=np.uint8)


def preprocess(gray: np.ndarray) -> np.ndarray:
    """Redimensiona a IMG_SIZE x IMG_SIZE y devuelve uint8."""
    if gray.shape != (IMG_SIZE, IMG_SIZE):
        gray = resize(gray, (IMG_SIZE, IMG_SIZE), anti_aliasing=True, preserve_range=True)
    return np.clip(gray, 0, 255).astype(np.uint8)


def _lbp_hist(gray: np.ndarray, p: int, r: int) -> np.ndarray:
    lbp = local_binary_pattern(gray, p, r, method="uniform")
    hist, _ = np.histogram(lbp, bins=np.arange(0, p + 3), density=True)
    return hist


def _glcm_feats(gray: np.ndarray) -> np.ndarray:
    q = (gray // 8).astype(np.uint8)  # 32 niveles para estabilidad
    glcm = graycomatrix(
        q, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32, symmetric=True, normed=True,
    )
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    # promedio sobre ángulos -> invariancia a rotación
    return np.concatenate([graycoprops(glcm, p).mean(axis=1) for p in props])


def _hog_feats(gray: np.ndarray) -> np.ndarray:
    small = resize(gray, (64, 64), anti_aliasing=True)
    return hog(small, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
               feature_vector=True)


def _intensity_feats(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float64) / 255.0
    flat = g.ravel()
    edges = sobel(g)
    std = flat.std()
    skew = float(stats.skew(flat)) if std > 1e-6 else 0.0
    kurt = float(stats.kurtosis(flat)) if std > 1e-6 else 0.0
    return np.array([
        flat.mean(), std, skew, kurt,
        *np.percentile(flat, [5, 50, 95]),
        edges.mean(), edges.std(), (edges > 0.1).mean(),
    ])


def extract_features(gray: np.ndarray) -> np.ndarray:
    """Vector de características (float64) a partir de una imagen gris uint8."""
    g = preprocess(gray)
    return np.concatenate([
        _lbp_hist(g, 8, 1),
        _lbp_hist(g, 16, 2),
        _glcm_feats(g),
        _hog_feats(g),
        _intensity_feats(g),
    ]).astype(np.float64)


def image_quality(gray: np.ndarray) -> dict:
    """Chequeos de calidad de captura (previos a la inferencia)."""
    h, w = gray.shape
    mean = float(gray.mean())
    std = float(gray.std())
    flags = []
    if min(h, w) < 32:
        flags.append("RESOLUCION_INSUFICIENTE")
    if mean < 15:
        flags.append("IMAGEN_SUBEXPUESTA")
    if mean > 240:
        flags.append("IMAGEN_SOBREEXPUESTA")
    if std < 3:
        flags.append("IMAGEN_UNIFORME_SIN_TEXTURA")
    return {"width": int(w), "height": int(h), "mean_intensity": round(mean, 2),
            "std_intensity": round(std, 2), "flags": flags}
