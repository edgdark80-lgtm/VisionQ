"""Servicio de inspección visual: carga del modelo e inferencia."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from PIL import UnidentifiedImageError

from app.features import FEATURE_VERSION, extract_features, image_quality, load_image_bytes

DEFECT_NAMES_ES = {
    "crazing": "Agrietamiento superficial (crazing)",
    "inclusion": "Inclusión",
    "patches": "Manchas (patches)",
    "pitted_surface": "Superficie picada",
    "rolled-in_scale": "Cascarilla laminada",
    "scratches": "Rayaduras",
}

# Política de disposición ILUSTRATIVA (simulada, no corresponde a una norma
# interna real). Se puede ajustar sin reentrenar el modelo.
DISPOSITION = {
    "crazing": "RECHAZO",
    "inclusion": "RECHAZO",
    "pitted_surface": "RECHAZO",
    "patches": "RETRABAJO",
    "rolled-in_scale": "RETRABAJO",
    "scratches": "RETRABAJO",
}


class InvalidImageError(ValueError):
    """La carga no es una imagen decodificable."""


class VisionService:
    def __init__(self, model_path: Path, confidence_threshold: float):
        bundle = joblib.load(model_path)
        if bundle["feature_version"] != FEATURE_VERSION:
            raise RuntimeError(
                f"Versión de features incompatible: modelo={bundle['feature_version']} código={FEATURE_VERSION}")
        self.pipeline = bundle["pipeline"]
        self.ood = bundle["ood"]
        self.ood_threshold = bundle["ood_threshold"]
        self.zmax_threshold = bundle.get("zmax_threshold", float("inf"))
        self.classes = bundle["classes"]
        self.meta = {k: bundle[k] for k in ("trained_at", "feature_version", "metrics")}
        self.confidence_threshold = confidence_threshold

    def predict_bytes(self, data: bytes, filename: str | None = None) -> dict:
        t0 = time.perf_counter()
        try:
            gray = load_image_bytes(data)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("El archivo no es una imagen válida o está corrupto") from exc

        quality = image_quality(gray)
        feats = extract_features(gray).reshape(1, -1)
        if not np.all(np.isfinite(feats)):
            feats = np.nan_to_num(feats)

        proba = self.pipeline.predict_proba(feats)[0]
        order = np.argsort(proba)[::-1]
        best = self.classes[order[0]]
        conf = float(proba[order[0]])

        scaled = self.pipeline.named_steps["scaler"].transform(feats)
        ood_score = float(self.ood.score_samples(scaled)[0])
        zmax = float(np.abs(scaled).max())
        is_ood = ood_score < self.ood_threshold or zmax > self.zmax_threshold

        # Regla de decisión con humano en el ciclo
        if quality["flags"]:
            decision, reason = "REVISION_MANUAL", "Calidad de captura insuficiente: " + ", ".join(quality["flags"])
        elif is_ood:
            decision, reason = "REVISION_MANUAL", "Imagen fuera de la distribución de entrenamiento"
        elif conf < self.confidence_threshold:
            decision, reason = "REVISION_MANUAL", f"Confianza {conf:.2f} < umbral {self.confidence_threshold:.2f}"
        else:
            decision = DISPOSITION[best]
            reason = f"Defecto '{best}' con confianza {conf:.2f}; política: {decision}"

        return {
            "filename": filename,
            "defect_class": best,
            "defect_name_es": DEFECT_NAMES_ES[best],
            "confidence": round(conf, 4),
            "top_k": [{"label": self.classes[i], "probability": round(float(proba[i]), 4)} for i in order[:3]],
            "decision": decision,
            "decision_reason": reason,
            "out_of_distribution": bool(is_ood),
            "ood_score": round(ood_score, 4),
            "quality": quality,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        }
