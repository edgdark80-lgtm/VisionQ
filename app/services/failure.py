"""Servicio de predicción de fallas de máquina (AI4I 2020)."""
from __future__ import annotations

import time
from pathlib import Path

import joblib
import pandas as pd

from app.failure_features import build_features, risk_indicators


class FailureService:
    def __init__(self, model_path: Path, threshold_override: float | None = None):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        # En inferencia, 1 hilo evita la sobrecarga de paralelizar lotes pequeños
        self.model.n_jobs = 1
        self.threshold = float(threshold_override if threshold_override is not None else bundle["threshold"])
        self.ranges = bundle["training_ranges"]
        self.meta = {"trained_at": bundle["trained_at"], "metrics": bundle["metrics"],
                     "threshold": self.threshold}

    def _risk_level(self, p: float) -> str:
        if p >= self.threshold:
            return "ALTO"
        if p >= self.threshold / 2:
            return "MEDIO"
        return "BAJO"

    @staticmethod
    def _action(level: str) -> str:
        return {
            "ALTO": "Programar mantenimiento inmediato e inspeccionar herramienta/enfriamiento",
            "MEDIO": "Aumentar frecuencia de monitoreo y planear mantenimiento en el siguiente turno",
            "BAJO": "Continuar operación normal",
        }[level]

    def predict(self, readings: list[dict]) -> list[dict]:
        t0 = time.perf_counter()
        df = pd.DataFrame(readings)
        probs = self.model.predict_proba(build_features(df))[:, 1]
        elapsed = (time.perf_counter() - t0) * 1000 / max(len(readings), 1)
        out = []
        for row, p in zip(readings, probs):
            p = float(p)
            level = self._risk_level(p)
            oor = [k for k, (lo, hi) in self.ranges.items() if not lo <= row[k] <= hi]
            out.append({
                "failure_probability": round(p, 4),
                "threshold": self.threshold,
                "prediction": "FALLA_PROBABLE" if p >= self.threshold else "OPERACION_NORMAL",
                "risk_level": level,
                "recommended_action": self._action(level),
                "risk_indicators": risk_indicators(row),
                "out_of_training_range": oor,
                "latency_ms": round(elapsed, 3),
            })
        return out
