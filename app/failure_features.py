"""Ingeniería de características del módulo de predicción de fallas (AI4I 2020).

Compartido por entrenamiento (training/train_failure.py) e inferencia
(app/services/failure.py) para garantizar consistencia.
"""
from __future__ import annotations

import math

import pandas as pd

RAW_COLUMNS = {
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}

FEATURES = [
    "type_L", "type_M", "type_H",
    "air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
    "torque_nm", "tool_wear_min",
    "temp_diff_k", "power_w", "strain_min_nm",
]

# Umbrales de sobre-esfuerzo (OSF) por variante de producto, según la
# documentación del dataset AI4I 2020 (Matzka, 2020).
OSF_LIMITS = {"L": 11000, "M": 12000, "H": 13000}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe columnas con nombres normalizados (ver RAW_COLUMNS) y devuelve X."""
    out = pd.DataFrame(index=df.index)
    t = df["type"].astype(str).str.upper()
    for v in ("L", "M", "H"):
        out[f"type_{v}"] = (t == v).astype(int)
    for c in ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
              "torque_nm", "tool_wear_min"]:
        out[c] = df[c].astype(float)
    out["temp_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * 2 * math.pi / 60
    out["strain_min_nm"] = out["tool_wear_min"] * out["torque_nm"]
    return out[FEATURES]


def risk_indicators(row: dict) -> list[dict]:
    """Indicadores interpretables por modo de falla (reglas documentadas del
    dataset). Complementan la probabilidad del modelo; no la sustituyen."""
    temp_diff = row["process_temperature_k"] - row["air_temperature_k"]
    power = row["torque_nm"] * row["rotational_speed_rpm"] * 2 * math.pi / 60
    strain = row["tool_wear_min"] * row["torque_nm"]
    limit = OSF_LIMITS.get(str(row["type"]).upper(), 12000)
    ind = []
    if temp_diff < 8.6 and row["rotational_speed_rpm"] < 1380:
        ind.append({"mode": "HDF", "name": "Falla por disipación de calor",
                    "detail": f"ΔT={temp_diff:.1f} K < 8.6 K y velocidad < 1380 rpm"})
    if power < 3500 or power > 9000:
        ind.append({"mode": "PWF", "name": "Falla de potencia",
                    "detail": f"Potencia={power:.0f} W fuera de [3500, 9000] W"})
    if strain > limit:
        ind.append({"mode": "OSF", "name": "Falla por sobre-esfuerzo",
                    "detail": f"Desgaste×Torque={strain:.0f} > {limit} min·Nm"})
    if row["tool_wear_min"] >= 200:
        ind.append({"mode": "TWF", "name": "Desgaste de herramienta",
                    "detail": f"Desgaste={row['tool_wear_min']:.0f} min (zona de reemplazo 200–240)"})
    return ind
