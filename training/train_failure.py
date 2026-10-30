"""Entrena el modelo de predicción de fallas de máquina (AI4I 2020).

Continúa la Actividad 5 (Random Forest como modelo campeón). Aquí se reentrena
de forma reproducible y se selecciona el umbral de decisión sobre un conjunto
de validación (no sobre prueba) para equilibrar precisión y exhaustividad.

Uso:
    python training/train_failure.py --csv data/ai4i2020.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.failure_features import FEATURES, RAW_COLUMNS, build_features  # noqa: E402

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data/ai4i2020.csv"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, encoding="utf-8-sig").rename(columns=RAW_COLUMNS)
    X = build_features(df)
    y = df["Machine failure"].astype(int).values

    # 60 / 20 / 20 estratificado: entrenamiento, validación (umbral), prueba
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)

    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr)

    p_va = rf.predict_proba(X_va)[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_score(y_va, (p_va >= t).astype(int), zero_division=0) for t in grid]
    thr = float(grid[int(np.argmax(f1s))])

    # Reentrena con train+val y evalúa una única vez en prueba
    rf.fit(pd.concat([X_tr, X_va]), np.concatenate([y_tr, y_va]))
    p_te = rf.predict_proba(X_te)[:, 1]
    y_hat = (p_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, y_hat).ravel()

    metrics = {
        "dataset": "AI4I 2020 Predictive Maintenance (Matzka, 2020) - 10 000 registros",
        "positive_rate": round(float(y.mean()), 4),
        "n_train_val": int(len(y_tmp)), "n_test": int(len(y_te)),
        "model": "RandomForestClassifier(n_estimators=300, class_weight=balanced_subsample)",
        "decision_threshold": thr, "threshold_selected_on": "validación (máx. F1)",
        "test_roc_auc": round(float(roc_auc_score(y_te, p_te)), 4),
        "test_pr_auc": round(float(average_precision_score(y_te, p_te)), 4),
        "test_precision": round(float(precision_score(y_te, y_hat)), 4),
        "test_recall": round(float(recall_score(y_te, y_hat)), 4),
        "test_f1": round(float(f1_score(y_te, y_hat)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance": dict(sorted(
            {f: round(float(i), 4) for f, i in zip(FEATURES, rf.feature_importances_)}.items(),
            key=lambda kv: -kv[1])),
        "training_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in
                            ["air_temperature_k", "process_temperature_k", "rotational_speed_rpm",
                             "torque_nm", "tool_wear_min"]},
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": rf, "threshold": thr, "features": FEATURES,
                 "training_ranges": metrics["training_ranges"],
                 "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "metrics": {k: metrics[k] for k in ("test_roc_auc", "test_pr_auc", "test_f1",
                                                      "test_precision", "test_recall")}},
                out / "failure_model.joblib", compress=3)
    (out / "failure_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in metrics.items() if k.startswith("test_") or k == "decision_threshold"}, indent=2))


if __name__ == "__main__":
    main()
