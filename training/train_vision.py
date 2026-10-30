"""Entrena el clasificador de defectos superficiales de VisionQ (NEU-DET).

Uso:
    bash training/download_data.sh          # descarga NEU-DET (~260 MB)
    python training/train_vision.py --data-dir data/neu_det/IMAGES

Incluye aumento de datos (ruido, desenfoque, rotación, brillo/contraste,
compresión JPEG y reescalado) para robustez ante variaciones de captura en
planta. La validación cruzada agrupa las copias aumentadas con su imagen
original (StratifiedGroupKFold) para evitar fuga de información.

Salida:
    models/vision_model.joblib   (pipeline + detector de fuera-de-distribución)
    models/vision_metrics.json   (métricas en conjunto de prueba)
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.features import FEATURE_VERSION, extract_features, load_image_bytes  # noqa: E402

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
SEED = 42
MODEL_VERSION = "2.0-aug"


def augment(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica 1-3 transformaciones aleatorias que simulan variación de captura."""
    im = Image.fromarray(gray)
    ops = rng.choice(["noise", "blur", "rot", "flip", "bright", "contrast", "jpeg", "scale"],
                     size=rng.integers(1, 4), replace=False)
    for op in ops:
        if op == "rot":
            im = im.rotate(int(rng.choice([90, 180, 270])))
        elif op == "flip":
            im = im.transpose(Image.FLIP_LEFT_RIGHT if rng.random() < 0.5 else Image.FLIP_TOP_BOTTOM)
        elif op == "blur":
            im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.5, 2.0))))
        elif op == "bright":
            im = ImageEnhance.Brightness(im).enhance(float(rng.uniform(0.7, 1.35)))
        elif op == "contrast":
            im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.5, 1.3)))
        elif op == "scale":
            f = float(rng.uniform(0.45, 0.85))
            w, h = im.size
            im = im.resize((int(w * f), int(h * f)))
            if rng.random() < 0.5:  # recorte parcial para simular otro encuadre
                w, h = im.size
                c = float(rng.uniform(0.7, 1.0))
                cw, ch = int(w * c), int(h * c)
                x0, y0 = int(rng.integers(0, w - cw + 1)), int(rng.integers(0, h - ch + 1))
                im = im.crop((x0, y0, x0 + cw, y0 + ch))
        elif op == "jpeg":
            b = io.BytesIO()
            im.save(b, format="JPEG", quality=int(rng.integers(15, 70)))
            im = Image.open(io.BytesIO(b.getvalue())).convert("L")
        elif op == "noise":
            a = np.asarray(im, dtype=np.float64)
            a = a + rng.normal(0, float(rng.uniform(3, 25)), a.shape)
            im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    return np.asarray(im, dtype=np.uint8)


def label_from_name(name: str) -> str:
    for c in CLASSES:
        if name.startswith(c):
            return c
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data/neu_det/IMAGES"))
    ap.add_argument("--out", default=str(ROOT / "models"))
    args = ap.parse_args()

    ap.add_argument("--n-aug", type=int, default=4, help="copias aumentadas por imagen (0 = sin aumento)")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    files = sorted(p for p in Path(args.data_dir).iterdir() if p.suffix.lower() in {".jpg", ".bmp", ".png"})
    labels = [label_from_name(p.name) for p in files]
    print(f"Imágenes encontradas: {len(files)}")
    f_tr, f_te = train_test_split(files, test_size=0.2, stratify=labels, random_state=SEED)

    t0 = time.time()
    X_tr, y_tr, g_tr = [], [], []
    for gid, p in enumerate(f_tr):
        gray = load_image_bytes(p.read_bytes())
        lab = label_from_name(p.name)
        for k in range(1 + args.n_aug):
            X_tr.append(extract_features(gray if k == 0 else augment(gray, rng)))
            y_tr.append(lab)
            g_tr.append(gid)
    X_tr, y_tr, g_tr = np.stack(X_tr), np.array(y_tr), np.array(g_tr)
    X_te = np.stack([extract_features(load_image_bytes(p.read_bytes())) for p in f_te])
    y_te = np.array([label_from_name(p.name) for p in f_te])
    print(f"Entrenamiento: {X_tr.shape} ({len(f_tr)} originales + aumentos); prueba: {X_te.shape}; "
          f"{time.time() - t0:.1f}s")

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", CalibratedClassifierCV(SVC(kernel="rbf", random_state=SEED),
                                                    method="sigmoid", ensemble=False))])
    grid = GridSearchCV(pipe, {"clf__estimator__C": [3, 10, 30], "clf__estimator__gamma": ["scale", 0.001]},
                        cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED),
                        scoring="f1_macro", n_jobs=-1)
    grid.fit(X_tr, y_tr, groups=g_tr)
    model = grid.best_estimator_
    print("Mejores hiperparámetros:", grid.best_params_, "CV f1_macro:", round(grid.best_score_, 4))

    y_pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=list(model.classes_))

    # Detector fuera de distribución sobre el espacio escalado
    scaler = model.named_steps["scaler"]
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED)
    iso.fit(scaler.transform(X_tr))
    tr_scores = iso.score_samples(scaler.transform(X_tr))  # incluye aumentos: capturas degradadas plausibles
    ood_threshold = float(np.percentile(tr_scores, 0.5))  # 99.5 % de entrenamiento queda "dentro"
    # Segundo criterio OOD: máximo |z| por imagen. El IsolationForest no detecta
    # bien entradas degeneradas (p. ej. ruido aleatorio) que disparan unas pocas
    # características a valores extremos; el máximo |z| sí las separa.
    zmax_tr = np.abs(scaler.transform(X_tr)).max(axis=1)
    zmax_threshold = float(np.percentile(zmax_tr, 99.5) * 1.5)
    print(f"max|z| entrenamiento: p50={np.percentile(zmax_tr,50):.2f} p99.5={np.percentile(zmax_tr,99.5):.2f} -> umbral {zmax_threshold:.2f}")

    metrics = {
        "dataset": "NEU-DET (Song & Yan, 2013) - 6 clases, 200x200",
        "model_version": MODEL_VERSION, "n_augmented_per_image": args.n_aug,
        "n_images": int(len(files)), "n_train_original": int(len(f_tr)),
        "n_train_with_aug": int(len(y_tr)), "n_test": int(len(y_te)),
        "feature_version": FEATURE_VERSION, "n_features": int(X_tr.shape[1]),
        "best_params": grid.best_params_, "cv_f1_macro": round(float(grid.best_score_), 4),
        "test_accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "test_f1_macro": round(float(f1_score(y_te, y_pred, average="macro")), 4),
        "mean_confidence_test": round(float(proba.max(axis=1).mean()), 4),
        "per_class": {c: {k: round(v, 4) for k, v in report[c].items()} for c in model.classes_},
        "confusion_matrix": {"labels": list(model.classes_), "matrix": cm.tolist()},
        "ood_threshold": ood_threshold,
        "zmax_threshold": zmax_threshold,
        "test_files": [p.name for p in f_te],
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "ood": iso, "ood_threshold": ood_threshold,
                 "zmax_threshold": zmax_threshold, "classes": list(model.classes_), "feature_version": FEATURE_VERSION,
                 "model_version": MODEL_VERSION,
                 "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                 "metrics": {k: metrics[k] for k in ("test_accuracy", "test_f1_macro", "cv_f1_macro")}},
                out / "vision_model.joblib", compress=3)
    (out / "vision_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(json.dumps({k: metrics[k] for k in ("test_accuracy", "test_f1_macro", "cv_f1_macro")}, indent=2))


if __name__ == "__main__":
    main()
