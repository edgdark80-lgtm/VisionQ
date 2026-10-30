"""Evaluación de robustez y comportamiento de los modelos de VisionQ.

Genera docs/evidence/evaluation_results.json y gráficas PNG. Uso:
    python scripts/evaluate_models.py
Requiere data/neu_det/IMAGES (ver training/download_data.sh) y data/ai4i2020.csv.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.failure_features import RAW_COLUMNS, build_features  # noqa: E402
from app.services.failure import FailureService  # noqa: E402
from app.services.vision import VisionService  # noqa: E402

OUT = ROOT / "docs/evidence"
OUT.mkdir(parents=True, exist_ok=True)
INK, STEEL, RED, AMBER, BLUE = "#1D2733", "#5D6D7C", "#B3261E", "#B57A00", "#2F5DA8"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

vision = VisionService(ROOT / "models/vision_model.joblib", confidence_threshold=0.60)
failure = FailureService(ROOT / "models/failure_model.joblib")
vm = json.loads((ROOT / "models/vision_metrics.json").read_text())
results: dict = {}


def label_of(name: str) -> str:
    for c in vision.classes:
        if name.startswith(c):
            return c
    raise ValueError(name)


def img_bytes(img: Image.Image, fmt="PNG", **kw) -> bytes:
    b = io.BytesIO()
    img.save(b, format=fmt, **kw)
    return b.getvalue()


def run_set(images: list[tuple[str, Image.Image]], fmt="PNG", **kw) -> dict:
    preds = [vision.predict_bytes(img_bytes(im, fmt, **kw), n) for n, im in images]
    y = [label_of(n) for n, _ in images]
    acc = np.mean([p["defect_class"] == t for p, t in zip(preds, y)])
    auto = [p for p in preds if p["decision"] != "REVISION_MANUAL"]
    auto_acc = np.mean([p["defect_class"] == label_of(p["filename"]) for p in auto]) if auto else float("nan")
    return {"n": len(preds), "accuracy": round(float(acc), 4),
            "manual_review_rate": round(1 - len(auto) / len(preds), 4),
            "accuracy_on_auto_decisions": round(float(auto_acc), 4),
            "ood_rate": round(float(np.mean([p["out_of_distribution"] for p in preds])), 4),
            "mean_confidence": round(float(np.mean([p["confidence"] for p in preds])), 4),
            "mean_latency_ms": round(float(np.mean([p["latency_ms"] for p in preds])), 2)}


# ------------------------------------------------------------------ visión
test_names = vm["test_files"]
img_dir = ROOT / "data/neu_det/IMAGES"
base = [(n, Image.open(img_dir / n).convert("L")) for n in test_names]
external = [(p.name, Image.open(p).convert("L")) for p in sorted((ROOT / "samples").glob("*.jpg"))]

results["vision_clean_internal_test"] = run_set(base)
results["vision_external_30"] = run_set(external)
ext_preds = [vision.predict_bytes(img_bytes(im), n) for n, im in external]
results["vision_external_errors"] = [
    {"file": p["filename"], "true": label_of(p["filename"]), "pred": p["defect_class"],
     "confidence": p["confidence"], "decision": p["decision"]}
    for p in ext_preds if p["defect_class"] != label_of(p["filename"])]

rng = np.random.default_rng(42)


def add_noise(im, sigma):
    a = np.asarray(im, dtype=np.float64) + rng.normal(0, sigma, (im.height, im.width))
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


perturbations = {
    "Ruido gaussiano σ=5": lambda im: add_noise(im, 5),
    "Ruido gaussiano σ=15": lambda im: add_noise(im, 15),
    "Ruido gaussiano σ=30": lambda im: add_noise(im, 30),
    "Desenfoque r=1": lambda im: im.filter(ImageFilter.GaussianBlur(1)),
    "Desenfoque r=2.5": lambda im: im.filter(ImageFilter.GaussianBlur(2.5)),
    "Brillo +30 %": lambda im: ImageEnhance.Brightness(im).enhance(1.3),
    "Brillo −30 %": lambda im: ImageEnhance.Brightness(im).enhance(0.7),
    "Contraste −50 %": lambda im: ImageEnhance.Contrast(im).enhance(0.5),
    "Rotación 90°": lambda im: im.rotate(90),
    "Rotación 180°": lambda im: im.rotate(180),
    "Espejo horizontal": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "Reescalado 50 %": lambda im: im.resize((100, 100)),
    "Recorte central 70 %": lambda im: im.crop((30, 30, 170, 170)),
}
pert = {"Original": results["vision_clean_internal_test"]}
for name, fn in perturbations.items():
    pert[name] = run_set([(n, fn(im)) for n, im in base])
pert["JPEG calidad 20"] = run_set(base, fmt="JPEG", quality=20)
results["vision_perturbations"] = pert

# Entradas ajenas al dominio (deben ir a revisión manual)
yy, xx = np.indices((200, 200))
ood_inputs = {
    "Ruido uniforme": rng.integers(0, 256, (200, 200)),
    "Ruido gaussiano puro": np.clip(rng.normal(128, 40, (200, 200)), 0, 255),
    "Tablero de ajedrez": (yy // 25 + xx // 25) % 2 * 255,
    "Gradiente lineal": np.tile(np.linspace(0, 255, 200), (200, 1)),
    "Imagen negra": np.zeros((200, 200)),
    "Imagen blanca": np.full((200, 200), 255),
    "Gris uniforme": np.full((200, 200), 128),
    "Círculos concéntricos": (np.sin(np.hypot(yy - 100, xx - 100) / 4) * 127 + 128),
    "Franjas horizontales": (yy // 10 % 2) * 200 + 25,
    "Texto/rejilla fina": ((yy % 8 == 0) | (xx % 8 == 0)) * 255,
}
ood_res = {}
for name, arr in ood_inputs.items():
    p = vision.predict_bytes(img_bytes(Image.fromarray(np.asarray(arr, dtype=np.uint8))), name)
    ood_res[name] = {"decision": p["decision"], "pred": p["defect_class"], "confidence": p["confidence"],
                     "ood": p["out_of_distribution"], "quality_flags": p["quality"]["flags"],
                     "reason": p["decision_reason"]}
results["vision_ood_inputs"] = ood_res
results["vision_ood_detection_rate"] = round(
    float(np.mean([v["decision"] == "REVISION_MANUAL" for v in ood_res.values()])), 4)

# Barrido de umbral de confianza: trade-off cobertura vs exactitud
probs = [vision.predict_bytes(img_bytes(im), n) for n, im in base]
conf = np.array([p["confidence"] for p in probs])
ok = np.array([p["defect_class"] == label_of(p["filename"]) for p in probs])
sweep = []
for t in np.round(np.arange(0.3, 0.96, 0.05), 2):
    m = conf >= t
    sweep.append({"threshold": float(t), "coverage": round(float(m.mean()), 4),
                  "accuracy_auto": round(float(ok[m].mean()) if m.any() else float("nan"), 4)})
results["vision_confidence_sweep"] = sweep

# ------------------------------------------------------------------ fallas
df = pd.read_csv(ROOT / "data/ai4i2020.csv", encoding="utf-8-sig").rename(columns=RAW_COLUMNS)
y = df["Machine failure"].values
idx_tmp, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=42)
te = df.iloc[idx_te].reset_index(drop=True)
p_te = failure.model.predict_proba(build_features(te))[:, 1]
yhat = (p_te >= failure.threshold).astype(int)
modes = {}
for m in ["TWF", "HDF", "PWF", "OSF", "RNF"]:
    mask = te[m].values == 1
    modes[m] = {"n_test": int(mask.sum()),
                "detected": int(yhat[mask].sum()),
                "recall": round(float(yhat[mask].mean()) if mask.any() else float("nan"), 4)}
results["failure_recall_by_mode"] = modes
cm = confusion_matrix(te["Machine failure"].values, yhat)
results["failure_confusion_matrix"] = cm.tolist()
thr_rows = []
for t in [0.2, 0.3, failure.threshold, 0.5, 0.7]:
    yh = (p_te >= t).astype(int)
    tp = int(((yh == 1) & (te["Machine failure"].values == 1)).sum())
    fp = int(((yh == 1) & (te["Machine failure"].values == 0)).sum())
    fn = int(((yh == 0) & (te["Machine failure"].values == 1)).sum())
    thr_rows.append({"threshold": float(t), "tp": tp, "fp": fp, "fn": fn,
                     "precision": round(tp / (tp + fp), 4) if tp + fp else None,
                     "recall": round(tp / (tp + fn), 4)})
results["failure_threshold_table"] = thr_rows

# Sensibilidad: barrido de desgaste y par sobre una lectura base
base_r = {"type": "M", "air_temperature_k": 300.0, "process_temperature_k": 310.0,
          "rotational_speed_rpm": 1500, "torque_nm": 40.0, "tool_wear_min": 100}
wear = np.arange(0, 251, 10)
torque = np.arange(10, 76, 5)
grid = np.array([[failure.predict([{**base_r, "tool_wear_min": float(w), "torque_nm": float(t)}])[0]["failure_probability"]
                  for w in wear] for t in torque])

(OUT / "evaluation_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))

# ------------------------------------------------------------------ gráficas
# 1. Matriz de confusión visión (prueba interna)
cmv = np.array(vm["confusion_matrix"]["matrix"])
labs = vm["confusion_matrix"]["labels"]
fig, ax = plt.subplots(figsize=(6.2, 5.2))
ax.imshow(cmv, cmap="Blues")
ax.set_xticks(range(6), labs, rotation=35, ha="right"); ax.set_yticks(range(6), labs)
for i in range(6):
    for j in range(6):
        ax.text(j, i, cmv[i, j], ha="center", va="center", color="white" if cmv[i, j] > cmv.max() / 2 else INK)
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title("Inspección visual: matriz de confusión (n=354)")
fig.tight_layout(); fig.savefig(OUT / "chart_vision_confusion.png", dpi=160); plt.close(fig)

# 2. Robustez ante perturbaciones
names = list(pert.keys())
accs = [pert[n]["accuracy"] for n in names]
rev = [pert[n]["manual_review_rate"] for n in names]
fig, ax = plt.subplots(figsize=(8, 5.4))
yv = np.arange(len(names))[::-1]
ax.barh(yv + 0.2, accs, height=0.38, color=INK, label="Exactitud")
ax.barh(yv - 0.2, rev, height=0.38, color=BLUE, label="Tasa de revisión manual")
ax.set_yticks(yv, names); ax.set_xlim(0, 1.05)
for v, a in zip(yv, accs):
    ax.text(a + 0.01, v + 0.2, f"{a:.2f}", va="center", fontsize=8)
ax.legend(loc="lower right", frameon=False)
ax.set_title("Robustez del modelo visual ante perturbaciones (prueba interna, n=354)")
fig.tight_layout(); fig.savefig(OUT / "chart_vision_robustness.png", dpi=160); plt.close(fig)

# 3. Barrido de umbral de confianza
fig, ax = plt.subplots(figsize=(6.4, 4))
ts = [r["threshold"] for r in sweep]
ax.plot(ts, [r["coverage"] for r in sweep], color=BLUE, marker="o", label="Cobertura automática")
ax.plot(ts, [r["accuracy_auto"] for r in sweep], color=INK, marker="s", label="Exactitud en decisiones automáticas")
ax.axvline(0.60, color=RED, ls="--", lw=1); ax.text(0.61, 0.9, "umbral 0.60", color=RED, fontsize=9)
ax.set_xlabel("Umbral de confianza"); ax.set_ylim(0.85, 1.005); ax.legend(frameon=False, loc="lower left")
ax.set_title("Compromiso cobertura / exactitud")
fig.tight_layout(); fig.savefig(OUT / "chart_vision_threshold.png", dpi=160); plt.close(fig)

# 4. Curva precisión-exhaustividad fallas
prec, rec, thr = precision_recall_curve(te["Machine failure"].values, p_te)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(rec, prec, color=INK)
i = np.argmin(np.abs(thr - failure.threshold))
ax.scatter([rec[i]], [prec[i]], color=RED, zorder=3)
ax.annotate(f"umbral {failure.threshold:.2f}", (rec[i], prec[i]), xytext=(-90, -25), textcoords="offset points", color=RED)
ax.set_xlabel("Exhaustividad (recall)"); ax.set_ylabel("Precisión")
ax.set_title("Predicción de fallas: curva precisión-exhaustividad (prueba, n=2000)")
fig.tight_layout(); fig.savefig(OUT / "chart_failure_pr.png", dpi=160); plt.close(fig)

# 5. Mapa de sensibilidad
fig, ax = plt.subplots(figsize=(7, 4.4))
im = ax.imshow(grid, origin="lower", aspect="auto", cmap="Reds", vmin=0, vmax=1,
               extent=[wear[0], wear[-1], torque[0], torque[-1]])
ax.contour(wear, torque, grid, levels=[failure.threshold], colors=INK, linewidths=1.2)
ax.set_xlabel("Desgaste de herramienta [min]"); ax.set_ylabel("Par [Nm]")
ax.set_title("Sensibilidad: probabilidad de falla (tipo M, 1500 rpm)")
fig.colorbar(im, ax=ax, label="Probabilidad")
fig.tight_layout(); fig.savefig(OUT / "chart_failure_sensitivity.png", dpi=160); plt.close(fig)

print(json.dumps({k: results[k] for k in ["vision_clean_internal_test", "vision_external_30",
                                          "vision_ood_detection_rate", "failure_recall_by_mode"]},
                 indent=1, ensure_ascii=False))
