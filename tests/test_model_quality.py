"""Validación del desempeño de los modelos (umbrales mínimos de aceptación)."""
import json
from pathlib import Path

from conftest import ROOT, label_of, sample_files

MIN_VISION_F1 = 0.95        # en partición de prueba interna
MIN_EXTERNAL_ACC = 0.85     # en 30 imágenes externas nunca vistas (Validation_Images)
MIN_FAILURE_ROC_AUC = 0.95
MIN_FAILURE_RECALL = 0.75


def test_vision_metrics_file():
    m = json.loads((ROOT / "models/vision_metrics.json").read_text())
    assert m["test_f1_macro"] >= MIN_VISION_F1
    for cls, s in m["per_class"].items():
        assert s["recall"] >= 0.85, cls


def test_vision_external_holdout(client):
    files = sample_files()
    assert len(files) == 30
    correct = 0
    for p in files:
        r = client.post("/api/v1/inspect", files={"file": (p.name, p.read_bytes(), "image/jpeg")}).json()
        correct += r["defect_class"] == label_of(p)
    assert correct / len(files) >= MIN_EXTERNAL_ACC


def test_failure_metrics_file():
    m = json.loads((ROOT / "models/failure_metrics.json").read_text())
    assert m["test_roc_auc"] >= MIN_FAILURE_ROC_AUC
    assert m["test_recall"] >= MIN_FAILURE_RECALL


def test_feature_pipeline_shared():
    """Entrenamiento e inferencia importan el mismo módulo de features."""
    for f in ["training/train_vision.py", "app/services/vision.py"]:
        assert "from app.features import" in Path(ROOT / f).read_text()
