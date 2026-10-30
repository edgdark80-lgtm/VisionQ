"""Casos extremos y de robustez de la API (entradas inválidas, límites, seguridad)."""
import numpy as np
import pytest
from conftest import NORMAL_READING, SAMPLES, to_bytes
from PIL import Image

INSPECT = "/api/v1/inspect"
FAIL = "/api/v1/failure/predict"


def post_img(client, data, name="x.png", ctype="image/png"):
    return client.post(INSPECT, files={"file": (name, data, ctype)})


# ---------------- Imágenes ----------------
def test_empty_file(client):
    r = post_img(client, b"")
    assert r.status_code == 400 and r.json()["error"] == "Archivo vacío"


def test_corrupt_image(client):
    r = post_img(client, b"\x89PNG\r\n\x1a\n esto no es una imagen", ctype="image/png")
    assert r.status_code == 400


def test_unsupported_mime(client):
    r = post_img(client, b"hola", name="a.txt", ctype="text/plain")
    assert r.status_code == 415


def test_file_too_large(client):
    big = np.random.default_rng(0).integers(0, 255, (2600, 2600), dtype=np.uint8)  # ~6.7 MB PNG
    r = post_img(client, to_bytes(big, "BMP"), name="big.bmp", ctype="image/bmp")
    assert r.status_code == 413


def test_tiny_image_flagged(client):
    r = post_img(client, to_bytes(np.full((16, 16), 120)))
    assert r.status_code == 200
    b = r.json()
    assert b["decision"] == "REVISION_MANUAL"
    assert "RESOLUCION_INSUFICIENTE" in b["quality"]["flags"]


@pytest.mark.parametrize("value,flag", [(0, "IMAGEN_SUBEXPUESTA"), (255, "IMAGEN_SOBREEXPUESTA")])
def test_black_and_white_images(client, value, flag):
    b = post_img(client, to_bytes(np.full((200, 200), value))).json()
    assert b["decision"] == "REVISION_MANUAL"
    assert flag in b["quality"]["flags"]


def test_random_noise_not_auto_dispositioned(client):
    noise = np.random.default_rng(1).integers(0, 256, (200, 200), dtype=np.uint8)
    b = post_img(client, to_bytes(noise)).json()
    assert b["decision"] == "REVISION_MANUAL"  # OOD o baja confianza


def test_non_steel_image_ood(client):
    # Imagen sintética de "tablero de ajedrez": ajena al dominio
    board = (np.indices((200, 200)).sum(axis=0) // 25 % 2 * 255).astype(np.uint8)
    b = post_img(client, to_bytes(board)).json()
    assert b["decision"] == "REVISION_MANUAL"
    assert b["out_of_distribution"] is True


def test_rgba_and_large_resolution_accepted(client):
    img = Image.open(SAMPLES / "scratches_195.jpg").convert("RGBA").resize((1600, 1600))
    b = post_img(client, to_bytes(img, "PNG")).json()
    assert b["defect_class"] == "scratches"


def test_format_invariance_jpg_png(client):
    src = SAMPLES / "inclusion_161.jpg"
    a = client.post(INSPECT, files={"file": ("a.jpg", src.read_bytes(), "image/jpeg")}).json()
    b = post_img(client, to_bytes(Image.open(src), "PNG")).json()
    assert a["defect_class"] == b["defect_class"]
    assert abs(a["confidence"] - b["confidence"]) < 0.02


def test_deterministic(client):
    data = (SAMPLES / "patches_128.jpg").read_bytes()
    r1 = client.post(INSPECT, files={"file": ("p.jpg", data, "image/jpeg")}).json()
    r2 = client.post(INSPECT, files={"file": ("p.jpg", data, "image/jpeg")}).json()
    assert r1["top_k"] == r2["top_k"]


def test_batch_over_limit(client):
    data = (SAMPLES / "patches_128.jpg").read_bytes()
    files = [("files", (f"{i}.jpg", data, "image/jpeg")) for i in range(17)]
    assert client.post("/api/v1/inspect/batch", files=files).status_code == 413


def test_batch_partial_errors(client):
    good = (SAMPLES / "patches_128.jpg").read_bytes()
    files = [("files", ("ok.jpg", good, "image/jpeg")), ("files", ("mal.txt", b"x", "text/plain"))]
    b = client.post("/api/v1/inspect/batch", files=files).json()
    assert b["count"] == 1 and len(b["errors"]) == 1


def test_missing_file_field(client):
    assert client.post(INSPECT).status_code == 422


# ---------------- Sensores ----------------
@pytest.mark.parametrize("field,value", [
    ("air_temperature_k", 200), ("air_temperature_k", 400), ("rotational_speed_rpm", -1),
    ("torque_nm", 500), ("tool_wear_min", -10), ("type", "X"), ("torque_nm", "abc"),
])
def test_failure_invalid_values(client, field, value):
    r = client.post(FAIL, json={**NORMAL_READING, field: value})
    assert r.status_code == 422
    assert any(field in d["campo"] for d in r.json()["detail"])


def test_failure_missing_field(client):
    body = {k: v for k, v in NORMAL_READING.items() if k != "torque_nm"}
    assert client.post(FAIL, json=body).status_code == 422


def test_failure_physical_inconsistency(client):
    r = client.post(FAIL, json={**NORMAL_READING, "air_temperature_k": 320, "process_temperature_k": 310})
    assert r.status_code == 422


def test_failure_extrapolation_flagged(client):
    # Dentro de límites físicos de la API pero fuera del rango de entrenamiento
    b = client.post(FAIL, json={**NORMAL_READING, "rotational_speed_rpm": 3500, "tool_wear_min": 300}).json()
    assert set(b["out_of_training_range"]) == {"rotational_speed_rpm", "tool_wear_min"}


def test_failure_boundary_values(client):
    for reading in [
        {"type": "L", "air_temperature_k": 250, "process_temperature_k": 250,
         "rotational_speed_rpm": 0, "torque_nm": 0, "tool_wear_min": 0},
        {"type": "H", "air_temperature_k": 350, "process_temperature_k": 400,
         "rotational_speed_rpm": 5000, "torque_nm": 150, "tool_wear_min": 400},
    ]:
        r = client.post(FAIL, json=reading)
        assert r.status_code == 200
        assert 0 <= r.json()["failure_probability"] <= 1


def test_failure_lowercase_type_rejected(client):
    assert client.post(FAIL, json={**NORMAL_READING, "type": "m"}).status_code == 422


def test_failure_empty_and_oversized_batch(client):
    assert client.post("/api/v1/failure/predict/batch", json={"readings": []}).status_code == 422
    r = client.post("/api/v1/failure/predict/batch", json={"readings": [NORMAL_READING] * 1001})
    assert r.status_code == 413


def test_failure_monotonic_tool_wear(client):
    """A mayor desgaste (resto constante) la probabilidad no debe disminuir de forma marcada."""
    probs = [client.post(FAIL, json={**NORMAL_READING, "torque_nm": 55, "tool_wear_min": w}).json()["failure_probability"]
             for w in (0, 100, 200, 240)]
    assert probs[-1] >= probs[0]


def test_malformed_json(client):
    r = client.post(FAIL, content=b"{type: M", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


# ---------------- Seguridad ----------------
def test_api_key_enforced(client, monkeypatch):
    from app import main
    monkeypatch.setattr(main.settings, "api_key", "secreto")
    assert client.post(FAIL, json=NORMAL_READING).status_code == 401
    assert client.post(FAIL, json=NORMAL_READING, headers={"X-API-Key": "malo"}).status_code == 401
    assert client.post(FAIL, json=NORMAL_READING, headers={"X-API-Key": "secreto"}).status_code == 200
    assert client.get("/api/v1/health").status_code == 200  # health sigue público


def test_method_not_allowed(client):
    assert client.get(INSPECT).status_code == 405
