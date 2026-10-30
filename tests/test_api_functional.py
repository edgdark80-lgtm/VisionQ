"""Pruebas funcionales de la API (contratos, flujo feliz e integración)."""
from conftest import NORMAL_READING, RISK_READING, SAMPLES, sample_files


def test_health_ok(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"] == {"vision": True, "failure": True}


def test_models_metadata(client):
    body = client.get("/api/v1/models").json()
    assert len(body["vision"]["classes"]) == 6
    assert 0 < body["failure"]["threshold"] < 1


def test_openapi_lists_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ["/api/v1/health", "/api/v1/models", "/api/v1/metrics", "/api/v1/inspect",
              "/api/v1/inspect/batch", "/api/v1/failure/predict", "/api/v1/failure/predict/batch"]:
        assert p in paths


def test_frontend_served(client):
    r = client.get("/", follow_redirects=True)
    assert r.status_code == 200 and "VisionQ" in r.text
    assert client.get("/ui/app.js").status_code == 200


def test_request_id_header_propagated(client):
    r = client.get("/api/v1/health", headers={"X-Request-ID": "prueba-123"})
    assert r.headers["X-Request-ID"] == "prueba-123"
    assert "X-Process-Time-ms" in r.headers


def test_inspect_contract(client):
    f = SAMPLES / "scratches_195.jpg"
    r = client.post("/api/v1/inspect", files={"file": (f.name, f.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    b = r.json()
    assert b["defect_class"] == "scratches"
    assert b["decision"] in {"RECHAZO", "RETRABAJO", "REVISION_MANUAL"}
    assert abs(sum(k["probability"] for k in b["top_k"]) - 1) < 0.05 or len(b["top_k"]) == 3
    assert b["top_k"][0]["label"] == b["defect_class"]
    assert 0 <= b["confidence"] <= 1


def test_inspect_batch(client):
    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in sample_files()[:5]]
    r = client.post("/api/v1/inspect/batch", files=files)
    assert r.status_code == 200
    assert r.json()["count"] == 5 and r.json()["errors"] == []


def test_failure_normal_vs_risk(client):
    n = client.post("/api/v1/failure/predict", json=NORMAL_READING).json()
    k = client.post("/api/v1/failure/predict", json=RISK_READING).json()
    assert n["prediction"] == "OPERACION_NORMAL" and n["risk_level"] == "BAJO"
    assert k["prediction"] == "FALLA_PROBABLE" and k["risk_level"] == "ALTO"
    assert {i["mode"] for i in k["risk_indicators"]} >= {"HDF", "OSF", "TWF"}


def test_failure_batch(client):
    r = client.post("/api/v1/failure/predict/batch",
                    json={"readings": [NORMAL_READING, RISK_READING] * 50})
    assert r.status_code == 200
    b = r.json()
    assert b["count"] == 100 and b["n_failures_predicted"] == 50


def test_metrics_counts_requests(client):
    client.get("/api/v1/health")
    m = client.get("/api/v1/metrics").json()
    assert m["requests"]["/api/v1/health"] >= 1
    assert "avg_latency_ms" in m
