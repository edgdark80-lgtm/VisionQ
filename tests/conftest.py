import io
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT / "samples"

NORMAL_READING = {"type": "M", "air_temperature_k": 298.1, "process_temperature_k": 308.6,
                  "rotational_speed_rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 0}
RISK_READING = {"type": "L", "air_temperature_k": 302.0, "process_temperature_k": 310.5,
                "rotational_speed_rpm": 1300, "torque_nm": 60, "tool_wear_min": 210}


@pytest.fixture(scope="session")
def client():
    from app.main import app
    with TestClient(app) as c:
        yield c


def sample_files():
    return sorted(SAMPLES.glob("*.jpg"))


def to_bytes(arr_or_img, fmt="PNG") -> bytes:
    img = arr_or_img if isinstance(arr_or_img, Image.Image) else Image.fromarray(np.asarray(arr_or_img, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def label_of(path: Path) -> str:
    for c in ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]:
        if path.name.startswith(c):
            return c
    raise ValueError(path)
