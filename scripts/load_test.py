"""Prueba de carga simple contra la API en ejecución (contenedor local o nube).

Uso:
    python scripts/load_test.py --base http://localhost:8000 --n 200 --concurrency 8
Mide latencia (p50/p95/p99), throughput y tasa de error para los endpoints
de inferencia. Escribe docs/evidence/load_test_results.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
IMAGES = sorted((ROOT / "samples").glob("*.jpg"))


def reading() -> dict:
    air = random.uniform(295, 304)
    return {"type": random.choice("LMH"), "air_temperature_k": round(air, 1),
            "process_temperature_k": round(air + random.uniform(8, 12), 1),
            "rotational_speed_rpm": random.randint(1170, 2800), "torque_nm": round(random.uniform(4, 76), 1),
            "tool_wear_min": random.randint(0, 250)}


async def worker(client, queue, endpoint, lat, errors):
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        t0 = time.perf_counter()
        try:
            if endpoint == "inspect":
                p = random.choice(IMAGES)
                r = await client.post("/api/v1/inspect", files={"file": (p.name, p.read_bytes(), "image/jpeg")})
            else:
                r = await client.post("/api/v1/failure/predict", json=reading())
            if r.status_code != 200:
                errors.append(r.status_code)
        except httpx.HTTPError as exc:
            errors.append(type(exc).__name__)
        lat.append((time.perf_counter() - t0) * 1000)


async def run(base: str, endpoint: str, n: int, conc: int) -> dict:
    queue: asyncio.Queue = asyncio.Queue()
    for _ in range(n):
        queue.put_nowait(1)
    lat: list[float] = []
    errors: list = []
    async with httpx.AsyncClient(base_url=base, timeout=30) as client:
        await client.get("/api/v1/health")  # calentamiento
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client, queue, endpoint, lat, errors) for _ in range(conc)])
        wall = time.perf_counter() - t0
    q = statistics.quantiles(lat, n=100)
    return {"endpoint": endpoint, "requests": n, "concurrency": conc,
            "throughput_rps": round(n / wall, 1), "error_rate": round(len(errors) / n, 4),
            "latency_ms": {"p50": round(q[49], 1), "p95": round(q[94], 1), "p99": round(q[98], 1),
                           "mean": round(statistics.mean(lat), 1), "max": round(max(lat), 1)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 8])
    args = ap.parse_args()
    random.seed(0)
    results = []
    for endpoint in ("inspect", "failure"):
        for c in args.concurrency:
            r = asyncio.run(run(args.base, endpoint, args.n, c))
            results.append(r)
            print(json.dumps(r, ensure_ascii=False))
    out = ROOT / "docs/evidence/load_test_results.json"
    out.write_text(json.dumps({"base": args.base, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
