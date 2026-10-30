"""VisionQ API (FastAPI).

Expone dos módulos de IA a través de una API REST versionada (/api/v1):
  * Inspección visual de defectos superficiales (NEU-DET).
  * Predicción de fallas de máquina (AI4I 2020).
Además sirve el frontend estático en "/" y la documentación OpenAPI en /docs.
"""
from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections import Counter
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas import (
    BatchFailureRequest,
    BatchFailureResponse,
    BatchInspectionResponse,
    ErrorResponse,
    FailurePrediction,
    HealthResponse,
    InspectionResult,
    MachineReading,
)
from app.services.failure import FailureService
from app.services.vision import InvalidImageError, VisionService

settings = get_settings()


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload, ensure_ascii=False)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("visionq")
logger.handlers = [handler]
logger.setLevel(settings.log_level)
logger.propagate = False

# ---------------------------------------------------------------- estado
STATE: dict = {"started": time.time(), "vision": None, "failure": None}
METRICS = {"requests": Counter(), "errors": Counter(), "latency_ms_sum": Counter(),
           "decisions": Counter(), "failure_predictions": Counter()}


@asynccontextmanager
async def lifespan(app: FastAPI):
    for key, cls, fname, kwargs in [
        ("vision", VisionService, "vision_model.joblib",
         {"confidence_threshold": settings.confidence_threshold}),
        ("failure", FailureService, "failure_model.joblib",
         {"threshold_override": settings.failure_threshold}),
    ]:
        try:
            STATE[key] = cls(settings.models_dir / fname, **kwargs)
            logger.info(f"Modelo cargado: {fname}")
        except Exception as exc:  # el servicio arranca en modo degradado
            logger.error(f"No se pudo cargar {fname}: {exc}")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=("API de VisionQ: inspección visual de defectos superficiales y predicción "
                 "de fallas de máquina. Proyecto académico (datos públicos NEU-DET y AI4I 2020)."),
    lifespan=lifespan,
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    path = request.url.path
    if path.startswith("/api/"):
        METRICS["requests"][path] += 1
        METRICS["latency_ms_sum"][path] += ms
        if response.status_code >= 400:
            METRICS["errors"][f"{path}:{response.status_code}"] += 1
        logger.info("request", extra={"extra_fields": {
            "request_id": rid, "method": request.method, "path": path,
            "status": response.status_code, "latency_ms": round(ms, 2)}})
    response.headers["X-Request-ID"] = rid
    response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
    return response


# ---------------------------------------------------------------- errores
def _err(request: Request, status: int, error: str, detail=None) -> JSONResponse:
    return JSONResponse(status_code=status, content=ErrorResponse(
        error=error, detail=detail, request_id=getattr(request.state, "request_id", None)).model_dump())


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    detail = [{"campo": ".".join(str(x) for x in e["loc"]), "mensaje": e["msg"]} for e in exc.errors()]
    return _err(request, 422, "Datos de entrada inválidos", detail)


@app.exception_handler(HTTPException)
async def http_handler(request: Request, exc: HTTPException):
    return _err(request, exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    logger.exception("Error no controlado")
    return _err(request, 500, "Error interno del servidor")


# ---------------------------------------------------------------- dependencias
async def require_api_key(x_api_key: str | None = Header(default=None)):
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="API key inválida o ausente (header X-API-Key)")


def vision_service() -> VisionService:
    if STATE["vision"] is None:
        raise HTTPException(status_code=503, detail="Modelo de inspección visual no disponible")
    return STATE["vision"]


def failure_service() -> FailureService:
    if STATE["failure"] is None:
        raise HTTPException(status_code=503, detail="Modelo de predicción de fallas no disponible")
    return STATE["failure"]


async def _read_upload(file: UploadFile) -> bytes:
    limit = int(settings.max_upload_mb * 1024 * 1024)
    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(status_code=415,
                            detail=f"Tipo no soportado: {file.content_type}. Permitidos: "
                                   + ", ".join(settings.allowed_image_types))
    data = await file.read(limit + 1)
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Archivo vacío")
    if len(data) > limit:
        raise HTTPException(status_code=413, detail=f"Archivo mayor a {settings.max_upload_mb} MB")
    return data


# ---------------------------------------------------------------- endpoints
@app.get("/api/v1/health", response_model=HealthResponse, tags=["Sistema"])
async def health():
    models = {"vision": STATE["vision"] is not None, "failure": STATE["failure"] is not None}
    return HealthResponse(status="ok" if all(models.values()) else "degraded",
                          version=settings.app_version, environment=settings.environment,
                          uptime_s=round(time.time() - STATE["started"], 1), models=models)


@app.get("/api/v1/models", tags=["Sistema"])
async def models_info():
    return {
        "vision": None if STATE["vision"] is None else {
            **STATE["vision"].meta, "classes": STATE["vision"].classes,
            "confidence_threshold": STATE["vision"].confidence_threshold},
        "failure": None if STATE["failure"] is None else STATE["failure"].meta,
    }


@app.get("/api/v1/metrics", tags=["Sistema"])
async def metrics():
    avg = {p: round(METRICS["latency_ms_sum"][p] / n, 2) for p, n in METRICS["requests"].items()}
    return {"requests": dict(METRICS["requests"]), "errors": dict(METRICS["errors"]),
            "avg_latency_ms": avg, "inspection_decisions": dict(METRICS["decisions"]),
            "failure_predictions": dict(METRICS["failure_predictions"])}


@app.post("/api/v1/inspect", response_model=InspectionResult, tags=["Inspección visual"],
          dependencies=[Depends(require_api_key)],
          responses={400: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
                     415: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
async def inspect(file: UploadFile = File(..., description="Imagen JPG/PNG/BMP/TIFF de la superficie"),
                  svc: VisionService = Depends(vision_service)):
    data = await _read_upload(file)
    try:
        result = await run_in_threadpool(svc.predict_bytes, data, file.filename)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    METRICS["decisions"][result["decision"]] += 1
    return result


@app.post("/api/v1/inspect/batch", response_model=BatchInspectionResponse, tags=["Inspección visual"],
          dependencies=[Depends(require_api_key)])
async def inspect_batch(request: Request, files: list[UploadFile] = File(...),
                        svc: VisionService = Depends(vision_service)):
    if len(files) > settings.max_batch_images:
        raise HTTPException(status_code=413, detail=f"Máximo {settings.max_batch_images} imágenes por lote")
    t0 = time.perf_counter()
    results, errors = [], []
    for f in files:
        try:
            data = await _read_upload(f)
            r = await run_in_threadpool(svc.predict_bytes, data, f.filename)
            METRICS["decisions"][r["decision"]] += 1
            results.append(r)
        except (HTTPException, InvalidImageError) as exc:
            errors.append(ErrorResponse(error=f.filename or "archivo",
                                        detail=getattr(exc, "detail", str(exc)),
                                        request_id=request.state.request_id))
    return {"count": len(results), "results": results, "errors": errors,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


@app.post("/api/v1/failure/predict", response_model=FailurePrediction, tags=["Predicción de fallas"],
          dependencies=[Depends(require_api_key)])
async def failure_predict(reading: MachineReading, svc: FailureService = Depends(failure_service)):
    result = (await run_in_threadpool(svc.predict, [reading.model_dump()]))[0]
    METRICS["failure_predictions"][result["prediction"]] += 1
    return result


@app.post("/api/v1/failure/predict/batch", response_model=BatchFailureResponse,
          tags=["Predicción de fallas"], dependencies=[Depends(require_api_key)])
async def failure_predict_batch(body: BatchFailureRequest, svc: FailureService = Depends(failure_service)):
    if len(body.readings) > settings.max_batch_records:
        raise HTTPException(status_code=413, detail=f"Máximo {settings.max_batch_records} registros por lote")
    t0 = time.perf_counter()
    results = await run_in_threadpool(svc.predict, [r.model_dump() for r in body.readings])
    for r in results:
        METRICS["failure_predictions"][r["prediction"]] += 1
    return {"count": len(results), "n_failures_predicted": sum(r["prediction"] == "FALLA_PROBABLE" for r in results),
            "results": results, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}


# ---------------------------------------------------------------- frontend
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/ui/")


if settings.frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=settings.frontend_dir, html=True), name="ui")
