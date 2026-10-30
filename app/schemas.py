"""Contratos de datos de la API (Pydantic v2). Generan el esquema OpenAPI."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------- Comunes ----------
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    environment: str
    uptime_s: float
    models: dict[str, bool]


class ErrorResponse(BaseModel):
    error: str
    detail: str | list | dict | None = None
    request_id: str | None = None


# ---------- Inspección visual ----------
class ClassScore(BaseModel):
    label: str
    probability: float


class ImageQuality(BaseModel):
    width: int
    height: int
    mean_intensity: float
    std_intensity: float
    flags: list[str]


class InspectionResult(BaseModel):
    filename: str | None
    defect_class: str = Field(description="Clase de defecto predicha (NEU-DET)")
    defect_name_es: str
    confidence: float
    top_k: list[ClassScore]
    decision: Literal["RECHAZO", "RETRABAJO", "REVISION_MANUAL"]
    decision_reason: str
    out_of_distribution: bool
    ood_score: float
    quality: ImageQuality
    latency_ms: float


class BatchInspectionResponse(BaseModel):
    count: int
    results: list[InspectionResult]
    errors: list[ErrorResponse]
    latency_ms: float


# ---------- Predicción de fallas ----------
class MachineReading(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "type": "M", "air_temperature_k": 298.1, "process_temperature_k": 308.6,
        "rotational_speed_rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 0}})

    type: Literal["L", "M", "H"] = Field(description="Variante de calidad del producto")
    air_temperature_k: float = Field(ge=250, le=350, description="Temperatura del aire [K]")
    process_temperature_k: float = Field(ge=250, le=400, description="Temperatura de proceso [K]")
    rotational_speed_rpm: float = Field(ge=0, le=5000, description="Velocidad de rotación [rpm]")
    torque_nm: float = Field(ge=0, le=150, description="Par [Nm]")
    tool_wear_min: float = Field(ge=0, le=400, description="Desgaste de herramienta [min]")

    @model_validator(mode="after")
    def check_physics(self):
        if self.process_temperature_k < self.air_temperature_k:
            raise ValueError("process_temperature_k debe ser >= air_temperature_k")
        return self


class RiskIndicator(BaseModel):
    mode: str
    name: str
    detail: str


class FailurePrediction(BaseModel):
    failure_probability: float
    threshold: float
    prediction: Literal["FALLA_PROBABLE", "OPERACION_NORMAL"]
    risk_level: Literal["BAJO", "MEDIO", "ALTO"]
    recommended_action: str
    risk_indicators: list[RiskIndicator]
    out_of_training_range: list[str] = Field(
        description="Variables fuera del rango observado en entrenamiento (extrapolación)")
    latency_ms: float


class BatchFailureRequest(BaseModel):
    readings: list[MachineReading] = Field(min_length=1)


class BatchFailureResponse(BaseModel):
    count: int
    n_failures_predicted: int
    results: list[FailurePrediction]
    latency_ms: float
