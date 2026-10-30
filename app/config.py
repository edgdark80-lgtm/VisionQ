"""Configuración por variables de entorno (12-factor). Prefijo: VISIONQ_."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VISIONQ_", env_file=".env", extra="ignore")

    app_name: str = "VisionQ API"
    app_version: str = "2.0.0"
    environment: str = "local"          # local | staging | production
    log_level: str = "INFO"

    models_dir: Path = BASE_DIR / "models"
    frontend_dir: Path = BASE_DIR / "frontend"

    # Inspección visual
    confidence_threshold: float = 0.60   # debajo -> REVISION_MANUAL
    max_upload_mb: float = 5.0
    max_batch_images: int = 16
    allowed_image_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/bmp", "image/tiff")

    # Predicción de fallas
    failure_threshold: float | None = None  # None -> usa el umbral guardado en el modelo
    max_batch_records: int = 1000

    # Seguridad / red
    api_key: str | None = None           # si se define, se exige X-API-Key en endpoints POST
    cors_origins: str = "*"              # lista separada por comas


@lru_cache
def get_settings() -> Settings:
    return Settings()
