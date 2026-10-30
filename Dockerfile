# =====================================================================
# VisionQ API + frontend — imagen de producción
# Multi-etapa: (1) builder instala dependencias en un venv aislado,
# (2) runtime copia solo el venv y el código; corre como usuario no root.
# Compatible con linux/amd64 y linux/arm64 (Apple Silicon M1/M2).
# =====================================================================
ARG BASE_IMAGE=python:3.11-slim

# ---------- Etapa 1: dependencias ----------
FROM ${BASE_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- Etapa 2: runtime ----------
FROM ${BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="VisionQ API" \
      org.opencontainers.image.description="Inspección visual de defectos y predicción de fallas (FastAPI + scikit-learn)" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.source="https://github.com/edgdark80-lgtm/VisionQ"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    WEB_CONCURRENCY=1 \
    VISIONQ_ENVIRONMENT=container

RUN useradd --create-home --uid 1000 visionq
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=visionq:visionq app/ ./app/
COPY --chown=visionq:visionq models/ ./models/
COPY --chown=visionq:visionq frontend/ ./frontend/

USER visionq
EXPOSE 8000

# Sonda de salud sin dependencias extra (no se requiere curl en la imagen slim)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request,sys; r=urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/api/v1/health',timeout=4); sys.exit(0 if r.status==200 else 1)"

# PORT es inyectado por Cloud Run / App Service / Railway; 8000 por defecto
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers ${WEB_CONCURRENCY} --proxy-headers"]
