#!/usr/bin/env bash
# Prueba de humo E2E contra un contenedor VisionQ en ejecución.
# Uso: bash scripts/smoke_test.sh [BASE_URL]   (por defecto http://localhost:8000)
# Guarda las respuestas en docs/evidence/ y termina con código != 0 si algo falla.
set -u
BASE="${1:-http://localhost:8000}"
OUT="$(dirname "$0")/../docs/evidence"
SAMPLES="$(dirname "$0")/../samples"
mkdir -p "$OUT"
PASS=0; FAIL=0
H=()
[ -n "${VISIONQ_API_KEY:-}" ] && H=(-H "X-API-Key: $VISIONQ_API_KEY")

check() {  # nombre  código_esperado  archivo_salida  args-curl...
  local name="$1" expected="$2" outfile="$3"; shift 3
  local code
  code=$(curl -s -o "$OUT/$outfile" -w "%{http_code}" "${H[@]}" "$@")
  if [ "$code" = "$expected" ]; then
    echo "  [OK]    $name -> HTTP $code"; PASS=$((PASS+1))
  else
    echo "  [FALLA] $name -> HTTP $code (esperado $expected)"; FAIL=$((FAIL+1))
  fi
}

echo "== VisionQ smoke test contra $BASE ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
check "GET  /api/v1/health"            200 smoke_health.json   "$BASE/api/v1/health"
check "GET  /api/v1/models"            200 smoke_models.json   "$BASE/api/v1/models"
check "GET  /ui/ (frontend)"           200 smoke_ui.html       "$BASE/ui/"
check "GET  /openapi.json"             200 smoke_openapi.json  "$BASE/openapi.json"
check "POST /api/v1/inspect (scratches)" 200 smoke_inspect.json \
      -F "file=@$SAMPLES/scratches_195.jpg;type=image/jpeg" "$BASE/api/v1/inspect"
check "POST /api/v1/inspect/batch (3 img)" 200 smoke_inspect_batch.json \
      -F "files=@$SAMPLES/inclusion_161.jpg;type=image/jpeg" \
      -F "files=@$SAMPLES/patches_128.jpg;type=image/jpeg" \
      -F "files=@$SAMPLES/rolled-in_scale_138.jpg;type=image/jpeg" "$BASE/api/v1/inspect/batch"
check "POST /api/v1/failure/predict (normal)" 200 smoke_failure_normal.json \
      -H "Content-Type: application/json" \
      -d '{"type":"M","air_temperature_k":298.1,"process_temperature_k":308.6,"rotational_speed_rpm":1551,"torque_nm":42.8,"tool_wear_min":0}' \
      "$BASE/api/v1/failure/predict"
check "POST /api/v1/failure/predict (riesgo)" 200 smoke_failure_risk.json \
      -H "Content-Type: application/json" \
      -d '{"type":"L","air_temperature_k":302.0,"process_temperature_k":310.5,"rotational_speed_rpm":1300,"torque_nm":60,"tool_wear_min":210}' \
      "$BASE/api/v1/failure/predict"
check "POST /api/v1/failure/predict (inválido -> 422)" 422 smoke_failure_invalid.json \
      -H "Content-Type: application/json" -d '{"type":"Z","air_temperature_k":-5}' \
      "$BASE/api/v1/failure/predict"
check "POST /api/v1/inspect (texto -> 415)" 415 smoke_inspect_415.json \
      -F "file=@$0;type=text/plain" "$BASE/api/v1/inspect"
check "GET  /api/v1/metrics"           200 smoke_metrics.json  "$BASE/api/v1/metrics"

echo "== Resultado: $PASS OK, $FAIL fallas"
[ "$FAIL" -eq 0 ]
