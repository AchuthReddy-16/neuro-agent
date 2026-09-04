#!/usr/bin/env bash
# Production FastAPI launcher for the neuro-agent research API (text + vision).
# Usage:
#   ./scripts/start_api_production.sh
#   NEURO_API_HOST=0.0.0.0 NEURO_API_PORT=8080 ./scripts/start_api_production.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export NEURO_WORKSPACE_ROOT="${NEURO_WORKSPACE_ROOT:-${PROJECT_ROOT}}"

# Production defaults (override via env)
export NEURO_API_ENABLE_VLM="${NEURO_API_ENABLE_VLM:-1}"
export NEURO_API_LOAD_AGENT="${NEURO_API_LOAD_AGENT:-1}"
export NEURO_API_VLM_TIMEOUT_S="${NEURO_API_VLM_TIMEOUT_S:-180}"
export NEURO_SERVING_MODE="${NEURO_SERVING_MODE:-hybrid}"
export NEURO_API_STORE_ROOT="${NEURO_API_STORE_ROOT:-${PROJECT_ROOT}/results/api_experiments}"

HOST="${NEURO_API_HOST:-0.0.0.0}"
PORT="${NEURO_API_PORT:-8080}"

# Prefer project venv
if [[ -x "${PROJECT_ROOT}/.venv/bin/uvicorn" ]]; then
  UVICORN="${PROJECT_ROOT}/.venv/bin/uvicorn"
elif command -v uvicorn >/dev/null 2>&1; then
  UVICORN="$(command -v uvicorn)"
else
  echo "ERROR: uvicorn not found. Create .venv and install deps, or put uvicorn on PATH." >&2
  exit 1
fi

TEXT_ADAPTER="${PROJECT_ROOT}/checkpoints/sft_corrected_v2/final"
VISION_ADAPTER="${PROJECT_ROOT}/checkpoints/multimodal_sft_corrected/final"

fail_missing() {
  echo "ERROR: required path missing: $1" >&2
  exit 1
}

[[ -d "${PROJECT_ROOT}/src/neuro_agent/api" ]] || fail_missing "${PROJECT_ROOT}/src/neuro_agent/api"
[[ -f "${TEXT_ADAPTER}/adapter_model.safetensors" ]] || fail_missing "${TEXT_ADAPTER}/adapter_model.safetensors"
[[ -f "${TEXT_ADAPTER}/adapter_config.json" ]] || fail_missing "${TEXT_ADAPTER}/adapter_config.json"
[[ -f "${VISION_ADAPTER}/adapter_model.safetensors" ]] || fail_missing "${VISION_ADAPTER}/adapter_model.safetensors"
[[ -f "${VISION_ADAPTER}/adapter_config.json" ]] || fail_missing "${VISION_ADAPTER}/adapter_config.json"

# HF / torch caches under workspace (persistent volume friendly)
export HF_HOME="${HF_HOME:-${NEURO_WORKSPACE_ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${NEURO_WORKSPACE_ROOT}/.cache/torch}"
mkdir -p "${HF_HOME}" "${HUGGINGFACE_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" "${NEURO_API_STORE_ROOT}"

if [[ -z "${NEURO_ALLOWED_ORIGINS:-${NEURO_API_CORS_ORIGINS:-}}" ]]; then
  echo "WARN: NEURO_ALLOWED_ORIGINS unset — using localhost CORS defaults only." >&2
  echo "      For Vercel, set NEURO_ALLOWED_ORIGINS=https://your-app.vercel.app" >&2
fi

echo "Starting neuro-agent API"
echo "  project:     ${PROJECT_ROOT}"
echo "  workspace:   ${NEURO_WORKSPACE_ROOT}"
echo "  bind:        ${HOST}:${PORT}"
echo "  text LoRA:   ${TEXT_ADAPTER}"
echo "  vision LoRA: ${VISION_ADAPTER}"
echo "  VLM:         NEURO_API_ENABLE_VLM=${NEURO_API_ENABLE_VLM}"
echo "  load agent:  NEURO_API_LOAD_AGENT=${NEURO_API_LOAD_AGENT}"

exec "${UVICORN}" neuro_agent.api.app:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --proxy-headers \
  --timeout-keep-alive 75
