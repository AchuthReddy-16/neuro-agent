#!/usr/bin/env bash
# Setup environment for neuro-agent on RunPod (persistent /workspace storage)
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE_ROOT="${NEURO_WORKSPACE_ROOT:-/workspace}"

echo "==> neuro-agent environment setup"
echo "    Project root:  ${PROJECT_ROOT}"
echo "    Workspace root: ${WORKSPACE_ROOT}"

# --- Hugging Face caches on persistent volume ---
export HF_HOME="${WORKSPACE_ROOT}/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${WORKSPACE_ROOT}/.cache/torch"

mkdir -p "${HF_HOME}/hub" "${HF_HOME}/datasets" "${HF_HOME}/transformers" "${TORCH_HOME}"
mkdir -p "${PROJECT_ROOT}/checkpoints" "${PROJECT_ROOT}/results"

# --- CUDA ---
if command -v nvidia-smi &>/dev/null; then
    echo "==> GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "==> WARNING: nvidia-smi not found"
fi

# --- Python venv (optional) ---
VENV_DIR="${PROJECT_ROOT}/.venv"
if [[ ! -d "${VENV_DIR}" ]]; then
    echo "==> Creating virtual environment at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "==> Installing package (editable, minimal deps)"
pip install --upgrade pip
pip install -e "${PROJECT_ROOT}"
pip install -e "${PROJECT_ROOT}[hardware]"

echo "==> Environment ready. Activate with:"
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "==> Verify hardware:"
echo "    python ${PROJECT_ROOT}/scripts/verify_hardware.py"
