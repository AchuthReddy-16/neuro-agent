#!/usr/bin/env bash
# Nsight Systems profiling for Qwen inference (prefill + decode).
# Requires: nsys (Nsight Systems CLI) in PATH.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/results/profiling/nsys"
mkdir -p "${OUTPUT_DIR}"

if ! command -v nsys &>/dev/null; then
    echo "ERROR: nsys (Nsight Systems) is NOT installed."
    echo "Install via: https://developer.nvidia.com/nsight-systems"
    echo "On RunPod, nsys may require: apt install nsight-systems-cli (if available)"
    exit 1
fi

echo "==> Nsight Systems version:"
nsys --version

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
REPORT="${OUTPUT_DIR}/qwen_inference_${TIMESTAMP}"

# Short representative run: load model + one generation
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

echo "==> Profiling one inference run (cuda,nvtx,osrt)..."
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --cpuctxsw=none \
    --force-overwrite=true \
    -o "${REPORT}" \
    "${PYTHON}" -c "
import sys
sys.path.insert(0, '${PROJECT_ROOT}/src')
from neuro_agent.config import load_benchmark_config
from neuro_agent.evaluation.benchmark import build_prompt
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.engine import generate_with_timings
from neuro_agent.inference.model_loader import load_model_and_tokenizer
from neuro_agent.paths import configure_hf_cache
configure_hf_cache()
cfg = load_benchmark_config()
mc, ic, bc, pc = cfg['model'], cfg['inference'], cfg['benchmark'], cfg['prompt']
config = InferenceConfig(model_name=mc['name'], dtype=mc['dtype'], seed=ic['seed'],
    do_sample=ic['do_sample'], max_new_tokens=min(ic['max_new_tokens'], 32),
    use_cache=ic['use_cache'])
model, tok, _ = load_model_and_tokenizer(config)
prompt, _ = build_prompt(tok, pc['base_text'], bc['prompt_token_length'])
generate_with_timings(model, tok, prompt, config)
print('Done')
"

echo "==> Report saved: ${REPORT}.nsys-rep"
echo "Open with: nsys-ui ${REPORT}.nsys-rep"
