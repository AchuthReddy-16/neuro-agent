#!/usr/bin/env bash
# Nsight Compute profiling for hottest CUDA kernels identified by PyTorch profiler.
# Workflow: PyTorch profiler -> identify hot kernels -> ncu only those kernels.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/results/profiling/ncu"
mkdir -p "${OUTPUT_DIR}"

if ! command -v ncu &>/dev/null; then
    echo "ERROR: ncu (Nsight Compute) is NOT installed."
    exit 1
fi

echo "==> Nsight Compute version:"
ncu --version

TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)
REPORT="${OUTPUT_DIR}/qwen_kernels_${TIMESTAMP}"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    PYTHON="python3"
fi

# Targeted metrics — not the full metric set
METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_issue_active.pct"

# Kernel name filter: profile attention/GEMM-heavy ops (adjust after PyTorch profiler)
# Use --kernel-name with regex; base regex covers common transformer kernels
KERNEL_REGEX=".*(gemm|Gemm|GEMM|attention|Attention|flash|Flash|norm|Norm|rope|RoPE).*"

echo "==> Profiling kernels matching: ${KERNEL_REGEX}"
echo "==> Metrics: ${METRICS}"

ncu \
    --target-processes all \
    --kernel-name "${KERNEL_REGEX}" \
    --metrics "${METRICS}" \
    --csv \
    --log-file "${REPORT}.log" \
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
    do_sample=ic['do_sample'], max_new_tokens=min(ic['max_new_tokens'], 16),
    use_cache=ic['use_cache'])
model, tok, _ = load_model_and_tokenizer(config)
prompt, _ = build_prompt(tok, pc['base_text'], min(bc['prompt_token_length'], 256))
generate_with_timings(model, tok, prompt, config)
print('Done')
" 2>&1 | tee "${OUTPUT_DIR}/ncu_run_${TIMESTAMP}.stdout"

echo "==> Nsight Compute report: ${REPORT}.ncu-rep"
echo "==> CSV/log: ${REPORT}.log"
