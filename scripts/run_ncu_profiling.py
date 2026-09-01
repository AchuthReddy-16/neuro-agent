#!/usr/bin/env python3
"""Run targeted Nsight Compute profiling and parse summaries."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "profiling" / "ncu"


# Kernel filters use NCU regex: prefix with "regex:" per NVIDIA docs
KERNEL_FILTERS = [
    ("gemm_matmul", "regex:ampere_bf16.*gemm.*"),
    ("attention_flash", "regex:flash_fwd.*"),
]

METRICS = [
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__occupancy_limit_blocks",
    "launch__occupancy_limit_warps",
    "smsp__sass_thread_inst_executed_op_hmma_pred_on.sum.per_cycle_elapsed",
    "smsp__warp_issue_stalled_long_scoreboard_per_issue_active.pct",
    "smsp__warp_issue_stalled_memory_dependency_per_issue_active.pct",
    "smsp__warp_issue_stalled_short_scoreboard_per_issue_active.pct",
]


def _run_ncu(label: str, kernel_regex: str, max_tokens: int = 8, prompt_len: int = 256) -> dict:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_base = OUTPUT_DIR / f"ncu_{label}_{ts}"
    metrics_str = ",".join(METRICS)

    python_code = f"""
import sys
sys.path.insert(0, '{PROJECT_ROOT / "src"}')
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
    do_sample=False, max_new_tokens={max_tokens}, use_cache=True)
model, tok, _ = load_model_and_tokenizer(config)
prompt, _ = build_prompt(tok, pc['base_text'], {prompt_len})
generate_with_timings(model, tok, prompt, config)
"""

    cmd = [
        "ncu",
        "--target-processes", "all",
        "--kernel-name-base", "demangled",
        "--kernel-name", kernel_regex,
        "--launch-count", "3",
        "--metrics", metrics_str,
        "--csv",
        "--log-file", str(report_base) + ".log",
        "-o", str(report_base),
        sys.executable, "-c", python_code,
    ]

    print(f"\n>>> NCU [{label}] regex={kernel_regex}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    stdout_path = OUTPUT_DIR / f"ncu_{label}_{ts}.stdout"
    stdout_path.write_text(result.stdout + "\n" + result.stderr)

    parsed = _parse_ncu_log(Path(str(report_base) + ".log"))
    log_text = Path(str(report_base) + ".log").read_text() if Path(str(report_base) + ".log").exists() else ""
    permission_error = "ERR_NVGPUCTRPERM" in log_text
    available_kernels = _extract_available_kernels(log_text)

    summary = {
        "label": label,
        "kernel_regex": kernel_regex,
        "returncode": result.returncode,
        "permission_error": permission_error,
        "report": str(report_base) + ".ncu-rep",
        "log": str(report_base) + ".log",
        "stdout": str(stdout_path),
        "kernels": parsed,
        "available_kernels": available_kernels,
    }
    return summary


def _extract_available_kernels(log_text: str) -> list[str]:
    kernels = []
    in_list = False
    for line in log_text.splitlines():
        if line.startswith("Available Kernels:"):
            in_list = True
            continue
        if in_list and line.strip() and line[0].isdigit():
            parts = line.split(".", 1)
            if len(parts) == 2:
                kernels.append(parts[1].strip())
    return kernels


def _parse_ncu_log(log_path: Path) -> list[dict]:
    """Parse NCU CSV log for kernel metric rows."""
    if not log_path.exists():
        return [{"error": f"log not found: {log_path}"}]

    kernels: list[dict] = []
    try:
        with log_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Kernel Name") or row.get("Kernel") or ""
                if not name:
                    continue
                entry = {"kernel_name": name}
                for k, v in row.items():
                    if k and k != "Kernel Name" and v:
                        try:
                            entry[k] = float(v)
                        except ValueError:
                            entry[k] = v
                kernels.append(entry)
    except Exception as exc:
        kernels.append({"error": str(exc)})

    # Sort by duration if available
    def duration_key(k: dict) -> float:
        for key in k:
            if "time_duration" in key.lower() or key == "gpu__time_duration.sum":
                try:
                    return float(k[key])
                except (TypeError, ValueError):
                    pass
        return 0.0

    kernels.sort(key=duration_key, reverse=True)
    return kernels[:10]


def _classify_from_pytorch(pytorch_summary: dict) -> dict:
    """Lightweight bottleneck classification from PyTorch profiler ops."""
    prefill_cuda = pytorch_summary.get("prefill", {}).get("top_cuda_ops", [])
    decode_cuda = pytorch_summary.get("decode", {}).get("top_cuda_ops", [])

    def cuda_us(ops: list, fragment: str) -> float:
        return sum(o.get("cuda_time_us", 0) for o in ops if fragment in o.get("name", ""))

    prefill_matmul = cuda_us(prefill_cuda, "matmul") + cuda_us(prefill_cuda, "mm") + cuda_us(prefill_cuda, "linear")
    prefill_launch = cuda_us(prefill_cuda, "cudaLaunchKernel")
    decode_matmul = cuda_us(decode_cuda, "matmul") + cuda_us(decode_cuda, "mm") + cuda_us(decode_cuda, "linear")

    prefill_class = "compute-bound" if prefill_matmul > prefill_launch * 2 else "mixed"
    return {
        "overall_classification": f"mixed (prefill {prefill_class}, decode mixed)",
        "prefill_classification": prefill_class,
        "decode_classification": "mixed",
        "prefill_matmul_cuda_us": prefill_matmul,
        "decode_matmul_cuda_us": decode_matmul,
    }


def classify_bottleneck(summaries: list[dict]) -> dict:
    """Classify inference bottleneck from NCU metrics."""
    all_kernels = []
    for s in summaries:
        all_kernels.extend(s.get("kernels", []))

    if not all_kernels:
        pytorch_path = PROJECT_ROOT / "results" / "profiling" / "pytorch" / "profiling_summary.json"
        if pytorch_path.exists():
            pt = json.loads(pytorch_path.read_text())
            pt_class = _classify_from_pytorch(pt)
            return {
                "classification": pt_class["overall_classification"],
                "reason": "NCU metrics unavailable; classified from PyTorch profiler.",
                "pytorch_fallback": pt_class,
            }
        return {"classification": "unknown", "reason": "No NCU kernel data parsed"}

    top = all_kernels[0]
    name = top.get("kernel_name", "")

    def _get(metric_fragment: str, default: float = 0.0) -> float:
        for k, v in top.items():
            if metric_fragment in k:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return default

    sm_util = _get("sm__throughput")
    dram_util = _get("dram__throughput")
    mem_throughput = _get("compute_memory_throughput")
    tc_util = _get("hmma") or _get("wmma")

    stall_mem = _get("stalled_memory_dependency")
    stall_scoreboard = _get("stalled_long_scoreboard")

    if sm_util > 60 and dram_util < 40:
        classification = "compute-bound"
    elif dram_util > 60 or mem_throughput > 60:
        classification = "memory-bound"
    elif sm_util < 30 and dram_util < 30:
        classification = "launch-bound"
    else:
        classification = "mixed"

    opt_target = "GEMM/matmul (linear layers)"
    if re.search(r"(?i)(attention|bmm|flash|softmax)", name):
        opt_target = "attention kernel (QK^T softmax V)"
    elif re.search(r"(?i)(norm|rms|layernorm)", name):
        opt_target = "RMSNorm/LayerNorm"
    elif re.search(r"(?i)(rope|rotary)", name):
        opt_target = "RoPE embedding"

    return {
        "classification": classification,
        "top_kernel": name,
        "sm_throughput_pct": sm_util,
        "dram_throughput_pct": dram_util,
        "memory_throughput_pct": mem_throughput,
        "tensor_core_indicator": tc_util,
        "stall_memory_dependency_pct": stall_mem,
        "stall_long_scoreboard_pct": stall_scoreboard,
        "recommended_optimization_target": opt_target,
        "reasoning": (
            f"Top kernel '{name[:80]}' — SM {sm_util:.1f}%, DRAM {dram_util:.1f}%, "
            f"memory throughput {mem_throughput:.1f}%."
        ),
    }


def main() -> None:
    if subprocess.run(["which", "ncu"], capture_output=True).returncode != 0:
        print("ERROR: ncu not available")
        sys.exit(1)

    subprocess.run(["ncu", "--version"], check=False)

    summaries = []
    for label, regex in KERNEL_FILTERS:
        try:
            summaries.append(_run_ncu(label, regex))
        except subprocess.TimeoutExpired:
            summaries.append({"label": label, "error": "timeout"})

    classification = classify_bottleneck(summaries)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ncu_version": subprocess.run(
            ["ncu", "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "kernel_filters": [{"label": l, "regex": r} for l, r in KERNEL_FILTERS],
        "metrics_collected": METRICS,
        "runs": summaries,
        "bottleneck_classification": classification,
    }

    out = OUTPUT_DIR / "ncu_summary.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nNCU summary saved to {out}")
    print(json.dumps(classification, indent=2))


if __name__ == "__main__":
    main()
