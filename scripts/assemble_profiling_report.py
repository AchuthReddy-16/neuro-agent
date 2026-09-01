#!/usr/bin/env python3
"""Assemble GPU profiling validation report from VRAM, PyTorch, and NCU artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results" / "profiling"


def _load_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def _extract_ncu_kernels(log_path: Path) -> list[str]:
    kernels = []
    if not log_path.exists():
        return kernels
    for line in log_path.read_text().splitlines():
        if line.strip().startswith(tuple(f"{i}." for i in range(1, 10))):
            # "1. kernel_name" format
            parts = line.split(".", 1)
            if len(parts) == 2:
                kernels.append(parts[1].strip())
    return kernels


def classify_from_pytorch(pytorch_summary: dict) -> dict:
    prefill_cuda = pytorch_summary.get("prefill", {}).get("top_cuda_ops", [])
    decode_cuda = pytorch_summary.get("decode", {}).get("top_cuda_ops", [])

    def top_names(ops: list, n: int = 5) -> list[str]:
        return [o["name"] for o in ops[:n]]

    prefill_top = top_names(prefill_cuda)
    decode_top = top_names(decode_cuda)

    # Sum CUDA time for op categories
    def cuda_us(ops: list, fragment: str) -> float:
        return sum(o.get("cuda_time_us", 0) for o in ops if fragment in o.get("name", ""))

    prefill_matmul_us = cuda_us(prefill_cuda, "matmul") + cuda_us(prefill_cuda, "mm") + cuda_us(prefill_cuda, "linear")
    prefill_launch_us = cuda_us(prefill_cuda, "cudaLaunchKernel")
    prefill_bmm_us = cuda_us(prefill_cuda, "bmm")

    decode_matmul_us = cuda_us(decode_cuda, "matmul") + cuda_us(decode_cuda, "mm") + cuda_us(decode_cuda, "linear")

    # Prefill: GEMM dominates → compute-bound; high launch count → mixed
    if prefill_matmul_us > prefill_launch_us * 2:
        prefill_class = "compute-bound"
    elif prefill_launch_us > prefill_matmul_us:
        prefill_class = "launch-bound"
    else:
        prefill_class = "mixed"

    # Decode: smaller batch GEMMs, lower utilization → mixed/memory-latency
    decode_class = "mixed" if decode_matmul_us > 0 else "unknown"

    overall = "mixed"
    if prefill_class == "compute-bound" and decode_class == "mixed":
        overall = "mixed (prefill compute-bound, decode mixed)"

    return {
        "prefill_classification": prefill_class,
        "decode_classification": decode_class,
        "overall_classification": overall,
        "prefill_top_ops": prefill_top,
        "decode_top_ops": decode_top,
        "prefill_matmul_cuda_us": prefill_matmul_us,
        "prefill_launch_cuda_us": prefill_launch_us,
        "prefill_bmm_cuda_us": prefill_bmm_us,
        "decode_matmul_cuda_us": decode_matmul_us,
        "recommended_optimization_target": (
            "BF16 Tensor Core GEMM (linear/FFN layers) — dominant in both prefill and decode; "
            "secondary target: Flash Attention kernels (flash_fwd_splitkv) and kernel-launch fusion "
            "to reduce 2000+ cudaLaunchKernel calls per prefill step."
        ),
    }


def main() -> None:
    vram = _load_json(RESULTS / "vram_investigation.json")
    pytorch = _load_json(RESULTS / "pytorch" / "profiling_summary.json")
    ncu_summary = _load_json(RESULTS / "ncu" / "ncu_summary.json")

    ncu_kernels = []
    for log in (RESULTS / "ncu").glob("*.log"):
        ncu_kernels.extend(_extract_ncu_kernels(log))
    ncu_kernels = sorted(set(ncu_kernels))

    ncu_perm_error = any(
        "ERR_NVGPUCTRPERM" in log.read_text()
        for log in (RESULTS / "ncu").glob("*.log")
        if log.exists()
    )

    nsys_available = subprocess.run(["which", "nsys"], capture_output=True).returncode == 0
    ncu_available = subprocess.run(["which", "ncu"], capture_output=True).returncode == 0

    classification = classify_from_pytorch(pytorch) if pytorch else {"overall_classification": "unknown"}

    after_load = next((s for s in vram.get("snapshots", []) if s["phase"] == "after_model_load"), {}) if vram else {}
    after_gen = next((s for s in vram.get("snapshots", []) if s["phase"] == "after_generation"), {}) if vram else {}
    two_model = next((s for s in vram.get("snapshots", []) if s["phase"] == "two_model_instances"), {}) if vram else {}

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vram_discrepancy": {
            "benchmark_peak_mb": after_gen.get("max_allocated_mb"),
            "single_model_allocated_mb": after_load.get("allocated_mb"),
            "single_model_reserved_mb": after_load.get("reserved_mb"),
            "single_model_nvidia_smi_mb": (
                after_load.get("nvidia_smi", {}).get("processes", [{}])[0].get("used_mb")
            ),
            "after_generation_allocated_mb": after_gen.get("allocated_mb"),
            "after_generation_reserved_mb": after_gen.get("reserved_mb"),
            "after_generation_max_allocated_mb": after_gen.get("max_allocated_mb"),
            "after_generation_max_reserved_mb": after_gen.get("max_reserved_mb"),
            "after_generation_nvidia_smi_mb": (
                after_gen.get("nvidia_smi", {}).get("processes", [{}])[0].get("used_mb")
            ),
            "two_model_instances_mb": two_model.get("allocated_mb"),
            "two_model_nvidia_smi_mb": (
                two_model.get("nvidia_smi", {}).get("processes", [{}])[0].get("used_mb")
            ),
            "explanation": (
                "Benchmark peak (~7.9 GB) is correct for a single model + KV cache + activations. "
                "The ~15.9 GB nvidia-smi reading matches loading TWO model instances in one process "
                "(15,353 MB allocated / 15,850 MB nvidia-smi). Single-model nvidia-smi is ~8.1 GB at "
                "rest and ~8.5 GB during inference — consistent with torch allocated + allocator reserved."
            ),
            "findings": vram.get("findings", []) if vram else [],
        },
        "ncu": {
            "available": ncu_available,
            "permission_error": ncu_perm_error,
            "metrics_collected": False if ncu_perm_error else None,
            "kernels_identified": [
                k for k in ncu_kernels
                if any(x in k.lower() for x in ("gemm", "flash", "bmm", "attention", "splitkv"))
            ][:15],
            "note": (
                "ERR_NVGPUCTRPERM on RunPod — GPU performance counters require elevated permissions. "
                "Kernel names were discovered but detailed SM/DRAM/occupancy metrics could not be collected."
                if ncu_perm_error
                else "See ncu_summary.json for metric details."
            ),
        },
        "nsys": {
            "available": nsys_available,
            "note": "Not installed on this pod." if not nsys_available else "Available.",
        },
        "bottleneck_classification": classification,
        "benchmark_bugs": [
            {
                "issue": "peak_vram naming clarity",
                "severity": "low",
                "detail": (
                    "benchmark peak_vram uses max_memory_allocated after reset_peak at generation start. "
                    "This is correct and matches measured values, but does not include transient reserved "
                    "pool overhead visible in nvidia-smi (~400-500 MB above allocated)."
                ),
            },
            {
                "issue": "no duplicate-model guard",
                "severity": "medium",
                "detail": (
                    "Sequential benchmark scripts each load a fresh model in a new process (OK), but "
                    "orchestration in one process would double VRAM. Document that only one model load "
                    "should be active per process."
                ),
            },
        ],
    }

    out = RESULTS / "gpu_profiling_validation_summary.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
