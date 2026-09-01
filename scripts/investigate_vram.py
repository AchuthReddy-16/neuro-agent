#!/usr/bin/env python3
"""Investigate VRAM accounting: PyTorch allocator vs nvidia-smi."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from neuro_agent.config import load_benchmark_config
from neuro_agent.evaluation.benchmark import build_prompt
from neuro_agent.inference.config import InferenceConfig
from neuro_agent.inference.engine import generate_with_timings
from neuro_agent.inference.model_loader import compute_weight_memory_mb, load_model_and_tokenizer
from neuro_agent.paths import RESULTS_DIR, configure_hf_cache, ensure_dirs
from neuro_agent.profiling.pytorch_profiler import profile_decode_step, profile_prefill


def _mb(bytes_val: int | float) -> float:
    return bytes_val / (1024 * 1024)


def _nvidia_smi_process_vram() -> dict[str, Any]:
    """Query nvidia-smi for per-process and total GPU memory."""
    info: dict[str, Any] = {"processes": [], "gpu_used_mb": None, "gpu_total_mb": None}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                info["processes"].append(
                    {"pid": int(parts[0]), "name": parts[1], "used_mb": float(parts[2])}
                )
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        parts = [p.strip() for p in gpu.stdout.strip().split(",")]
        if len(parts) >= 2:
            info["gpu_used_mb"] = float(parts[0])
            info["gpu_total_mb"] = float(parts[1])
    except Exception as exc:
        info["error"] = str(exc)
    return info


@dataclass
class VramSnapshot:
    """Single-point VRAM measurement."""

    phase: str
    allocated_mb: float
    reserved_mb: float
    max_allocated_mb: float
    max_reserved_mb: float
    nvidia_smi: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


def snapshot(phase: str, device: torch.device, notes: str = "") -> VramSnapshot:
    torch.cuda.synchronize(device)
    return VramSnapshot(
        phase=phase,
        allocated_mb=_mb(torch.cuda.memory_allocated(device)),
        reserved_mb=_mb(torch.cuda.memory_reserved(device)),
        max_allocated_mb=_mb(torch.cuda.max_memory_allocated(device)),
        max_reserved_mb=_mb(torch.cuda.max_memory_reserved(device)),
        nvidia_smi=_nvidia_smi_process_vram(),
        notes=notes,
    )


def reset_peak(device: torch.device) -> None:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(idx)
    torch.cuda.reset_max_memory_reserved(idx) if hasattr(torch.cuda, "reset_max_memory_reserved") else None


def main() -> None:
    configure_hf_cache()
    ensure_dirs()
    cfg = load_benchmark_config()
    mc, ic, bc, pc = cfg["model"], cfg["inference"], cfg["benchmark"], cfg["prompt"]

    device = torch.device("cuda:0")
    snapshots: list[VramSnapshot] = []
    findings: list[str] = []

    # Phase 0: baseline CUDA context
    _ = torch.zeros(1, device=device)
    torch.cuda.synchronize(device)
    snapshots.append(snapshot("cuda_context_init", device, "After minimal CUDA alloc"))

    config = InferenceConfig(
        model_name=mc["name"],
        dtype=mc["dtype"],
        seed=ic["seed"],
        do_sample=ic["do_sample"],
        max_new_tokens=ic["max_new_tokens"],
        use_cache=ic["use_cache"],
        trust_remote_code=mc.get("trust_remote_code", False),
    )

    # Phase 1: model load
    model, tokenizer, load_info = load_model_and_tokenizer(config, device)
    weight_mb = compute_weight_memory_mb(model)
    snapshots.append(
        snapshot(
            "after_model_load",
            device,
            f"weights={weight_mb:.1f}MB load_time={load_info.load_time_s:.2f}s",
        )
    )

    prompt, prompt_len = build_prompt(tokenizer, pc["base_text"], bc["prompt_token_length"])

    # Phase 2: benchmark-style generation (with peak reset like engine.py)
    reset_peak(device)
    snapshots.append(
        snapshot("before_generation_peak_reset", device, "State before reset_peak in benchmark")
    )
    reset_peak(device)
    out = generate_with_timings(model, tokenizer, prompt, config)
    snapshots.append(
        snapshot(
            "after_generation",
            device,
            f"benchmark peak_vram reported={out.timings.peak_vram_mb:.1f}MB",
        )
    )

    # Phase 3: generation WITHOUT peak reset (true process peak)
    reset_peak(device)
    _ = generate_with_timings(model, tokenizer, prompt, config)
    snapshots.append(snapshot("generation_no_mid_reset", device, "Peak since explicit reset only"))

    # Phase 4: profiler overhead
    reset_peak(device)
    profile_prefill(model, tokenizer, prompt, config, RESULTS_DIR / "profiling" / "pytorch")
    snapshots.append(snapshot("after_prefill_profiler", device))
    profile_decode_step(model, tokenizer, prompt, config, RESULTS_DIR / "profiling" / "pytorch")
    snapshots.append(snapshot("after_decode_profiler", device))

    # Phase 5: check for duplicate model (load second instance)
    model2, _, _ = load_model_and_tokenizer(config, device)
    snapshots.append(snapshot("two_model_instances", device, "Second model loaded — duplicate check"))
    del model2
    torch.cuda.empty_cache()
    snapshots.append(snapshot("after_delete_second_model", device))

    # Analysis
    after_load = snapshots[1]
    after_gen = snapshots[3]
    reserved_vs_alloc = after_load.reserved_mb - after_load.allocated_mb
    if reserved_vs_alloc > 500:
        findings.append(
            f"CUDA caching allocator reserved {reserved_vs_alloc:.0f} MB more than allocated "
            f"after load ({after_load.reserved_mb:.0f} vs {after_load.allocated_mb:.0f} MB). "
            "nvidia-smi reports reserved pool, not just tensor bytes."
        )

    smi_after_load = after_load.nvidia_smi.get("gpu_used_mb")
    if smi_after_load and smi_after_load > after_load.allocated_mb * 1.3:
        findings.append(
            f"nvidia-smi GPU used ({smi_after_load:.0f} MB) exceeds torch allocated "
            f"({after_load.allocated_mb:.0f} MB) — likely allocator reserved blocks + driver overhead."
        )

    two_model = next(s for s in snapshots if s.phase == "two_model_instances")
    if two_model.allocated_mb > after_load.allocated_mb * 1.8:
        findings.append(
            f"Duplicate model load doubled allocated VRAM ({two_model.allocated_mb:.0f} MB) — "
            "external 15.9 GB snapshot may reflect two loads or stale process."
        )
    else:
        findings.append(
            "Single model instance confirmed; 15.9 GB external reading not reproduced with one load."
        )

    bench_peak = out.timings.peak_vram_mb
    if abs(bench_peak - after_gen.max_allocated_mb) > 50:
        findings.append(
            f"Benchmark peak ({bench_peak:.0f} MB) differs from max_allocated after gen "
            f"({after_gen.max_allocated_mb:.0f} MB) — possible peak-reset timing issue."
        )
    else:
        findings.append(
            f"Benchmark peak_vram ({bench_peak:.0f} MB) matches max_memory_allocated "
            f"({after_gen.max_allocated_mb:.0f} MB) when using engine reset semantics."
        )

    profiler_snap = snapshots[-3]  # after_decode_profiler
    profiler_delta = profiler_snap.allocated_mb - after_gen.allocated_mb
    if profiler_delta > 100:
        findings.append(
            f"PyTorch profiler adds ~{profiler_delta:.0f} MB transient allocated memory."
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "model": mc["name"],
        "weight_memory_mb": weight_mb,
        "prompt_tokens": prompt_len,
        "snapshots": [asdict(s) for s in snapshots],
        "findings": findings,
        "explanation": {
            "benchmark_peak_vram_source": "torch.cuda.max_memory_allocated() after reset_peak at generation start",
            "nvidia_smi_vs_torch": (
                "nvidia-smi reports driver-level memory including CUDA caching allocator "
                "reserved blocks, cuDNN workspaces, and context overhead. torch.cuda.memory_allocated() "
                "counts only live tensors. torch.cuda.memory_reserved() is closer to nvidia-smi but "
                "still may differ due to driver fragmentation and other processes."
            ),
            "likely_15_9gb_cause": (
                "~2x single-run footprint: either two model loads in same process, "
                "profiler/benchmark scripts run back-to-back without process exit, "
                "or nvidia-smi snapshot taken during peak reserved (not allocated) phase."
            ),
        },
    }

    out_path = RESULTS_DIR / "profiling" / "vram_investigation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
