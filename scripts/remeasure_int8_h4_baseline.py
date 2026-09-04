#!/usr/bin/env python3
"""Remeasure original INT8 (no torch_dtype) and torch.compile(mode=default)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import importlib.util

_opt_path = PROJECT_ROOT / "scripts" / "optimize_int8_runtime.py"
_spec = importlib.util.spec_from_file_location("optimize_int8_runtime", _opt_path)
_opt = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_opt)

PROMPT_BASE = _opt.PROMPT_BASE
PROMPT_TOKENS = _opt.PROMPT_TOKENS
RESULTS_DIR = _opt.RESULTS_DIR
benchmark_eager = _opt.benchmark_eager
cleanup = _opt.cleanup
load_int8 = _opt.load_int8
profile_generate = _opt.profile_generate
unload = _opt.unload
from neuro_agent.inference.engine import generate_with_timings, make_prompt_of_token_length

CMP_PATH = PROJECT_ROOT / "results" / "model_comparison" / "int8_before_vs_after_optimization.json"


def main() -> None:
    print("Remeasure original INT8 (int8_compute_dtype=None, no torch_dtype)")
    cleanup()
    model, tokenizer, info, cfg = load_int8(compute_dtype=None)
    prompt, ntok = make_prompt_of_token_length(tokenizer, PROMPT_BASE, PROMPT_TOKENS)
    print(f"prompt_tokens={ntok} load={info.load_time_s:.2f}s")
    base_load_s = info.load_time_s
    base = benchmark_eager(model, tokenizer, prompt, cfg)
    base_prof = profile_generate(model, tokenizer, prompt, cfg)
    print(
        f"BASE TTFT={base['ttft_ms']} decode={base['decode_tok_per_s']} "
        f"lat/tok={base['latency_per_token_ms']} E2E={base['e2e_ms']} "
        f"peak={base['peak_vram_mb']} casts={base_prof['cast_count']} "
        f"bf16_casts={base_prof['bf16_copy_cast_count']} launches={base_prof['kernel_launch_count']}"
    )
    unload(model)

    print("\nRetry torch.compile surrounding mode=default on original INT8")
    compile_ok = False
    compile_err = None
    h43 = None
    h43_prof = None
    try:
        model, tokenizer, info, cfg = load_int8(compute_dtype=None, compile_surrounding=True)
        for _ in range(2):
            generate_with_timings(model, tokenizer, prompt, cfg)
        h43 = benchmark_eager(model, tokenizer, prompt, cfg)
        h43_prof = profile_generate(model, tokenizer, prompt, cfg)
        compile_ok = True
        print(
            f"COMPILE TTFT={h43['ttft_ms']} decode={h43['decode_tok_per_s']} "
            f"E2E={h43['e2e_ms']} launches={h43_prof['kernel_launch_count']}"
        )
        unload(model)
    except Exception as exc:  # noqa: BLE001
        compile_err = f"{type(exc).__name__}: {exc}"
        print(f"COMPILE FAIL {compile_err}")
        cleanup()

    report = json.loads(CMP_PATH.read_text())
    h41_after = report["h4_1"]["after"]
    kept_h41 = True
    decode_gain = (
        (h41_after["decode_tok_per_s"] - base["decode_tok_per_s"])
        / max(base["decode_tok_per_s"], 1e-6)
        * 100.0
    )
    h43_kept = False
    if compile_ok and h43:
        h43_kept = h43["decode_tok_per_s"] >= base["decode_tok_per_s"] * 1.05

    report["h4_1"]["before"] = {
        "ttft_ms": base["ttft_ms"],
        "decode_tok_per_s": base["decode_tok_per_s"],
        "latency_per_token_ms": base["latency_per_token_ms"],
        "e2e_ms": base["e2e_ms"],
        "cast_count": base_prof["cast_count"],
        "bf16_copy_cast_count": base_prof["bf16_copy_cast_count"],
        "kernel_launch_count": base_prof["kernel_launch_count"],
        "peak_vram_mb": base["peak_vram_mb"],
        "note": "Original H.1B/H.3 INT8 load (no torch_dtype). First H.4 pass accidentally passed torch_dtype=bfloat16 and is discarded.",
    }
    report["h4_1"]["kept"] = kept_h41
    report["h4_1"]["decode_gain_pct"] = round(decode_gain, 2)

    report["h4_3"] = {
        "torch_compile_compatible": compile_ok,
        "error": compile_err,
        "mode": "default",
        "before": {
            "decode_tok_per_s": base["decode_tok_per_s"],
            "e2e_ms": base["e2e_ms"],
        },
        "after": None
        if not h43
        else {
            "ttft_ms": h43["ttft_ms"],
            "decode_tok_per_s": h43["decode_tok_per_s"],
            "e2e_ms": h43["e2e_ms"],
            "kernel_launch_count": h43_prof["kernel_launch_count"] if h43_prof else None,
        },
        "kept": h43_kept,
        "note": "reduce-overhead CUDA Graphs overwrite RMSNorm outputs; retried with mode=default.",
    }

    report["final_comparison"]["h4_baseline_remeasured"] = {
        "ttft_ms": base["ttft_ms"],
        "decode_tok_per_s": base["decode_tok_per_s"],
        "latency_per_token_ms": base["latency_per_token_ms"],
        "e2e_ms": base["e2e_ms"],
        "peak_vram_mb": base["peak_vram_mb"],
        "cast_count": base_prof["cast_count"],
        "bf16_copy_cast_count": base_prof["bf16_copy_cast_count"],
        "kernel_launch_count": base_prof["kernel_launch_count"],
        "load_path": "bitsandbytes load_in_8bit without torch_dtype (original)",
    }
    report["h4_2"]["before_decode_tok_per_s"] = base["decode_tok_per_s"]
    report["variants"]["baseline_int8_original"] = {
        **base,
        **{f"prof_{k}": v for k, v in base_prof.items()},
        "load_time_s": round(base_load_s, 3),
        "note": "true original INT8 path",
    }
    if h43:
        report["variants"]["h4_3_torch_compile_default"] = {
            **h43,
            **({f"prof_{k}": v for k, v in (h43_prof or {}).items()}),
            "kept": h43_kept,
        }

    # Combined remains fp16-only (graphs/compile not kept)
    report["best_combined"]["flags"]["compile_surrounding"] = h43_kept
    report["decode_gain_pct_vs_h4_baseline"] = round(decode_gain, 2)
    fusion_needed = decode_gain < 15.0
    report["h4_5_fusion"]["investigated"] = fusion_needed
    report["h4_5_fusion"]["reason_to_skip_or_run"] = (
        f"Combined decode gain vs original INT8 is {decode_gain:.1f}%. "
        + (
            "Below 15% material-improvement bar — document fusion limits, do not write kernels."
            if fusion_needed
            else "Material decode gain from H4-1; skip custom fusion kernels."
        )
    )

    CMP_PATH.write_text(json.dumps(report, indent=2, default=str))
    (RESULTS_DIR / "h4_report.json").write_text(json.dumps(report, indent=2, default=str))
    (RESULTS_DIR / "h4_baseline_original.json").write_text(
        json.dumps({"bench": base, "prof": base_prof}, indent=2, default=str)
    )
    print(f"Updated {CMP_PATH}")


if __name__ == "__main__":
    main()
