#!/usr/bin/env python3
"""Stage H.4 — Optimize the measured bitsandbytes INT8 bottleneck.

Sequential runtime variants only. Does not replace INT8 with another backend.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_agent.evaluation.llm_eval import (  # noqa: E402
    _build_prompt,
    _generate_response,
    extract_subjects,
    load_eval_examples,
    verify_heldout_integrity,
)
from neuro_agent.evaluation.verifiers import verify_example  # noqa: E402
from neuro_agent.inference.config import InferenceConfig  # noqa: E402
from neuro_agent.inference.engine import (  # noqa: E402
    generate_with_timings,
    make_prompt_of_token_length,
)
from neuro_agent.inference.model_loader import load_model_and_tokenizer  # noqa: E402
from neuro_agent.paths import configure_hf_cache  # noqa: E402
from neuro_agent.quantization.int8_runtime import apply_int8_runtime  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results" / "optimization" / "int8"
CMP_PATH = PROJECT_ROOT / "results" / "model_comparison" / "int8_before_vs_after_optimization.json"
BNB_LOG = RESULTS_DIR / "bnb_warnings.log"
H3_PATH = PROJECT_ROOT / "results" / "model_comparison" / "bf16_vs_int8_profile.json"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = str(PROJECT_ROOT / "checkpoints" / "sft_corrected_v2" / "final")
WARMUP = 2
TIMED = 5
MAX_NEW = 64
PROMPT_TOKENS = 512
PROMPT_BASE = (
    "You are a neuroscience research assistant. Answer the following question "
    "with precise scientific detail. Describe high-frequency gamma oscillations, "
    "mu rhythm suppression over C3/C4, and motor imagery BCI decoding."
)

ORIGINAL_INT8 = {
    "source": "H.1B preserved baseline",
    "quality": 0.866,
    "ttft_ms": 112.5,
    "decode_tok_per_s": 12.5,
    "e2e_ms": 5450.0,
    "peak_vram_gb": 4.60,
}

TASK_FAMILIES = [
    "numerical_reasoning",
    "channel_ranking",
    "band_power_analysis",
    "movement_task_classification",
    "execution_vs_imagery",
    "statistical_comparison",
    "tool_selection",
    "factual_grounding",
]


def _redirect_bnb_warnings() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(BNB_LOG)
    handler.setLevel(logging.ERROR)
    for name in ("bitsandbytes", "bitsandbytes.autograd", "bitsandbytes.nn"):
        log = logging.getLogger(name)
        log.setLevel(logging.ERROR)
        log.handlers.clear()
        log.addHandler(handler)
        log.propagate = False
    # MatMul8bitLt emits one warning per linear when activations are BF16; that
    # I/O alone can dominate a Python decode loop and invalidate timing.
    import warnings

    warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
    warnings.filterwarnings("ignore", message=".*inputs will be cast.*")


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def assert_gpu_clear() -> dict:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
        text=True,
    )
    procs = []
    for line in smi.strip().splitlines()[1:]:
        if line.strip() and "No running" not in line:
            procs.append(line.strip())
    return {"nvidia_smi_compute_apps": procs, "clear": len(procs) == 0, "raw": smi}


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_int8(compute_dtype: str | None = None, compile_surrounding: bool = False):
    config = InferenceConfig(
        model_name=MODEL_NAME,
        dtype="bfloat16",
        seed=42,
        do_sample=False,
        max_new_tokens=MAX_NEW,
        use_cache=True,
        temperature=0.0,
        top_p=1.0,
        adapter_path=ADAPTER_PATH,
        quantization="int8",
        int8_compute_dtype=compute_dtype,
        compile_surrounding=compile_surrounding,
    )
    cleanup()
    model, tokenizer, info = load_model_and_tokenizer(config)
    model.eval()
    return model, tokenizer, info, config


def benchmark_eager(model, tokenizer, prompt: str, config: InferenceConfig) -> dict:
    device = next(model.parameters()).device
    for _ in range(WARMUP):
        generate_with_timings(model, tokenizer, prompt, config)
        torch.cuda.synchronize(device)

    prefill, ttft, decode_tps, lat_tok, e2e, peak = [], [], [], [], [], []
    allocated = []
    for _ in range(TIMED):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        out = generate_with_timings(model, tokenizer, prompt, config)
        t = out.timings
        prefill.append(t.prefill_latency_ms)
        ttft.append(t.ttft_ms)
        decode_tps.append(t.decode_tokens_per_second)
        lat_tok.append(t.decode_latency_per_token_ms)
        e2e.append(t.end_to_end_latency_ms)
        peak.append(t.peak_vram_mb)
        allocated.append(torch.cuda.memory_allocated(device) / (1024 * 1024))
        torch.cuda.synchronize(device)

    return {
        "prompt_tokens": out.timings.prompt_token_count,
        "generated_tokens": out.timings.generated_token_count,
        "ttft_ms": round(_mean(ttft), 2),
        "prefill_ms": round(_mean(prefill), 2),
        "decode_tok_per_s": round(_mean(decode_tps), 2),
        "latency_per_token_ms": round(_mean(lat_tok), 2),
        "e2e_ms": round(_mean(e2e), 2),
        "allocated_vram_mb": round(_mean(allocated), 1),
        "peak_vram_mb": round(_mean(peak), 1),
        "warmup": WARMUP,
        "timed_iters": TIMED,
        "all_ttft_ms": [round(x, 2) for x in ttft],
        "all_prefill_ms": [round(x, 2) for x in prefill],
        "all_decode_tok_per_s": [round(x, 2) for x in decode_tps],
        "all_e2e_ms": [round(x, 2) for x in e2e],
        "decode_loop": "eager_generate_with_timings",
    }


def profile_generate(model, tokenizer, prompt: str, config: InferenceConfig) -> dict:
    generate_with_timings(model, tokenizer, prompt, config)
    torch.cuda.synchronize()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        generate_with_timings(model, tokenizer, prompt, config)
        torch.cuda.synchronize()

    events = []
    for evt in prof.key_averages():
        self_dev = getattr(evt, "self_device_time_total", 0) or 0
        if self_dev > 0:
            events.append(
                {
                    "name": evt.key,
                    "count": evt.count,
                    "self_cuda_time_us": round(self_dev, 1),
                }
            )
    events.sort(key=lambda e: e["self_cuda_time_us"], reverse=True)
    launch_count = sum(e["count"] for e in events)
    casts = [
        e
        for e in events
        if e["name"] in {"aten::_to_copy", "aten::to"}
        or "_to_copy" in e["name"]
        or "BFloat16" in e["name"]
        or "cast" in e["name"].lower()
    ]
    bf16_casts = [e for e in events if "BFloat16" in e["name"] and "copy" in e["name"].lower()]
    return {
        "kernel_launch_count": launch_count,
        "cast_count": sum(e["count"] for e in casts),
        "bf16_copy_cast_count": sum(e["count"] for e in bf16_casts),
        "cast_self_cuda_us": round(sum(e["self_cuda_time_us"] for e in casts), 1),
        "top10_ops": events[:10],
    }


def _try_static_cache(model, max_cache_len: int):
    from transformers import StaticCache

    cfg = model.config
    try:
        return StaticCache(config=cfg, max_cache_len=max_cache_len)
    except TypeError:
        return StaticCache(
            config=cfg,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=next(model.parameters()).device,
        )


def test_cuda_graphs(model, tokenizer, prompt: str, config: InferenceConfig) -> dict:
    """Attempt CUDA Graph capture of fixed-shape single-token decode."""
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    prompt_len = input_ids.shape[1]
    result: dict = {
        "compatible": False,
        "reason": None,
        "exception": None,
        "static_cache_ok": False,
        "capture_ok": False,
        "benchmark": None,
    }

    try:
        cache = _try_static_cache(model, prompt_len + MAX_NEW)
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True, past_key_values=cache)
        result["static_cache_ok"] = True
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        past = out.past_key_values
        static_in = next_token.clone()

        # Warmup eager static-cache decode (allocates caches / bnb state)
        for _ in range(3):
            with torch.inference_mode():
                step = model(input_ids=static_in, past_key_values=past, use_cache=True)
            past = step.past_key_values
            static_in.copy_(torch.argmax(step.logits[:, -1, :], dim=-1, keepdim=True))
        torch.cuda.synchronize(device)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.inference_mode():
                step = model(input_ids=static_in, past_key_values=past, use_cache=True)
            past = step.past_key_values
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize(device)

        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                with torch.inference_mode():
                    step = model(input_ids=static_in, past_key_values=past, use_cache=True)
                past = step.past_key_values
            result["capture_ok"] = True
            result["compatible"] = True
        except Exception as exc:  # noqa: BLE001
            result["reason"] = (
                "CUDA Graph capture failed on the bitsandbytes INT8 + HF decode step. "
                f"{type(exc).__name__}: {exc}"
            )
            result["exception"] = traceback.format_exc(limit=8)
            # Likely causes for this stack:
            result["likely_causes"] = [
                "MatMul8bitLt uses data-dependent LLM.int8() outlier columns (llm_int8_threshold=6.0)",
                "int8_vectorwise_quant / int8_mixed_scaled_mm allocate and branch per forward",
                "Dynamic / Static cache updates plus bitsandbytes state mutations are not graph-safe",
            ]
            return result

        # Replay graph for remaining tokens and time vs eager
        def _graph_generate_once() -> dict:
            cache_i = _try_static_cache(model, prompt_len + MAX_NEW)
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                out_i = model(input_ids=input_ids, use_cache=True, past_key_values=cache_i)
            torch.cuda.synchronize(device)
            t_prefill = time.perf_counter()
            tok = torch.argmax(out_i.logits[:, -1, :], dim=-1, keepdim=True)
            static_in.copy_(tok)
            generated = 1
            decode_t0 = time.perf_counter()
            for _ in range(MAX_NEW - 1):
                graph.replay()
                tok = torch.argmax(step.logits[:, -1, :], dim=-1, keepdim=True)
                static_in.copy_(tok)
                generated += 1
            torch.cuda.synchronize(device)
            t_end = time.perf_counter()
            prefill_ms = (t_prefill - t0) * 1000.0
            decode_ms = (t_end - decode_t0) * 1000.0
            e2e_ms = (t_end - t0) * 1000.0
            return {
                "prefill_ms": prefill_ms,
                "ttft_ms": prefill_ms,
                "decode_ms": decode_ms,
                "e2e_ms": e2e_ms,
                "generated_tokens": generated,
                "decode_tok_per_s": generated / (decode_ms / 1000.0) if decode_ms > 0 else 0.0,
                "latency_per_token_ms": decode_ms / generated if generated else 0.0,
            }

        # Note: graph replay binds the captured `step`/`past` from the capture
        # iteration; if those tensors are stale this path is invalid.
        for _ in range(WARMUP):
            _graph_generate_once()
        runs = [_graph_generate_once() for _ in range(TIMED)]
        result["benchmark"] = {
            "ttft_ms": round(_mean([r["ttft_ms"] for r in runs]), 2),
            "prefill_ms": round(_mean([r["prefill_ms"] for r in runs]), 2),
            "decode_tok_per_s": round(_mean([r["decode_tok_per_s"] for r in runs]), 2),
            "latency_per_token_ms": round(_mean([r["latency_per_token_ms"] for r in runs]), 2),
            "e2e_ms": round(_mean([r["e2e_ms"] for r in runs]), 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 1),
            "allocated_vram_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 1),
            "decode_loop": "cuda_graph_static_cache",
            "graph_correctness_caveat": (
                "Replay uses tensors captured in the graph; treat as throughput "
                "probe only if logits stay bound to the captured step output."
            ),
        }
        return result
    except Exception as exc:  # noqa: BLE001
        result["reason"] = (
            "INT8 + HuggingFace decode is not CUDA-Graph compatible in this stack. "
            f"{type(exc).__name__}: {exc}"
        )
        result["exception"] = traceback.format_exc(limit=12)
        result["likely_causes"] = [
            "StaticCache construction or HF forward rejected bitsandbytes INT8 tensors",
            "bitsandbytes MatMul8bitLt mixed-precision control flow is data-dependent",
            "Allocator / graph memory pool incompatibility with INT8 workspace buffers",
        ]
        return result


def quality_sanity(model, tokenizer, config: InferenceConfig) -> dict:
    dataset = PROJECT_ROOT / "data" / "processed" / "eval_heldout_corrected.jsonl"
    examples = load_eval_examples(dataset)
    cfg_yaml_cats = TASK_FAMILIES
    buckets: dict[str, list] = defaultdict(list)
    for ex in examples:
        cat = ex.get("category")
        if cat in cfg_yaml_cats and len(buckets[cat]) < 3:
            buckets[cat].append(ex)
    subset = []
    for cat in cfg_yaml_cats:
        subset.extend(buckets.get(cat, []))
    # 24 examples (3×8). Top up to 32 if extra rows exist.
    extras = [ex for ex in examples if ex not in subset and ex.get("category") in cfg_yaml_cats]
    subset.extend(extras[: max(0, 32 - len(subset))])

    held_out = {"S026", "S027", "S028", "S029", "S030"}
    forbidden = {f"S{i:03d}" for i in range(1, 26)}
    integrity = verify_heldout_integrity(subset, held_out, forbidden)

    system = (
        "You are a neuroscience research assistant. Answer using only the provided "
        "context. Respond with a concise direct answer only. Do not add unsupported "
        "interpretation."
    )
    qcfg = InferenceConfig(
        model_name=config.model_name,
        dtype=config.dtype,
        seed=42,
        do_sample=False,
        max_new_tokens=128,
        use_cache=True,
        temperature=0.0,
        top_p=1.0,
        adapter_path=config.adapter_path,
        quantization="int8",
        int8_compute_dtype=config.int8_compute_dtype,
        compile_surrounding=config.compile_surrounding,
    )

    records = []
    for ex in subset:
        prompt = _build_prompt(ex, system, tokenizer)
        response, ntok = _generate_response(model, tokenizer, prompt, qcfg)
        ver = verify_example(ex, response)
        records.append(
            {
                "id": ex["id"],
                "category": ex["category"],
                "passed": ver.passed,
                "reason": ver.reason,
                "response": response[:400],
                "generated_tokens": ntok,
                "subjects": extract_subjects(ex),
            }
        )

    passed = sum(1 for r in records if r["passed"])
    per_task = {}
    for cat in cfg_yaml_cats:
        rows = [r for r in records if r["category"] == cat]
        per_task[cat] = {
            "n": len(rows),
            "pass_rate": (sum(1 for r in rows if r["passed"]) / len(rows)) if rows else None,
        }
    return {
        "n_examples": len(records),
        "verifier_pass_rate": passed / len(records) if records else 0.0,
        "per_task": per_task,
        "integrity": integrity,
        "original_int8_quality": ORIGINAL_INT8["quality"],
        "records": records,
    }


def unload(model) -> None:
    del model
    cleanup()


def improved(candidate: dict | None, baseline: dict, key: str = "decode_tok_per_s", min_pct: float = 5.0) -> bool:
    if not candidate or key not in candidate or key not in baseline:
        return False
    if baseline[key] <= 0:
        return False
    return (candidate[key] - baseline[key]) / baseline[key] * 100.0 >= min_pct


def main() -> None:
    configure_hf_cache()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CMP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _redirect_bnb_warnings()

    print("=" * 70)
    print("H.4 — INT8 runtime optimization (bitsandbytes path only)")
    print("=" * 70)

    gpu_pre = assert_gpu_clear()
    print(f"GPU clear: {gpu_pre['clear']}")
    if not gpu_pre["clear"]:
        print(gpu_pre["nvidia_smi_compute_apps"])

    h3 = json.loads(H3_PATH.read_text()) if H3_PATH.exists() else {}
    h3_int8 = (h3.get("benchmarks") or {}).get("int8") or {}
    h3_repro = {
        "source": "H.3 bf16_vs_int8_profile.json",
        "prefill_ms": h3_int8.get("prefill_ms", 98.74),
        "decode_tok_per_s": h3_int8.get("decode_tok_per_s", 13.08),
        "ttft_ms": h3_int8.get("ttft_ms", 98.74),
        "e2e_ms": h3_int8.get("e2e_ms", 4992.59),
        "latency_per_token_ms": h3_int8.get("latency_per_token_ms", 76.47),
        "peak_vram_mb": h3_int8.get("peak_vram_mb", 4963.6),
        "kernel_launches": (h3.get("kernel_launch_counts") or {}).get("int8", 945473),
    }

    tokenizer_probe_cfg = InferenceConfig(
        model_name=MODEL_NAME,
        dtype="bfloat16",
        quantization="int8",
        adapter_path=ADAPTER_PATH,
        max_new_tokens=MAX_NEW,
        do_sample=False,
    )
    # Prompt built after first load via that tokenizer.

    variants: dict[str, dict] = {}

    # ── Baseline INT8 (original path) ──
    print("\n── H4 baseline INT8 (BF16 activations) ──")
    model, tokenizer, info, cfg = load_int8(compute_dtype=None)
    prompt, actual_len = make_prompt_of_token_length(tokenizer, PROMPT_BASE, PROMPT_TOKENS)
    print(f"Prompt tokens: {actual_len}  load={info.load_time_s:.2f}s")
    base_bench = benchmark_eager(model, tokenizer, prompt, cfg)
    base_prof = profile_generate(model, tokenizer, prompt, cfg)
    variants["baseline_int8"] = {
        "description": "Existing bitsandbytes load_in_8bit; BF16 non-quantized compute",
        "load_time_s": round(info.load_time_s, 3),
        "allocated_after_load_mb": info.allocated_after_load_mb,
        **base_bench,
        **{f"prof_{k}": v for k, v in base_prof.items()},
    }
    print(
        f"  TTFT={base_bench['ttft_ms']} ms  decode={base_bench['decode_tok_per_s']} tok/s  "
        f"E2E={base_bench['e2e_ms']} ms  peak={base_bench['peak_vram_mb']} MB"
    )
    unload(model)

    # ── H4-1 FP16 compute / activations ──
    print("\n── H4-1 FP16 activations into Linear8bitLt ──")
    model, tokenizer, info, cfg = load_int8(compute_dtype="float16")
    h41_bench = benchmark_eager(model, tokenizer, prompt, cfg)
    h41_prof = profile_generate(model, tokenizer, prompt, cfg)
    variants["h4_1_fp16_compute"] = {
        "description": "Cast non-quantized params/activations to FP16; INT8 weights unchanged",
        "load_time_s": round(info.load_time_s, 3),
        "allocated_after_load_mb": info.allocated_after_load_mb,
        **h41_bench,
        **{f"prof_{k}": v for k, v in h41_prof.items()},
    }
    print(
        f"  TTFT={h41_bench['ttft_ms']} ms  decode={h41_bench['decode_tok_per_s']} tok/s  "
        f"E2E={h41_bench['e2e_ms']} ms  casts={h41_prof['cast_count']} "
        f"(baseline {base_prof['cast_count']})"
    )
    h41_keep = improved(h41_bench, base_bench) or (
        h41_prof["bf16_copy_cast_count"] < base_prof["bf16_copy_cast_count"] * 0.5
        and h41_bench["decode_tok_per_s"] >= base_bench["decode_tok_per_s"] * 0.98
    )
    variants["h4_1_fp16_compute"]["kept"] = bool(h41_keep)
    unload(model)

    # ── H4-2 CUDA Graphs (independent, on baseline dtype) ──
    print("\n── H4-2 CUDA Graphs on INT8 decode ──")
    model, tokenizer, info, cfg = load_int8(compute_dtype=None)
    graph_info = test_cuda_graphs(model, tokenizer, prompt, cfg)
    variants["h4_2_cuda_graphs"] = {
        "description": "CUDA Graph capture/replay of fixed-shape single-token INT8 decode",
        "load_time_s": round(info.load_time_s, 3),
        **graph_info,
        "kept": bool(graph_info.get("compatible") and improved(graph_info.get("benchmark"), base_bench)),
    }
    print(f"  compatible={graph_info.get('compatible')} reason={graph_info.get('reason')}")
    unload(model)

    # ── H4-3 torch.compile surrounding modules (independent) ──
    print("\n── H4-3 torch.compile surrounding (RMSNorm/rotary) ──")
    compile_ok = False
    compile_error = None
    h43_bench = None
    h43_prof = None
    try:
        model, tokenizer, info, cfg = load_int8(compute_dtype=None, compile_surrounding=True)
        # Extra warmup for compiler
        for _ in range(2):
            generate_with_timings(model, tokenizer, prompt, cfg)
            torch.cuda.synchronize()
        h43_bench = benchmark_eager(model, tokenizer, prompt, cfg)
        h43_prof = profile_generate(model, tokenizer, prompt, cfg)
        compile_ok = True
        unload(model)
    except Exception as exc:  # noqa: BLE001
        compile_error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
        cleanup()

    variants["h4_3_torch_compile"] = {
        "description": "torch.compile on RMSNorm/rotary only; Linear8bitLt eager",
        "compatible": compile_ok,
        "error": compile_error,
        **(h43_bench or {}),
        **({f"prof_{k}": v for k, v in (h43_prof or {}).items()}),
        "kept": bool(compile_ok and improved(h43_bench, base_bench)),
    }
    if h43_bench:
        print(
            f"  TTFT={h43_bench['ttft_ms']} ms  decode={h43_bench['decode_tok_per_s']} tok/s  "
            f"E2E={h43_bench['e2e_ms']} ms"
        )
    else:
        print(f"  compile failed: {compile_error}")

    # ── H4-4 combined successful ──
    use_fp16 = variants["h4_1_fp16_compute"]["kept"]
    use_compile = variants["h4_3_torch_compile"]["kept"]
    use_graph = variants["h4_2_cuda_graphs"]["kept"]
    print(
        f"\n── H4-4 combined (fp16={use_fp16}, compile={use_compile}, graphs={use_graph}) ──"
    )
    model, tokenizer, info, cfg = load_int8(
        compute_dtype="float16" if use_fp16 else None,
        compile_surrounding=use_compile,
    )
    if use_compile:
        for _ in range(2):
            generate_with_timings(model, tokenizer, prompt, cfg)
            torch.cuda.synchronize()
    if use_graph:
        graph_combo = test_cuda_graphs(model, tokenizer, prompt, cfg)
        combo_bench = graph_combo.get("benchmark") or benchmark_eager(model, tokenizer, prompt, cfg)
        combo_prof = profile_generate(model, tokenizer, prompt, cfg)
        combo_extra = {"cuda_graph": graph_combo}
    else:
        combo_bench = benchmark_eager(model, tokenizer, prompt, cfg)
        combo_prof = profile_generate(model, tokenizer, prompt, cfg)
        combo_extra = {}
    variants["h4_4_combined"] = {
        "description": "Independently successful H4-1/H4-2/H4-3 flags only",
        "flags": {"fp16_compute": use_fp16, "compile_surrounding": use_compile, "cuda_graphs": use_graph},
        "load_time_s": round(info.load_time_s, 3),
        **combo_bench,
        **{f"prof_{k}": v for k, v in combo_prof.items()},
        **combo_extra,
    }
    print(
        f"  TTFT={combo_bench['ttft_ms']} ms  decode={combo_bench['decode_tok_per_s']} tok/s  "
        f"E2E={combo_bench['e2e_ms']} ms  peak={combo_bench['peak_vram_mb']} MB"
    )

    # ── H4-5 fusion investigation only if needed ──
    decode_gain_pct = (
        (combo_bench["decode_tok_per_s"] - base_bench["decode_tok_per_s"])
        / max(base_bench["decode_tok_per_s"], 1e-6)
        * 100.0
    )
    fusion_needed = decode_gain_pct < 15.0
    fusion = {
        "investigated": fusion_needed,
        "reason_to_skip_or_run": (
            f"Combined decode gain vs H.4 baseline is {decode_gain_pct:.1f}%. "
            + (
                "Below 15% material-improvement bar — document fusion limits, do not write kernels."
                if fusion_needed
                else "Material decode gain achieved; skip custom fusion kernels."
            )
        ),
        "would_require_new_backend": True,
        "notes": [
            "H.3: MatMul8bitLt wrapper ~71.6% self CUDA; each linear is quantize→int8_gemm→dequant.",
            "bitsandbytes implements this in C++/CUDA (int8_vectorwise_quant, int8_linear_matmul, int8_mm_dequant).",
            "Fusing those three would mean replacing bitsandbytes internals or adding a new INT8 GEMM path.",
            "That is a new quantization backend, which is out of scope for H.4. STOP — no Triton/CUDA rewrite.",
        ],
    }

    print("\n── Quality sanity (24–32 examples, 8 task families) ──")
    quality = quality_sanity(model, tokenizer, cfg)
    print(
        f"  n={quality['n_examples']} pass_rate={quality['verifier_pass_rate']:.3f} "
        f"(original INT8 quality {ORIGINAL_INT8['quality']})"
    )
    unload(model)

    optimized = variants["h4_4_combined"]
    remaining = (
        "Dominant cost remains bitsandbytes MatMul8bitLt: each decode step still "
        "launches quantize + INT8 GEMM + dequantize (+ outlier mixed-precision) "
        "instead of a single Tensor-Core GEMM. Cast reduction and surrounding "
        "compile/graphs cannot remove that multi-kernel linear path."
    )

    pass_fail = "PASS"
    reasons = []
    if not variants["baseline_int8"].get("decode_tok_per_s"):
        pass_fail = "FAIL"
        reasons.append("baseline INT8 benchmark missing")
    if quality["n_examples"] < 16:
        pass_fail = "FAIL"
        reasons.append("quality sanity too small")
    reasons.append("H.4 completed measured H4-1/H4-2/H4-3/H4-4 on the existing bitsandbytes INT8 path")

    comparison = {
        "stage": "H.4",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL_NAME,
        "adapter": ADAPTER_PATH,
        "contract": {
            "prompt_tokens_target": PROMPT_TOKENS,
            "generated_tokens": MAX_NEW,
            "warmup": WARMUP,
            "timed_iters": TIMED,
            "backend": "bitsandbytes load_in_8bit (not replaced)",
        },
        "gpu_preflight": gpu_pre,
        "h4_1": {
            "what_changed": (
                "Non-quantized tensors (embeddings, RMSNorm, LoRA, biases) cast to FP16 "
                "so Linear8bitLt / MatMul8bitLt receive FP16 activations. "
                "A.to(float16) inside bitsandbytes becomes a no-op instead of BF16→FP16."
            ),
            "before": {
                "ttft_ms": variants["baseline_int8"]["ttft_ms"],
                "decode_tok_per_s": variants["baseline_int8"]["decode_tok_per_s"],
                "latency_per_token_ms": variants["baseline_int8"]["latency_per_token_ms"],
                "e2e_ms": variants["baseline_int8"]["e2e_ms"],
                "cast_count": variants["baseline_int8"].get("prof_cast_count"),
                "bf16_copy_cast_count": variants["baseline_int8"].get("prof_bf16_copy_cast_count"),
                "kernel_launch_count": variants["baseline_int8"].get("prof_kernel_launch_count"),
            },
            "after": {
                "ttft_ms": variants["h4_1_fp16_compute"]["ttft_ms"],
                "decode_tok_per_s": variants["h4_1_fp16_compute"]["decode_tok_per_s"],
                "latency_per_token_ms": variants["h4_1_fp16_compute"]["latency_per_token_ms"],
                "e2e_ms": variants["h4_1_fp16_compute"]["e2e_ms"],
                "cast_count": variants["h4_1_fp16_compute"].get("prof_cast_count"),
                "bf16_copy_cast_count": variants["h4_1_fp16_compute"].get("prof_bf16_copy_cast_count"),
                "kernel_launch_count": variants["h4_1_fp16_compute"].get("prof_kernel_launch_count"),
            },
            "kept": variants["h4_1_fp16_compute"]["kept"],
        },
        "h4_2": {
            "cuda_graph_compatible": variants["h4_2_cuda_graphs"].get("compatible"),
            "reason": variants["h4_2_cuda_graphs"].get("reason"),
            "likely_causes": variants["h4_2_cuda_graphs"].get("likely_causes"),
            "before_decode_tok_per_s": variants["baseline_int8"]["decode_tok_per_s"],
            "after": variants["h4_2_cuda_graphs"].get("benchmark"),
            "kept": variants["h4_2_cuda_graphs"]["kept"],
        },
        "h4_3": {
            "torch_compile_compatible": variants["h4_3_torch_compile"].get("compatible"),
            "error": variants["h4_3_torch_compile"].get("error"),
            "before": {
                "decode_tok_per_s": variants["baseline_int8"]["decode_tok_per_s"],
                "e2e_ms": variants["baseline_int8"]["e2e_ms"],
            },
            "after": {
                "ttft_ms": variants["h4_3_torch_compile"].get("ttft_ms"),
                "decode_tok_per_s": variants["h4_3_torch_compile"].get("decode_tok_per_s"),
                "e2e_ms": variants["h4_3_torch_compile"].get("e2e_ms"),
                "kernel_launch_count": variants["h4_3_torch_compile"].get("prof_kernel_launch_count"),
            },
            "kept": variants["h4_3_torch_compile"]["kept"],
        },
        "best_combined": {
            "flags": variants["h4_4_combined"]["flags"],
            "metrics": {
                "ttft_ms": optimized["ttft_ms"],
                "prefill_ms": optimized.get("prefill_ms"),
                "decode_tok_per_s": optimized["decode_tok_per_s"],
                "latency_per_token_ms": optimized["latency_per_token_ms"],
                "e2e_ms": optimized["e2e_ms"],
                "allocated_vram_mb": optimized.get("allocated_vram_mb"),
                "peak_vram_mb": optimized.get("peak_vram_mb"),
                "kernel_launch_count": optimized.get("prof_kernel_launch_count"),
                "cast_count": optimized.get("prof_cast_count"),
            },
        },
        "final_comparison": {
            "original_int8_h1b": ORIGINAL_INT8,
            "h3_reproduced_int8": h3_repro,
            "h4_baseline_remeasured": {
                "ttft_ms": variants["baseline_int8"]["ttft_ms"],
                "decode_tok_per_s": variants["baseline_int8"]["decode_tok_per_s"],
                "latency_per_token_ms": variants["baseline_int8"]["latency_per_token_ms"],
                "e2e_ms": variants["baseline_int8"]["e2e_ms"],
                "peak_vram_mb": variants["baseline_int8"]["peak_vram_mb"],
            },
            "optimized_int8": {
                "ttft_ms": optimized["ttft_ms"],
                "decode_tok_per_s": optimized["decode_tok_per_s"],
                "latency_per_token_ms": optimized["latency_per_token_ms"],
                "e2e_ms": optimized["e2e_ms"],
                "peak_vram_mb": optimized.get("peak_vram_mb"),
                "allocated_vram_mb": optimized.get("allocated_vram_mb"),
                "quality_sanity_pass_rate": quality["verifier_pass_rate"],
            },
        },
        "quality_sanity": {
            "n": quality["n_examples"],
            "pass_rate": quality["verifier_pass_rate"],
            "per_task": quality["per_task"],
            "original_int8_quality": ORIGINAL_INT8["quality"],
        },
        "remaining_bottleneck": remaining,
        "deeper_kernel_backend_required": True,
        "h4_5_fusion": fusion,
        "variants": variants,
        "pass_fail": pass_fail,
        "pass_reasons": reasons,
        "decode_gain_pct_vs_h4_baseline": round(decode_gain_pct, 2),
    }

    (RESULTS_DIR / "h4_variants.json").write_text(json.dumps(variants, indent=2, default=str))
    (RESULTS_DIR / "h4_quality_sanity.json").write_text(json.dumps(quality, indent=2, default=str))
    (RESULTS_DIR / "h4_report.json").write_text(json.dumps(comparison, indent=2, default=str))
    CMP_PATH.write_text(json.dumps(comparison, indent=2, default=str))
    print(f"\nSaved {CMP_PATH}")
    print(f"Saved {RESULTS_DIR / 'h4_report.json'}")
    print(f"H.4 {pass_fail}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
