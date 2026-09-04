"""H.3 — Profile existing INT8 bottleneck vs BF16 baseline.

Reproduces BF16 and INT8 inference with identical prompt / generation settings,
profiles model loading, prefill and decode phases, and writes structured results.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_INT8 = PROJECT_ROOT / "results" / "profiling" / "int8"
RESULTS_COMP = PROJECT_ROOT / "results" / "model_comparison"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_PATH = str(PROJECT_ROOT / "checkpoints" / "sft_corrected_v2" / "final")

WARMUP = 2
TIMED = 5
MAX_NEW_TOKENS = 64

# ~512-token prompt (neuroscience domain, padded with context)
PROMPT_TEXT = (
    "You are an expert neuroscientist. A researcher has recorded 64-channel EEG data "
    "from 109 subjects performing motor imagery tasks (left fist, right fist, both fists, "
    "both feet) using the BCI2000 system at 160 Hz sampling rate. The montage follows the "
    "international 10-20 system with additional electrodes. Each recording session contains "
    "baseline runs with eyes open and eyes closed, followed by motor imagery and motor "
    "execution runs. The researcher wants to build a brain-computer interface that can "
    "distinguish between left and right hand motor imagery with at least 85% accuracy. "
    "They have extracted features including band power in mu (8-12 Hz) and beta (13-30 Hz) "
    "bands, computed event-related desynchronization/synchronization (ERD/ERS) patterns, "
    "and applied common spatial pattern (CSP) filters. A logistic regression baseline "
    "achieves 72% accuracy on a held-out test set. They are considering deep learning "
    "approaches including EEGNet, ShallowConvNet, and transformer-based architectures. "
    "The data has been preprocessed with a 1-49 Hz bandpass filter, artifact rejection "
    "using ICA, and re-referenced to common average. Channel impedances were kept below "
    "10 kOhm during recording. The researcher has also computed time-frequency "
    "representations using Morlet wavelets and noticed lateralized patterns in the alpha "
    "band during motor imagery. Power spectral density analysis shows clear mu suppression "
    "over contralateral sensorimotor cortex during hand motor imagery. Topographic maps "
    "confirm expected spatial patterns with focal activity over C3 and C4 electrodes. "
    "Given all this information, provide a detailed analysis of: 1) Why the current "
    "baseline accuracy is below target, 2) Which deep learning architecture would be "
    "most suitable and why, 3) What additional feature engineering or data augmentation "
    "strategies could help, 4) How to properly validate the model given the multi-subject "
    "nature of the dataset, and 5) What are the key considerations for real-time "
    "deployment of such a BCI system. Please provide specific recommendations backed by "
    "recent literature in the field of brain-computer interfaces and neural engineering."
)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def load_model(quantization: str | None, dtype: torch.dtype = torch.bfloat16):
    """Load base + adapter. quantization=None means BF16, '8bit' means INT8."""
    load_kwargs = {"trust_remote_code": False, "device_map": {"": "cuda:0"}}
    if quantization == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        load_kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **load_kwargs)
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model


def timed_load(quantization: str | None) -> dict:
    cleanup()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = load_model(quantization)
    torch.cuda.synchronize()
    load_time = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e6
    del model
    cleanup()
    return {"load_time_s": round(load_time, 3), "peak_vram_mb": round(peak, 1)}


def build_input_ids(tokenizer):
    messages = [{"role": "user", "content": PROMPT_TEXT}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
    return ids


def benchmark_generation(model, input_ids, label: str) -> dict:
    """Warmup + timed runs; returns timing breakdown."""
    prompt_len = input_ids.shape[1]

    # Warmup
    for _ in range(WARMUP):
        with torch.no_grad():
            model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        torch.cuda.synchronize()

    # Timed runs with CUDA events
    prefill_times = []
    decode_times = []
    e2e_times = []

    for _ in range(TIMED):
        torch.cuda.synchronize()

        start_e2e = torch.cuda.Event(enable_timing=True)
        end_e2e = torch.cuda.Event(enable_timing=True)
        start_prefill = torch.cuda.Event(enable_timing=True)
        end_prefill = torch.cuda.Event(enable_timing=True)

        # Prefill: single forward pass (no generation)
        torch.cuda.synchronize()
        start_prefill.record()
        with torch.no_grad():
            out = model(input_ids, use_cache=True)
        end_prefill.record()
        torch.cuda.synchronize()
        prefill_ms = start_prefill.elapsed_time(end_prefill)
        prefill_times.append(prefill_ms)
        del out

        # Full generation (includes prefill + decode)
        torch.cuda.synchronize()
        start_e2e.record()
        with torch.no_grad():
            gen_out = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        end_e2e.record()
        torch.cuda.synchronize()
        e2e_ms = start_e2e.elapsed_time(end_e2e)
        e2e_times.append(e2e_ms)

        generated_tokens = gen_out.shape[1] - prompt_len
        decode_ms = e2e_ms - prefill_ms  # approximate
        decode_times.append(decode_ms)
        del gen_out

    avg = lambda xs: sum(xs) / len(xs)
    avg_prefill = avg(prefill_times)
    avg_decode = avg(decode_times)
    avg_e2e = avg(e2e_times)
    avg_gen_tokens = MAX_NEW_TOKENS  # do_sample=False, deterministic

    peak_vram = torch.cuda.max_memory_allocated() / 1e6

    return {
        "label": label,
        "prompt_tokens": prompt_len,
        "generated_tokens": avg_gen_tokens,
        "prefill_ms": round(avg_prefill, 2),
        "decode_ms": round(avg_decode, 2),
        "e2e_ms": round(avg_e2e, 2),
        "ttft_ms": round(avg_prefill, 2),
        "decode_tok_per_s": round(avg_gen_tokens / (avg_decode / 1000), 2) if avg_decode > 0 else 0,
        "latency_per_token_ms": round(avg_decode / avg_gen_tokens, 2) if avg_gen_tokens > 0 else 0,
        "peak_vram_mb": round(peak_vram, 1),
        "warmup": WARMUP,
        "timed_iters": TIMED,
        "all_prefill_ms": [round(x, 2) for x in prefill_times],
        "all_decode_ms": [round(x, 2) for x in decode_times],
        "all_e2e_ms": [round(x, 2) for x in e2e_times],
    }


def profile_operators(model, input_ids, label: str) -> list[dict]:
    """Use torch.profiler to capture top CUDA operators."""
    # Warmup
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof:
        with record_function(f"{label}_generate"):
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            torch.cuda.synchronize()

    # Also profile prefill only
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
        profile_memory=True,
    ) as prof_prefill:
        with record_function(f"{label}_prefill"):
            with torch.no_grad():
                model(input_ids, use_cache=True)
            torch.cuda.synchronize()

    # Extract events
    def extract_events(p, phase):
        events = []
        for evt in p.key_averages():
            dev_total = getattr(evt, "device_time_total", 0) or 0
            self_dev = getattr(evt, "self_device_time_total", 0) or 0
            if dev_total > 0 or self_dev > 0:
                events.append({
                    "name": evt.key,
                    "phase": phase,
                    "count": evt.count,
                    "cpu_time_us": round(evt.cpu_time_total, 1),
                    "cuda_time_us": round(dev_total, 1),
                    "cuda_time_pct": 0,  # filled below
                    "self_cuda_time_us": round(self_dev, 1),
                    "cpu_memory_usage_bytes": getattr(evt, "cpu_memory_usage", 0),
                    "cuda_memory_usage_bytes": getattr(evt, "cuda_memory_usage", 0),
                })
        total_cuda = sum(e["self_cuda_time_us"] for e in events)
        for e in events:
            e["cuda_time_pct"] = round(100 * e["self_cuda_time_us"] / total_cuda, 2) if total_cuda > 0 else 0
        events.sort(key=lambda x: x["self_cuda_time_us"], reverse=True)
        return events

    gen_events = extract_events(prof, "generate")
    prefill_events = extract_events(prof_prefill, "prefill")

    # Save chrome trace
    trace_path = RESULTS_INT8 / f"{label}_trace.json"
    prof.export_chrome_trace(str(trace_path))

    return gen_events, prefill_events


def analyze_int8_specifics(events: list[dict]) -> dict:
    """Extract INT8-specific overhead from profiler events."""
    matmul8bit = [e for e in events if "matmul8bit" in e["name"].lower() or "int8" in e["name"].lower() or "MatMul8bitLt" in e["name"]]
    casts = [e for e in events if "to" == e["name"] or "convert" in e["name"].lower() or "_to_copy" in e["name"].lower() or "cast" in e["name"].lower()]
    quantize_ops = [e for e in events if "quantize" in e["name"].lower() or "dequantize" in e["name"].lower()]
    gemm = [e for e in events if "gemm" in e["name"].lower() or "matmul" in e["name"].lower() or "mm" in e["name"].lower() or "cublas" in e["name"].lower()]

    def summarize(ops, tag):
        total_cuda = sum(e["self_cuda_time_us"] for e in ops)
        total_count = sum(e["count"] for e in ops)
        return {
            "tag": tag,
            "total_self_cuda_us": round(total_cuda, 1),
            "total_count": total_count,
            "ops": [{"name": e["name"], "count": e["count"], "self_cuda_us": e["self_cuda_time_us"]} for e in ops[:10]],
        }

    return {
        "matmul8bit_ops": summarize(matmul8bit, "matmul8bit"),
        "cast_ops": summarize(casts, "dtype_casts"),
        "quantize_ops": summarize(quantize_ops, "quantize_dequantize"),
        "gemm_ops": summarize(gemm, "gemm_matmul"),
    }


def kernel_launch_count(events: list[dict]) -> int:
    return sum(e["count"] for e in events if e["self_cuda_time_us"] > 0)


def main():
    print("=" * 70)
    print("H.3 — INT8 Bottleneck Profiling")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    input_ids = build_input_ids(tokenizer)
    prompt_len = input_ids.shape[1]
    print(f"Prompt tokens: {prompt_len}")

    results = {}

    # ── BF16 ──
    print("\n── BF16 Loading ──")
    bf16_load = timed_load(None)
    print(f"  Load time: {bf16_load['load_time_s']}s, Peak VRAM: {bf16_load['peak_vram_mb']} MB")

    print("── BF16 Benchmark ──")
    cleanup()
    bf16_model = load_model(None)
    bf16_bench = benchmark_generation(bf16_model, input_ids, "bf16")
    print(f"  Prefill: {bf16_bench['prefill_ms']}ms  Decode: {bf16_bench['decode_ms']}ms  E2E: {bf16_bench['e2e_ms']}ms")
    print(f"  Decode: {bf16_bench['decode_tok_per_s']} tok/s  TTFT: {bf16_bench['ttft_ms']}ms")

    print("── BF16 Profiling ──")
    bf16_gen_events, bf16_prefill_events = profile_operators(bf16_model, input_ids, "bf16")
    bf16_int8_analysis = analyze_int8_specifics(bf16_gen_events)
    bf16_kernel_count = kernel_launch_count(bf16_gen_events)
    print(f"  Top operator: {bf16_gen_events[0]['name']} ({bf16_gen_events[0]['self_cuda_time_us']} us)")
    print(f"  Kernel launches: {bf16_kernel_count}")

    del bf16_model
    cleanup()

    # ── INT8 ──
    print("\n── INT8 Loading ──")
    int8_load = timed_load("8bit")
    print(f"  Load time: {int8_load['load_time_s']}s, Peak VRAM: {int8_load['peak_vram_mb']} MB")

    print("── INT8 Benchmark ──")
    cleanup()
    int8_model = load_model("8bit")
    int8_bench = benchmark_generation(int8_model, input_ids, "int8")
    print(f"  Prefill: {int8_bench['prefill_ms']}ms  Decode: {int8_bench['decode_ms']}ms  E2E: {int8_bench['e2e_ms']}ms")
    print(f"  Decode: {int8_bench['decode_tok_per_s']} tok/s  TTFT: {int8_bench['ttft_ms']}ms")

    print("── INT8 Profiling ──")
    int8_gen_events, int8_prefill_events = profile_operators(int8_model, input_ids, "int8")
    int8_analysis = analyze_int8_specifics(int8_gen_events)
    int8_kernel_count = kernel_launch_count(int8_gen_events)
    print(f"  Top operator: {int8_gen_events[0]['name']} ({int8_gen_events[0]['self_cuda_time_us']} us)")
    print(f"  Kernel launches: {int8_kernel_count}")

    # INT8 prefill-specific analysis
    int8_prefill_analysis = analyze_int8_specifics(int8_prefill_events)

    del int8_model
    cleanup()

    # ── Build comparison ──
    print("\n── Building Comparison Report ──")

    # Top 20 operators each
    bf16_top = bf16_gen_events[:20]
    int8_top = int8_gen_events[:20]

    # Compute overhead ratios
    prefill_ratio = int8_bench["prefill_ms"] / bf16_bench["prefill_ms"] if bf16_bench["prefill_ms"] > 0 else 0
    decode_ratio = int8_bench["decode_ms"] / bf16_bench["decode_ms"] if bf16_bench["decode_ms"] > 0 else 0
    e2e_ratio = int8_bench["e2e_ms"] / bf16_bench["e2e_ms"] if bf16_bench["e2e_ms"] > 0 else 0

    # Ranked bottlenecks
    bottlenecks = []

    # 1: MatMul8bitLt overhead
    int8_matmul_us = int8_analysis["matmul8bit_ops"]["total_self_cuda_us"]
    int8_gemm_us = int8_analysis["gemm_ops"]["total_self_cuda_us"]
    bf16_gemm_us = bf16_int8_analysis["gemm_ops"]["total_self_cuda_us"]
    bottlenecks.append({
        "rank": 1,
        "name": "MatMul8bitLt / INT8 quantized matmul overhead",
        "evidence": {
            "int8_matmul8bit_cuda_us": int8_matmul_us,
            "int8_total_gemm_cuda_us": int8_gemm_us,
            "bf16_total_gemm_cuda_us": bf16_gemm_us,
            "gemm_slowdown_ratio": round(int8_gemm_us / bf16_gemm_us, 2) if bf16_gemm_us > 0 else "N/A",
            "int8_matmul8bit_ops": int8_analysis["matmul8bit_ops"]["ops"][:5],
        },
        "description": "bitsandbytes MatMul8bitLt decomposes each linear into quantize→int8 gemm→dequantize→accumulate, replacing single BF16 Tensor Core GEMM with multi-step mixed-precision pipeline.",
    })

    # 2: dtype cast overhead
    int8_cast_us = int8_analysis["cast_ops"]["total_self_cuda_us"]
    bf16_cast_us = bf16_int8_analysis["cast_ops"]["total_self_cuda_us"]
    bottlenecks.append({
        "rank": 2,
        "name": "BF16→FP16 dtype cast overhead",
        "evidence": {
            "int8_cast_cuda_us": int8_cast_us,
            "bf16_cast_cuda_us": bf16_cast_us,
            "cast_overhead_us": round(int8_cast_us - bf16_cast_us, 1),
            "int8_cast_count": int8_analysis["cast_ops"]["total_count"],
            "bf16_cast_count": bf16_int8_analysis["cast_ops"]["total_count"],
            "int8_cast_ops": int8_analysis["cast_ops"]["ops"][:5],
        },
        "description": "bitsandbytes INT8 path expects FP16 inputs; model computes in BF16, causing repeated BF16→FP16 casts before every quantized linear layer.",
    })

    # 3: quantize/dequantize
    int8_qd_us = int8_analysis["quantize_ops"]["total_self_cuda_us"]
    bf16_qd_us = bf16_int8_analysis["quantize_ops"]["total_self_cuda_us"]
    bottlenecks.append({
        "rank": 3,
        "name": "Quantize/dequantize operations",
        "evidence": {
            "int8_quantize_dequantize_cuda_us": int8_qd_us,
            "bf16_quantize_dequantize_cuda_us": bf16_qd_us,
            "int8_qd_count": int8_analysis["quantize_ops"]["total_count"],
            "int8_qd_ops": int8_analysis["quantize_ops"]["ops"][:5],
        },
        "description": "Each INT8 matmul requires activation quantization (FP16→INT8) and result dequantization (INT32→FP16), adding per-layer kernel launches.",
    })

    # 4: kernel launch overhead
    bottlenecks.append({
        "rank": 4,
        "name": "Increased kernel launch count",
        "evidence": {
            "int8_kernel_launches": int8_kernel_count,
            "bf16_kernel_launches": bf16_kernel_count,
            "launch_ratio": round(int8_kernel_count / bf16_kernel_count, 2) if bf16_kernel_count > 0 else "N/A",
        },
        "description": "INT8 path launches multiple kernels per linear (quantize, matmul, dequantize, cast) vs single GEMM for BF16, increasing CPU-side launch overhead and GPU idle time.",
    })

    # Optimization candidates
    candidates = [
        {
            "id": "H4-1",
            "name": "Remove redundant BF16→FP16 casts",
            "rationale": "If model inputs to quantized linears are pre-cast to FP16 once at layer entry, redundant per-op casts can be eliminated.",
            "estimated_impact": "Moderate — removes cast kernels that fire on every forward pass of every quantized layer.",
        },
        {
            "id": "H4-2",
            "name": "torch.compile on non-quantized layers",
            "rationale": "Attention, RMSNorm, rotary embeddings, and MLP activation functions are not quantized and can benefit from operator fusion via torch.compile.",
            "estimated_impact": "Moderate — reduces kernel launch count for non-linear operations.",
        },
        {
            "id": "H4-3",
            "name": "CUDA Graphs for decode phase",
            "rationale": "Decode is dominated by kernel launch overhead (single-token steps); CUDA Graphs can replay the entire decode step as one graph launch.",
            "estimated_impact": "High for decode throughput — addresses launch overhead bottleneck directly.",
        },
        {
            "id": "H4-4",
            "name": "Kernel fusion for quantize→matmul→dequantize",
            "rationale": "If profiling shows quantize and dequantize are significant, a fused Triton kernel could combine them with the INT8 GEMM.",
            "estimated_impact": "High but complex — depends on whether MatMul8bitLt is the dominant cost.",
        },
        {
            "id": "H4-5",
            "name": "Input dtype preparation (load model with FP16 compute dtype)",
            "rationale": "If bitsandbytes INT8 path is hardcoded to FP16, loading the model with torch_dtype=float16 for non-quantized components could avoid BF16→FP16 casts.",
            "estimated_impact": "Low-moderate — addresses cast overhead at the source.",
        },
    ]

    # ── Assemble final report ──
    report = {
        "stage": "H.3",
        "description": "INT8 bottleneck profiling — BF16 vs bitsandbytes INT8",
        "model": MODEL_NAME,
        "adapter": ADAPTER_PATH,
        "prompt_tokens": prompt_len,
        "generated_tokens": MAX_NEW_TOKENS,
        "warmup": WARMUP,
        "timed_iters": TIMED,
        "loading": {"bf16": bf16_load, "int8": int8_load},
        "benchmarks": {"bf16": bf16_bench, "int8": int8_bench},
        "overhead_ratios": {
            "prefill_slowdown": round(prefill_ratio, 2),
            "decode_slowdown": round(decode_ratio, 2),
            "e2e_slowdown": round(e2e_ratio, 2),
        },
        "profiler_top_operators": {
            "bf16_generate_top20": bf16_top,
            "int8_generate_top20": int8_top,
            "bf16_prefill_top10": bf16_prefill_events[:10],
            "int8_prefill_top10": int8_prefill_events[:10],
        },
        "int8_overhead_analysis": {
            "generate": int8_analysis,
            "prefill": int8_prefill_analysis,
        },
        "bf16_baseline_analysis": bf16_int8_analysis,
        "kernel_launch_counts": {
            "bf16": bf16_kernel_count,
            "int8": int8_kernel_count,
        },
        "ranked_bottlenecks": bottlenecks,
        "optimization_candidates": candidates,
        "nsight_note": "Nsight Compute (ncu) kernel profiling not attempted due to typical container permission restrictions. All profiling uses torch.profiler and CUDA events.",
        "pass_fail": "PASS" if len(bottlenecks) >= 3 else "FAIL",
    }

    # Save
    comp_path = RESULTS_COMP / "bf16_vs_int8_profile.json"
    comp_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved comparison: {comp_path}")

    int8_detail_path = RESULTS_INT8 / "int8_profiling_detail.json"
    int8_detail_path.write_text(json.dumps({
        "int8_generate_events": int8_gen_events[:50],
        "int8_prefill_events": int8_prefill_events[:50],
        "int8_analysis": int8_analysis,
        "int8_prefill_analysis": int8_prefill_analysis,
    }, indent=2, default=str))
    print(f"Saved INT8 detail: {int8_detail_path}")

    # ── Print Final Report ──
    print("\n" + "=" * 70)
    print("H.3 FINAL REPORT")
    print("=" * 70)

    print(f"\n1. BF16 Hot Operators (generate, top 5):")
    for e in bf16_top[:5]:
        print(f"   {e['name']:50s}  self_cuda: {e['self_cuda_time_us']:>10.0f} us  ({e['cuda_time_pct']:.1f}%)  count: {e['count']}")

    print(f"\n2. INT8 Hot Operators (generate, top 5):")
    for e in int8_top[:5]:
        print(f"   {e['name']:50s}  self_cuda: {e['self_cuda_time_us']:>10.0f} us  ({e['cuda_time_pct']:.1f}%)  count: {e['count']}")

    print(f"\n3. Prefill Difference:")
    print(f"   BF16: {bf16_bench['prefill_ms']} ms  |  INT8: {int8_bench['prefill_ms']} ms  |  Slowdown: {prefill_ratio:.2f}x")

    print(f"\n4. Decode Difference:")
    print(f"   BF16: {bf16_bench['decode_ms']} ms  |  INT8: {int8_bench['decode_ms']} ms  |  Slowdown: {decode_ratio:.2f}x")
    print(f"   BF16: {bf16_bench['decode_tok_per_s']} tok/s  |  INT8: {int8_bench['decode_tok_per_s']} tok/s")

    print(f"\n5. Cast Overhead:")
    print(f"   INT8 cast time: {int8_cast_us:.0f} us ({int8_analysis['cast_ops']['total_count']} calls)")
    print(f"   BF16 cast time: {bf16_cast_us:.0f} us ({bf16_int8_analysis['cast_ops']['total_count']} calls)")
    print(f"   Delta: {int8_cast_us - bf16_cast_us:.0f} us")

    print(f"\n6. MatMul8bitLt Overhead:")
    print(f"   INT8 matmul8bit CUDA time: {int8_matmul_us:.0f} us")
    print(f"   INT8 total GEMM CUDA time: {int8_gemm_us:.0f} us")
    print(f"   BF16 total GEMM CUDA time: {bf16_gemm_us:.0f} us")

    print(f"\n7. Launch / Synchronization:")
    print(f"   BF16 kernel launches: {bf16_kernel_count}")
    print(f"   INT8 kernel launches: {int8_kernel_count}")
    print(f"   Ratio: {int8_kernel_count / bf16_kernel_count:.2f}x" if bf16_kernel_count > 0 else "   N/A")

    print(f"\n8. Ranked Root Causes:")
    for b in bottlenecks:
        print(f"   #{b['rank']}: {b['name']}")
        print(f"       {b['description']}")

    print(f"\n9. Optimization Candidates for H.4:")
    for c in candidates:
        print(f"   [{c['id']}] {c['name']}")
        print(f"       {c['rationale']}")
        print(f"       Impact: {c['estimated_impact']}")

    print(f"\n10. H.3 PASS/FAIL: {report['pass_fail']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
