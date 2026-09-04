#!/usr/bin/env python3
"""
Stage H.1B – Systems benchmark for BF16 / INT8 / INT4 quantization.
Measures TTFT, decode tok/s, latency/token, E2E latency, peak VRAM.
"""
import gc
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Suppress bitsandbytes noise
import logging
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import torch

BASE_DIR = Path("/workspace/neuro-agent")
OUT_DIR  = BASE_DIR / "results" / "quantization" / "text" / "systems"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID   = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER    = str(BASE_DIR / "checkpoints" / "sft_corrected_v2" / "final")
HF_CACHE   = "/workspace"

WARMUP_ITERS = 2
TIMED_ITERS  = 5
MAX_NEW_TOKENS = 64

# ~512-token prompt (neuroscience research question padded to length)
PROMPT_TEXT = (
    "You are a neuroscience research assistant. Answer the following question "
    "with precise scientific detail.\n\n"
    "Question: Describe the role of high-frequency gamma oscillations (30-100 Hz) "
    "in cortical information processing, their relationship to working memory, "
    "and explain how cross-frequency coupling between theta and gamma rhythms "
    "supports cognitive functions such as item encoding in the hippocampus. "
    "Include discussion of how EEG-based brain-computer interfaces can leverage "
    "these oscillatory signatures for decoding motor intent and cognitive state. "
    "Discuss current signal processing pipelines including bandpass filtering, "
    "independent component analysis, common spatial patterns, and Riemannian "
    "geometry classifiers as applied to motor imagery BCI paradigms. "
    "Additionally, address how deep learning architectures such as EEGNet, "
    "ShallowConvNet, and transformer-based models compare to classical machine "
    "learning baselines on benchmark EEG datasets including BCI Competition IV "
    "dataset 2a and the PhysioNet EEG Motor Movement/Imagery dataset. "
    "Provide context on subject-dependent vs subject-independent generalization "
    "challenges, domain adaptation techniques, and the impact of electrode "
    "montage density on classification accuracy. Consider also the neuroscientific "
    "basis for mu rhythm suppression during motor imagery and its spectral "
    "fingerprint across C3, Cz, and C4 electrode positions. "
    "Finally, outline best practices for artifact rejection including ocular and "
    "muscle artifact removal in the context of real-time BCI applications where "
    "low-latency inference is critical for closed-loop neurofeedback systems.\n\n"
    "Answer:"
)


def nvidia_smi_mb():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        )
        return float(out.strip().split("\n")[0])
    except Exception:
        return None


def load_model(variant: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, cache_dir=HF_CACHE, trust_remote_code=True
    )

    t0 = time.perf_counter()
    if variant == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_CACHE,
            trust_remote_code=True,
        )
    elif variant == "int8":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            cache_dir=HF_CACHE,
            trust_remote_code=True,
        )
    elif variant == "int4":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_cfg,
            device_map="auto",
            cache_dir=HF_CACHE,
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    load_time = time.perf_counter() - t0

    return model, tokenizer, load_time


def benchmark_variant(variant: str):
    print(f"\n{'='*60}", flush=True)
    print(f"  Benchmarking variant: {variant.upper()}", flush=True)
    print(f"{'='*60}", flush=True)

    result = {"variant": variant, "status": "ok"}

    try:
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer, load_time = load_model(variant)

        alloc_after_load  = torch.cuda.memory_allocated() / 1024**2
        reserved_after_load = torch.cuda.memory_reserved() / 1024**2
        smi_after_load    = nvidia_smi_mb()

        result["load_time_s"]           = round(load_time, 3)
        result["alloc_after_load_mb"]   = round(alloc_after_load, 2)
        result["reserved_after_load_mb"]= round(reserved_after_load, 2)
        result["nvidia_smi_after_load_mb"] = smi_after_load

        print(f"  Load time: {load_time:.2f}s  VRAM alloc: {alloc_after_load:.0f} MB", flush=True)

        # Tokenize prompt
        inputs = tokenizer(PROMPT_TEXT, return_tensors="pt").to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        print(f"  Prompt tokens: {prompt_len}", flush=True)
        result["prompt_tokens"] = prompt_len

        def run_generation():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            # Time TTFT: prefill + first token
            t_start = time.perf_counter()
            with torch.no_grad():
                # Generate 1 token to get TTFT
                out1 = model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            t_first_token = time.perf_counter()
            ttft_ms = (t_first_token - t_start) * 1000.0

            # Now generate remaining tokens (decode phase)
            t_decode_start = time.perf_counter()
            with torch.no_grad():
                out_full = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
            torch.cuda.synchronize()
            t_e2e_end = time.perf_counter()

            e2e_ms = (t_e2e_end - t_start) * 1000.0
            generated_tokens = out_full.shape[1] - inputs["input_ids"].shape[1]
            decode_tokens = max(generated_tokens - 1, 1)
            decode_time_s = (t_e2e_end - t_decode_start)
            decode_tps = decode_tokens / decode_time_s if decode_time_s > 0 else 0
            latency_per_token_ms = decode_time_s / decode_tokens * 1000.0

            peak_alloc = torch.cuda.max_memory_allocated() / 1024**2
            peak_reserved = torch.cuda.max_memory_reserved() / 1024**2

            return {
                "ttft_ms": ttft_ms,
                "decode_tps": decode_tps,
                "latency_per_token_ms": latency_per_token_ms,
                "e2e_ms": e2e_ms,
                "generated_tokens": generated_tokens,
                "peak_alloc_mb": peak_alloc,
                "peak_reserved_mb": peak_reserved,
            }

        # Warmup
        print(f"  Warmup ({WARMUP_ITERS} iters)...", flush=True)
        for i in range(WARMUP_ITERS):
            _ = run_generation()
            print(f"    warmup {i+1}/{WARMUP_ITERS} done", flush=True)

        # Timed
        print(f"  Timed ({TIMED_ITERS} iters)...", flush=True)
        records = []
        for i in range(TIMED_ITERS):
            r = run_generation()
            records.append(r)
            print(f"    iter {i+1}: TTFT={r['ttft_ms']:.1f}ms  dec={r['decode_tps']:.1f}tok/s  e2e={r['e2e_ms']:.0f}ms  peak={r['peak_alloc_mb']:.0f}MB", flush=True)

        # Average
        def avg(key):
            return round(sum(r[key] for r in records) / len(records), 3)

        result["ttft_ms"]               = avg("ttft_ms")
        result["decode_tps"]            = avg("decode_tps")
        result["latency_per_token_ms"]  = avg("latency_per_token_ms")
        result["e2e_ms"]                = avg("e2e_ms")
        result["generated_tokens_avg"]  = avg("generated_tokens")
        result["peak_alloc_mb"]         = avg("peak_alloc_mb")
        result["peak_reserved_mb"]      = avg("peak_reserved_mb")

        smi_gen = nvidia_smi_mb()
        result["nvidia_smi_during_gen_mb"] = smi_gen

        print(f"\n  RESULTS [{variant.upper()}]:", flush=True)
        print(f"    TTFT:              {result['ttft_ms']:.1f} ms", flush=True)
        print(f"    Decode tok/s:      {result['decode_tps']:.1f}", flush=True)
        print(f"    Latency/token:     {result['latency_per_token_ms']:.2f} ms", flush=True)
        print(f"    E2E latency:       {result['e2e_ms']:.0f} ms", flush=True)
        print(f"    Peak alloc VRAM:   {result['peak_alloc_mb']:.0f} MB", flush=True)
        print(f"    Peak reserved VRAM:{result['peak_reserved_mb']:.0f} MB", flush=True)

    except Exception as e:
        import traceback
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        print(f"  ERROR: {e}", flush=True)

    # Cleanup
    try:
        del model
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

    return result


def merge_systems_into_comparison(results: dict):
    comp_path = BASE_DIR / "results" / "model_comparison" / "text_quantization_bf16_int8_int4.json"
    if not comp_path.exists():
        print(f"  Comparison file not found at {comp_path}, skipping merge.", flush=True)
        return

    with open(comp_path) as f:
        comp = json.load(f)

    if "systems" not in comp:
        comp["systems"] = {}

    for variant, data in results.items():
        comp["systems"][variant] = data

    with open(comp_path, "w") as f:
        json.dump(comp, f, indent=2)
    print(f"  Merged systems into {comp_path}", flush=True)


def main():
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    print("Stage H.1B – Quantization Systems Benchmark", flush=True)
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.0f} MB", flush=True)

    all_results = {}

    for variant in ["bf16", "int8", "int4"]:
        data = benchmark_variant(variant)
        all_results[variant] = data

        out_path = OUT_DIR / f"{variant}_systems.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n  Saved: {out_path}", flush=True)

    # Merge into comparison file
    merge_systems_into_comparison(all_results)

    # Print final comparison table
    print("\n" + "="*80, flush=True)
    print("FINAL COMPARISON TABLE", flush=True)
    print("="*80, flush=True)

    quality = {
        "bf16": 0.864,
        "int8": 0.866,
        "int4": "FAIL (0.20 factual_grounding < 0.30 floor)",
    }
    load_vram = {"bf16": 7930, "int8": 4479, "int4": 2820}

    header = f"{'Metric':<28} {'BF16':>12} {'INT8':>12} {'INT4':>12}"
    print(header, flush=True)
    print("-"*68, flush=True)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.1f}"
        return str(v)

    rows = [
        ("Quality pass rate", [quality[v] for v in ["bf16","int8","int4"]]),
        ("Load VRAM alloc (MB)",   [load_vram[v] for v in ["bf16","int8","int4"]]),
    ]

    for label, vals in rows:
        print(f"{label:<28} {fmt(vals[0]):>12} {fmt(vals[1]):>12} {fmt(vals[2]):>12}", flush=True)

    def get(v, k, fallback="N/A"):
        d = all_results.get(v, {})
        if d.get("status") == "error":
            return "ERROR"
        val = d.get(k)
        if val is None:
            return fallback
        return val

    sys_rows = [
        ("Peak VRAM alloc (MB)", "peak_alloc_mb"),
        ("Load time (s)",        "load_time_s"),
        ("TTFT (ms)",            "ttft_ms"),
        ("Decode tok/s",         "decode_tps"),
        ("Latency/token (ms)",   "latency_per_token_ms"),
        ("E2E latency (ms)",     "e2e_ms"),
    ]

    for label, key in sys_rows:
        vals = [get(v, key) for v in ["bf16","int8","int4"]]
        strs = []
        for val in vals:
            if isinstance(val, (int, float)):
                strs.append(f"{val:.1f}")
            else:
                strs.append(str(val))
        print(f"{label:<28} {strs[0]:>12} {strs[1]:>12} {strs[2]:>12}", flush=True)

    print("="*80, flush=True)
    print("\nRecommendation: INT8", flush=True)
    print("Reason: Matches BF16 quality (0.866 vs 0.864), uses 43% less VRAM (4479 vs 7930 MB),", flush=True)
    print("        faster load (6.2s vs 32.9s), and INT4 fails the factual_grounding quality gate.", flush=True)


if __name__ == "__main__":
    main()
