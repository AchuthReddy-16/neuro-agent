#!/usr/bin/env python3
"""H.7E/H.7F — vLLM W8A8 INT8 smoke + single-request systems benchmark.

No LoRA at runtime, no bitsandbytes, no prefix cache, concurrency=1.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT = PROJECT_ROOT / "checkpoints/text_w8a8_int8_compressed"
OUT_DIR = PROJECT_ROOT / "results/quantization/w8a8_int8"
SMOKE_PATH = OUT_DIR / "vllm_smoke.json"
BENCH_PATH = OUT_DIR / "single_request_benchmark.json"

NUM_WARMUPS = 2
NUM_TIMED = 5
MAX_TOKENS = 64

BENCH_PROMPT = (
    "You are a neuroscience research assistant. "
    "A researcher presents the following EEG study scenario:\n\n"
    "Study design: 64-channel EEG recorded at 1000 Hz from 32 participants "
    "performing a motor imagery task (left fist, right fist, both fists, rest). "
    "Data were band-pass filtered 0.1–40 Hz, epoched −200 to 800 ms around cue onset, "
    "baseline corrected using the pre-stimulus interval, and rejected for EOG artifacts "
    "exceeding ±100 µV. Independent component analysis removed cardiac and ocular "
    "artifacts. Source localisation used LORETA with a standard 3-shell spherical head "
    "model. Spectral analysis focused on mu (8–12 Hz) and beta (13–30 Hz) bands.\n\n"
    "The researcher asks:\n"
    "1. What neural mechanisms underlie event-related desynchronisation (ERD) and "
    "event-related synchronisation (ERS) during motor imagery?\n"
    "2. Which cortical areas are expected to show maximal ERD during left-hand vs "
    "right-hand imagery, and why is the pattern contralateral?\n"
    "3. How does beta rebound after movement/imagery termination relate to cortical "
    "inhibition and idling rhythms?\n"
    "4. What are the limitations of LORETA for EEG source localisation, and how do "
    "they affect interpretation of the spatial patterns?\n"
    "5. If the goal is BCI classification, which features derived from this paradigm "
    "are most discriminative, and what classifiers perform best in the literature?\n\n"
    "Please provide a detailed, evidence-based response covering all five questions. "
    "Cite specific frequency bands, brain regions (using standard anatomical terminology), "
    "and note any methodological caveats the researcher should be aware of.\n\n"
    "Response:"
)


def nvidia_smi() -> dict:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used, util = out.split(",")
        return {"nvidia_smi_mb": float(used), "gpu_util_pct": float(util)}
    except Exception as exc:
        return {"nvidia_smi_error": str(exc)}


def main() -> None:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    smoke: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CKPT),
        "enable_lora": False,
        "quantization_arg": None,
    }

    cfg = json.loads((CKPT / "config.json").read_text())
    qcfg = cfg.get("quantization_config") or cfg.get("compression_config") or {}
    smoke["checkpoint_quantization_config"] = qcfg
    quant_method = str(qcfg.get("quant_method") or qcfg.get("format") or "")
    smoke["quant_method"] = quant_method
    if "compressed-tensors" not in quant_method and "compressed_tensors" not in json.dumps(qcfg):
        smoke["status"] = "FAIL"
        smoke["reason"] = "checkpoint missing compressed-tensors quant_method"
        SMOKE_PATH.write_text(json.dumps(smoke, indent=2, default=str))
        print(json.dumps(smoke, indent=2, default=str))
        sys.exit(2)

    tokenizer = AutoTokenizer.from_pretrained(str(CKPT))
    prompt_ids = tokenizer(BENCH_PROMPT, add_special_tokens=False)["input_ids"]
    smoke["bench_prompt_tokens"] = len(prompt_ids)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    llm = LLM(
        model=str(CKPT),
        quantization=None,
        enable_lora=False,
        dtype="auto",
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        enable_prefix_caching=False,
        tensor_parallel_size=1,
        trust_remote_code=False,
        enforce_eager=True,
    )
    load_s = time.perf_counter() - t_load
    smi_load = nvidia_smi()
    allocated_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved_mb = torch.cuda.memory_reserved() / (1024 * 1024)

    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    quant_cfg = None
    try:
        vcfg = llm.llm_engine.vllm_config
        quant_cfg = str(vcfg.quant_config)
        quant_cfg_cls = type(vcfg.quant_config).__name__
    except Exception:
        quant_cfg_cls = None

    sampling = SamplingParams(temperature=0.0, max_tokens=32)
    outs = llm.generate(["Say the word neuroscience."], sampling_params=sampling)
    text = outs[0].outputs[0].text
    ntok = len(outs[0].outputs[0].token_ids)

    # Detect accidental bitsandbytes / LoRA
    import gc

    bnb_modules = []
    try:
        runner = llm.llm_engine.model_executor.driver_worker.model_runner
        model_obj = runner.model
        for name, mod in model_obj.named_modules():
            cls = type(mod).__name__
            if "8bit" in cls.lower() or "bitsandbytes" in cls.lower() or "Linear8bit" in cls:
                bnb_modules.append((name, cls))
    except Exception as exc:
        smoke["module_scan_error"] = str(exc)

    smoke.update(
        {
            "status": "ok" if text.strip() and not bnb_modules else "FAIL",
            "load_time_s": round(load_s, 2),
            "generation_ok": bool(text.strip()),
            "sample_text": text[:400],
            "sample_tokens": ntok,
            "vllm_quant_config": quant_cfg,
            "vllm_quant_config_class": quant_cfg_cls,
            "bitsandbytes_modules": bnb_modules,
            "lora_applied_at_runtime": False,
            "allocated_after_load_mb": round(allocated_mb, 1),
            "reserved_after_load_mb": round(reserved_mb, 1),
            "nvidia_smi_after_load": smi_load,
            "silent_bf16_fallback": False,
        }
    )
    if bnb_modules:
        smoke["status"] = "FAIL"
        smoke["reason"] = "bitsandbytes modules present"
    if quant_cfg_cls and "CompressedTensors" not in str(quant_cfg_cls) and "compressed" not in str(quant_cfg).lower():
        smoke["status"] = "FAIL"
        smoke["reason"] = f"unexpected quant config {quant_cfg_cls}"

    SMOKE_PATH.write_text(json.dumps(smoke, indent=2, default=str))
    print(json.dumps({k: smoke[k] for k in smoke if k != "checkpoint_quantization_config"}, indent=2, default=str))
    if smoke["status"] != "ok":
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        sys.exit(2)

    # H.7F single-request bench
    sampling64 = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    def run_once():
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        outputs = llm.generate([BENCH_PROMPT], sampling_params=sampling64)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = outputs[0]
        n_tokens = len(out.outputs[0].token_ids)
        e2e_ms = (t1 - t0) * 1000
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        return e2e_ms, n_tokens, peak_mb, out.outputs[0].text

    print(f"Warmups {NUM_WARMUPS}...", flush=True)
    for _ in range(NUM_WARMUPS):
        run_once()

    e2e_list, peak_list, token_counts = [], [], []
    utils = []
    print(f"Timed {NUM_TIMED}...", flush=True)
    for i in range(NUM_TIMED):
        e2e_ms, n_tok, peak, _gen = run_once()
        smi = nvidia_smi()
        e2e_list.append(e2e_ms)
        peak_list.append(peak)
        token_counts.append(n_tok)
        utils.append(smi.get("gpu_util_pct"))
        print(f"  iter {i+1}: {e2e_ms:.0f}ms {n_tok} tok peak={peak:.0f}MB", flush=True)

    avg_e2e = sum(e2e_list) / len(e2e_list)
    avg_peak = sum(peak_list) / len(peak_list)
    avg_tokens = sum(token_counts) / len(token_counts)
    decode_tok_s = avg_tokens / (avg_e2e / 1000) if avg_e2e > 0 else 0
    latency_per_token_ms = avg_e2e / avg_tokens if avg_tokens else 0
    model_vram = smoke["allocated_after_load_mb"]

    bench = {
        "backend": "vllm_w8a8_int8_compressed_tensors",
        "checkpoint": str(CKPT),
        "prompt_tokens": smoke["bench_prompt_tokens"],
        "max_new_tokens": MAX_TOKENS,
        "num_warmups": NUM_WARMUPS,
        "num_timed": NUM_TIMED,
        "concurrency": 1,
        "prefix_caching": False,
        "enable_lora": False,
        "load_time_s": smoke["load_time_s"],
        "model_vram_mb": round(model_vram, 1),
        "peak_vram_mb": round(avg_peak, 1),
        "peak_vram_gb": round(avg_peak / 1024, 3),
        "e2e_ms": round(avg_e2e, 1),
        "generated_tokens_avg": round(avg_tokens, 1),
        "decode_tok_per_s": round(decode_tok_s, 2),
        "latency_per_token_ms": round(latency_per_token_ms, 2),
        "ttft_ms": None,
        "ttft_note": "offline LLM.generate does not expose TTFT; E2E includes prefill+decode",
        "gpu_util_pct_samples": utils,
        "e2e_all_ms": [round(x, 1) for x in e2e_list],
        "failures": 0,
        "kernel_path": {
            "quant_config_class": quant_cfg_cls,
            "expected": "CompressedTensorsW8A8Int8 -> CutlassInt8ScaledMMLinearKernel",
        },
        "comparisons_existing": {
            "hf_bf16": {"quality": 0.864, "peak_gb": 8.16, "decode_tok_s": 52.74, "e2e_ms": 1264.5},
            "hf_bnb_int8": {"quality": 0.866, "peak_gb": 4.71, "decode_tok_s": 12.5, "e2e_ms": 5450},
            "h4_fair_bnb_int8": {"peak_gb": 4.70, "decode_tok_s": 18.73, "e2e_ms": 3497},
            "vllm_bf16": {"peak_gb": 8.24, "decode_tok_s": 61.3, "e2e_ms": 1043.8},
            "vllm_fp8_reference_only": {"peak_gb": 5.18, "decode_tok_s": 54.1, "e2e_ms": 1182.5},
        },
    }
    BENCH_PATH.write_text(json.dumps(bench, indent=2))
    print(json.dumps(bench, indent=2))

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    print("H7EF_OK")


if __name__ == "__main__":
    main()
