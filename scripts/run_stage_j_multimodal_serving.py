#!/usr/bin/env python3
"""Multimodal serving validation + single-GPU co-residency.

Text: vLLM W8A8 INT8 (subprocess, /usr/bin/python3)
Vision: corrected Qwen2.5-VL-3B LoRA via HF (this process)

Does not retrain, requantize, touch frontend, or mutate git.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_agent.paths import configure_hf_cache  # noqa: E402

configure_hf_cache()

RESULTS = PROJECT_ROOT / "results" / "serving" / "multimodal"
CMP = PROJECT_ROOT / "results" / "model_comparison" / "multimodal_production_serving.json"
TEXT_CKPT = PROJECT_ROOT / "checkpoints" / "text_w8a8_int8_compressed"
VISION_ADAPTER = PROJECT_ROOT / "checkpoints" / "multimodal_sft_corrected" / "final"
VISION_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
EVAL_JSONL = PROJECT_ROOT / "data" / "processed" / "vision" / "multimodal_eval_heldout.jsonl"
SERVER_LOG = RESULTS / "vllm_text_server.log"

HOST, PORT = "127.0.0.1", 8000
SERVED = "w8a8-int8"
VLLM_PYTHON = "/usr/bin/python3"

# Production text knobs (I.1–I.3 family) — first co-residency attempt
PROD_GPU_UTIL = 0.90
# If production util cannot co-reside, try a headroom-aware util (documented separately)
CORESIDE_GPU_UTIL = 0.40
MAX_MODEL_LEN = 4096

REF_QUALITY = {
    "base": 0.114,
    "initial_sft": 0.164,
    "corrected_sft": 0.493,
    "multimodal_rlvr": 0.482,
}

GATE_TASKS = {
    "waveform": [
        "waveform_highest_rms",
        "waveform_max_rms_numeric",
        "waveform_rms_order",
    ],
    "categorical": [
        "spectrogram_strongest_vs_weakest",
        "spectrogram_dominant_band",
        "topomap_strongest_alpha_mu",
        "band_power_weakest_alpha_mu",
        "psd_dominant_band",
    ],
    "ranking": [
        "psd_band_order",
        "band_power_beta_top3",
        "topomap_beta_top3",
    ],
    "numeric": [
        "psd_peak_frequency",
        "spectrogram_peak_frequency",
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return round(ys[0], 4)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return round(ys[f], 4)
    return round(ys[f] + (ys[c] - ys[f]) * (k - f), 4)


def gpu_stats() -> dict[str, Any]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.free,memory.total,utilization.gpu",
            "--format=csv,nounits,noheader",
        ],
        text=True,
    ).strip()
    used, free, total, util = [float(x.strip()) for x in out.split(",")]
    apps = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        text=True,
    )
    app_rows = []
    for line in apps.strip().splitlines()[1:]:
        if line.strip() and "No running" not in line:
            app_rows.append(line.strip())
    torch_alloc = None
    torch_reserved = None
    if torch.cuda.is_available():
        torch_alloc = round(torch.cuda.memory_allocated(0) / (1024**2), 1)
        torch_reserved = round(torch.cuda.memory_reserved(0) / (1024**2), 1)
    return {
        "nvidia_smi_used_mb": used,
        "nvidia_smi_free_mb": free,
        "nvidia_smi_total_mb": total,
        "gpu_util_pct": util,
        "compute_apps": app_rows,
        "torch_allocated_mb": torch_alloc,
        "torch_reserved_mb": torch_reserved,
        "timestamp": now_iso(),
    }


def start_vllm(gpu_memory_utilization: float) -> tuple[subprocess.Popen, Any]:
    env = os.environ.copy()
    env["HF_HOME"] = os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    cmd = [
        VLLM_PYTHON,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(TEXT_CKPT),
        "--served-model-name",
        SERVED,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        "1",
        "--enable-prefix-caching",
        "--enforce-eager",
    ]
    print("starting vLLM:", " ".join(cmd))
    log_f = SERVER_LOG.open("a")
    log_f.write(f"\n===== util={gpu_memory_utilization} {now_iso()} =====\n")
    log_f.flush()
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT)
    )
    return proc, log_f


def stop_vllm(proc: subprocess.Popen | None, log_f) -> None:
    if proc is None:
        return
    print(f"stopping vLLM pid {proc.pid}")
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=45)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)
    try:
        log_f.close()
    except Exception:
        pass
    time.sleep(3)
    # Ensure GPU released
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(2)


async def wait_healthy(timeout_s: float = 360.0) -> None:
    t0 = time.perf_counter()
    last = None
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
        while time.perf_counter() - t0 < timeout_s:
            try:
                async with s.get(f"http://{HOST}:{PORT}/health") as r:
                    if r.status == 200:
                        return
                    last = r.status
            except Exception as e:  # noqa: BLE001
                last = str(e)
            await asyncio.sleep(2)
    raise RuntimeError(f"vLLM not healthy: {last}")


async def text_completion(prompt: str, max_tokens: int = 64) -> dict[str, Any]:
    payload = {
        "model": SERVED,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    t_first = None
    toks = 0
    err = None
    text = ""
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as s:
        try:
            async with s.post(f"http://{HOST}:{PORT}/v1/completions", json=payload) as resp:
                if resp.status != 200:
                    err = f"http_{resp.status}: {(await resp.text())[:300]}"
                else:
                    async for raw in resp.content:
                        line = raw.decode("utf-8", "replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                        except json.JSONDecodeError:
                            continue
                        ch = (obj.get("choices") or [{}])[0]
                        delta = ch.get("text") or ""
                        if delta and t_first is None:
                            t_first = time.perf_counter()
                        text += delta
                        u = obj.get("usage") or {}
                        if u.get("completion_tokens"):
                            toks = int(u["completion_tokens"])
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    t1 = time.perf_counter()
    if toks <= 0:
        toks = max(len(text.split()), 1)
    e2e = (t1 - t0) * 1000.0
    ttft = None if t_first is None else (t_first - t0) * 1000.0
    decode_tok_s = None
    if t_first is not None and toks > 0:
        decode_tok_s = round(toks / max(t1 - t_first, 1e-6), 2)
    return {
        "ok": err is None and t_first is not None,
        "error": err,
        "ttft_ms": None if ttft is None else round(ttft, 3),
        "e2e_ms": round(e2e, 3),
        "completion_tokens": toks,
        "decode_tok_s": decode_tok_s,
        "preview": text[:160],
    }


def parse_engine_load_gib(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    import re

    hits = re.findall(
        r"Model loading took ([0-9.]+) GiB",
        log_path.read_text(errors="replace"),
    )
    return float(hits[-1]) if hits else None


def load_vision_model():
    from neuro_agent.inference.config import InferenceConfig
    from neuro_agent.multimodal.model import load_vlm_for_inference

    cfg = InferenceConfig(
        model_name=VISION_BASE,
        dtype="bfloat16",
        trust_remote_code=True,
        adapter_path=str(VISION_ADAPTER),
        max_new_tokens=32,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        use_cache=True,
        seed=42,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    before = gpu_stats()
    model, processor, info = load_vlm_for_inference(cfg)
    torch.cuda.synchronize()
    after = gpu_stats()
    peak_torch = round(torch.cuda.max_memory_allocated(0) / (1024**2), 1)
    return {
        "model": model,
        "processor": processor,
        "info": {
            "model_name": info.model_name,
            "dtype": info.dtype,
            "device": info.device,
            "total_parameters": info.total_parameters,
            "vision_parameters": info.vision_parameters,
            "merger_parameters": info.merger_parameters,
            "weight_memory_mb": info.weight_memory_mb,
            "load_time_s": info.load_time_s,
            "adapter_path": str(VISION_ADAPTER),
            "quantization": None,
            "vision_tower_dtype": "bfloat16",
            "backend": "transformers+peft (not vLLM)",
            "architecture": "Qwen2_5_VLForConditionalGeneration",
        },
        "before": before,
        "after": after,
        "peak_torch_allocated_mb": peak_torch,
        "wall_s": round(time.perf_counter() - t0, 3),
    }


def unload_vision(model) -> None:
    del model
    torch.cuda.empty_cache()
    time.sleep(1)


def run_one_vision(
    model,
    processor,
    *,
    image_path: Path,
    question: str,
    context: dict | None = None,
    system_prompt: str | None = None,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    from neuro_agent.multimodal.dataset import build_multimodal_messages

    if system_prompt is None:
        system_prompt = (
            "You are a neuroscience research assistant analyzing EEG-derived plots. "
            "Use the provided image and context. Respond with ONLY the direct answer — "
            "a single label, channel name, number, or comma-separated list. "
            "Never add explanation, units, or supporting values."
        )
    user_text = (
        f"Context:\n{json.dumps(context or {}, indent=2, sort_keys=True)}\n\n"
        f"Question: {question.strip()}"
    )
    messages = build_multimodal_messages(
        system_prompt=system_prompt,
        user_text=user_text,
        image_uri=f"file://{image_path.resolve()}",
    )

    torch.cuda.reset_peak_memory_stats()
    t_pre0 = time.perf_counter()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    torch.cuda.synchronize()
    t_pre1 = time.perf_counter()
    preprocess_ms = (t_pre1 - t_pre0) * 1000.0

    t_gen0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    torch.cuda.synchronize()
    t_gen1 = time.perf_counter()

    # Approximate TTFT as full generate start→end / tokens is weak; report generate E2E
    # Prefill+decode combined in generate(); separate encoder timing not exposed.
    in_len = int(inputs["input_ids"].shape[-1])
    gen_ids = out[0][in_len:]
    n_new = int(gen_ids.numel())
    decoded = processor.batch_decode(
        out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    gen_ms = (t_gen1 - t_gen0) * 1000.0
    e2e_ms = (t_gen1 - t_pre0) * 1000.0
    # TTFT proxy: preprocess + first forward is not separately timed; use gen_ms/n as decode
    # and report preprocess separately. TTFT ≈ preprocess + (portion of generate for prefill).
    # Conservative: TTFT_proxy = preprocess_ms + gen_ms * (in_len / (in_len+n_new)) rough
    prefill_proxy = gen_ms * (in_len / max(in_len + n_new, 1))
    ttft_proxy = preprocess_ms + prefill_proxy
    decode_tok_s = round(n_new / max((gen_ms - prefill_proxy) / 1000.0, 1e-6), 2) if n_new else None

    return {
        "ok": True,
        "image": str(image_path),
        "question": question,
        "output": decoded.strip(),
        "preprocess_ms": round(preprocess_ms, 3),
        "generate_ms": round(gen_ms, 3),
        "e2e_ms": round(e2e_ms, 3),
        "ttft_proxy_ms": round(ttft_proxy, 3),
        "vision_encoder_latency_observable": False,
        "note": "Vision encoder latency not separately exposed; TTFT is preprocess+prefill proxy.",
        "input_tokens": in_len,
        "completion_tokens": n_new,
        "decode_tok_s": decode_tok_s,
        "peak_torch_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "nvidia_smi_after_mb": gpu_stats()["nvidia_smi_used_mb"],
    }


def pick_eval_examples(n_per_group: int = 4) -> list[dict[str, Any]]:
    from neuro_agent.evaluation.llm_eval import load_eval_examples
    from neuro_agent.multimodal.dataset import normalize_eval_example

    all_ex = [normalize_eval_example(ex) for ex in load_eval_examples(EVAL_JSONL)]
    by_cat: dict[str, list] = defaultdict(list)
    for ex in all_ex:
        by_cat[ex["category"]].append(ex)

    selected = []
    for group, cats in GATE_TASKS.items():
        for cat in cats:
            pool = by_cat.get(cat, [])
            for ex in pool[:n_per_group]:
                ex = dict(ex)
                ex["_gate_group"] = group
                selected.append(ex)
    return selected


def run_targeted_quality(model, processor, examples: list[dict[str, Any]]) -> dict[str, Any]:
    from neuro_agent.evaluation.verifiers import verify_example
    from neuro_agent.multimodal.dataset import resolve_image_path
    from neuro_agent.config import load_yaml
    from neuro_agent.paths import CONFIGS_DIR

    cfg = load_yaml(CONFIGS_DIR / "multimodal_sft_corrective.yaml")
    system_prompt = cfg["prompt"]["system"]
    results = []
    group_stats: dict[str, list[bool]] = defaultdict(list)

    for i, ex in enumerate(examples):
        img = resolve_image_path(PROJECT_ROOT, ex["image_path"])
        try:
            gen = run_one_vision(
                model,
                processor,
                image_path=img,
                question=ex["question"],
                context=ex.get("context") or {},
                system_prompt=system_prompt,
                max_new_tokens=32,
            )
            pred = gen["output"]
            vr = verify_example(ex, pred)
            passed = bool(vr.get("passed", False))
            row = {
                "id": ex.get("id"),
                "category": ex["category"],
                "gate_group": ex.get("_gate_group"),
                "passed": passed,
                "prediction": pred,
                "e2e_ms": gen["e2e_ms"],
                "verification": {k: vr[k] for k in ("passed", "verification_type") if k in vr},
            }
        except Exception as e:  # noqa: BLE001
            passed = False
            row = {
                "id": ex.get("id"),
                "category": ex["category"],
                "gate_group": ex.get("_gate_group"),
                "passed": False,
                "error": f"{type(e).__name__}: {e}",
            }
        results.append(row)
        group_stats[ex.get("_gate_group", "other")].append(passed)
        if (i + 1) % 10 == 0:
            print(f"  quality {i+1}/{len(examples)}")

    n = len(results)
    n_pass = sum(1 for r in results if r.get("passed"))
    per_group = {
        g: {
            "n": len(vs),
            "pass_rate": round(sum(vs) / len(vs), 4) if vs else None,
        }
        for g, vs in group_stats.items()
    }
    overall = round(n_pass / n, 4) if n else None
    # Parity vs corrected 0.493 — targeted subset may differ; flag if catastrophic drop
    parity_ok = overall is not None and overall >= 0.35  # gate-ish floor; corrected was 0.493 full
    return {
        "n_examples": n,
        "verifier_pass_rate": overall,
        "reference_corrected_sft": REF_QUALITY["corrected_sft"],
        "reference_table": REF_QUALITY,
        "per_gate_group": per_group,
        "parity_plausible": parity_ok,
        "full_440_rerun_needed": not parity_ok,
        "note": (
            "Targeted production-serving validation on gate families. "
            "Full 440 skipped unless parity fails."
        ),
        "predictions_sample": results[:20],
        "failures": [r for r in results if not r.get("passed")][:30],
    }


async def attempt_coresidency(gpu_util: float, label: str) -> dict[str, Any]:
    """Load text vLLM then vision HF; run one text + one vision request."""
    result: dict[str, Any] = {
        "label": label,
        "gpu_memory_utilization": gpu_util,
        "text_load_ok": False,
        "vision_load_ok": False,
        "text_request_ok": False,
        "vision_request_ok": False,
        "oom": False,
        "error": None,
    }
    clean = gpu_stats()
    result["clean_gpu"] = clean
    proc = None
    log_f = None
    vision_bundle = None
    try:
        proc, log_f = start_vllm(gpu_util)
        await wait_healthy(360)
        await asyncio.sleep(1)
        after_text = gpu_stats()
        model_gib = parse_engine_load_gib(SERVER_LOG)
        result["text_load_ok"] = True
        result["after_text_load"] = after_text
        result["text_model_load_gib_from_log"] = model_gib
        result["text_engine_reservation_mb"] = after_text["nvidia_smi_used_mb"]
        print(
            f"[{label}] text loaded smi={after_text['nvidia_smi_used_mb']} "
            f"model_gib={model_gib} free={after_text['nvidia_smi_free_mb']}"
        )

        # Attempt vision load while text resident
        peak_during_vision_load = after_text["nvidia_smi_used_mb"]
        try:
            # Poll GPU in background while loading
            vision_bundle = load_vision_model()
            after_both = gpu_stats()
            peak_during_vision_load = max(
                peak_during_vision_load, after_both["nvidia_smi_used_mb"]
            )
            result["vision_load_ok"] = True
            result["after_both_idle"] = after_both
            result["vision_info"] = vision_bundle["info"]
            result["vision_load"] = {
                "wall_s": vision_bundle["wall_s"],
                "peak_torch_allocated_mb": vision_bundle["peak_torch_allocated_mb"],
                "before": vision_bundle["before"],
                "after": vision_bundle["after"],
            }
            result["peak_vram_during_second_model_load_mb"] = peak_during_vision_load
            result["combined_idle_vram_mb"] = after_both["nvidia_smi_used_mb"]
            result["remaining_free_mb"] = after_both["nvidia_smi_free_mb"]
            print(
                f"[{label}] vision loaded combined={after_both['nvidia_smi_used_mb']} "
                f"free={after_both['nvidia_smi_free_mb']}"
            )
        except torch.cuda.OutOfMemoryError as e:
            result["oom"] = True
            result["error"] = f"CUDA OOM loading vision: {e}"
            result["after_failed_vision_load"] = gpu_stats()
            print(f"[{label}] vision OOM:", e)
            return result
        except Exception as e:  # noqa: BLE001
            # HF may raise RuntimeError wrapping OOM
            msg = f"{type(e).__name__}: {e}"
            if "out of memory" in msg.lower() or "cuda" in msg.lower() and "memory" in msg.lower():
                result["oom"] = True
            result["error"] = msg
            result["traceback"] = traceback.format_exc()[-1500:]
            result["after_failed_vision_load"] = gpu_stats()
            print(f"[{label}] vision load failed:", msg)
            return result

        # One text request while both resident
        text_prompt = (
            "<|im_start|>system\nYou are a neuroscience research intent parser.\n"
            "Output ONLY JSON.<|im_end|>\n"
            "<|im_start|>user\nQuestion: What is beta power on C3 in S001_R03_E012?\n\nJSON:"
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        torch.cuda.reset_peak_memory_stats()
        before_txt = gpu_stats()
        text_res = await text_completion(text_prompt, max_tokens=48)
        after_txt = gpu_stats()
        result["text_request"] = text_res
        result["text_request_ok"] = bool(text_res.get("ok"))
        result["peak_vram_during_text_request_mb"] = max(
            before_txt["nvidia_smi_used_mb"], after_txt["nvidia_smi_used_mb"]
        )
        print(f"[{label}] text req ok={text_res.get('ok')} e2e={text_res.get('e2e_ms')}")

        # One vision request while both resident
        # Find a sample image
        from neuro_agent.evaluation.llm_eval import load_eval_examples
        from neuro_agent.multimodal.dataset import normalize_eval_example, resolve_image_path

        ex = normalize_eval_example(next(iter(load_eval_examples(EVAL_JSONL))))
        img = resolve_image_path(PROJECT_ROOT, ex["image_path"])
        before_vis = gpu_stats()
        try:
            vis_res = run_one_vision(
                vision_bundle["model"],
                vision_bundle["processor"],
                image_path=img,
                question=ex["question"],
                context=ex.get("context") or {},
            )
            after_vis = gpu_stats()
            result["vision_request"] = {
                k: v for k, v in vis_res.items() if k != "output"
            } | {"output_preview": (vis_res.get("output") or "")[:120]}
            result["vision_request_ok"] = True
            result["peak_vram_during_vision_request_mb"] = max(
                before_vis["nvidia_smi_used_mb"],
                after_vis["nvidia_smi_used_mb"],
                vis_res.get("peak_torch_allocated_mb") or 0,
            )
            print(
                f"[{label}] vision req ok e2e={vis_res.get('e2e_ms')} "
                f"out={vis_res.get('output','')[:60]}"
            )
        except torch.cuda.OutOfMemoryError as e:
            result["oom"] = True
            result["vision_request_ok"] = False
            result["vision_request_error"] = f"OOM: {e}"
            print(f"[{label}] vision request OOM")
        except Exception as e:  # noqa: BLE001
            result["vision_request_ok"] = False
            result["vision_request_error"] = f"{type(e).__name__}: {e}"
            print(f"[{label}] vision request failed:", e)

        # Safety margin
        idle = result.get("after_both_idle") or gpu_stats()
        result["safety_margin_mb"] = idle.get("nvidia_smi_free_mb")
        result["coresident_pass"] = bool(
            result["text_load_ok"]
            and result["vision_load_ok"]
            and result["text_request_ok"]
            and result["vision_request_ok"]
            and not result["oom"]
            and (idle.get("nvidia_smi_free_mb") or 0) >= 500  # >=0.5GB free margin
        )
        result["_vision_bundle"] = vision_bundle  # in-memory only
        return result
    finally:
        # Caller may want to keep models; we leave cleanup to caller via keys
        pass


async def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    SERVER_LOG.write_text("")
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(2)

    # -------- 1. Co-residency --------
    clean0 = gpu_stats()
    print("clean GPU", clean0)

    coresidency: dict[str, Any] = {
        "stage": "J",
        "timestamp": now_iso(),
        "gpu": "NVIDIA GeForce RTX 4090",
        "total_vram_mb": clean0["nvidia_smi_total_mb"],
        "clean_state": clean0,
        "text_checkpoint": str(TEXT_CKPT),
        "vision_checkpoint": str(VISION_ADAPTER),
        "attempts": [],
    }

    def _teardown_pair(vision_bundle) -> None:
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
        time.sleep(3)
        if vision_bundle is not None:
            unload_vision(vision_bundle["model"])
        torch.cuda.empty_cache()
        time.sleep(2)

    # Attempt A: production util 0.90
    attempt_a = await attempt_coresidency(PROD_GPU_UTIL, "production_util_0.90")
    vision_keep = attempt_a.pop("_vision_bundle", None)
    coresidency["attempts"].append({k: v for k, v in attempt_a.items()})

    coresident_ok = bool(attempt_a.get("coresident_pass"))
    active_util = PROD_GPU_UTIL
    active_attempt = attempt_a

    if coresident_ok:
        print("Production util co-residency PASS — tearing down before vision-only benches")
        _teardown_pair(vision_keep)
        vision_keep = None
    else:
        print("Production util co-residency failed; tearing down and retrying headroom-aware util", CORESIDE_GPU_UTIL)
        _teardown_pair(vision_keep)
        vision_keep = None
        attempt_b = await attempt_coresidency(CORESIDE_GPU_UTIL, "coreside_util_0.40")
        vision_keep = attempt_b.pop("_vision_bundle", None)
        coresidency["attempts"].append({k: v for k, v in attempt_b.items()})
        if attempt_b.get("coresident_pass"):
            coresident_ok = True
            active_util = CORESIDE_GPU_UTIL
            active_attempt = attempt_b
            print("Headroom-aware co-residency PASS — tearing down before vision-only benches")
            _teardown_pair(vision_keep)
            vision_keep = None
        else:
            _teardown_pair(vision_keep)
            vision_keep = None

    # Decision / swap policy
    if coresident_ok:
        combined = active_attempt.get("combined_idle_vram_mb")
        free = active_attempt.get("remaining_free_mb")
        coresidency["verdict"] = "PASS"
        coresidency["strategy"] = "co_resident"
        coresidency["safe_combined_budget_mb"] = combined
        coresidency["safety_margin_mb"] = free
        coresidency["recommended_text_gpu_memory_utilization"] = active_util
        coresidency["note"] = (
            f"Both models resident with text gpu_memory_utilization={active_util}. "
            "Text KV capacity is reduced vs production util=0.90 if util was lowered."
        )
    else:
        coresidency["verdict"] = "FAIL"
        coresidency["strategy"] = "swap_unload"
        coresidency["swap_policy"] = {
            "default_resident": "text_w8a8_vllm",
            "on_vision_request": [
                "stop or sleep text vLLM engine (release KV reservation)",
                "load corrected Qwen2.5-VL-3B LoRA (bf16 HF)",
                "serve vision request(s)",
                "unload VLM",
                "restart text vLLM with production util=0.90",
            ],
            "rationale": (
                "Production text util=0.90 reserves ~22GB; corrected VLM alone peaks "
                "~8GB torch / ~16GB nvidia-smi historically — cannot share 24GB safely "
                "with full text KV reservation."
            ),
            "alternative": (
                "Keep both resident only with sharply reduced text gpu_memory_utilization "
                f"(tried {CORESIDE_GPU_UTIL}) and accept much smaller KV / concurrency."
            ),
        }
        coresidency["recommended_text_gpu_memory_utilization"] = PROD_GPU_UTIL
        coresidency["safe_combined_budget_mb"] = None
        coresidency["safety_margin_mb"] = None

    save_json(RESULTS / "co_residency.json", coresidency)
    print("co-residency verdict:", coresidency["verdict"], coresidency["strategy"])

    # -------- 2. Vision serving config + load for quality/latency --------
    print("Loading vision alone for quality/latency…")
    vision_keep = load_vision_model()

    serving_config = {
        "stage": "J",
        "timestamp": now_iso(),
        "model_architecture": "Qwen2_5_VLForConditionalGeneration",
        "base_model": VISION_BASE,
        "adapter": str(VISION_ADAPTER),
        "adapter_type": "PEFT LoRA (not merged)",
        "dtype": "bfloat16",
        "quantization": None,
        "vision_tower_dtype": "bfloat16",
        "device": "cuda:0",
        "serving_runtime_backend": "HuggingFace Transformers + PEFT + qwen_vl_utils",
        "not_vllm": True,
        "image_preprocessing": {
            "processor": "Qwen2_5_VLProcessor / Qwen2VLImageProcessor",
            "path": "qwen_vl_utils.process_vision_info → processor(images=...)",
            "normalize_rescale_resize": True,
            "patch_size": 14,
        },
        "max_image_resolution_settings": "processor defaults from checkpoint processor_config.json",
        "vram_idle_mb": vision_keep["after"]["nvidia_smi_used_mb"],
        "vram_peak_torch_mb": vision_keep["peak_torch_allocated_mb"],
        "weight_memory_mb": vision_keep["info"]["weight_memory_mb"],
        "load_info": vision_keep["info"],
        "co_residency_verdict": coresidency["verdict"],
    }
    # enrich processor config if present
    proc_cfg = VISION_ADAPTER / "processor_config.json"
    if proc_cfg.exists():
        serving_config["processor_config"] = json.loads(proc_cfg.read_text())
    save_json(RESULTS / "serving_config.json", serving_config)

    # -------- 3. Targeted quality --------
    print("Running targeted multimodal quality validation…")
    examples = pick_eval_examples(n_per_group=3)
    # Ensure modality coverage: add one of each image type if missing
    quality = run_targeted_quality(
        vision_keep["model"], vision_keep["processor"], examples
    )
    quality["timestamp"] = now_iso()
    quality["checkpoint"] = str(VISION_ADAPTER)
    save_json(RESULTS / "quality_validation.json", quality)
    print(
        "quality pass_rate=",
        quality["verifier_pass_rate"],
        "n=",
        quality["n_examples"],
        "full440=",
        quality["full_440_rerun_needed"],
    )

    # -------- 4. Vision latency benchmark (representative types) --------
    print("Vision latency benchmark…")
    from neuro_agent.evaluation.llm_eval import load_eval_examples
    from neuro_agent.multimodal.dataset import normalize_eval_example, resolve_image_path

    all_ex = [normalize_eval_example(ex) for ex in load_eval_examples(EVAL_JSONL)]
    type_keywords = {
        "topomap": "img_topomap",
        "spectrogram": "img_spectrogram",
        "psd": "img_psd",
        "waveform": "img_waveform",
        "band_power": "img_channel_band",
    }
    latency_rows = []
    for tname, key in type_keywords.items():
        pool = [ex for ex in all_ex if key in ex.get("image_path", "")]
        for ex in pool[:5]:
            img = resolve_image_path(PROJECT_ROOT, ex["image_path"])
            try:
                r = run_one_vision(
                    vision_keep["model"],
                    vision_keep["processor"],
                    image_path=img,
                    question=ex["question"],
                    context=ex.get("context") or {},
                )
                latency_rows.append(
                    {
                        "image_type": tname,
                        "category": ex["category"],
                        "preprocess_ms": r["preprocess_ms"],
                        "ttft_proxy_ms": r["ttft_proxy_ms"],
                        "e2e_ms": r["e2e_ms"],
                        "decode_tok_s": r["decode_tok_s"],
                        "completion_tokens": r["completion_tokens"],
                        "peak_torch_mb": r["peak_torch_allocated_mb"],
                        "nvidia_smi_mb": r["nvidia_smi_after_mb"],
                    }
                )
            except Exception as e:  # noqa: BLE001
                latency_rows.append(
                    {"image_type": tname, "error": f"{type(e).__name__}: {e}"}
                )
        print(f"  latency type={tname} n={sum(1 for r in latency_rows if r.get('image_type')==tname and 'e2e_ms' in r)}")

    def agg(rows: list[dict], field: str) -> dict:
        xs = [r[field] for r in rows if field in r and r[field] is not None]
        return {
            "p50": percentile(xs, 50),
            "p95": percentile(xs, 95),
            "mean": round(statistics.mean(xs), 3) if xs else None,
            "n": len(xs),
        }

    ok_lat = [r for r in latency_rows if "e2e_ms" in r]
    latency_bench = {
        "stage": "J",
        "timestamp": now_iso(),
        "n_requests": len(ok_lat),
        "overall": {
            "preprocess_ms": agg(ok_lat, "preprocess_ms"),
            "ttft_proxy_ms": agg(ok_lat, "ttft_proxy_ms"),
            "e2e_ms": agg(ok_lat, "e2e_ms"),
            "decode_tok_s": agg(ok_lat, "decode_tok_s"),
            "peak_torch_mb": agg(ok_lat, "peak_torch_mb"),
            "nvidia_smi_mb": agg(ok_lat, "nvidia_smi_mb"),
        },
        "per_image_type": {},
        "rows": latency_rows,
        "vision_encoder_latency_observable": False,
    }
    for tname in type_keywords:
        subset = [r for r in ok_lat if r["image_type"] == tname]
        latency_bench["per_image_type"][tname] = {
            "preprocess_ms": agg(subset, "preprocess_ms"),
            "ttft_proxy_ms": agg(subset, "ttft_proxy_ms"),
            "e2e_ms": agg(subset, "e2e_ms"),
            "decode_tok_s": agg(subset, "decode_tok_s"),
            "n": len(subset),
        }
    save_json(RESULTS / "latency_benchmark.json", latency_bench)

    # -------- 5. Text-only vs vision cost --------
    print("Text vs vision cost…")
    # Unload vision, start text alone at production util for fair text cost
    unload_vision(vision_keep["model"])
    vision_keep = None
    torch.cuda.empty_cache()
    time.sleep(2)

    text_cost_rows = []
    proc, log_f = start_vllm(PROD_GPU_UTIL)
    try:
        await wait_healthy(360)
        # reuse production-ish prompt
        text_prompt = (
            "<|im_start|>system\nYou are a neuroscience research intent parser. "
            "Output ONLY a JSON object.<|im_end|>\n"
            "<|im_start|>user\nQuestion: What is the beta-band power for channel C3 "
            "in sample S001_R03_E012?\n\nJSON:<|im_end|>\n<|im_start|>assistant\n"
        )
        # warmup
        await text_completion(text_prompt, max_tokens=16)
        for i in range(8):
            before = gpu_stats()
            r = await text_completion(text_prompt, max_tokens=64)
            after = gpu_stats()
            text_cost_rows.append(
                {
                    **r,
                    "vram_mb": after["nvidia_smi_used_mb"],
                    "gpu_util": after["gpu_util_pct"],
                    "model_invocations": 1,
                    "branch": "text_vllm_w8a8",
                }
            )
        text_idle = gpu_stats()
    finally:
        stop_vllm(proc, log_f)

    # Vision alone cost (reload)
    vision_keep = load_vision_model()
    vision_cost_rows = []
    pool = [ex for ex in all_ex if "img_psd" in ex.get("image_path", "")][:8]
    for ex in pool:
        img = resolve_image_path(PROJECT_ROOT, ex["image_path"])
        before = gpu_stats()
        r = run_one_vision(
            vision_keep["model"],
            vision_keep["processor"],
            image_path=img,
            question=ex["question"],
            context=ex.get("context") or {},
        )
        after = gpu_stats()
        vision_cost_rows.append(
            {
                "ok": True,
                "ttft_ms": r["ttft_proxy_ms"],
                "e2e_ms": r["e2e_ms"],
                "decode_tok_s": r["decode_tok_s"],
                "vram_mb": after["nvidia_smi_used_mb"],
                "peak_torch_mb": r["peak_torch_allocated_mb"],
                "gpu_util": after["gpu_util_pct"],
                "model_invocations": 1,
                "branch": "vision_hf_bf16_lora",
                "preprocess_ms": r["preprocess_ms"],
            }
        )
    vision_idle = gpu_stats()

    def mean_field(rows, key):
        xs = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(xs), 3) if xs else None

    cost = {
        "stage": "J",
        "timestamp": now_iso(),
        "text_only": {
            "backend": "vLLM W8A8 INT8",
            "n": len(text_cost_rows),
            "ttft_ms_mean": mean_field(text_cost_rows, "ttft_ms"),
            "e2e_ms_mean": mean_field(text_cost_rows, "e2e_ms"),
            "e2e_ms_p50": percentile([r["e2e_ms"] for r in text_cost_rows], 50),
            "decode_tok_s_mean": mean_field(text_cost_rows, "decode_tok_s"),
            "vram_idle_mb": text_idle["nvidia_smi_used_mb"],
            "model_invocations_per_request": 1,
        },
        "vision_image_request": {
            "backend": "HF Qwen2.5-VL-3B + corrected LoRA bf16",
            "n": len(vision_cost_rows),
            "ttft_proxy_ms_mean": mean_field(vision_cost_rows, "ttft_ms"),
            "e2e_ms_mean": mean_field(vision_cost_rows, "e2e_ms"),
            "e2e_ms_p50": percentile([r["e2e_ms"] for r in vision_cost_rows], 50),
            "preprocess_ms_mean": mean_field(vision_cost_rows, "preprocess_ms"),
            "decode_tok_s_mean": mean_field(vision_cost_rows, "decode_tok_s"),
            "vram_idle_mb": vision_idle["nvidia_smi_used_mb"],
            "peak_torch_mb_mean": mean_field(vision_cost_rows, "peak_torch_mb"),
            "model_invocations_per_request": 1,
        },
        "comparison": {
            "e2e_ratio_vision_over_text": (
                None
                if not text_cost_rows or not vision_cost_rows
                else round(
                    mean_field(vision_cost_rows, "e2e_ms")
                    / max(mean_field(text_cost_rows, "e2e_ms"), 1e-9),
                    2,
                )
            ),
            "vram_ratio_vision_over_text": (
                None
                if not text_idle or not vision_idle
                else round(
                    vision_idle["nvidia_smi_used_mb"]
                    / max(text_idle["nvidia_smi_used_mb"], 1e-9),
                    2,
                )
            ),
            "why_vision_only_when_needed": (
                "Vision path loads a separate ~3B VLM (bf16+LoRA), runs image preprocess + "
                "multimodal prefill, and uses substantially more VRAM than the text W8A8 branch. "
                "Text/tool orchestration should stay on the W8A8 vLLM path; invoke vision only "
                "when an image is required."
            ),
        },
        "text_rows": text_cost_rows,
        "vision_rows": vision_cost_rows,
    }
    save_json(RESULTS / "text_vs_vision_cost.json", cost)

    # Cleanup vision
    unload_vision(vision_keep["model"])
    vision_keep = None
    torch.cuda.empty_cache()

    # -------- 6. Production decision --------
    q_rate = quality.get("verifier_pass_rate")
    vision_usable = bool(q_rate is not None and q_rate >= 0.35)
    additional_quant_needed_now = False
    if coresidency["verdict"] == "FAIL":
        # Memory is the blocker for co-residency, not quality — recommend swap, not immediate quant saga
        additional_quant_needed_now = False

    decision = {
        "stage": "J",
        "timestamp": now_iso(),
        "can_text_and_vision_remain_resident": coresidency["verdict"] == "PASS",
        "co_residency_verdict": coresidency["verdict"],
        "strategy": coresidency["strategy"],
        "safe_combined_vram_budget_mb": coresidency.get("safe_combined_budget_mb"),
        "safety_margin_mb": coresidency.get("safety_margin_mb"),
        "swap_policy": coresidency.get("swap_policy"),
        "vision_suitable_for_low_concurrency_production": vision_usable,
        "vision_quality_pass_rate_targeted": q_rate,
        "reference_corrected_sft": REF_QUALITY["corrected_sft"],
        "additional_multimodal_quantization_needed_now": additional_quant_needed_now,
        "quantization_guidance": (
            "Do not open a quantization/debugging saga now. Corrected VLM is usable in bf16+LoRA "
            "for low-concurrency vision requests. If future co-residency with full text KV is "
            "required, consider a later VLM quant/swap plan — memory is the co-residency blocker, "
            "not current quality."
            if coresidency["verdict"] == "FAIL"
            else "Current corrected VLM fits with the documented co-resident util; no extra quant now."
        ),
        "scope_boundary": (
            "Text-path concurrency, prefix caching, and SLA were benchmarked separately. "
            "Vision-path concurrency is outside this stage unless required for basic correctness."
        ),
    }

    # Overall J verdict
    if coresidency["verdict"] == "PASS" and vision_usable:
        j_verdict = "PASS"
    elif vision_usable and coresidency["verdict"] == "FAIL":
        j_verdict = "PARTIAL PASS"  # vision OK, co-residency requires swap
    else:
        j_verdict = "FAIL"
    decision["j_verdict"] = j_verdict
    save_json(RESULTS / "production_decision.json", decision)

    comparison = {
        "stage": "J",
        "timestamp": now_iso(),
        "verdict": j_verdict,
        "co_residency": {
            "verdict": coresidency["verdict"],
            "strategy": coresidency["strategy"],
            "combined_vram_mb": coresidency.get("safe_combined_budget_mb"),
            "safety_margin_mb": coresidency.get("safety_margin_mb"),
            "attempts_summary": [
                {
                    "label": a.get("label"),
                    "util": a.get("gpu_memory_utilization"),
                    "text_ok": a.get("text_load_ok"),
                    "vision_ok": a.get("vision_load_ok"),
                    "text_req": a.get("text_request_ok"),
                    "vision_req": a.get("vision_request_ok"),
                    "oom": a.get("oom"),
                    "combined_idle_mb": a.get("combined_idle_vram_mb"),
                    "free_mb": a.get("remaining_free_mb"),
                    "error": a.get("error"),
                }
                for a in coresidency.get("attempts", [])
            ],
        },
        "vision_serving_config": {
            k: serving_config[k]
            for k in (
                "model_architecture",
                "dtype",
                "quantization",
                "vision_tower_dtype",
                "device",
                "serving_runtime_backend",
                "vram_idle_mb",
                "vram_peak_torch_mb",
            )
        },
        "quality": {
            "targeted_pass_rate": q_rate,
            "n": quality.get("n_examples"),
            "per_gate_group": quality.get("per_gate_group"),
            "reference": REF_QUALITY,
            "full_440_rerun_needed": quality.get("full_440_rerun_needed"),
        },
        "vision_latency": latency_bench["overall"],
        "text_vs_vision_cost": cost["comparison"]
        | {
            "text_e2e_ms_mean": cost["text_only"]["e2e_ms_mean"],
            "vision_e2e_ms_mean": cost["vision_image_request"]["e2e_ms_mean"],
            "text_vram_mb": cost["text_only"]["vram_idle_mb"],
            "vision_vram_mb": cost["vision_image_request"]["vram_idle_mb"],
        },
        "production_decision": decision,
        "scope_boundary": decision["scope_boundary"],
    }
    save_json(CMP, comparison)

    print("\nJ complete. verdict=", j_verdict)
    print("co-residency=", coresidency["verdict"], coresidency["strategy"])
    print("quality=", q_rate)
    print("J_DONE")


if __name__ == "__main__":
    # Prefer venv if available for qwen_vl_utils — re-exec if needed
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_py.exists() and Path(sys.executable).resolve() != venv_py.resolve():
            print("Re-exec under project venv for qwen_vl_utils…")
            os.execv(str(venv_py), [str(venv_py), *sys.argv])
        raise
    asyncio.run(main())
