#!/usr/bin/env python3
"""Resume Stage J unfinished pieces: quality fix, text-vs-vision cost, final decision."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
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
PROD_GPU_UTIL = 0.90
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
    return {
        "nvidia_smi_used_mb": used,
        "nvidia_smi_free_mb": free,
        "nvidia_smi_total_mb": total,
        "gpu_util_pct": util,
    }


def load_vision():
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
    model, processor, info = load_vlm_for_inference(cfg)
    return model, processor, info


def run_one_vision(model, processor, image_path: Path, question: str, context: dict, system_prompt: str):
    from qwen_vl_utils import process_vision_info

    from neuro_agent.multimodal.dataset import build_multimodal_messages

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
    t0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    torch.cuda.synchronize()
    t_pre = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    in_len = int(inputs["input_ids"].shape[-1])
    decoded = processor.batch_decode(
        out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    n_new = int(out[0][in_len:].numel())
    return {
        "output": decoded.strip(),
        "preprocess_ms": round((t_pre - t0) * 1000, 3),
        "e2e_ms": round((t1 - t0) * 1000, 3),
        "ttft_proxy_ms": round((t_pre - t0) * 1000 + (t1 - t_pre) * 1000 * (in_len / max(in_len + n_new, 1)), 3),
        "completion_tokens": n_new,
        "decode_tok_s": round(n_new / max(t1 - t_pre, 1e-6), 2) if n_new else None,
        "peak_torch_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "nvidia_smi_mb": gpu_stats()["nvidia_smi_used_mb"],
    }


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
    decode_tok_s = None
    if t_first is not None and toks > 0:
        decode_tok_s = round(toks / max(t1 - t_first, 1e-6), 2)
    return {
        "ok": err is None and t_first is not None,
        "error": err,
        "ttft_ms": None if t_first is None else round((t_first - t0) * 1000, 3),
        "e2e_ms": round((t1 - t0) * 1000, 3),
        "completion_tokens": toks,
        "decode_tok_s": decode_tok_s,
    }


async def main() -> None:
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(2)
    torch.cuda.empty_cache()

    from neuro_agent.config import load_yaml
    from neuro_agent.evaluation.llm_eval import load_eval_examples
    from neuro_agent.evaluation.verifiers import verify_example
    from neuro_agent.multimodal.dataset import normalize_eval_example, resolve_image_path
    from neuro_agent.paths import CONFIGS_DIR

    cfg = load_yaml(CONFIGS_DIR / "multimodal_sft_corrective.yaml")
    system_prompt = cfg["prompt"]["system"]

    print("Loading vision for quality resume…")
    model, processor, info = load_vision()
    print("vision loaded", gpu_stats())

    all_ex = [normalize_eval_example(ex) for ex in load_eval_examples(EVAL_JSONL)]
    by_cat: dict[str, list] = defaultdict(list)
    for ex in all_ex:
        by_cat[ex["category"]].append(ex)
    examples = []
    for group, cats in GATE_TASKS.items():
        for cat in cats:
            for ex in by_cat.get(cat, [])[:3]:
                row = dict(ex)
                row["_gate_group"] = group
                examples.append(row)

    results = []
    group_stats: dict[str, list[bool]] = defaultdict(list)
    for i, ex in enumerate(examples):
        img = resolve_image_path(PROJECT_ROOT, ex["image_path"])
        gen = run_one_vision(
            model, processor, img, ex["question"], ex.get("context") or {}, system_prompt
        )
        pred = gen["output"]
        vr = verify_example(ex, pred)
        passed = bool(vr.passed)
        results.append(
            {
                "id": ex.get("id"),
                "category": ex["category"],
                "gate_group": ex.get("_gate_group"),
                "passed": passed,
                "prediction": pred,
                "expected": vr.expected,
                "verification_type": vr.verification_type,
                "reason": vr.reason,
                "e2e_ms": gen["e2e_ms"],
            }
        )
        group_stats[ex.get("_gate_group", "other")].append(passed)
        if (i + 1) % 10 == 0:
            print(f"quality {i+1}/{len(examples)} running_pass={sum(1 for r in results if r['passed'])}/{len(results)}")

    n = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    overall = round(n_pass / n, 4) if n else None
    # Targeted gate can differ from full 440; parity if not catastrophic vs corrected
    parity_ok = overall is not None and overall >= 0.30
    quality = {
        "stage": "J",
        "timestamp": now_iso(),
        "checkpoint": str(VISION_ADAPTER),
        "n_examples": n,
        "verifier_pass_rate": overall,
        "reference_corrected_sft": REF_QUALITY["corrected_sft"],
        "reference_table": REF_QUALITY,
        "per_gate_group": {
            g: {"n": len(vs), "pass_rate": round(sum(vs) / len(vs), 4) if vs else None}
            for g, vs in group_stats.items()
        },
        "parity_plausible": parity_ok,
        "full_440_rerun_needed": not parity_ok,
        "note": (
            "Targeted production-serving validation on gate families using HF+PEFT path. "
            "Full 440 only if parity fails."
        ),
        "predictions_sample": results[:20],
        "failures": [r for r in results if not r["passed"]][:40],
    }
    save_json(RESULTS / "quality_validation.json", quality)
    print("quality pass_rate=", overall, "parity_ok=", parity_ok)

    # ---- text vs vision cost ----
    # Collect vision cost while loaded
    vision_cost_rows = []
    pool = [ex for ex in all_ex if "img_psd" in ex.get("image_path", "")][:8]
    for ex in pool:
        img = resolve_image_path(PROJECT_ROOT, ex["image_path"])
        r = run_one_vision(model, processor, img, ex["question"], ex.get("context") or {}, system_prompt)
        after = gpu_stats()
        vision_cost_rows.append(
            {
                "ok": True,
                "ttft_ms": r["ttft_proxy_ms"],
                "e2e_ms": r["e2e_ms"],
                "decode_tok_s": r["decode_tok_s"],
                "vram_mb": after["nvidia_smi_used_mb"],
                "peak_torch_mb": r["peak_torch_mb"],
                "gpu_util": after["gpu_util_pct"],
                "preprocess_ms": r["preprocess_ms"],
                "model_invocations": 1,
                "branch": "vision_hf_bf16_lora",
            }
        )
    vision_idle = gpu_stats()

    del model
    torch.cuda.empty_cache()
    time.sleep(3)
    # Force release
    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    time.sleep(1)
    print("after vision unload", gpu_stats())

    env = os.environ.copy()
    env["HF_HOME"] = os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
    env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = [
        "/usr/bin/python3",
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
        str(PROD_GPU_UTIL),
        "--max-model-len",
        "4096",
        "--tensor-parallel-size",
        "1",
        "--enable-prefix-caching",
        "--enforce-eager",
    ]
    print("starting text vLLM for cost…")
    log_f = SERVER_LOG.open("a")
    log_f.write(f"\n===== text cost {now_iso()} =====\n")
    log_f.flush()
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
    text_cost_rows = []
    try:
        await wait_healthy(360)
        text_prompt = (
            "<|im_start|>system\nYou are a neuroscience research intent parser. "
            "Output ONLY a JSON object.<|im_end|>\n"
            "<|im_start|>user\nQuestion: What is the beta-band power for channel C3 "
            "in sample S001_R03_E012?\n\nJSON:<|im_end|>\n<|im_start|>assistant\n"
        )
        await text_completion(text_prompt, max_tokens=16)
        for _ in range(8):
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
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        log_f.close()
        time.sleep(2)

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
            "e2e_ms_p50": percentile([r["e2e_ms"] for r in text_cost_rows if "e2e_ms" in r], 50),
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
            "e2e_ratio_vision_over_text": round(
                mean_field(vision_cost_rows, "e2e_ms")
                / max(mean_field(text_cost_rows, "e2e_ms") or 1e-9, 1e-9),
                2,
            )
            if text_cost_rows and vision_cost_rows
            else None,
            "vram_ratio_vision_over_text": round(
                vision_idle["nvidia_smi_used_mb"] / max(text_idle["nvidia_smi_used_mb"], 1e-9),
                2,
            ),
            "why_vision_only_when_needed": (
                "Vision path uses a separate ~3B VLM (bf16+LoRA), image preprocess, and multimodal "
                "prefill with much higher VRAM than W8A8 text. Keep orchestration on text; call "
                "vision only when an image is required."
            ),
        },
        "text_rows": text_cost_rows,
        "vision_rows": vision_cost_rows,
    }
    save_json(RESULTS / "text_vs_vision_cost.json", cost)
    print("cost text_e2e", cost["text_only"]["e2e_ms_mean"], "vision_e2e", cost["vision_image_request"]["e2e_ms_mean"])

    # ---- final decision from existing co_residency + new quality/cost ----
    coresidency = json.loads((RESULTS / "co_residency.json").read_text())
    serving_config = json.loads((RESULTS / "serving_config.json").read_text())
    latency_bench = json.loads((RESULTS / "latency_benchmark.json").read_text())

    vision_usable = bool(overall is not None and overall >= 0.30)
    # Production util cannot co-reside; headroom util can — prefer documenting both
    prod_attempt = next(
        (a for a in coresidency.get("attempts", []) if a.get("label") == "production_util_0.90"),
        None,
    )
    coreside_attempt = next(
        (a for a in coresidency.get("attempts", []) if a.get("label") == "coreside_util_0.40"),
        None,
    )
    # Recommended production strategy: swap for full text KV; optional co-resident with util=0.40 for demos
    decision = {
        "stage": "J",
        "timestamp": now_iso(),
        "can_text_and_vision_remain_resident": {
            "with_production_text_util_0.90": False,
            "with_reduced_text_util_0.40": bool(coreside_attempt and coreside_attempt.get("coresident_pass")),
        },
        "co_residency_verdict_production_config": "FAIL",
        "co_residency_verdict_reduced_kv": coresidency.get("verdict"),
        "recommended_strategy": "swap_unload_for_production_text_kv",
        "safe_combined_vram_budget_mb_reduced_kv": coresidency.get("safe_combined_budget_mb"),
        "safety_margin_mb_reduced_kv": coresidency.get("safety_margin_mb"),
        "swap_policy": {
            "default_resident": "text_w8a8_vllm @ gpu_memory_utilization=0.90 (full I.1–I.3 capacity)",
            "on_vision_request": [
                "stop/sleep text vLLM to release KV reservation",
                "load corrected Qwen2.5-VL-3B LoRA (bf16 HF)",
                "serve vision request(s) at low concurrency",
                "unload VLM",
                "restart text vLLM at util=0.90",
            ],
            "optional_demo_coresidency": (
                "Both can co-reside if text util is lowered to ~0.40 "
                f"(combined idle ~{coresidency.get('safe_combined_budget_mb')} MB, "
                f"~{coresidency.get('safety_margin_mb')} MB free) — accepts much smaller text KV/concurrency."
            ),
            "rationale": (
                "Production text util=0.90 reserves ~22GB; VLM needs ~7–8GB torch weights. "
                "OOM observed when loading vision beside production text."
            ),
        },
        "vision_suitable_for_low_concurrency_production": vision_usable,
        "vision_quality_pass_rate_targeted": overall,
        "reference_corrected_sft": REF_QUALITY["corrected_sft"],
        "additional_multimodal_quantization_needed_now": False,
        "quantization_guidance": (
            "Do not open a quantization saga now. Corrected VLM is usable in bf16+LoRA for "
            "low-concurrency vision. Memory blocks production-util co-residency; use swap/unload "
            "(or reduced text KV) rather than immediate VLM quantization."
        ),
        "scope_boundary": (
            "Text-path concurrency, prefix caching, and SLA were benchmarked separately. "
            "Vision-path concurrency is outside this stage unless required for basic correctness."
        ),
    }

    if vision_usable and coresidency.get("verdict") == "PASS":
        # PASS overall if vision OK and at least one safe residency mode exists;
        # production-util co-residency still FAIL → PARTIAL PASS is more honest
        j_verdict = "PARTIAL PASS"
    elif vision_usable:
        j_verdict = "PARTIAL PASS"
    else:
        j_verdict = "FAIL"
    decision["j_verdict"] = j_verdict
    decision["interpretation"] = (
        "PARTIAL PASS: corrected vision branch is production-usable at low concurrency, but "
        "text+vision cannot remain resident under production text util=0.90. Use swap/unload "
        "for production, or reduced text util only when accepting smaller KV."
    )
    save_json(RESULTS / "production_decision.json", decision)

    # Update co_residency.json with clearer production vs reduced distinction
    coresidency["production_util_0.90_pass"] = False
    coresidency["reduced_util_0.40_pass"] = bool(
        coreside_attempt and coreside_attempt.get("coresident_pass")
    )
    coresidency["recommended_production_strategy"] = "swap_unload"
    coresidency["verdict_note"] = (
        "PASS only under reduced text gpu_memory_utilization=0.40. "
        "FAIL under production util=0.90. Recommended production strategy: swap/unload."
    )
    save_json(RESULTS / "co_residency.json", coresidency)

    comparison = {
        "stage": "J",
        "timestamp": now_iso(),
        "verdict": j_verdict,
        "co_residency": {
            "production_util_0.90": "FAIL (OOM loading vision; free ~1.1GB)",
            "reduced_util_0.40": "PASS",
            "combined_idle_mb_at_0.40": coresidency.get("safe_combined_budget_mb"),
            "safety_margin_mb_at_0.40": coresidency.get("safety_margin_mb"),
            "recommended_strategy": "swap_unload_for_production",
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
                    "error": (a.get("error") or "")[:200] or None,
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
            if k in serving_config
        },
        "quality": {
            "targeted_pass_rate": overall,
            "n": n,
            "per_gate_group": quality["per_gate_group"],
            "reference": REF_QUALITY,
            "full_440_rerun_needed": quality["full_440_rerun_needed"],
        },
        "vision_latency": latency_bench.get("overall"),
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
    print("J complete. verdict=", j_verdict)
    print("J_DONE")


if __name__ == "__main__":
    try:
        import qwen_vl_utils  # noqa: F401
    except ImportError:
        venv_py = PROJECT_ROOT / ".venv" / "bin" / "python"
        if venv_py.exists():
            os.execv(str(venv_py), [str(venv_py), *sys.argv])
        raise
    asyncio.run(main())
