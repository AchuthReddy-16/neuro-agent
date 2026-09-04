#!/usr/bin/env python3
"""Optimize text↔vision swap via co-residency at reduced text util.

Profiling/optimization measurement only. No retrain/requantize/kernel/git mutation.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import requests
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT = PROJECT_ROOT / "results" / "optimization" / "model_residency"
CMP = PROJECT_ROOT / "results" / "model_comparison"
TEXT_CKPT = PROJECT_ROOT / "checkpoints" / "text_w8a8_int8_compressed"
VISION_ADAPTER = PROJECT_ROOT / "checkpoints" / "multimodal_sft_corrected" / "final"
VISION_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
VLLM_PYTHON = "/usr/bin/python3"
HOST = "127.0.0.1"
PORT = 8000
SERVED = "qwen3-w8a8-int8"
MAX_MODEL_LEN = 4096

IMG_ROOT = PROJECT_ROOT / "data/processed/vision/images/test"
IMAGES = {
    "topomap": next(IMG_ROOT.glob("img_topomap_*.png")),
    "spectrogram": next(IMG_ROOT.glob("img_spectrogram_*.png")),
    "psd": next(IMG_ROOT.glob("img_psd_*.png")),
    "waveform": next(IMG_ROOT.glob("img_waveform_*.png")),
}

# Light concurrency workload (same family as I.1, shorter run)
MAX_TOKENS = 64
TEMPERATURE = 0.0
BENCH_PROMPT = (
    "You are a neuroscience research assistant. "
    "Summarize in one short sentence how mu-band ERD relates to motor imagery. "
    "Context: 64-channel EEG, 8–12 Hz mu, contralateral sensorimotor cortex, "
    "beta rebound after imagery, BCI classification relevance. "
    "Answer briefly."
)

I1_REF = {
    1: {
        "util": 0.90,
        "rps": 2.0527,
        "output_tok_s": 131.38,
        "ttft_p50_ms": 27.3875,
        "ttft_p95_ms": 29.1837,
        "e2e_p50_ms": 484.9265,
        "e2e_p95_ms": 486.4591,
        "source": "results/serving/load/w8a8_int8/concurrency_1.json",
    },
    8: {
        "util": 0.90,
        "rps": 11.9668,
        "output_tok_s": 765.88,
        "ttft_p50_ms": 122.622,
        "ttft_p95_ms": 153.8022,
        "e2e_p50_ms": 667.9405,
        "e2e_p95_ms": 674.2661,
        "source": "results/serving/load/w8a8_int8/concurrency_8.json",
    },
    16: {
        "util": 0.90,
        "rps": 18.4472,
        "output_tok_s": 1176.01,
        "ttft_p50_ms": 173.2065,
        "ttft_p95_ms": 282.3705,
        "e2e_p50_ms": 853.428,
        "e2e_p95_ms": 923.4758,
        "source": "results/serving/load/w8a8_int8/concurrency_16.json",
    },
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pctile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def gpu_stats() -> dict[str, Any]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    used, total, util = [float(x.strip()) for x in out.split(",")]
    free = total - used
    torch_alloc = (
        round(torch.cuda.memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0
    )
    torch_reserved = (
        round(torch.cuda.memory_reserved(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0
    )
    return {
        "nvidia_smi_used_mb": used,
        "nvidia_smi_total_mb": total,
        "nvidia_smi_free_mb": free,
        "gpu_util_pct": util,
        "torch_allocated_mb": torch_alloc,
        "torch_reserved_mb": torch_reserved,
        "timestamp": now_iso(),
    }


def start_vllm(gpu_memory_utilization: float) -> tuple[subprocess.Popen, Any]:
    log_path = OUT / f"vllm_util_{gpu_memory_utilization:.2f}.log"
    log_f = log_path.open("a")
    log_f.write(f"\n===== util={gpu_memory_utilization} {now_iso()} =====\n")
    log_f.flush()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HOME"] = os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
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
    print("starting vLLM util=", gpu_memory_utilization)
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT), env=env
    )
    return proc, log_f


def stop_vllm(proc: subprocess.Popen | None, log_f) -> None:
    if proc is not None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=60)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=15)
            except Exception:
                pass
    if log_f is not None:
        try:
            log_f.close()
        except Exception:
            pass
    time.sleep(2)
    # ensure dead
    for _ in range(30):
        try:
            requests.get(f"http://{HOST}:{PORT}/health", timeout=0.4)
            time.sleep(0.5)
        except Exception:
            break
    g = gpu_stats()
    # wait for VRAM drop if needed
    t0 = time.time()
    while g["nvidia_smi_used_mb"] > 3000 and time.time() - t0 < 90:
        time.sleep(1)
        g = gpu_stats()


def wait_healthy(timeout_s: float = 360.0) -> float:
    t0 = time.perf_counter()
    last = ""
    while time.perf_counter() - t0 < timeout_s:
        try:
            r = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if r.status_code == 200:
                return (time.perf_counter() - t0) * 1000.0
            last = f"status={r.status_code}"
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(1.5)
    raise RuntimeError(f"vLLM not healthy: {last}")


def text_completion_once(prompt: str, max_tokens: int = 64) -> dict[str, Any]:
    payload = {
        "model": SERVED,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = None
    text = ""
    toks = 0
    err = None
    try:
        with requests.post(
            f"http://{HOST}:{PORT}/v1/completions",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            if resp.status_code != 200:
                err = f"http_{resp.status_code}: {resp.text[:200]}"
            else:
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = (obj.get("choices") or [{}])[0].get("text") or ""
                    if delta and t_first is None:
                        t_first = time.perf_counter()
                    text += delta
                    u = obj.get("usage") or {}
                    if u.get("completion_tokens"):
                        toks = int(u["completion_tokens"])
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    t1 = time.perf_counter()
    if toks <= 0:
        toks = max(len(text.split()), 1 if text else 0)
    e2e = (t1 - t0) * 1000.0
    ttft = None if t_first is None else (t_first - t0) * 1000.0
    decode_tok_s = None
    if t_first is not None and toks > 0:
        decode_tok_s = round(toks / max(t1 - t_first, 1e-6), 2)
    return {
        "ok": err is None and bool(text),
        "error": err,
        "ttft_ms": None if ttft is None else round(ttft, 3),
        "e2e_ms": round(e2e, 3),
        "completion_tokens": toks,
        "decode_tok_s": decode_tok_s,
        "preview": text[:120],
    }


def load_baseline_swap() -> dict[str, Any]:
    vis = json.loads(
        (PROJECT_ROOT / "results/profiling/final_system/vision_path_profile.json").read_text()
    )
    return {
        "stage": "K.2_baseline_from_K.1",
        "timestamp": now_iso(),
        "mode": "full_swap",
        "text_util": 0.90,
        "swap_overhead_mean_ms": vis["measured_swap_unload_reload"][
            "mean_total_swap_overhead_excluding_infer_ms"
        ],
        "E": vis["E"]["swap_breakdown_ms"],
        "F": vis["F"]["swap_breakdown_ms"],
        "vision_e2e_with_swap_ms": {
            "E": vis["E"]["trace"]["e2e_ms"],
            "F": vis["F"]["trace"]["e2e_ms"],
        },
        "components_mean_approx_ms": {
            "text_unload": round(
                (vis["E"]["swap_breakdown_ms"]["text_unload_release"]
                 + vis["F"]["swap_breakdown_ms"]["text_unload_release"])
                / 2,
                3,
            ),
            "vlm_load": round(
                (vis["E"]["swap_breakdown_ms"]["vlm_load"]
                 + vis["F"]["swap_breakdown_ms"]["vlm_load"])
                / 2,
                3,
            ),
            "vlm_infer_generate": round(
                (vis["E"]["swap_breakdown_ms"]["vlm_generate"]
                 + vis["F"]["swap_breakdown_ms"]["vlm_generate"])
                / 2,
                3,
            ),
            "vlm_unload": round(
                (vis["E"]["swap_breakdown_ms"]["vlm_unload"]
                 + vis["F"]["swap_breakdown_ms"]["vlm_unload"])
                / 2,
                3,
            ),
            "text_restore": round(
                (vis["E"]["swap_breakdown_ms"]["text_vllm_restore"]
                 + vis["F"]["swap_breakdown_ms"]["text_vllm_restore"])
                / 2,
                3,
            ),
        },
        "source": "results/profiling/final_system/vision_path_profile.json",
    }


def load_vision():
    from neuro_agent.inference.config import InferenceConfig
    from neuro_agent.multimodal.model import load_vlm_for_inference

    cfg = InferenceConfig(
        model_name=VISION_BASE,
        dtype="bfloat16",
        trust_remote_code=True,
        adapter_path=str(VISION_ADAPTER),
        max_new_tokens=64,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        use_cache=True,
        seed=42,
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = gpu_stats()
    t0 = time.perf_counter()
    try:
        model, processor, info = load_vlm_for_inference(cfg)
        torch.cuda.synchronize()
        ok = True
        err = None
    except Exception as exc:  # noqa: BLE001
        model = processor = info = None
        ok = False
        err = f"{type(exc).__name__}: {exc}"
    wall_ms = (time.perf_counter() - t0) * 1000.0
    after = gpu_stats()
    return {
        "ok": ok,
        "error": err,
        "model": model,
        "processor": processor,
        "wall_ms": round(wall_ms, 3),
        "before": before,
        "after": after,
        "peak_torch_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1)
        if ok
        else None,
        "info": None
        if info is None
        else {
            "model_name": info.model_name,
            "dtype": info.dtype,
            "weight_memory_mb": info.weight_memory_mb,
            "load_time_s": info.load_time_s,
        },
    }


def unload_vision(model) -> None:
    del model
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    time.sleep(1)


def run_vision_infer(model, processor, image_path: Path, question: str) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    from neuro_agent.multimodal.dataset import build_multimodal_messages

    system_prompt = (
        "You are a neuroscience research assistant analyzing EEG-derived plots. "
        "Answer briefly based on the image."
    )
    messages = build_multimodal_messages(
        system_prompt=system_prompt,
        user_text=f"Question: {question.strip()}",
        image_uri=f"file://{image_path.resolve()}",
    )
    torch.cuda.reset_peak_memory_stats()
    t_pre0 = time.perf_counter()
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    torch.cuda.synchronize()
    preprocess_ms = (time.perf_counter() - t_pre0) * 1000.0

    t_gen0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    gen_ms = (time.perf_counter() - t_gen0) * 1000.0
    in_len = int(inputs["input_ids"].shape[-1])
    n_new = int(out[0][in_len:].numel())
    decoded = processor.batch_decode(
        out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return {
        "ok": True,
        "preprocess_ms": round(preprocess_ms, 3),
        "generate_ms": round(gen_ms, 3),
        "e2e_ms": round(preprocess_ms + gen_ms, 3),
        "completion_tokens": n_new,
        "output": decoded.strip()[:200],
        "peak_torch_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "gpu": gpu_stats(),
    }


async def _one_async(session: aiohttp.ClientSession, prompt: str) -> dict[str, Any]:
    payload = {
        "model": SERVED,
        "prompt": prompt,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }
    t0 = time.perf_counter()
    t_first = None
    text = ""
    toks = 0
    err = None
    try:
        async with session.post(f"http://{HOST}:{PORT}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                err = f"http_{resp.status}"
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
                    delta = (obj.get("choices") or [{}])[0].get("text") or ""
                    if delta and t_first is None:
                        t_first = time.perf_counter()
                    text += delta
                    u = obj.get("usage") or {}
                    if u.get("completion_tokens"):
                        toks = int(u["completion_tokens"])
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    t1 = time.perf_counter()
    if toks <= 0:
        toks = max(len(text.split()), 1 if text else 0)
    return {
        "ok": err is None and t_first is not None,
        "error": err,
        "ttft_ms": None if t_first is None else (t_first - t0) * 1000.0,
        "e2e_ms": (t1 - t0) * 1000.0,
        "completion_tokens": toks,
    }


async def bench_concurrency(concurrency: int, n_requests: int) -> dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:

        async def run_one(i: int):
            async with sem:
                # slight prompt diversity
                prompt = BENCH_PROMPT + f" [req={i}]"
                return await _one_async(session, prompt)

        t_wall0 = time.perf_counter()
        results = await asyncio.gather(*[run_one(i) for i in range(n_requests)])
        wall_s = time.perf_counter() - t_wall0

    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in ok]
    toks = sum(r["completion_tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "n_submitted": n_requests,
        "completed_requests": len(ok),
        "failed_requests": len(fail),
        "wall_s": round(wall_s, 3),
        "requests_per_sec": round(len(ok) / max(wall_s, 1e-6), 4),
        "output_tokens_per_sec": round(toks / max(wall_s, 1e-6), 2),
        "ttft_ms": {
            "p50": None if not ttfts else round(pctile(ttfts, 0.50), 3),
            "p95": None if not ttfts else round(pctile(ttfts, 0.95), 3),
            "mean": None if not ttfts else round(statistics.mean(ttfts), 3),
        },
        "e2e_ms": {
            "p50": None if not e2es else round(pctile(e2es, 0.50), 3),
            "p95": None if not e2es else round(pctile(e2es, 0.95), 3),
            "mean": None if not e2es else round(statistics.mean(e2es), 3),
        },
        "gpu_peak_during": gpu_stats(),
        "failures": [r.get("error") for r in fail[:5]],
        "oom_suspected": any(
            r.get("error") and "out of memory" in str(r.get("error")).lower() for r in fail
        ),
    }


def run_vision_suite(model, processor) -> dict[str, Any]:
    from neuro_agent.tools.ranking import rank_channels_for_sample

    questions = {
        "topomap": "Where is power concentrated in this topomap?",
        "spectrogram": "What temporal-frequency pattern is visible in this spectrogram?",
        "psd": "What pattern is visible in this PSD figure?",
        "waveform": "Describe the waveform morphology in this figure briefly.",
    }
    rows = {}
    for kind, q in questions.items():
        print(f"  vision warm {kind}")
        rows[kind] = run_vision_infer(model, processor, IMAGES[kind], q)

    # combined vision+tool (tools on CPU; VLM already warm — no swap)
    t0 = time.perf_counter()
    ranking = rank_channels_for_sample("S001_R01_E000", "beta_power", top_k=3)
    tool_ms = (time.perf_counter() - t0) * 1000.0
    top = list(ranking.ranking)[:3]
    q = (
        f"Tools ranked beta channels as {top}. "
        "Does this topomap visually support that ranking?"
    )
    print("  vision+tool combined")
    vis = run_vision_infer(model, processor, IMAGES["topomap"], q)
    rows["vision_tool_combined"] = {
        **vis,
        "tool_ms": round(tool_ms, 3),
        "ranking": top,
        "total_e2e_ms": round(tool_ms + vis["e2e_ms"], 3),
        "swap_overhead_ms": 0.0,
        "note": "co-resident: no text unload / VLM load / text restore",
    }

    e2es = [rows[k]["e2e_ms"] for k in ("topomap", "spectrogram", "psd", "waveform") if rows[k].get("ok")]
    return {
        "rows": rows,
        "summary": {
            "n": len(e2es),
            "e2e_ms_p50": None if not e2es else round(pctile(e2es, 0.50), 3),
            "e2e_ms_mean": None if not e2es else round(statistics.mean(e2es), 3),
            "combined_total_e2e_ms": rows["vision_tool_combined"]["total_e2e_ms"],
            "swap_overhead_ms": 0.0,
        },
        "gpu_after": gpu_stats(),
    }


def audit_sleep_wake() -> dict[str, Any]:
    """Bounded check whether this vLLM supports sleep/wake without full reload."""
    notes = []
    try:
        import vllm

        ver = getattr(vllm, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        return {"supported": False, "vllm_version": None, "error": str(exc)}

    # Probe API / docs surface
    found = []
    try:
        import vllm.engine.arg_utils as arg_utils  # noqa: F401

        for mod_name in (
            "vllm.entrypoints.openai.api_server",
            "vllm.engine.llm_engine",
            "vllm.v1.engine.async_llm",
        ):
            try:
                mod = __import__(mod_name, fromlist=["*"])
                names = dir(mod)
                hits = [n for n in names if "sleep" in n.lower() or "wake" in n.lower() or "hibernate" in n.lower()]
                if hits:
                    found.append({"module": mod_name, "symbols": hits})
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        notes.append(str(exc))

    # Check CLI help for sleep-related flags
    try:
        help_out = subprocess.check_output(
            [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=60,
        )
        sleep_lines = [
            ln.strip()
            for ln in help_out.splitlines()
            if any(k in ln.lower() for k in ("sleep", "wake", "hibernate", "offload"))
        ]
    except Exception as exc:  # noqa: BLE001
        sleep_lines = []
        notes.append(f"help_probe: {exc}")

    supported = bool(found) or bool(sleep_lines)
    return {
        "vllm_version": ver,
        "supported_surface_found": supported,
        "symbol_hits": found,
        "cli_help_hits": sleep_lines[:20],
        "notes": notes
        + [
            "K.2 policy: only run sleep/wake smoke if co-residency text capacity is unacceptable.",
            "No smoke executed unless decision path requires it.",
        ],
    }


def decide(
    baseline: dict[str, Any],
    cores: dict[float, dict[str, Any]],
    sleep_audit: dict[str, Any],
) -> dict[str, Any]:
    safe = {
        u: r
        for u, r in cores.items()
        if r.get("safe") and r.get("vision_load_ok") and r.get("text_request_ok") and r.get("vision_request_ok")
    }
    if not safe:
        return {
            "strategy": "B_text_primary_full_swap",
            "reason": "No safe co-resident configuration found; keep K.1 full swap.",
            "selected_util": None,
            "k2_verdict": "FAIL",
        }

    best_util = max(safe.keys())
    best = safe[best_util]
    # Text capacity impact at c=8 (production-ish)
    text_c8 = (best.get("text_capacity") or {}).get("8") or {}
    i1_c8 = I1_REF[8]
    tok_ratio = None
    if text_c8.get("output_tokens_per_sec") and i1_c8["output_tok_s"]:
        tok_ratio = text_c8["output_tokens_per_sec"] / i1_c8["output_tok_s"]
    e2e_p95_delta = None
    if text_c8.get("e2e_ms", {}).get("p95") is not None:
        e2e_p95_delta = text_c8["e2e_ms"]["p95"] - i1_c8["e2e_p95_ms"]

    vis_mean = (best.get("vision_warm") or {}).get("summary", {}).get("e2e_ms_mean")
    swap_base = baseline["swap_overhead_mean_ms"]
    # co-resident vision E2E ≈ warm infer only; swap overhead ~0
    vision_before = baseline["vision_e2e_with_swap_ms"]["E"]  # includes swap wall
    # fairer: swap_overhead + warm_infer_proxy from K.1 generate
    k1_infer = baseline["components_mean_approx_ms"]["vlm_infer_generate"]
    vision_after = vis_mean
    swap_after = 0.0
    reduction = (swap_base - swap_after) / swap_base * 100.0 if swap_base else None

    # Decision heuristics
    # Unacceptable text loss: tok/s < 50% of I.1 at c=8 OR cannot run c=8
    unacceptable = False
    if not text_c8 or text_c8.get("failed_requests", 0) > 0 or text_c8.get("completed_requests", 0) == 0:
        unacceptable = True
    if tok_ratio is not None and tok_ratio < 0.50:
        unacceptable = True

    c16 = (best.get("text_capacity") or {}).get("16")
    can_c16 = bool(c16 and c16.get("completed_requests", 0) > 0 and not c16.get("oom_suspected"))

    if unacceptable:
        strategy = "C_hybrid_policy"
        reason = (
            "Co-residency removes ~58s swap but text capacity loss at reduced util is severe; "
            "use hybrid: util=0.90 text-primary for normal traffic; reduced-util co-resident "
            "mode for vision-heavy/demo sessions."
        )
        sleep_note = {
            "audit": sleep_audit,
            "smoke_run": False,
            "note": (
                "Sleep/wake smoke skipped unless a clear supported API exists; "
                f"surface_found={sleep_audit.get('supported_surface_found')}."
            ),
        }
        verdict = "PARTIAL PASS"
    else:
        c16_ok = can_c16
        if c16_ok and tok_ratio is not None and tok_ratio >= 0.70:
            strategy = "A_co_resident_production"
            reason = (
                f"Highest safe util={best_util:.2f} keeps VLM warm (swap≈0). Vision E2E drops from "
                f"~{vision_before/1000:.1f}s (with swap) to ~{(vision_after or 0)/1000:.2f}s warm. "
                f"Text c=8 tok/s ratio vs I.1 util=0.90 ≈ {round(tok_ratio, 3)}; c=16 still runnable."
            )
            verdict = "PASS"
        else:
            strategy = "C_hybrid_policy"
            reason = (
                f"Co-resident util={best_util:.2f} is safe and eliminates ~58s swap, but text "
                f"capacity vs I.1 util=0.90 is reduced "
                f"(c8 tok/s ratio={None if tok_ratio is None else round(tok_ratio, 3)}, "
                f"c16_ok={c16_ok}). Recommend hybrid: util=0.90 for text-heavy production; "
                f"util={best_util:.2f} co-resident for vision-enabled mode."
            )
            verdict = "PARTIAL PASS"
        sleep_note = {
            "skipped": True,
            "reason": "Co-residency path chosen; no sleep/wake smoke required for K.2.",
            "audit": sleep_audit,
        }

    return {
        "strategy": strategy,
        "selected_util": best_util,
        "reason": reason,
        "k2_verdict": verdict,
        "metrics": {
            "swap_overhead_before_ms": swap_base,
            "swap_overhead_after_ms": swap_after,
            "swap_reduction_pct": None if reduction is None else round(reduction, 2),
            "vision_e2e_before_with_swap_ms": vision_before,
            "vision_e2e_after_warm_ms": vision_after,
            "k1_vlm_generate_mean_ms": k1_infer,
            "text_c8_tok_s_ratio_vs_i1": None if tok_ratio is None else round(tok_ratio, 4),
            "text_c8_e2e_p95_delta_ms_vs_i1": None
            if e2e_p95_delta is None
            else round(e2e_p95_delta, 3),
            "can_run_c16_coresident": can_c16,
            "combined_idle_vram_mb": (best.get("after_both_idle") or {}).get("nvidia_smi_used_mb"),
            "free_vram_mb": (best.get("after_both_idle") or {}).get("nvidia_smi_free_mb"),
        },
        "sleep_wake": sleep_note,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CMP.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_swap()
    (OUT / "baseline_swap.json").write_text(json.dumps(baseline, indent=2))
    print("baseline swap mean ms", baseline["swap_overhead_mean_ms"])

    utils_to_test = [0.40, 0.45, 0.50]
    cores: dict[float, dict[str, Any]] = {}
    stop_further = False

    for util in utils_to_test:
        if stop_further:
            print(f"skip util={util} — prior config unsafe")
            break

        print(f"\n===== CO-RESIDENT util={util:.2f} =====")
        result: dict[str, Any] = {
            "stage": "K.2",
            "timestamp": now_iso(),
            "gpu_memory_utilization": util,
            "safe": False,
            "oom": False,
            "error": None,
        }
        proc = log_f = None
        model = None
        try:
            clean = gpu_stats()
            result["clean_gpu"] = clean
            proc, log_f = start_vllm(util)
            boot_ms = wait_healthy()
            result["text_load_ok"] = True
            result["text_boot_ms"] = round(boot_ms, 3)
            result["after_text_load"] = gpu_stats()

            # text smoke
            tr = text_completion_once(BENCH_PROMPT, max_tokens=32)
            result["text_request_ok"] = bool(tr["ok"])
            result["text_smoke"] = tr
            if not tr["ok"]:
                result["error"] = tr.get("error")
                stop_further = True
            else:
                # load vision while text resident
                print("loading VLM alongside text...")
                vis_load = load_vision()
                result["vision_load_ok"] = vis_load["ok"]
                result["vision_load_ms"] = vis_load["wall_ms"]
                result["vision_load_error"] = vis_load.get("error")
                result["after_both_idle"] = vis_load["after"]
                result["vision_info"] = vis_load.get("info")
                model = vis_load.get("model")
                processor = vis_load.get("processor")

                if not vis_load["ok"]:
                    result["oom"] = "out of memory" in str(vis_load.get("error") or "").lower()
                    result["error"] = vis_load.get("error")
                    stop_further = True
                else:
                    free = vis_load["after"]["nvidia_smi_free_mb"]
                    result["safety_margin_mb"] = free
                    # unsafe if free < 1.5 GB
                    if free < 1500:
                        result["safe"] = False
                        result["error"] = f"insufficient free VRAM margin: {free} MB"
                        stop_further = True
                    else:
                        # vision smoke
                        vs = run_vision_infer(
                            model,
                            processor,
                            IMAGES["topomap"],
                            "Where is power concentrated in this topomap?",
                        )
                        result["vision_request_ok"] = bool(vs.get("ok"))
                        result["vision_smoke"] = vs
                        result["peak_vram_after_vision_mb"] = gpu_stats()["nvidia_smi_used_mb"]

                        if not vs.get("ok"):
                            result["error"] = "vision request failed"
                            stop_further = True
                        else:
                            result["safe"] = True
                            # text capacity with both resident
                            cap: dict[str, Any] = {}
                            for c, n in ((1, 16), (8, 24)):
                                print(f"  text capacity c={c} n={n}")
                                cap[str(c)] = asyncio.run(bench_concurrency(c, n))
                            # c=16 only if free margin still healthy after c=8
                            free_now = gpu_stats()["nvidia_smi_free_mb"]
                            if free_now >= 2000 and not cap["8"].get("oom_suspected"):
                                print("  text capacity c=16 n=32")
                                cap["16"] = asyncio.run(bench_concurrency(16, 32))
                            else:
                                cap["16"] = {
                                    "skipped": True,
                                    "reason": f"free_mb={free_now} or c8 oom risk",
                                }
                            result["text_capacity"] = cap

                            # warm vision suite (no reload)
                            print("  warm vision suite")
                            result["vision_warm"] = run_vision_suite(model, processor)
                            result["gpu_final"] = gpu_stats()

        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
            result["oom"] = "out of memory" in str(exc).lower()
            result["traceback"] = traceback.format_exc()[-2000:]
            stop_further = True
            print("ERROR", result["error"])
        finally:
            if model is not None:
                try:
                    unload_vision(model)
                except Exception:
                    pass
            model = None
            stop_vllm(proc, log_f)
            # force free
            torch.cuda.empty_cache()
            time.sleep(2)

        tag = f"{int(util*100):03d}"
        path = OUT / f"co_resident_{tag}.json"
        # strip non-serializable
        serial = {k: v for k, v in result.items()}
        path.write_text(json.dumps(serial, indent=2, default=str))
        cores[util] = result
        print(
            f"util={util} safe={result.get('safe')} text_ok={result.get('text_request_ok')} "
            f"vision_ok={result.get('vision_request_ok')} free={result.get('safety_margin_mb')}"
        )
        if not result.get("safe"):
            stop_further = True

    # Aggregates
    sleep_audit = audit_sleep_wake()
    decision = decide(baseline, cores, sleep_audit)
    (OUT / "production_decision.json").write_text(json.dumps(decision, indent=2))

    # text capacity impact table
    text_impact = {
        "timestamp": now_iso(),
        "i1_reference_util_0.90": I1_REF,
        "co_resident": {},
    }
    for util, r in cores.items():
        if not r.get("text_capacity"):
            continue
        text_impact["co_resident"][f"{util:.2f}"] = r["text_capacity"]
    (OUT / "text_capacity_impact.json").write_text(json.dumps(text_impact, indent=2))

    # vision latency impact
    selected = decision.get("selected_util")
    vis_impact = {
        "timestamp": now_iso(),
        "baseline_full_swap": {
            "swap_overhead_mean_ms": baseline["swap_overhead_mean_ms"],
            "vision_e2e_with_swap_ms": baseline["vision_e2e_with_swap_ms"],
            "components": baseline["components_mean_approx_ms"],
        },
        "co_resident_warm": {},
    }
    for util, r in cores.items():
        if r.get("vision_warm"):
            vis_impact["co_resident_warm"][f"{util:.2f}"] = r["vision_warm"]["summary"]
            vis_impact["co_resident_warm"][f"{util:.2f}_rows"] = {
                k: {
                    "e2e_ms": v.get("e2e_ms"),
                    "preprocess_ms": v.get("preprocess_ms"),
                    "generate_ms": v.get("generate_ms"),
                }
                for k, v in r["vision_warm"]["rows"].items()
            }
    (OUT / "vision_latency_impact.json").write_text(json.dumps(vis_impact, indent=2))

    # comparison rollup
    sel = cores.get(selected) if selected is not None else None
    comparison = {
        "stage": "K.2",
        "timestamp": now_iso(),
        "configs_tested": sorted(cores.keys()),
        "highest_safe_util": selected,
        "baseline_A_full_swap": {
            "text_util": 0.90,
            "swap_overhead_ms": baseline["swap_overhead_mean_ms"],
            "vision_e2e_ms": baseline["vision_e2e_with_swap_ms"]["E"],
        },
        "co_resident_B": None
        if sel is None
        else {
            "text_util": selected,
            "swap_overhead_ms": 0.0,
            "combined_idle_vram_mb": (sel.get("after_both_idle") or {}).get("nvidia_smi_used_mb"),
            "free_vram_mb": (sel.get("after_both_idle") or {}).get("nvidia_smi_free_mb"),
            "vision_warm_e2e_mean_ms": (sel.get("vision_warm") or {}).get("summary", {}).get(
                "e2e_ms_mean"
            ),
            "text_capacity": sel.get("text_capacity"),
        },
        "delta": decision.get("metrics"),
        "production_decision": decision,
        "sleep_wake_audit": sleep_audit,
        "artifacts": {
            "baseline_swap": str(OUT / "baseline_swap.json"),
            "co_resident_files": sorted(str(p) for p in OUT.glob("co_resident_*.json")),
            "text_capacity_impact": str(OUT / "text_capacity_impact.json"),
            "vision_latency_impact": str(OUT / "vision_latency_impact.json"),
            "production_decision": str(OUT / "production_decision.json"),
        },
    }
    (CMP / "model_swap_vs_co_resident.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps({"k2_verdict": decision["k2_verdict"], "strategy": decision["strategy"], "util": selected, "metrics": decision.get("metrics")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
        raise
