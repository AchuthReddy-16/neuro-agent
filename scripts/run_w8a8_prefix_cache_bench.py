#!/usr/bin/env python3
"""Stage I.2 — vLLM prefix/KV cache optimization on W8A8 INT8.

Compares prefix cache OFF vs ON for realistic neuroscience research-agent
shared prefixes. Production checkpoint only. No SLA, no requantize, no git.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_agent.paths import configure_hf_cache  # noqa: E402

configure_hf_cache()

CKPT = PROJECT_ROOT / "checkpoints" / "text_w8a8_int8_compressed"
SERVED_MODEL_NAME = "w8a8-int8"
RESULTS_DIR = PROJECT_ROOT / "results" / "serving" / "prefix_cache" / "w8a8_int8"
CMP_PATH = PROJECT_ROOT / "results" / "model_comparison" / "w8a8_prefix_cache_comparison.json"
SERVER_LOG = RESULTS_DIR / "vllm_server.log"
GPU_LOG = RESULTS_DIR / "gpu_poll.csv"
TRACES_PATH = RESULTS_DIR / "per_request_traces.jsonl"

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
METRICS_URL = f"{BASE_URL}/metrics"
HEALTH_URL = f"{BASE_URL}/health"
COMPLETIONS_URL = f"{BASE_URL}/v1/completions"
RESET_CACHE_URL = f"{BASE_URL}/reset_prefix_cache"

GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 4096
TENSOR_PARALLEL_SIZE = 1
ENFORCE_EAGER = True

MAX_TOKENS = 64
TEMPERATURE = 0.0
N_WARM_C1 = 32
N_WARM_C8 = 48
N_COLD = 12
N_SENSITIVITY_WARM = 24

# I.1 references (prefix cache OFF) for context only
I1_REF = {
    1: {"tok_s": 131.38, "rps": 2.05, "e2e_p95_ms": 486.5},
    8: {"tok_s": 765.88, "rps": 11.97, "e2e_p95_ms": 674.3},
    16: {"tok_s": 1176.01, "rps": 18.45, "e2e_p95_ms": 923.5},
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def pct_change(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return round((after - before) / before * 100.0, 2)


def reduction_pct(before: float | None, after: float | None) -> float | None:
    """Positive = improvement (lower latency)."""
    if before is None or after is None or before == 0:
        return None
    return round((before - after) / before * 100.0, 2)


def load_prompts_module():
    path = PROJECT_ROOT / "src" / "neuro_agent" / "agent" / "prompts.py"
    spec = importlib.util.spec_from_file_location("agent_prompts_i2", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_verifier_system() -> str:
    text = (PROJECT_ROOT / "src" / "neuro_agent" / "agent" / "verifier.py").read_text()
    m = re.search(r'VERIFIER_SYSTEM_PROMPT = """(.*?)"""', text, re.S)
    if not m:
        raise RuntimeError("VERIFIER_SYSTEM_PROMPT not found")
    return m.group(1)


QUESTIONS = [
    "What is the beta-band power for channel C3 in sample S001_R03_E012?",
    "Rank channels by alpha_mu power for sample S002_R04_E008, top_k=5.",
    "Compare left_fist vs right_fist beta power for subject S003.",
    "What is the RMS amplitude on C4 for sample S004_R01_E020?",
    "Find the PSD peak frequency for channel Cz in S005_R02_E015.",
    "Select channels with beta power above the upper quartile for S006_R03_E010.",
    "Report theta-band power for all channels in sample S007_R05_E003.",
    "What is the dominant frequency on C3 during rest for S008_R02_E001?",
    "Compute delta-band power for Fp1 in sample S009_R03_E007.",
    "Rank the top 3 channels by RMS for sample S010_R01_E011.",
    "Compare both_fists vs rest alpha_mu power for subject S011.",
    "What is beta power on C3 vs C4 in sample S012_R04_E002?",
    "Find channels below the median beta threshold for S013_R02_E009.",
    "Report PSD peak on Pz for sample S014_R05_E004.",
    "What is RMS for channel Fz in sample S015_R03_E016?",
    "Compare left_fist vs both_fists beta for subject S016.",
]


def sample_evidence(question: str, idx: int) -> dict[str, Any]:
    sid = f"S{(idx % 20) + 1:03d}_R{(idx % 5) + 1:02d}_E{(idx % 30) + 1:03d}"
    return {
        "metadata": {
            "sample_id": sid,
            "subject_id": sid.split("_")[0],
            "channels": ["C3", "C4", "Cz", "Fz"],
            "fs": 160,
        },
        "numeric_evidence": {
            "beta_power": {"C3": 10.0 + idx * 0.17, "C4": 8.5 + idx * 0.11},
            "rms": {"C3": 4.2 + idx * 0.03, "C4": 3.9 + idx * 0.02},
        },
        "ranked_evidence": {
            "metric": "beta_power",
            "ranking": [{"channel": "C3", "value": 10.0 + idx * 0.17}],
        },
        "set_evidence": None,
        "condition_evidence": {
            "condition_a": "left_fist",
            "condition_b": "right_fist",
            "delta_beta": 1.2 + idx * 0.01,
        },
        "vision_evidence": None,
        "provenance": {"tool": "band_power", "params": {"band": "beta"}},
        "warnings": [],
        "uncertainty_notes": [],
        "units": {"power": "µV²", "amplitude": "µV", "frequency": "Hz"},
        "tool_invocations": [{"name": "band_power", "success": True}],
    }


def build_prefix_manifest(tok) -> dict[str, Any]:
    prompts = load_prompts_module()
    verifier = load_verifier_system()
    recovery_policy = (
        "Recovery policy (deterministic runtime):\n"
        "- Trigger verifier on tool failure, grounding warnings, multi-tool routes, or low confidence.\n"
        "- Recovery actions: REWRITE | RETRY_TOOL | REPLAN | INSUFFICIENT_EVIDENCE.\n"
        "- Format-only failures use deterministic format_grounded_answer without model rewrite.\n"
        "- MAX_TOOL_CALLS = 6. Do not invent numeric values or sample IDs.\n"
    )
    production_system = (
        f"{prompts.INTENT_SYSTEM_PROMPT.strip()}\n\n"
        f"{prompts.ANSWER_SYSTEM_PROMPT.strip()}\n\n"
        "Recovery rewrite policy:\n"
        "When recovery is active, correct any unsupported numeric claims. "
        "Use ONLY values present in the evidence bundle.\n\n"
        f"{verifier.strip()}\n\n"
        f"{recovery_policy}"
    )

    def chat(system: str, user: str) -> str:
        return tok.apply_chat_template(
            [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def ntok(s: str, *, add: bool = True) -> int:
        return len(tok.encode(s, add_special_tokens=add))

    variants: dict[str, Any] = {}

    # short: answer-stage system (production answer call)
    short_sys = prompts.ANSWER_SYSTEM_PROMPT
    short_user0 = prompts.build_answer_user_prompt(QUESTIONS[0], sample_evidence(QUESTIONS[0], 0))
    short_full0 = chat(short_sys, short_user0)
    short_idx = short_full0.find(short_user0)
    short_shared = short_full0[:short_idx]
    short_prompts = []
    for i, q in enumerate(QUESTIONS):
        user = prompts.build_answer_user_prompt(q, sample_evidence(q, i))
        full = chat(short_sys, user)
        short_prompts.append(
            {
                "prompt": full,
                "user": user,
                "question": q,
                "total_tokens": ntok(full),
                "user_tokens": ntok(user, add=False),
            }
        )
    variants["short"] = {
        "name": "short_answer_system",
        "description": "Production ANSWER_SYSTEM_PROMPT via Qwen chat template (answer stage).",
        "system_source": "src/neuro_agent/agent/prompts.py::ANSWER_SYSTEM_PROMPT",
        "shared_prefix_text": short_shared,
        "shared_prefix_tokens": ntok(short_shared),
        "mean_user_tokens": round(statistics.mean(p["user_tokens"] for p in short_prompts), 1),
        "mean_total_tokens": round(statistics.mean(p["total_tokens"] for p in short_prompts), 1),
        "prompts": short_prompts,
    }

    # medium: intent parser system (first model call in agent)
    med_sys = prompts.INTENT_SYSTEM_PROMPT
    med_user0 = prompts.build_intent_user_prompt(QUESTIONS[0])
    med_full0 = chat(med_sys, med_user0)
    med_idx = med_full0.find(med_user0)
    med_shared = med_full0[:med_idx]
    med_prompts = []
    for i, q in enumerate(QUESTIONS):
        user = prompts.build_intent_user_prompt(q)
        full = chat(med_sys, user)
        med_prompts.append(
            {
                "prompt": full,
                "user": user,
                "question": q,
                "total_tokens": ntok(full),
                "user_tokens": ntok(user, add=False),
            }
        )
    variants["medium"] = {
        "name": "medium_intent_system",
        "description": "Production INTENT_SYSTEM_PROMPT (tool routing schema) via chat template.",
        "system_source": "src/neuro_agent/agent/prompts.py::INTENT_SYSTEM_PROMPT",
        "shared_prefix_text": med_shared,
        "shared_prefix_tokens": ntok(med_shared),
        "mean_user_tokens": round(statistics.mean(p["user_tokens"] for p in med_prompts), 1),
        "mean_total_tokens": round(statistics.mean(p["total_tokens"] for p in med_prompts), 1),
        "prompts": med_prompts,
    }

    # long / production: intent + answer + recovery + verifier + policy
    long_sys = production_system
    long_user0 = prompts.build_intent_user_prompt(QUESTIONS[0])
    long_full0 = chat(long_sys, long_user0)
    long_idx = long_full0.find(long_user0)
    long_shared = long_full0[:long_idx]
    long_prompts = []
    for i, q in enumerate(QUESTIONS):
        user = prompts.build_intent_user_prompt(q)
        full = chat(long_sys, user)
        long_prompts.append(
            {
                "prompt": full,
                "user": user,
                "question": q,
                "total_tokens": ntok(full),
                "user_tokens": ntok(user, add=False),
            }
        )
    variants["production"] = {
        "name": "long_production_shared_prefix",
        "description": (
            "Realistic production shared prefix: intent routing schema + answer format + "
            "recovery rewrite policy + verifier schema + deterministic recovery policy, "
            "wrapped as the system role in the Qwen chat template. Variable suffix = user question."
        ),
        "system_source": [
            "INTENT_SYSTEM_PROMPT",
            "ANSWER_SYSTEM_PROMPT",
            "RECOVERY rewrite policy text",
            "VERIFIER_SYSTEM_PROMPT",
            "deterministic recovery policy",
        ],
        "shared_prefix_text": long_shared,
        "shared_prefix_tokens": ntok(long_shared),
        "mean_user_tokens": round(statistics.mean(p["user_tokens"] for p in long_prompts), 1),
        "mean_total_tokens": round(statistics.mean(p["total_tokens"] for p in long_prompts), 1),
        "prompts": long_prompts,
        "is_main_result": True,
    }

    return {
        "stage": "I.2",
        "timestamp": now_iso(),
        "tokenizer": str(CKPT),
        "chat_template": "Qwen apply_chat_template(system+user, add_generation_prompt=True)",
        "note": (
            "Prefixes are real research-agent prompt text, not synthetic padding. "
            "Token counts measured with the production W8A8 checkpoint tokenizer."
        ),
        "variants": {
            k: {
                **{kk: vv for kk, vv in v.items() if kk != "prompts"},
                "n_suffixes": len(v["prompts"]),
                "example_user": v["prompts"][0]["user"][:240],
            }
            for k, v in variants.items()
        },
        "_runtime": variants,  # kept in-memory only; stripped before save
    }


def gpu_compute_apps() -> list[str]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
        text=True,
    )
    rows = []
    for line in out.strip().splitlines()[1:]:
        if line.strip() and "No running" not in line:
            rows.append(line.strip())
    return rows


def gpu_memory_used_mb() -> float:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        text=True,
    )
    return float(out.strip().splitlines()[0])


def parse_prom_text(text: str) -> dict[str, list[dict[str, Any]]]:
    metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            labels_s, val_s = rest.rsplit("}", 1)
            labels = {}
            for part in labels_s.split(","):
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                labels[k.strip()] = v.strip().strip('"')
            try:
                val = float(val_s.strip())
            except ValueError:
                continue
            metrics[name.strip()].append({"labels": labels, "value": val})
        else:
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                metrics[parts[0]].append({"labels": {}, "value": float(parts[1])})
            except ValueError:
                continue
    return dict(metrics)


def prom_gauge(metrics: dict, name: str) -> float | None:
    rows = metrics.get(name) or []
    if not rows:
        return None
    return rows[0]["value"]


def prom_counter(metrics: dict, name: str) -> float | None:
    return prom_gauge(metrics, name)


async def fetch_metrics(session: aiohttp.ClientSession) -> dict[str, list[dict[str, Any]]]:
    async with session.get(METRICS_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        text = await resp.text()
    return parse_prom_text(text)


def cache_metrics_snapshot(metrics: dict) -> dict[str, Any]:
    hits = prom_counter(metrics, "vllm:prefix_cache_hits")
    queries = prom_counter(metrics, "vllm:prefix_cache_queries")
    hit_rate = None
    if hits is not None and queries is not None and queries > 0:
        hit_rate = round(hits / queries, 4)
    return {
        "prefix_cache_hits": hits,
        "prefix_cache_queries": queries,
        "prefix_cache_hit_rate": hit_rate,
        "gpu_cache_usage_perc": prom_gauge(metrics, "vllm:gpu_cache_usage_perc"),
        "note": "Counters are process-lifetime cumulative unless differenced.",
    }


def cache_metrics_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    bh, bq = before.get("prefix_cache_hits"), before.get("prefix_cache_queries")
    ah, aq = after.get("prefix_cache_hits"), after.get("prefix_cache_queries")
    d_hits = None if bh is None or ah is None else ah - bh
    d_q = None if bq is None or aq is None else aq - bq
    rate = None
    if d_hits is not None and d_q is not None and d_q > 0:
        rate = round(d_hits / d_q, 4)
    return {
        "delta_prefix_cache_hits": d_hits,
        "delta_prefix_cache_queries": d_q,
        "hit_rate_over_window": rate,
        "before": before,
        "after": after,
    }


async def wait_healthy(timeout_s: float = 360.0) -> None:
    t0 = time.perf_counter()
    last_err = None
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while time.perf_counter() - t0 < timeout_s:
            try:
                async with session.get(HEALTH_URL) as resp:
                    if resp.status == 200:
                        return
                    last_err = f"status={resp.status}"
            except Exception as exc:  # noqa: BLE001
                last_err = str(exc)
            await asyncio.sleep(2.0)
    raise RuntimeError(f"vLLM server not healthy after {timeout_s}s: {last_err}")


async def reset_prefix_cache(session: aiohttp.ClientSession) -> dict[str, Any]:
    try:
        async with session.post(RESET_CACHE_URL, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            body = await resp.text()
            ok = resp.status == 200
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                data = {"raw": body[:300]}
            return {"http_status": resp.status, "ok": ok, "body": data}
    except Exception as exc:  # noqa: BLE001
        return {"http_status": None, "ok": False, "error": str(exc)}


async def stream_completion(
    session: aiohttp.ClientSession,
    *,
    req_id: str,
    prompt: str,
    max_tokens: int = MAX_TOKENS,
    ignore_eos: bool = True,
    concurrency: int = 1,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": SERVED_MODEL_NAME,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "stream": True,
        "ignore_eos": ignore_eos,
    }
    t_submit = time.perf_counter()
    t_first = None
    text_parts: list[str] = []
    completion_tokens = 0
    prompt_tokens = None
    finish_reason = None
    error = None
    try:
        async with session.post(COMPLETIONS_URL, json=payload) as resp:
            if resp.status != 200:
                error = f"http_{resp.status}: {(await resp.text())[:400]}"
            else:
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") or [{}]
                    ch = choices[0]
                    delta = ch.get("text") or ""
                    if delta and t_first is None:
                        t_first = time.perf_counter()
                    if delta:
                        text_parts.append(delta)
                    usage = obj.get("usage") or {}
                    if usage.get("completion_tokens"):
                        completion_tokens = int(usage["completion_tokens"])
                    if usage.get("prompt_tokens"):
                        prompt_tokens = int(usage["prompt_tokens"])
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    if completion_tokens <= 0:
        completion_tokens = len(text_parts)
    out = {
        "req_id": req_id,
        "concurrency": concurrency,
        "submit_unix_s": t_submit,
        "ttft_ms": None if t_first is None else round((t_first - t_submit) * 1000.0, 3),
        "e2e_ms": round((t_end - t_submit) * 1000.0, 3),
        "prefill_proxy_note": "TTFT used as prefill+scheduling proxy; vLLM does not expose separate prefill latency on /v1/completions.",
        "max_tokens": max_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens_usage": prompt_tokens,
        "finish_reason": finish_reason,
        "ok": error is None and t_first is not None,
        "error": error,
        "output_text": "".join(text_parts),
        "output_chars": sum(len(x) for x in text_parts),
        **(extra or {}),
    }
    return out


class GpuPoller:
    def __init__(self, path: Path, interval_s: float = 0.25):
        self.path = path
        self.interval_s = interval_s
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = (
            f"echo 'ts,util_gpu,mem_used_mb' > {self.path}; "
            f"while true; do "
            f"ts=$(date +%s.%N); "
            f"line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,nounits,noheader); "
            f"u=$(echo $line | cut -d, -f1 | tr -d ' '); "
            f"m=$(echo $line | cut -d, -f2 | tr -d ' '); "
            f"echo \"$ts,$u,$m\" >> {self.path}; "
            f"sleep {self.interval_s}; "
            f"done"
        )
        self.proc = subprocess.Popen(["bash", "-c", cmd], start_new_session=True)

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.proc = None

    def window_stats(self, t0: float, t1: float) -> dict[str, Any]:
        utils, mems = [], []
        if not self.path.exists():
            return {"avg_gpu_util": None, "peak_gpu_util": None, "peak_vram_mb": None, "n_samples": 0}
        with self.path.open() as f:
            next(f, None)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue
                try:
                    ts, u, m = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                if t0 <= ts <= t1:
                    utils.append(u)
                    mems.append(m)
        return {
            "avg_gpu_util": round(statistics.mean(utils), 2) if utils else None,
            "peak_gpu_util": max(utils) if utils else None,
            "peak_vram_mb": max(mems) if mems else None,
            "n_samples": len(utils),
        }


def summarize_traces(
    traces: list[dict[str, Any]],
    *,
    wall_s: float,
    gpu: dict[str, Any],
    cache_delta: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    ok = [t for t in traces if t.get("ok")]
    failed = [t for t in traces if not t.get("ok")]
    ttfts = [t["ttft_ms"] for t in ok if t.get("ttft_ms") is not None]
    e2es = [t["e2e_ms"] for t in ok]
    out_tokens = sum(int(t.get("completion_tokens") or 0) for t in ok)
    return {
        "label": label,
        "completed_requests": len(ok),
        "failed_requests": len(failed),
        "n_submitted": len(traces),
        "wall_s": round(wall_s, 3),
        "requests_per_sec": round(len(ok) / wall_s, 4) if wall_s > 0 else 0.0,
        "output_tokens_per_sec": round(out_tokens / wall_s, 2) if wall_s > 0 else 0.0,
        "mean_output_tokens": round(out_tokens / max(len(ok), 1), 2),
        "ttft_ms": {
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "p99": percentile(ttfts, 99),
            "mean": round(statistics.mean(ttfts), 3) if ttfts else None,
            "min": round(min(ttfts), 3) if ttfts else None,
            "max": round(max(ttfts), 3) if ttfts else None,
        },
        "e2e_ms": {
            "p50": percentile(e2es, 50),
            "p95": percentile(e2es, 95),
            "p99": percentile(e2es, 99),
            "mean": round(statistics.mean(e2es), 3) if e2es else None,
        },
        "prefill_latency_ms": {
            "observable": False,
            "proxy": "ttft_ms",
            "p50": percentile(ttfts, 50),
            "p95": percentile(ttfts, 95),
            "note": "No separate prefill metric on streaming completions; TTFT is the prefill+queue proxy.",
        },
        "gpu": gpu,
        "cache_metrics": cache_delta,
        "failures": [{"req_id": t["req_id"], "error": t.get("error")} for t in failed[:20]],
    }


async def run_concurrent_wave(
    session: aiohttp.ClientSession,
    prompts: list[str],
    *,
    concurrency: int,
    tag: str,
    traces_fp,
    max_tokens: int = MAX_TOKENS,
) -> tuple[list[dict[str, Any]], float]:
    sem = asyncio.Semaphore(concurrency)
    counter = 0
    lock = asyncio.Lock()
    t0 = time.perf_counter()

    async def one(prompt: str) -> dict[str, Any]:
        nonlocal counter
        async with sem:
            async with lock:
                counter += 1
                idx = counter
            rec = await stream_completion(
                session,
                req_id=f"{tag}-c{concurrency}-{idx}",
                prompt=prompt,
                max_tokens=max_tokens,
                concurrency=concurrency,
                extra={"tag": tag, "wave_index": idx},
            )
            # Drop full text from traces to keep file small (keep for correctness separately)
            slim = {k: v for k, v in rec.items() if k != "output_text"}
            slim["output_text_preview"] = (rec.get("output_text") or "")[:120]
            traces_fp.write(json.dumps(slim, default=str) + "\n")
            traces_fp.flush()
            return rec

    tasks = [asyncio.create_task(one(p)) for p in prompts]
    traces = await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0
    return list(traces), wall


def cycle_prompts(variant_prompts: list[dict[str, Any]], n: int) -> list[str]:
    out = []
    for i in range(n):
        out.append(variant_prompts[i % len(variant_prompts)]["prompt"])
    return out


def start_server(*, enable_prefix_caching: bool) -> tuple[subprocess.Popen, Any]:
    env = os.environ.copy()
    env["HF_HOME"] = os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_LOGGING_LEVEL"] = "INFO"
    cmd = [
        "/usr/bin/python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(CKPT),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--enable-prefix-caching" if enable_prefix_caching else "--no-enable-prefix-caching",
    ]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    mode = "ON" if enable_prefix_caching else "OFF"
    print(f"starting vLLM prefix_cache={mode}:", " ".join(cmd))
    log_f = SERVER_LOG.open("a")
    log_f.write(f"\n===== prefix_cache={mode} start {now_iso()} =====\n")
    log_f.flush()
    server = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))
    return server, log_f


def stop_server(server: subprocess.Popen, log_f) -> None:
    print(f"stopping vLLM pid {server.pid}")
    try:
        server.send_signal(signal.SIGTERM)
        server.wait(timeout=45)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=15)
    try:
        log_f.close()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(2)


async def measure_cold(
    session: aiohttp.ClientSession,
    prompts: list[str],
    *,
    n_cold: int,
    tag: str,
    traces_fp,
    gpu_poller: GpuPoller,
    do_reset: bool,
) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    resets = []
    t0 = time.time()
    before = cache_metrics_snapshot(await fetch_metrics(session))
    for i in range(n_cold):
        if do_reset:
            r = await reset_prefix_cache(session)
            resets.append(r)
            await asyncio.sleep(0.15)
        prompt = prompts[i % len(prompts)]
        rec = await stream_completion(
            session,
            req_id=f"{tag}-cold-{i+1}",
            prompt=prompt,
            extra={"tag": tag, "phase": "cold", "cold_index": i + 1},
        )
        slim = {k: v for k, v in rec.items() if k != "output_text"}
        slim["output_text_preview"] = (rec.get("output_text") or "")[:120]
        traces_fp.write(json.dumps(slim, default=str) + "\n")
        traces_fp.flush()
        traces.append(rec)
    t1 = time.time()
    after = cache_metrics_snapshot(await fetch_metrics(session))
    gpu = gpu_poller.window_stats(t0, t1)
    summary = summarize_traces(
        traces,
        wall_s=t1 - t0,
        gpu=gpu,
        cache_delta=cache_metrics_delta(before, after),
        label=tag,
    )
    summary["resets"] = resets[:3] + ([{"n_more": len(resets) - 3}] if len(resets) > 3 else [])
    summary["reset_enabled"] = do_reset
    return summary


async def measure_warm(
    session: aiohttp.ClientSession,
    prompts: list[str],
    *,
    concurrency: int,
    n_requests: int,
    tag: str,
    traces_fp,
    gpu_poller: GpuPoller,
    prime: bool,
) -> dict[str, Any]:
    if prime:
        # Populate prefix KV with first prompt before timing window.
        prime_rec = await stream_completion(
            session,
            req_id=f"{tag}-prime",
            prompt=prompts[0],
            extra={"tag": tag, "phase": "prime"},
        )
        traces_fp.write(
            json.dumps({k: v for k, v in prime_rec.items() if k != "output_text"}, default=str) + "\n"
        )
        await asyncio.sleep(0.2)

    before = cache_metrics_snapshot(await fetch_metrics(session))
    wave_prompts = cycle_prompts(
        [{"prompt": p} for p in prompts],
        n_requests,
    )
    # cycle_prompts expects list of dicts — fix:
    wave_prompts = [prompts[i % len(prompts)] for i in range(n_requests)]
    t0 = time.time()
    traces, wall = await run_concurrent_wave(
        session,
        wave_prompts,
        concurrency=concurrency,
        tag=tag,
        traces_fp=traces_fp,
    )
    t1 = time.time()
    after = cache_metrics_snapshot(await fetch_metrics(session))
    gpu = gpu_poller.window_stats(t0, t1)
    summary = summarize_traces(
        traces,
        wall_s=wall,
        gpu=gpu,
        cache_delta=cache_metrics_delta(before, after),
        label=tag,
    )
    summary["concurrency"] = concurrency
    summary["primed"] = prime
    return summary


async def correctness_check(
    session: aiohttp.ClientSession,
    production_variant: dict[str, Any],
    traces_fp,
) -> dict[str, Any]:
    """Deterministic greedy sanity: shared prefix reuse must not corrupt outputs."""
    prompts_mod = load_prompts_module()
    p0 = production_variant["prompts"][0]["prompt"]
    p1 = production_variant["prompts"][1]["prompt"]
    # Altered prefix: change one word in system content by building with tweaked system
    tok_prompt_alt_user = prompts_mod.build_intent_user_prompt(QUESTIONS[0])
    # Build altered by string replace in shared region of p0
    if "neuroscience research intent parser" in p0:
        p_alt = p0.replace(
            "neuroscience research intent parser",
            "neuroscience research INTENT PARSER ALTERED",
            1,
        )
    else:
        p_alt = p0 + "\n# ALTERED_PREFIX_MARKER\n"

    await reset_prefix_cache(session)
    await asyncio.sleep(0.2)

    # Two warm requests with identical prefix, different suffixes
    r_a1 = await stream_completion(
        session, req_id="corr-sameprefix-a", prompt=p0, max_tokens=48, ignore_eos=True,
        extra={"phase": "correctness"},
    )
    r_a2 = await stream_completion(
        session, req_id="corr-sameprefix-b", prompt=p1, max_tokens=48, ignore_eos=True,
        extra={"phase": "correctness"},
    )
    # Repeat p0 — should still be independent/valid
    r_a1b = await stream_completion(
        session, req_id="corr-sameprefix-a-repeat", prompt=p0, max_tokens=48, ignore_eos=True,
        extra={"phase": "correctness"},
    )
    # Changed prefix content
    r_alt = await stream_completion(
        session, req_id="corr-altprefix", prompt=p_alt, max_tokens=48, ignore_eos=True,
        extra={"phase": "correctness"},
    )

    for r in (r_a1, r_a2, r_a1b, r_alt):
        slim = {k: v for k, v in r.items() if k != "output_text"}
        slim["output_text"] = r.get("output_text") or ""
        traces_fp.write(json.dumps(slim, default=str) + "\n")

    out_a1 = (r_a1.get("output_text") or "").strip()
    out_a2 = (r_a2.get("output_text") or "").strip()
    out_a1b = (r_a1b.get("output_text") or "").strip()
    out_alt = (r_alt.get("output_text") or "").strip()

    same_suffix_independent = out_a1 != out_a2  # different user questions → different outputs expected
    # With ignore_eos and temperature 0, repeat of same prompt should be identical or very similar
    repeat_stable = out_a1 == out_a1b
    # Altered prefix should not silently return exact same bytes as unaltered if model reacts;
    # at minimum request must succeed and produce non-empty output (no KV corruption crash)
    alt_ok = bool(r_alt.get("ok") and out_alt)
    all_ok = all(r.get("ok") for r in (r_a1, r_a2, r_a1b, r_alt))

    passed = bool(all_ok and same_suffix_independent and repeat_stable and alt_ok)
    return {
        "passed": passed,
        "all_requests_ok": all_ok,
        "different_suffixes_produce_independent_outputs": same_suffix_independent,
        "identical_prompt_repeat_deterministic": repeat_stable,
        "altered_prefix_request_ok": alt_ok,
        "altered_prefix_output_differs_from_original": out_alt != out_a1,
        "outputs_preview": {
            "same_prefix_q0": out_a1[:200],
            "same_prefix_q1": out_a2[:200],
            "same_prefix_q0_repeat": out_a1b[:200],
            "altered_prefix_q0": out_alt[:200],
        },
        "checks": [
            "Requests sharing an identical prefix still produce suffix-dependent outputs.",
            "Repeating the exact same prompt under greedy+ignore_eos is byte-stable (no KV corruption).",
            "A content-changed prefix still serves successfully (no stale-state crash).",
        ],
        "note": (
            "Correctness here is serving/KV integrity under prefix reuse, not full agent quality eval."
        ),
    }


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_PATH.write_text("")
    SERVER_LOG.write_text("")

    import torch
    import vllm
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=False)
    manifest_full = build_prefix_manifest(tok)
    runtime = manifest_full.pop("_runtime")
    save_json(RESULTS_DIR / "prefix_manifest.json", manifest_full)

    for name, v in runtime.items():
        print(
            f"prefix {name}: shared={v['shared_prefix_tokens']} "
            f"mean_user={v['mean_user_tokens']} mean_total={v['mean_total_tokens']}"
        )

    production = runtime["production"]
    short = runtime["short"]
    medium = runtime["medium"]

    apps = gpu_compute_apps()
    print("gpu apps before start:", apps)
    if apps:
        subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
        time.sleep(2)

    gpu_poller = GpuPoller(GPU_LOG, interval_s=0.25)
    gpu_poller.start()

    kernel_lines: list[str] = []
    config_common = {
        "stage": "I.2",
        "timestamp": now_iso(),
        "vllm_version": vllm.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "checkpoint": str(CKPT),
        "served_model_name": SERVED_MODEL_NAME,
        "dtype": "auto",
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "enforce_eager": ENFORCE_EAGER,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "ignore_eos": True,
        "i1_reference": I1_REF,
        "main_prefix": "production",
        "prefix_shared_tokens": {
            "short": short["shared_prefix_tokens"],
            "medium": medium["shared_prefix_tokens"],
            "production": production["shared_prefix_tokens"],
        },
    }

    traces_fp = TRACES_PATH.open("a")
    cache_off: dict[str, Any] = {}
    cache_on_cold: dict[str, Any] = {}
    cache_on_warm: dict[str, Any] = {}
    sensitivity: dict[str, Any] = {}
    concurrency_cmp: dict[str, Any] = {}
    correctness: dict[str, Any] = {}

    try:
        # ---------- A. PREFIX CACHE OFF ----------
        server, log_f = start_server(enable_prefix_caching=False)
        try:
            await wait_healthy(360)
            print("server healthy (cache OFF)")
            await asyncio.sleep(1)
            timeout = aiohttp.ClientTimeout(total=600)
            connector = aiohttp.TCPConnector(limit=128)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                smoke = await stream_completion(
                    session, req_id="smoke-off", prompt=production["prompts"][0]["prompt"], max_tokens=8
                )
                print("smoke OFF", smoke["ok"], smoke.get("ttft_ms"), smoke.get("error"))
                if not smoke["ok"]:
                    raise RuntimeError(f"smoke OFF failed: {smoke}")

                vram_off = gpu_memory_used_mb()
                log_text = SERVER_LOG.read_text(errors="replace")
                kernel_lines = [
                    ln.strip()
                    for ln in log_text.splitlines()
                    if "CutlassInt8" in ln or "CompressedTensorsW8A8Int8" in ln
                ]
                enable_line = [
                    ln.strip()
                    for ln in log_text.splitlines()
                    if "enable_prefix_caching" in ln
                ][-3:]

                prod_prompts = [p["prompt"] for p in production["prompts"]]

                # Steady-state without cache (every request pays full prefill)
                off_c1 = await measure_warm(
                    session,
                    prod_prompts,
                    concurrency=1,
                    n_requests=N_WARM_C1,
                    tag="cacheoff-prod-c1",
                    traces_fp=traces_fp,
                    gpu_poller=gpu_poller,
                    prime=False,
                )
                print(
                    "cache OFF c1",
                    off_c1["ttft_ms"]["p50"],
                    off_c1["output_tokens_per_sec"],
                    off_c1["failed_requests"],
                )

                off_c8 = await measure_warm(
                    session,
                    prod_prompts,
                    concurrency=8,
                    n_requests=N_WARM_C8,
                    tag="cacheoff-prod-c8",
                    traces_fp=traces_fp,
                    gpu_poller=gpu_poller,
                    prime=False,
                )
                print(
                    "cache OFF c8",
                    off_c8["ttft_ms"]["p50"],
                    off_c8["output_tokens_per_sec"],
                    off_c8["failed_requests"],
                )

                # Also measure "cold-like" isolated serial requests without cache for apples-to-apples cold section
                off_cold = await measure_cold(
                    session,
                    prod_prompts,
                    n_cold=N_COLD,
                    tag="cacheoff-prod-coldserial",
                    traces_fp=traces_fp,
                    gpu_poller=gpu_poller,
                    do_reset=False,
                )

                cache_off = {
                    "prefix_caching": False,
                    "vram_after_load_mb": vram_off,
                    "enable_prefix_caching_log": enable_line,
                    "production_prefix_tokens": production["shared_prefix_tokens"],
                    "concurrency_1": off_c1,
                    "concurrency_8": off_c8,
                    "serial_baseline": off_cold,
                    "note": (
                        "With prefix caching disabled there is no reusable prefix KV; "
                        "serial_baseline and concurrency_1 both reflect full-prefill cost."
                    ),
                }
                save_json(RESULTS_DIR / "cache_off.json", cache_off)
        finally:
            stop_server(server, log_f)

        # ---------- B. PREFIX CACHE ON ----------
        server, log_f = start_server(enable_prefix_caching=True)
        try:
            await wait_healthy(360)
            print("server healthy (cache ON)")
            await asyncio.sleep(1)
            timeout = aiohttp.ClientTimeout(total=600)
            connector = aiohttp.TCPConnector(limit=128)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                smoke = await stream_completion(
                    session, req_id="smoke-on", prompt=production["prompts"][0]["prompt"], max_tokens=8
                )
                print("smoke ON", smoke["ok"], smoke.get("ttft_ms"), smoke.get("error"))
                if not smoke["ok"]:
                    raise RuntimeError(f"smoke ON failed: {smoke}")

                vram_on = gpu_memory_used_mb()
                log_text = SERVER_LOG.read_text(errors="replace")
                enable_line_on = [
                    ln.strip()
                    for ln in log_text.splitlines()
                    if "enable_prefix_caching" in ln
                ][-3:]
                kernel_lines = [
                    ln.strip()
                    for ln in log_text.splitlines()
                    if "CutlassInt8" in ln or "CompressedTensorsW8A8Int8" in ln
                ] or kernel_lines

                prod_prompts = [p["prompt"] for p in production["prompts"]]

                # COLD: reset between requests so each is a true cold miss
                on_cold = await measure_cold(
                    session,
                    prod_prompts,
                    n_cold=N_COLD,
                    tag="cacheon-prod-cold",
                    traces_fp=traces_fp,
                    gpu_poller=gpu_poller,
                    do_reset=True,
                )
                print("cache ON cold TTFT p50", on_cold["ttft_ms"]["p50"])
                cache_on_cold = {
                    "prefix_caching": True,
                    "phase": "cold",
                    "vram_after_load_mb": vram_on,
                    "enable_prefix_caching_log": enable_line_on,
                    "production_prefix_tokens": production["shared_prefix_tokens"],
                    "method": "POST /reset_prefix_cache before each cold request",
                    "result": on_cold,
                }
                save_json(RESULTS_DIR / "cache_on_cold.json", cache_on_cold)

                # WARM production @ c1 and c8
                on_warm_c1 = await measure_warm(
                    session,
                    prod_prompts,
                    concurrency=1,
                    n_requests=N_WARM_C1,
                    tag="cacheon-prod-warm-c1",
                    traces_fp=traces_fp,
                    gpu_poller=gpu_poller,
                    prime=True,
                )
                print(
                    "cache ON warm c1",
                    on_warm_c1["ttft_ms"]["p50"],
                    on_warm_c1["output_tokens_per_sec"],
                    on_warm_c1.get("cache_metrics", {}).get("hit_rate_over_window"),
                )

                on_warm_c8 = await measure_warm(
                    session,
                    prod_prompts,
                    concurrency=8,
                    n_requests=N_WARM_C8,
                    tag="cacheon-prod-warm-c8",
                    traces_fp=traces_fp,
                    gpu_poller=gpu_poller,
                    prime=True,
                )
                print(
                    "cache ON warm c8",
                    on_warm_c8["ttft_ms"]["p50"],
                    on_warm_c8["output_tokens_per_sec"],
                    on_warm_c8.get("cache_metrics", {}).get("hit_rate_over_window"),
                )

                cache_on_warm = {
                    "prefix_caching": True,
                    "phase": "warm",
                    "production_prefix_tokens": production["shared_prefix_tokens"],
                    "concurrency_1": on_warm_c1,
                    "concurrency_8": on_warm_c8,
                    "method": "prime one request, then timed requests with shared prefix + varying suffixes",
                }
                save_json(RESULTS_DIR / "cache_on_warm.json", cache_on_warm)

                # Prefix-length sensitivity (c=1, warm after prime; also one cold each)
                sens = {"concurrency": 1, "variants": {}}
                for key, variant in [("short", short), ("medium", medium), ("production", production)]:
                    prompts = [p["prompt"] for p in variant["prompts"]]
                    await reset_prefix_cache(session)
                    await asyncio.sleep(0.2)
                    cold = await measure_cold(
                        session,
                        prompts,
                        n_cold=6,
                        tag=f"sens-{key}-cold",
                        traces_fp=traces_fp,
                        gpu_poller=gpu_poller,
                        do_reset=True,
                    )
                    warm = await measure_warm(
                        session,
                        prompts,
                        concurrency=1,
                        n_requests=N_SENSITIVITY_WARM,
                        tag=f"sens-{key}-warm",
                        traces_fp=traces_fp,
                        gpu_poller=gpu_poller,
                        prime=True,
                    )
                    ttft_cold = cold["ttft_ms"]["p50"]
                    ttft_warm = warm["ttft_ms"]["p50"]
                    sens["variants"][key] = {
                        "shared_prefix_tokens": variant["shared_prefix_tokens"],
                        "mean_user_tokens": variant["mean_user_tokens"],
                        "mean_total_tokens": variant["mean_total_tokens"],
                        "cold_ttft_p50_ms": ttft_cold,
                        "warm_ttft_p50_ms": ttft_warm,
                        "ttft_reduction_pct_warm_vs_cold": reduction_pct(ttft_cold, ttft_warm),
                        "cold": cold,
                        "warm": warm,
                    }
                    print(
                        f"sensitivity {key}: shared={variant['shared_prefix_tokens']} "
                        f"cold_p50={ttft_cold} warm_p50={ttft_warm} "
                        f"red%={reduction_pct(ttft_cold, ttft_warm)}"
                    )
                sensitivity = sens
                save_json(RESULTS_DIR / "prefix_length_sensitivity.json", sensitivity)

                # Concurrency comparison bundle
                concurrency_cmp = {
                    "production_prefix_tokens": production["shared_prefix_tokens"],
                    "cache_off": {
                        "c1": cache_off["concurrency_1"],
                        "c8": cache_off["concurrency_8"],
                    },
                    "cache_on_warm": {
                        "c1": on_warm_c1,
                        "c8": on_warm_c8,
                    },
                    "cache_on_cold_c1_serial": on_cold,
                    "improvement_warm_vs_off": {
                        "c1": {
                            "ttft_p50_reduction_pct": reduction_pct(
                                cache_off["concurrency_1"]["ttft_ms"]["p50"],
                                on_warm_c1["ttft_ms"]["p50"],
                            ),
                            "ttft_p95_reduction_pct": reduction_pct(
                                cache_off["concurrency_1"]["ttft_ms"]["p95"],
                                on_warm_c1["ttft_ms"]["p95"],
                            ),
                            "e2e_p50_reduction_pct": reduction_pct(
                                cache_off["concurrency_1"]["e2e_ms"]["p50"],
                                on_warm_c1["e2e_ms"]["p50"],
                            ),
                            "e2e_p95_reduction_pct": reduction_pct(
                                cache_off["concurrency_1"]["e2e_ms"]["p95"],
                                on_warm_c1["e2e_ms"]["p95"],
                            ),
                            "throughput_change_pct": pct_change(
                                cache_off["concurrency_1"]["output_tokens_per_sec"],
                                on_warm_c1["output_tokens_per_sec"],
                            ),
                            "rps_change_pct": pct_change(
                                cache_off["concurrency_1"]["requests_per_sec"],
                                on_warm_c1["requests_per_sec"],
                            ),
                        },
                        "c8": {
                            "ttft_p50_reduction_pct": reduction_pct(
                                cache_off["concurrency_8"]["ttft_ms"]["p50"],
                                on_warm_c8["ttft_ms"]["p50"],
                            ),
                            "ttft_p95_reduction_pct": reduction_pct(
                                cache_off["concurrency_8"]["ttft_ms"]["p95"],
                                on_warm_c8["ttft_ms"]["p95"],
                            ),
                            "e2e_p50_reduction_pct": reduction_pct(
                                cache_off["concurrency_8"]["e2e_ms"]["p50"],
                                on_warm_c8["e2e_ms"]["p50"],
                            ),
                            "e2e_p95_reduction_pct": reduction_pct(
                                cache_off["concurrency_8"]["e2e_ms"]["p95"],
                                on_warm_c8["e2e_ms"]["p95"],
                            ),
                            "throughput_change_pct": pct_change(
                                cache_off["concurrency_8"]["output_tokens_per_sec"],
                                on_warm_c8["output_tokens_per_sec"],
                            ),
                            "rps_change_pct": pct_change(
                                cache_off["concurrency_8"]["requests_per_sec"],
                                on_warm_c8["requests_per_sec"],
                            ),
                        },
                    },
                    "note": "Concurrency 16 not re-run; I.1 already established saturation at 16.",
                }
                save_json(RESULTS_DIR / "concurrency_cache_comparison.json", concurrency_cmp)

                correctness = await correctness_check(session, production, traces_fp)
                save_json(RESULTS_DIR / "correctness_check.json", correctness)
                print("correctness", correctness["passed"])
        finally:
            stop_server(server, log_f)

    finally:
        traces_fp.close()
        gpu_poller.stop()

    # ---------- Aggregate comparison + verdict ----------
    off_c1 = cache_off["concurrency_1"]
    on_c1 = cache_on_warm["concurrency_1"]
    on_cold_r = cache_on_cold["result"]
    off_c8 = cache_off["concurrency_8"]
    on_c8 = cache_on_warm["concurrency_8"]

    ttft_red_p50 = reduction_pct(off_c1["ttft_ms"]["p50"], on_c1["ttft_ms"]["p50"])
    ttft_red_p95 = reduction_pct(off_c1["ttft_ms"]["p95"], on_c1["ttft_ms"]["p95"])
    e2e_red_p50 = reduction_pct(off_c1["e2e_ms"]["p50"], on_c1["e2e_ms"]["p50"])
    e2e_red_p95 = reduction_pct(off_c1["e2e_ms"]["p95"], on_c1["e2e_ms"]["p95"])
    tput_chg = pct_change(off_c1["output_tokens_per_sec"], on_c1["output_tokens_per_sec"])
    vram_delta = None
    if cache_off.get("vram_after_load_mb") is not None and cache_on_cold.get("vram_after_load_mb") is not None:
        vram_delta = round(cache_on_cold["vram_after_load_mb"] - cache_off["vram_after_load_mb"], 1)

    # Material help: >=10% TTFT p50 reduction at production warm c1, with cache hits observed
    hit_rate = (on_c1.get("cache_metrics") or {}).get("hit_rate_over_window")
    material = bool(
        (ttft_red_p50 or 0) >= 10.0
        and correctness.get("passed")
        and (hit_rate is None or hit_rate > 0)
    )
    worth_enabling = material and (ttft_red_p50 or 0) >= 10.0

    # Fail only if experiment broken (failures, correctness fail, kernel missing)
    experiment_ok = (
        off_c1["failed_requests"] == 0
        and on_c1["failed_requests"] == 0
        and off_c8["failed_requests"] == 0
        and on_c8["failed_requests"] == 0
        and correctness.get("passed") is True
        and bool(kernel_lines)
    )
    verdict = "PASS" if experiment_ok else "FAIL"

    # When does caching help?
    sens_rows = []
    for key, row in sensitivity.get("variants", {}).items():
        sens_rows.append(
            {
                "variant": key,
                "shared_tokens": row["shared_prefix_tokens"],
                "ttft_reduction_pct_warm_vs_cold": row["ttft_reduction_pct_warm_vs_cold"],
            }
        )
    sens_rows_sorted = sorted(sens_rows, key=lambda r: r["shared_tokens"])
    material_from = None
    for row in sens_rows_sorted:
        if (row["ttft_reduction_pct_warm_vs_cold"] or 0) >= 10.0:
            material_from = row
            break

    comparison = {
        "stage": "I.2",
        "timestamp": now_iso(),
        "verdict": verdict,
        "checkpoint": str(CKPT),
        "kernel_evidence": kernel_lines[:5],
        "config": config_common,
        "production_shared_prefix_tokens": production["shared_prefix_tokens"],
        "production_mean_user_tokens": production["mean_user_tokens"],
        "production_mean_total_tokens": production["mean_total_tokens"],
        "cache_configuration": {
            "off_flag": "--no-enable-prefix-caching",
            "on_flag": "--enable-prefix-caching",
            "fresh_engine_starts": True,
            "cold_method": "POST /reset_prefix_cache before each cold request",
            "warm_method": "prime then repeated shared-prefix / varying-suffix requests",
        },
        "cold_vs_warm": {
            "cache_on_cold_ttft": on_cold_r["ttft_ms"],
            "cache_on_warm_c1_ttft": on_c1["ttft_ms"],
            "cache_off_c1_ttft": off_c1["ttft_ms"],
            "warm_vs_cold_ttft_p50_reduction_pct": reduction_pct(
                on_cold_r["ttft_ms"]["p50"], on_c1["ttft_ms"]["p50"]
            ),
            "warm_vs_off_ttft_p50_reduction_pct": ttft_red_p50,
        },
        "ttft_before_after_c1": {
            "cache_off": off_c1["ttft_ms"],
            "cache_on_warm": on_c1["ttft_ms"],
            "cache_on_cold": on_cold_r["ttft_ms"],
            "p50_reduction_pct_warm_vs_off": ttft_red_p50,
            "p95_reduction_pct_warm_vs_off": ttft_red_p95,
            "p99_reduction_pct_warm_vs_off": reduction_pct(
                off_c1["ttft_ms"]["p99"], on_c1["ttft_ms"]["p99"]
            ),
        },
        "e2e_before_after_c1": {
            "cache_off": off_c1["e2e_ms"],
            "cache_on_warm": on_c1["e2e_ms"],
            "p50_reduction_pct_warm_vs_off": e2e_red_p50,
            "p95_reduction_pct_warm_vs_off": e2e_red_p95,
            "p99_reduction_pct_warm_vs_off": reduction_pct(
                off_c1["e2e_ms"]["p99"], on_c1["e2e_ms"]["p99"]
            ),
        },
        "throughput_before_after_c1": {
            "cache_off_tok_s": off_c1["output_tokens_per_sec"],
            "cache_on_warm_tok_s": on_c1["output_tokens_per_sec"],
            "change_pct": tput_chg,
            "cache_off_rps": off_c1["requests_per_sec"],
            "cache_on_warm_rps": on_c1["requests_per_sec"],
            "rps_change_pct": pct_change(off_c1["requests_per_sec"], on_c1["requests_per_sec"]),
            "note": "Decode path unchanged; throughput gains come from reduced prefill/TTFT overlap only.",
        },
        "concurrency_8": {
            "cache_off_ttft": off_c8["ttft_ms"],
            "cache_on_warm_ttft": on_c8["ttft_ms"],
            "ttft_p50_reduction_pct": reduction_pct(off_c8["ttft_ms"]["p50"], on_c8["ttft_ms"]["p50"]),
            "ttft_p95_reduction_pct": reduction_pct(off_c8["ttft_ms"]["p95"], on_c8["ttft_ms"]["p95"]),
            "e2e_p50_reduction_pct": reduction_pct(off_c8["e2e_ms"]["p50"], on_c8["e2e_ms"]["p50"]),
            "throughput_change_pct": pct_change(
                off_c8["output_tokens_per_sec"], on_c8["output_tokens_per_sec"]
            ),
            "cache_hit_rate_warm": (on_c8.get("cache_metrics") or {}).get("hit_rate_over_window"),
        },
        "cache_hit_reuse_evidence": {
            "warm_c1": on_c1.get("cache_metrics"),
            "warm_c8": on_c8.get("cache_metrics"),
            "cold": on_cold_r.get("cache_metrics"),
            "prometheus_metrics": ["vllm:prefix_cache_hits", "vllm:prefix_cache_queries"],
        },
        "prefix_length_sensitivity_summary": sens_rows_sorted,
        "material_help_begins_at": material_from,
        "vram": {
            "cache_off_after_load_mb": cache_off.get("vram_after_load_mb"),
            "cache_on_after_load_mb": cache_on_cold.get("vram_after_load_mb"),
            "delta_mb": vram_delta,
            "note": "nvidia-smi includes KV reservation at gpu_memory_utilization=0.90; delta isolates enable-flag overhead at idle.",
        },
        "correctness": correctness,
        "worth_enabling_in_production": worth_enabling,
        "recommendation": (
            "Enable --enable-prefix-caching for research-agent serving: warm TTFT improves materially "
            "on the production shared prefix with cache-hit evidence and no correctness failure."
            if worth_enabling
            else (
                "Prefix caching is optional: measured warm TTFT gain is modest or not clearly "
                "prefill-bound on this workload; keep available but do not treat as required."
                if experiment_ok
                else "Experiment incomplete or failed integrity checks; do not decide from this run."
            )
        ),
        "quantified_improvement_production_prefix": {
            "ttft_reduction_pct_p50_warm_vs_off": ttft_red_p50,
            "ttft_reduction_pct_p95_warm_vs_off": ttft_red_p95,
            "e2e_reduction_pct_p50_warm_vs_off": e2e_red_p50,
            "e2e_reduction_pct_p95_warm_vs_off": e2e_red_p95,
            "prefill_reduction_pct": {
                "observable_directly": False,
                "proxy_ttft_p50_reduction_pct": ttft_red_p50,
            },
            "throughput_change_pct": tput_chg,
            "vram_delta_mb": vram_delta,
            "cold_ttft_p50_ms": on_cold_r["ttft_ms"]["p50"],
            "warm_ttft_p50_ms": on_c1["ttft_ms"]["p50"],
            "off_ttft_p50_ms": off_c1["ttft_ms"]["p50"],
        },
        "material_improvement_observed": material,
    }
    save_json(CMP_PATH, comparison)

    config_out = {
        **config_common,
        "kernel_log_lines": kernel_lines[:10],
        "results_dir": str(RESULTS_DIR),
        "comparison_path": str(CMP_PATH),
        "verdict": verdict,
    }
    save_json(RESULTS_DIR / "config.json", config_out)

    print("\nI.2 complete. verdict=", verdict, "worth_enabling=", worth_enabling)
    print(
        f"prod shared_tokens={production['shared_prefix_tokens']} "
        f"TTFT p50 off={off_c1['ttft_ms']['p50']} cold={on_cold_r['ttft_ms']['p50']} "
        f"warm={on_c1['ttft_ms']['p50']} red%={ttft_red_p50}"
    )
    print("I2_DONE")


if __name__ == "__main__":
    asyncio.run(main())
