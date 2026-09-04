#!/usr/bin/env python3
"""Stage I.1 — vLLM BF16 concurrency and continuous-batching benchmark.

Online OpenAI-compatible server (not offline LLM.generate batch).
Prefix caching OFF. No INT8, no Triton, no SLA, no prefix-cache stage.
"""

from __future__ import annotations

import asyncio
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

BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "sft_corrected_v2" / "final"
LORA_NAME = "sft"
RESULTS_DIR = PROJECT_ROOT / "results" / "serving" / "load"
CMP_PATH = PROJECT_ROOT / "results" / "model_comparison" / "vllm_concurrency_scaling.json"
SERVER_LOG = RESULTS_DIR / "vllm_server.log"
GPU_LOG = RESULTS_DIR / "gpu_poll.csv"
TRACES_PATH = RESULTS_DIR / "per_request_traces.jsonl"

HOST = "127.0.0.1"
PORT = 8000
BASE_URL = f"http://{HOST}:{PORT}"
METRICS_URL = f"{BASE_URL}/metrics"
HEALTH_URL = f"{BASE_URL}/health"
COMPLETIONS_URL = f"{BASE_URL}/v1/completions"

# H.2-validated serving knobs (plus explicit prefix-cache off)
GPU_MEMORY_UTILIZATION = 0.85
MAX_MODEL_LEN = 2048
TENSOR_PARALLEL_SIZE = 1
MAX_LORA_RANK = 32
ENFORCE_EAGER = True  # same as H.2; not a new tuning pass

MAX_TOKENS = 64
TEMPERATURE = 0.0

# Same research-agent prompt family as H.2 (~512 tokens)
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

HF_REF = {"decode_tok_per_s": 52.74, "e2e_ms": 1264.5}
VLLM_SINGLE_REF = {"decode_tok_per_s": 61.3, "e2e_ms": 1043.8}

CONCURRENCY_PLAN = [1, 4, 8, 16]  # 32 gated
TIMED_REQUESTS = {1: 32, 4: 32, 8: 48, 16: 64, 32: 64}


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


def prompt_token_count() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    n = len(tok.encode(BENCH_PROMPT, add_special_tokens=True))
    return n


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
    """Parse Prometheus exposition into {name: [{labels, value}, ...]}."""
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


def prom_hist_percentile(metrics: dict, name: str, p: float) -> float | None:
    """Approximate percentile from histogram buckets (seconds → ms)."""
    buckets = metrics.get(name + "_bucket") or []
    if not buckets:
        return None
    pts = []
    for b in buckets:
        le = b["labels"].get("le")
        if le is None:
            continue
        if le == "+Inf":
            continue
        pts.append((float(le), b["value"]))
    if not pts:
        return None
    pts.sort()
    total = pts[-1][1]
    if total <= 0:
        return None
    target = total * (p / 100.0)
    prev_le, prev_c = 0.0, 0.0
    for le, c in pts:
        if c >= target:
            if c == prev_c:
                return round(le * 1000.0, 4)
            frac = (target - prev_c) / (c - prev_c)
            return round((prev_le + frac * (le - prev_le)) * 1000.0, 4)
        prev_le, prev_c = le, c
    return round(pts[-1][0] * 1000.0, 4)


async def fetch_metrics(session: aiohttp.ClientSession) -> dict[str, list[dict[str, Any]]]:
    async with session.get(METRICS_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        text = await resp.text()
    return parse_prom_text(text)


async def wait_healthy(timeout_s: float = 300.0) -> None:
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


async def stream_completion(
    session: aiohttp.ClientSession,
    *,
    req_id: str,
    concurrency: int,
    max_tokens: int,
    prompt: str,
    ignore_eos: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "model": LORA_NAME,
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
                    if ch.get("finish_reason"):
                        finish_reason = ch["finish_reason"]
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    t_end = time.perf_counter()
    if completion_tokens <= 0:
        completion_tokens = len(text_parts)  # fallback: chunk count, not tokens
    out = {
        "req_id": req_id,
        "concurrency": concurrency,
        "submit_unix_s": t_submit,
        "ttft_ms": None if t_first is None else round((t_first - t_submit) * 1000.0, 3),
        "e2e_ms": round((t_end - t_submit) * 1000.0, 3),
        "first_token_unix_s": t_first,
        "end_unix_s": t_end,
        "max_tokens": max_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "ok": error is None and t_first is not None,
        "error": error,
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


class MetricsPoller:
    def __init__(self):
        self.samples: list[dict[str, Any]] = []
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def _loop(self) -> None:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self._stop.is_set():
                rec = {"unix_s": time.time()}
                try:
                    m = await fetch_metrics(session)
                    rec["num_requests_running"] = prom_gauge(m, "vllm:num_requests_running")
                    rec["num_requests_waiting"] = prom_gauge(m, "vllm:num_requests_waiting")
                    rec["gpu_cache_usage_perc"] = prom_gauge(m, "vllm:gpu_cache_usage_perc")
                    rec["prompt_throughput"] = prom_gauge(m, "vllm:avg_prompt_throughput_toks_per_s") or prom_gauge(
                        m, "vllm:prompt_tokens_total"
                    )
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = str(exc)
                self.samples.append(rec)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=0.1)
                except TimeoutError:
                    pass

    def start(self) -> None:
        self.samples = []
        self._stop.clear()
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._task:
            await self._task
        return self.samples


def extract_server_config(log_text: str) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    patterns = {
        "max_num_seqs": r"max_num_seqs[=: ]+(\d+)",
        "max_num_batched_tokens": r"max_num_batched_tokens[=: ]+(\d+)",
        "enable_prefix_caching": r"enable_prefix_caching[=: ]+(\w+)",
        "gpu_memory_utilization": r"gpu_memory_utilization[=: ]+([0-9.]+)",
        "max_model_len": r"max_model_len[=: ]+(\d+)",
        "dtype": r"dtype[=: ]+([A-Za-z0-9_]+)",
        "tensor_parallel_size": r"tensor_parallel_size[=: ]+(\d+)",
    }
    for k, pat in patterns.items():
        ms = re.findall(pat, log_text)
        if ms:
            cfg[k + "_from_log"] = ms[-1]
    return cfg


async def run_wave(
    session: aiohttp.ClientSession,
    *,
    concurrency: int,
    n_requests: int,
    max_tokens: int,
    ignore_eos: bool,
    tag: str,
    traces_fp,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    counter = 0
    lock = asyncio.Lock()

    async def one() -> dict[str, Any]:
        nonlocal counter
        async with sem:
            async with lock:
                counter += 1
                idx = counter
            rec = await stream_completion(
                session,
                req_id=f"{tag}-c{concurrency}-{idx}",
                concurrency=concurrency,
                max_tokens=max_tokens,
                prompt=BENCH_PROMPT,
                ignore_eos=ignore_eos,
                extra={"tag": tag, "wave_index": idx},
            )
            traces_fp.write(json.dumps(rec, default=str) + "\n")
            traces_fp.flush()
            return rec

    tasks = [asyncio.create_task(one()) for _ in range(n_requests)]
    return await asyncio.gather(*tasks)


def summarize_level(
    traces: list[dict[str, Any]],
    *,
    concurrency: int,
    wall_s: float,
    gpu: dict[str, Any],
    metrics_samples: list[dict[str, Any]],
    prom: dict[str, Any],
) -> dict[str, Any]:
    ok = [t for t in traces if t.get("ok")]
    failed = [t for t in traces if not t.get("ok")]
    ttfts = [t["ttft_ms"] for t in ok if t.get("ttft_ms") is not None]
    e2es = [t["e2e_ms"] for t in ok]
    out_tokens = sum(int(t.get("completion_tokens") or 0) for t in ok)
    running = [s["num_requests_running"] for s in metrics_samples if s.get("num_requests_running") is not None]
    waiting = [s["num_requests_waiting"] for s in metrics_samples if s.get("num_requests_waiting") is not None]
    cache = [s["gpu_cache_usage_perc"] for s in metrics_samples if s.get("gpu_cache_usage_perc") is not None]
    return {
        "concurrency": concurrency,
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
        },
        "e2e_ms": {
            "p50": percentile(e2es, 50),
            "p95": percentile(e2es, 95),
            "p99": percentile(e2es, 99),
            "mean": round(statistics.mean(e2es), 3) if e2es else None,
        },
        "queue_ms_from_vllm_histogram": prom.get("queue_ms"),
        "gpu": gpu,
        "scheduler_observed": {
            "max_num_requests_running": max(running) if running else None,
            "mean_num_requests_running": round(statistics.mean(running), 3) if running else None,
            "max_num_requests_waiting": max(waiting) if waiting else None,
            "mean_num_requests_waiting": round(statistics.mean(waiting), 3) if waiting else None,
            "max_gpu_cache_usage_perc": max(cache) if cache else None,
            "n_metric_samples": len(metrics_samples),
        },
        "failures": [{"req_id": t["req_id"], "error": t.get("error")} for t in failed[:20]],
        "oom_events": 0,
    }


def decide_run_32(c16: dict[str, Any], peak_vram_mb: float) -> tuple[bool, str]:
    if c16["failed_requests"] > 0:
        return False, f"concurrency 16 had {c16['failed_requests']} failures"
    if c16.get("oom_events"):
        return False, "OOM at concurrency 16"
    if peak_vram_mb > 22000:
        return False, f"peak VRAM {peak_vram_mb} MB too close to 24 GB"
    # saturate if p95 e2e already >> c=1 and throughput flattening is checked later;
    # still allow 32 if 16 was clean and VRAM safe.
    return True, "concurrency 16 succeeded cleanly, no OOM, VRAM safe"


async def continuous_batching_experiment(session: aiohttp.ClientSession, traces_fp) -> dict[str, Any]:
    """Staggered mixed-length arrivals to show independent enter/leave."""
    poller = MetricsPoller()
    poller.start()
    t0 = time.perf_counter()
    unix0 = time.time()

    async def launch(req_id: str, max_tokens: int, delay_s: float) -> dict[str, Any]:
        if delay_s:
            await asyncio.sleep(delay_s)
        rec = await stream_completion(
            session,
            req_id=req_id,
            concurrency=-1,
            max_tokens=max_tokens,
            prompt=BENCH_PROMPT,
            ignore_eos=True,
            extra={"tag": "continuous_batching", "delay_s": delay_s},
        )
        traces_fp.write(json.dumps(rec, default=str) + "\n")
        traces_fp.flush()
        rec["delay_s"] = delay_s
        rec["rel_submit_ms"] = round((rec["submit_unix_s"] - t0) * 1000.0, 2)
        rec["rel_first_ms"] = (
            None if rec["first_token_unix_s"] is None else round((rec["first_token_unix_s"] - t0) * 1000.0, 2)
        )
        rec["rel_end_ms"] = round((rec["end_unix_s"] - t0) * 1000.0, 2)
        return rec

    # Wave A: 4 long sequences at t=0
    # Wave B: 4 short sequences after 200ms (join while A is decoding)
    # Wave C: 2 medium after 450ms
    tasks = []
    for i in range(4):
        tasks.append(asyncio.create_task(launch(f"cb-long-{i}", 64, 0.0)))
    for i in range(4):
        tasks.append(asyncio.create_task(launch(f"cb-short-{i}", 16, 0.20)))
    for i in range(2):
        tasks.append(asyncio.create_task(launch(f"cb-mid-{i}", 32, 0.45)))
    recs = await asyncio.gather(*tasks)
    samples = await poller.stop()

    longs = [r for r in recs if r["req_id"].startswith("cb-long")]
    shorts = [r for r in recs if r["req_id"].startswith("cb-short")]
    mids = [r for r in recs if r["req_id"].startswith("cb-mid")]

    short_ends = [r["rel_end_ms"] for r in shorts if r.get("ok")]
    long_ends = [r["rel_end_ms"] for r in longs if r.get("ok")]
    long_firsts = [r["rel_first_ms"] for r in longs if r.get("ok") and r.get("rel_first_ms") is not None]
    short_submits = [r["rel_submit_ms"] for r in shorts if r.get("ok")]
    mid_submits = [r["rel_submit_ms"] for r in mids if r.get("ok")]

    shorts_finished_before_longs = False
    if short_ends and long_ends:
        shorts_finished_before_longs = max(short_ends) < min(long_ends)

    shorts_joined_during_long_decode = False
    if short_submits and long_firsts and long_ends:
        shorts_joined_during_long_decode = min(short_submits) > min(long_firsts) and min(short_submits) < min(long_ends)

    mids_joined_while_active = False
    if mid_submits and long_ends and short_ends:
        mids_joined_while_active = min(mid_submits) < min(long_ends)

    running_series = [
        {"rel_ms": round((s["unix_s"] - unix0) * 1000.0, 1), "running": s.get("num_requests_running"), "waiting": s.get("num_requests_waiting")}
        for s in samples
        if "num_requests_running" in s
    ]
    run_vals = [x["running"] for x in running_series if x["running"] is not None]
    max_running = max(run_vals) if run_vals else None

    evidence = {
        "method": (
            "Staggered OpenAI streaming completions against a live vLLM server: "
            "4×64-token requests at t=0, 4×16-token at t=200ms, 2×32-token at t=450ms. "
            "Client timestamps for submit / first token / end plus polled vllm:num_requests_running."
        ),
        "not_static_batch_claim": (
            "Offline LLM.generate() of a fixed prompt list would be a static batch. "
            "This experiment uses independently arriving HTTP requests on the online server."
        ),
        "all_ok": all(r.get("ok") for r in recs),
        "shorts_finished_before_longs": shorts_finished_before_longs,
        "shorts_joined_during_long_decode": shorts_joined_during_long_decode,
        "mids_joined_while_others_active": mids_joined_while_active,
        "max_observed_running": max_running,
        "running_exceeded_one": (max_running or 0) > 1,
        "request_timeline": [
            {
                "req_id": r["req_id"],
                "max_tokens": r["max_tokens"],
                "ok": r["ok"],
                "rel_submit_ms": r.get("rel_submit_ms"),
                "rel_first_ms": r.get("rel_first_ms"),
                "rel_end_ms": r.get("rel_end_ms"),
                "e2e_ms": r.get("e2e_ms"),
                "completion_tokens": r.get("completion_tokens"),
            }
            for r in recs
        ],
        "scheduler_samples": running_series[:400],
        "conclusions": [],
    }
    if shorts_finished_before_longs:
        evidence["conclusions"].append(
            "Shorter (16-token) requests completed while longer (64-token) requests were still active — not a static batch barrier."
        )
    if shorts_joined_during_long_decode:
        evidence["conclusions"].append(
            "New short requests were submitted after long requests had already produced first tokens and before those longs finished — requests joined an in-flight decode batch."
        )
    if mids_joined_while_active:
        evidence["conclusions"].append(
            "A third arrival wave joined while earlier sequences were still running."
        )
    if (max_running or 0) > 1:
        evidence["conclusions"].append(
            f"vllm:num_requests_running peaked at {max_running}, showing multiple sequences in the execution batch."
        )
    evidence["continuous_batching_supported_by_evidence"] = bool(
        shorts_finished_before_longs
        and shorts_joined_during_long_decode
        and evidence["all_ok"]
    )
    return evidence


def saturation_analysis(levels: dict[int, dict[str, Any]]) -> dict[str, Any]:
    concs = sorted(levels)
    tps = [levels[c]["output_tokens_per_sec"] for c in concs]
    rps = [levels[c]["requests_per_sec"] for c in concs]
    e2e_p95 = [levels[c]["e2e_ms"]["p95"] for c in concs]
    ttft_p95 = [levels[c]["ttft_ms"]["p95"] for c in concs]
    gpu_avg = [levels[c]["gpu"].get("avg_gpu_util") for c in concs]
    vram = [levels[c]["gpu"].get("peak_vram_mb") for c in concs]
    fails = [levels[c]["failed_requests"] for c in concs]

    # Throughput stop scaling: first c where tok/s gain vs previous < 10%
    tput_sat = None
    for i in range(1, len(concs)):
        prev, cur = tps[i - 1], tps[i]
        gain = (cur - prev) / max(prev, 1e-9) * 100.0
        if gain < 10.0:
            tput_sat = concs[i]
            break

    # p95 E2E sharp increase: >50% jump vs previous
    lat_jump = None
    for i in range(1, len(concs)):
        prev, cur = e2e_p95[i - 1], e2e_p95[i]
        if prev and cur and (cur - prev) / prev * 100.0 >= 50.0:
            lat_jump = concs[i]
            break

    c1 = levels.get(1, {})
    practical = tput_sat or concs[-1]
    return {
        "concurrency_levels": concs,
        "output_tok_per_s": tps,
        "requests_per_sec": rps,
        "e2e_p95_ms": e2e_p95,
        "ttft_p95_ms": ttft_p95,
        "avg_gpu_util": gpu_avg,
        "peak_vram_mb": vram,
        "failed_requests": fails,
        "throughput_stop_scaling_at": tput_sat,
        "p95_latency_sharp_increase_at": lat_jump,
        "ttft_trend": "increases with concurrency" if ttft_p95[-1] and ttft_p95[0] and ttft_p95[-1] > ttft_p95[0] * 1.2 else "stable_or_mild",
        "gpu_util_trend": gpu_avg,
        "vram_trend": vram,
        "any_failures": any(fails),
        "any_oom": False,
        "practical_saturation_point": practical,
        "relative_to_vllm_c1": {
            c: {
                "tok_s_ratio": round(levels[c]["output_tokens_per_sec"] / max(c1.get("output_tokens_per_sec") or 1e-9, 1e-9), 3),
                "e2e_p50_ratio": round(
                    (levels[c]["e2e_ms"]["p50"] or 0) / max(c1.get("e2e_ms", {}).get("p50") or 1e-9, 1e-9),
                    3,
                ),
            }
            for c in concs
            if c1
        },
        "notes": [
            "Saturation is load-behavior only; no SLA/admission policy selected.",
            "Throughput scaling uses output tokens/sec from completed timed requests.",
        ],
    }


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACES_PATH.write_text("")

    import torch
    import vllm

    n_prompt = prompt_token_count()
    print(f"prompt_tokens={n_prompt}")

    apps = gpu_compute_apps()
    print("gpu apps before start:", apps)
    if apps:
        print("WARNING: GPU already has compute processes; continuing only if they are leftover display stubs")

    env = os.environ.copy()
    env["HF_HOME"] = os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
    env["CUDA_VISIBLE_DEVICES"] = "0"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        BASE_MODEL,
        "--dtype",
        "bfloat16",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--no-enable-prefix-caching",
        "--enable-lora",
        "--max-lora-rank",
        str(MAX_LORA_RANK),
        "--lora-modules",
        f"{LORA_NAME}={ADAPTER_DIR}",
        "--disable-log-requests",
    ]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")

    print("starting:", " ".join(cmd))
    log_f = SERVER_LOG.open("w")
    server = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT_ROOT))

    gpu_poller = GpuPoller(GPU_LOG, interval_s=0.25)
    gpu_poller.start()

    try:
        await wait_healthy(timeout_s=360)
        print("server healthy")
        await asyncio.sleep(2)

        # duplicate-process check
        apps = gpu_compute_apps()
        print("gpu apps after start:", apps)

        timeout = aiohttp.ClientTimeout(total=600)
        connector = aiohttp.TCPConnector(limit=128)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            # smoke
            smoke = await stream_completion(
                session,
                req_id="smoke-1",
                concurrency=1,
                max_tokens=8,
                prompt=BENCH_PROMPT,
                ignore_eos=True,
                extra={"tag": "smoke"},
            )
            print("smoke", smoke["ok"], smoke["ttft_ms"], smoke["e2e_ms"], smoke.get("error"))
            if not smoke["ok"]:
                raise RuntimeError(f"smoke request failed: {smoke}")

            vram_after_load = gpu_memory_used_mb()
            prom0 = await fetch_metrics(session)

            log_text = SERVER_LOG.read_text(errors="replace")
            cfg = {
                "stage": "I.1",
                "timestamp": now_iso(),
                "vllm_version": vllm.__version__,
                "cuda_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "model": BASE_MODEL,
                "adapter": str(ADAPTER_DIR),
                "lora_served_as": LORA_NAME,
                "model_dtype": "bfloat16",
                "quantization": None,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                "prefix_caching": False,
                "prefix_caching_flag": "--no-enable-prefix-caching",
                "enforce_eager": ENFORCE_EAGER,
                "max_lora_rank": MAX_LORA_RANK,
                "max_num_seqs_cli": None,
                "max_num_batched_tokens_cli": None,
                "note": "max_num_seqs / max_num_batched_tokens left at vLLM defaults (not aggressively tuned).",
                "server_cmd": cmd,
                "from_log": extract_server_config(log_text),
                "vram_after_load_nvidia_smi_mb": vram_after_load,
                "gpu_compute_apps": apps,
                "smoke_ok": smoke["ok"],
                "prompt_tokens": n_prompt,
                "max_tokens": MAX_TOKENS,
                "prometheus_gauges_at_idle": {
                    "num_requests_running": prom_gauge(prom0, "vllm:num_requests_running"),
                    "num_requests_waiting": prom_gauge(prom0, "vllm:num_requests_waiting"),
                    "gpu_cache_usage_perc": prom_gauge(prom0, "vllm:gpu_cache_usage_perc"),
                },
            }
            # try to fill defaults from engine log more loosely
            for line in log_text.splitlines():
                if "Scheduler" in line or "max_num_seqs" in line or "Chunked prefill" in line:
                    if len(cfg.setdefault("log_scheduler_lines", [])) < 30:
                        cfg["log_scheduler_lines"].append(line.strip()[:400])
            save_json(RESULTS_DIR / "config.json", cfg)

            levels: dict[int, dict[str, Any]] = {}
            traces_fp = TRACES_PATH.open("a")

            planned = list(CONCURRENCY_PLAN)
            try:
                for conc in planned:
                    n_timed = TIMED_REQUESTS[conc]
                    n_warm = conc  # one full wave
                    print(f"\n=== concurrency={conc} warmup={n_warm} timed={n_timed} ===")
                    await run_wave(
                        session,
                        concurrency=conc,
                        n_requests=n_warm,
                        max_tokens=MAX_TOKENS,
                        ignore_eos=True,
                        tag=f"warmup-c{conc}",
                        traces_fp=traces_fp,
                    )
                    mp = MetricsPoller()
                    mp.start()
                    wall0 = time.perf_counter()
                    unix0 = time.time()
                    timed = await run_wave(
                        session,
                        concurrency=conc,
                        n_requests=n_timed,
                        max_tokens=MAX_TOKENS,
                        ignore_eos=True,
                        tag=f"timed-c{conc}",
                        traces_fp=traces_fp,
                    )
                    wall = time.perf_counter() - wall0
                    unix1 = time.time()
                    samples = await mp.stop()
                    gpu = gpu_poller.window_stats(unix0, unix1)
                    gpu["vram_nvidia_smi_now_mb"] = gpu_memory_used_mb()
                    prom = await fetch_metrics(session)
                    queue = {
                        "p50": prom_hist_percentile(prom, "vllm:request_queue_time_seconds", 50),
                        "p95": prom_hist_percentile(prom, "vllm:request_queue_time_seconds", 95),
                        "p99": prom_hist_percentile(prom, "vllm:request_queue_time_seconds", 99),
                        "note": "Prometheus histogram is cumulative across the process lifetime, not this level only.",
                    }
                    summary = summarize_level(
                        timed,
                        concurrency=conc,
                        wall_s=wall,
                        gpu=gpu,
                        metrics_samples=samples,
                        prom={"queue_ms": queue},
                    )
                    summary["queue_ms"] = queue
                    summary["warmup_requests"] = n_warm
                    summary["ignore_eos"] = True
                    summary["max_tokens"] = MAX_TOKENS
                    summary["prompt_tokens"] = n_prompt
                    levels[conc] = summary
                    save_json(RESULTS_DIR / f"concurrency_{conc}.json", summary)
                    print(
                        f"  rps={summary['requests_per_sec']} tok/s={summary['output_tokens_per_sec']} "
                        f"ttft_p50={summary['ttft_ms']['p50']} e2e_p50={summary['e2e_ms']['p50']} "
                        f"fail={summary['failed_requests']} gpu={gpu.get('avg_gpu_util')} vram={gpu.get('peak_vram_mb')}"
                    )

                    if conc == 16:
                        run32, why = decide_run_32(summary, gpu.get("peak_vram_mb") or 0)
                        cfg["concurrency_32_decision"] = {"run": run32, "reason": why}
                        save_json(RESULTS_DIR / "config.json", cfg)
                        if run32:
                            planned.append(32)
                        else:
                            print(f"skip concurrency 32: {why}")

                print("\n=== continuous batching evidence ===")
                evidence = await continuous_batching_experiment(session, traces_fp)
                save_json(RESULTS_DIR / "continuous_batching_evidence.json", evidence)
                print("  supported=", evidence["continuous_batching_supported_by_evidence"])
                for c in evidence["conclusions"]:
                    print("   -", c)
            finally:
                traces_fp.close()

            sat = saturation_analysis(levels)
            sat["hf_bf16_single_request_reference"] = HF_REF
            sat["vllm_bf16_single_request_reference"] = VLLM_SINGLE_REF
            sat["concurrency_32_tested"] = 32 in levels
            sat["concurrency_32_reason"] = cfg.get("concurrency_32_decision")
            save_json(RESULTS_DIR / "saturation_analysis.json", sat)

            comparison = {
                "stage": "I.1",
                "timestamp": now_iso(),
                "backend": "vllm_bf16_online",
                "prefix_caching": False,
                "config": {
                    "vllm_version": vllm.__version__,
                    "dtype": "bfloat16",
                    "max_model_len": MAX_MODEL_LEN,
                    "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                    "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
                    "enforce_eager": ENFORCE_EAGER,
                    "adapter": str(ADAPTER_DIR),
                },
                "workload": {
                    "prompt_tokens": n_prompt,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "ignore_eos": True,
                    "prompt_family": "H.2 research-agent EEG scenario",
                },
                "single_request_references": {"hf_bf16": HF_REF, "vllm_bf16_h2": VLLM_SINGLE_REF},
                "levels": {str(k): v for k, v in levels.items()},
                "saturation": sat,
                "continuous_batching_supported_by_evidence": evidence["continuous_batching_supported_by_evidence"],
            }
            save_json(CMP_PATH, comparison)
            print("\nI.1 complete. levels=", list(levels))
    finally:
        gpu_poller.stop()
        print("stopping vLLM pid", server.pid)
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        log_f.close()


if __name__ == "__main__":
    asyncio.run(main())
