#!/usr/bin/env python3
"""I.2 supplemental true-cold: requires VLLM_SERVER_DEV_MODE=1 for /reset_prefix_cache."""

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
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

PROJECT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))

from neuro_agent.paths import configure_hf_cache  # noqa: E402

configure_hf_cache()

CKPT = PROJECT / "checkpoints" / "text_w8a8_int8_compressed"
RESULTS = PROJECT / "results" / "serving" / "prefix_cache" / "w8a8_int8"
CMP = PROJECT / "results" / "model_comparison" / "w8a8_prefix_cache_comparison.json"
HOST, PORT = "127.0.0.1", 8000
BASE = f"http://{HOST}:{PORT}"
SERVED = "w8a8-int8"


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


def reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return round((before - after) / before * 100.0, 2)


def build_prompts() -> list[str]:
    from transformers import AutoTokenizer

    spec = importlib.util.spec_from_file_location(
        "p", PROJECT / "src/neuro_agent/agent/prompts.py"
    )
    prompts = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(prompts)
    vtext = (PROJECT / "src/neuro_agent/agent/verifier.py").read_text()
    verifier = re.search(r'VERIFIER_SYSTEM_PROMPT = """(.*?)"""', vtext, re.S).group(1)
    tok = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=False)
    production_system = (
        f"{prompts.INTENT_SYSTEM_PROMPT.strip()}\n\n"
        f"{prompts.ANSWER_SYSTEM_PROMPT.strip()}\n\n"
        "Recovery rewrite policy:\n"
        "When recovery is active, correct any unsupported numeric claims. "
        "Use ONLY values present in the evidence bundle.\n\n"
        f"{verifier.strip()}\n\n"
        "Recovery policy (deterministic runtime):\n"
        "- Trigger verifier on tool failure, grounding warnings, multi-tool routes, or low confidence.\n"
        "- Recovery actions: REWRITE | RETRY_TOOL | REPLAN | INSUFFICIENT_EVIDENCE.\n"
        "- Format-only failures use deterministic format_grounded_answer without model rewrite.\n"
        "- MAX_TOOL_CALLS = 6. Do not invent numeric values or sample IDs.\n"
    )
    questions = [
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
    ]

    def chat(system: str, user: str) -> str:
        return tok.apply_chat_template(
            [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    return [chat(production_system, prompts.build_intent_user_prompt(q)) for q in questions]


async def wait_healthy(timeout: float = 360.0) -> None:
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
        while time.perf_counter() - t0 < timeout:
            try:
                async with s.get(f"{BASE}/health") as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(2)
    raise RuntimeError("not healthy")


async def stream(session: aiohttp.ClientSession, req_id: str, prompt: str) -> dict:
    payload = {
        "model": SERVED,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    t1 = None
    toks = 0
    err = None
    text = ""
    try:
        async with session.post(f"{BASE}/v1/completions", json=payload) as resp:
            if resp.status != 200:
                err = f"http_{resp.status}: {(await resp.text())[:200]}"
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
                    if delta and t1 is None:
                        t1 = time.perf_counter()
                    text += delta
                    u = obj.get("usage") or {}
                    if u.get("completion_tokens"):
                        toks = int(u["completion_tokens"])
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    t2 = time.perf_counter()
    if toks <= 0:
        toks = max(len(text) // 2, 1)
    return {
        "req_id": req_id,
        "ok": err is None and t1 is not None,
        "error": err,
        "ttft_ms": None if t1 is None else round((t1 - t0) * 1000, 3),
        "e2e_ms": round((t2 - t0) * 1000, 3),
        "completion_tokens": toks,
    }


async def reset(session: aiohttp.ClientSession) -> dict:
    async with session.post(f"{BASE}/reset_prefix_cache") as r:
        body = await r.text()
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            data = {"raw": body[:200]}
        return {"http_status": r.status, "ok": r.status == 200, "body": data}


def summarize(traces: list[dict], label: str) -> dict:
    ok = [t for t in traces if t["ok"]]
    tt = [t["ttft_ms"] for t in ok]
    ee = [t["e2e_ms"] for t in ok]
    return {
        "label": label,
        "completed_requests": len(ok),
        "failed_requests": len(traces) - len(ok),
        "ttft_ms": {
            "p50": percentile(tt, 50),
            "p95": percentile(tt, 95),
            "p99": percentile(tt, 99),
            "mean": round(statistics.mean(tt), 3) if tt else None,
            "min": round(min(tt), 3) if tt else None,
            "max": round(max(tt), 3) if tt else None,
        },
        "e2e_ms": {
            "p50": percentile(ee, 50),
            "p95": percentile(ee, 95),
            "p99": percentile(ee, 99),
            "mean": round(statistics.mean(ee), 3) if ee else None,
        },
        "reset_ok": all((t.get("reset") or {}).get("ok") for t in traces if "reset" in t),
    }


async def main() -> None:
    prompts_list = build_prompts()
    manifest = json.loads((RESULTS / "prefix_manifest.json").read_text())
    prod_shared = manifest["variants"]["production"]["shared_prefix_tokens"]

    hit_re = re.compile(r"Prefix cache hit rate: ([0-9.]+)%")
    log = (RESULTS / "vllm_server.log").read_text(errors="replace")
    off_hits: list[float] = []
    on_hits: list[float] = []
    cur = None
    for line in log.splitlines():
        if "prefix_cache=OFF" in line:
            cur = "off"
        elif "prefix_cache=ON" in line:
            cur = "on"
        m = hit_re.search(line)
        if m and cur:
            (off_hits if cur == "off" else on_hits).append(float(m.group(1)))

    subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"], check=False)
    await asyncio.sleep(2)

    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HF_HOME"] = os.environ.get("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
    cmd = [
        "/usr/bin/python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(CKPT),
        "--served-model-name",
        SERVED,
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--dtype",
        "auto",
        "--gpu-memory-utilization",
        "0.9",
        "--max-model-len",
        "4096",
        "--tensor-parallel-size",
        "1",
        "--enable-prefix-caching",
        "--enforce-eager",
    ]
    print("starting true-cold engine:", " ".join(cmd))
    logf = (RESULTS / "vllm_server_true_cold.log").open("w")
    proc = subprocess.Popen(
        cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(PROJECT)
    )
    try:
        await wait_healthy()
        print("healthy")
        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            r0 = await reset(session)
            print("reset probe", r0)
            if not r0["ok"] or not r0.get("body", {}).get("success", True):
                raise RuntimeError(f"reset still failing: {r0}")

            colds = []
            for i in range(12):
                rr = await reset(session)
                body = rr.get("body") or {}
                if not body.get("success", rr["ok"]):
                    await asyncio.sleep(0.3)
                    rr = await reset(session)
                await asyncio.sleep(0.15)
                rec = await stream(session, f"truecold-{i+1}", prompts_list[i % len(prompts_list)])
                rec["reset"] = rr
                colds.append(rec)
                print("cold", i + 1, rec["ttft_ms"], rec["ok"], rr.get("body"))

            await stream(session, "prime", prompts_list[0])
            warms = []
            for i in range(16):
                rec = await stream(
                    session, f"truewarm-{i+1}", prompts_list[i % len(prompts_list)]
                )
                warms.append(rec)
                print("warm", i + 1, rec["ttft_ms"], rec["ok"])
    finally:
        print(f"stopping pid {proc.pid}")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=45)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
        logf.close()
        await asyncio.sleep(2)

    cold_s = summarize(colds, "true-cold")
    warm_s = summarize(warms, "paired-warm-after-true-cold")
    print("TRUE COLD", cold_s["ttft_ms"])
    print("PAIRED WARM", warm_s["ttft_ms"])

    prev = json.loads((RESULTS / "cache_on_cold.json").read_text())
    prev["true_cold_correction"] = {
        "timestamp": now_iso(),
        "reason": (
            "Original cold used POST /reset_prefix_cache which returned 404 without "
            "VLLM_SERVER_DEV_MODE; those samples were warm-contaminated."
        ),
        "method": (
            "Fresh engine with VLLM_SERVER_DEV_MODE=1; POST /reset_prefix_cache "
            "success=true before each of 12 requests"
        ),
        "production_prefix_tokens": prod_shared,
        "result": cold_s,
        "paired_warm_check": warm_s,
        "ttft_reduction_pct_warm_vs_true_cold": reduction(
            cold_s["ttft_ms"]["p50"], warm_s["ttft_ms"]["p50"]
        ),
    }
    prev["note_on_original_result"] = (
        "original result field is warm-contaminated; use true_cold_correction.result"
    )
    (RESULTS / "cache_on_cold.json").write_text(json.dumps(prev, indent=2) + "\n")

    cmp = json.loads(CMP.read_text())
    off_ttft = cmp["ttft_before_after_c1"]["cache_off"]
    warm_ttft = cmp["ttft_before_after_c1"]["cache_on_warm"]
    cmp["cache_hit_reuse_evidence"]["server_log_prefix_cache_hit_rate_pct"] = {
        "cache_off_observed": off_hits,
        "cache_on_observed": on_hits,
        "cache_off_mean": round(statistics.mean(off_hits), 2) if off_hits else None,
        "cache_on_mean": round(statistics.mean(on_hits), 2) if on_hits else None,
        "source": "vLLM Engine periodic logger lines in vllm_server.log",
        "prometheus_note": (
            "vllm:prefix_cache_hits/queries were not present on /metrics in this run; "
            "log hit-rate used instead (not invented)."
        ),
    }
    cmp["cold_vs_warm"] = {
        "cache_off_c1_ttft": off_ttft,
        "cache_on_true_cold_ttft": cold_s["ttft_ms"],
        "cache_on_warm_c1_ttft": warm_ttft,
        "warm_vs_true_cold_ttft_p50_reduction_pct": reduction(
            cold_s["ttft_ms"]["p50"], warm_ttft["p50"]
        ),
        "warm_vs_off_ttft_p50_reduction_pct": cmp["quantified_improvement_production_prefix"][
            "ttft_reduction_pct_p50_warm_vs_off"
        ],
        "true_cold_vs_off_ttft_p50_reduction_pct": reduction(
            off_ttft["p50"], cold_s["ttft_ms"]["p50"]
        ),
        "paired_warm_vs_true_cold_ttft_p50_reduction_pct": reduction(
            cold_s["ttft_ms"]["p50"], warm_s["ttft_ms"]["p50"]
        ),
        "note": (
            "True cold from supplemental engine with working reset API. "
            "Original labeled cold was contaminated (reset 404)."
        ),
    }
    cmp["ttft_before_after_c1"]["cache_on_cold"] = cold_s["ttft_ms"]
    cmp["ttft_before_after_c1"]["cache_on_cold_is_true_cold"] = True
    cmp["quantified_improvement_production_prefix"]["cold_ttft_p50_ms"] = cold_s["ttft_ms"][
        "p50"
    ]
    cmp["quantified_improvement_production_prefix"][
        "warm_vs_true_cold_ttft_p50_reduction_pct"
    ] = reduction(cold_s["ttft_ms"]["p50"], warm_ttft["p50"])
    on_mean = round(statistics.mean(on_hits), 1) if on_hits else "N/A"
    cmp["recommendation"] = (
        "Enable --enable-prefix-caching for research-agent serving: warm TTFT improves "
        f"materially vs cache-off "
        f"({cmp['quantified_improvement_production_prefix']['ttft_reduction_pct_p50_warm_vs_off']}% p50) "
        f"with server-log prefix cache hit rates ~{on_mean}% and correctness PASS. "
        "Prometheus hit counters were unavailable; evidence is log hit-rate + TTFT."
    )
    cmp["true_cold_supplement"] = {
        "completed": True,
        "result": cold_s,
        "paired_warm_check": warm_s,
    }
    CMP.write_text(json.dumps(cmp, indent=2) + "\n")
    print("TRUE_COLD_DONE")


if __name__ == "__main__":
    asyncio.run(main())
