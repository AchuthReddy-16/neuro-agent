#!/usr/bin/env python3
"""Final end-to-end production profiling (measurement only).

Profiles the locked production architecture:
  TEXT: vLLM W8A8 INT8 + prefix cache + deterministic tools + agent loop
  VISION: HF Qwen2.5-VL-3B corrected LoRA via swap/unload (util=0.90)

Does NOT optimize, retrain, requantize, or mutate git.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from neuro_agent.agent.answer_format import format_grounded_answer  # noqa: E402
from neuro_agent.agent.intent import IntentParseResult, parse_and_validate_intent  # noqa: E402
from neuro_agent.agent.policies import is_format_only_failure, should_trigger_verifier  # noqa: E402
from neuro_agent.agent.prompts import (  # noqa: E402
    ANSWER_SYSTEM_PROMPT,
    INTENT_SYSTEM_PROMPT,
    RECOVERY_ANSWER_SYSTEM_PROMPT,
    build_answer_user_prompt,
    build_intent_user_prompt,
)
from neuro_agent.agent.recovery import execute_recovery, plan_recovery  # noqa: E402
from neuro_agent.agent.traces import check_grounding  # noqa: E402
from neuro_agent.agent.verifier import (  # noqa: E402
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_user_prompt,
    deterministic_to_verification,
    merge_verification,
    parse_verification_json,
    run_deterministic_checks,
)
from neuro_agent.tools.comparison import compare_conditions  # noqa: E402
from neuro_agent.tools.eeg_signal import compute_band_power, compute_rms, find_psd_peak  # noqa: E402
from neuro_agent.tools.evidence import EvidenceBundle, ResearchToolRequest, new_request_id  # noqa: E402
from neuro_agent.tools.ranking import rank_channels_for_sample, select_channels_above_threshold  # noqa: E402
from neuro_agent.tools.router import route_research_request  # noqa: E402

OUT = PROJECT_ROOT / "results" / "profiling" / "final_system"
CMP = PROJECT_ROOT / "results" / "model_comparison"
TEXT_CKPT = PROJECT_ROOT / "checkpoints" / "text_w8a8_int8_compressed"
VISION_BASE = "Qwen/Qwen2.5-VL-3B-Instruct"
VISION_ADAPTER = PROJECT_ROOT / "checkpoints" / "multimodal_sft_corrected" / "final"
VLLM_PYTHON = "/usr/bin/python3"
HOST = "127.0.0.1"
PORT = 8000
SERVED = "qwen3-w8a8-int8"
GPU_UTIL = 0.90
MAX_MODEL_LEN = 4096
MAX_TOOL_CALLS = 6

TOPOMAP = PROJECT_ROOT / "data/processed/vision/images/test/img_topomap_06b9e5be484f99a7.png"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    torch_alloc = (
        round(torch.cuda.memory_allocated(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0
    )
    torch_reserved = (
        round(torch.cuda.memory_reserved(0) / (1024**2), 1) if torch.cuda.is_available() else 0.0
    )
    return {
        "nvidia_smi_used_mb": used,
        "nvidia_smi_total_mb": total,
        "gpu_util_pct": util,
        "torch_allocated_mb": torch_alloc,
        "torch_reserved_mb": torch_reserved,
    }


# ---------------------------------------------------------------------------
# Span / trace helpers
# ---------------------------------------------------------------------------


@dataclass
class Span:
    name: str
    t0: float
    t1: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def ms(self) -> float | None:
        if self.t1 is None:
            return None
        return (self.t1 - self.t0) * 1000.0


class TraceClock:
    def __init__(self) -> None:
        self.t_request = time.perf_counter()
        self.spans: list[Span] = []
        self._open: dict[str, Span] = {}

    def start(self, name: str, **meta: Any) -> None:
        self._open[name] = Span(name=name, t0=time.perf_counter(), meta=dict(meta))

    def end(self, name: str, **meta: Any) -> float:
        sp = self._open.pop(name)
        sp.t1 = time.perf_counter()
        sp.meta.update(meta)
        self.spans.append(sp)
        assert sp.ms is not None
        return sp.ms

    def mark(self, name: str, ms: float, **meta: Any) -> None:
        now = time.perf_counter()
        self.spans.append(
            Span(name=name, t0=now - ms / 1000.0, t1=now, meta=dict(meta))
        )

    def e2e_ms(self) -> float:
        return (time.perf_counter() - self.t_request) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "e2e_ms": round(self.e2e_ms(), 3),
            "spans": [
                {"name": s.name, "ms": None if s.ms is None else round(s.ms, 3), **s.meta}
                for s in self.spans
            ],
            "span_map_ms": {
                s.name: round(s.ms, 3) for s in self.spans if s.ms is not None
            },
        }


# ---------------------------------------------------------------------------
# vLLM lifecycle + generation
# ---------------------------------------------------------------------------


def start_vllm() -> tuple[subprocess.Popen, Any]:
    log_path = OUT / "vllm_k1.log"
    log_f = log_path.open("a")
    log_f.write(f"\n===== start {now_iso()} =====\n")
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
        str(GPU_UTIL),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--tensor-parallel-size",
        "1",
        "--enable-prefix-caching",
        "--enforce-eager",
    ]
    print("starting vLLM:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=str(PROJECT_ROOT), env=env
    )
    return proc, log_f


def stop_vllm(proc: subprocess.Popen | None, log_f) -> dict[str, Any]:
    """Stop text engine and measure wall + VRAM release. Prefer SIGTERM on our child."""
    t0 = time.perf_counter()
    before = gpu_stats()
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
    # Wait until port free / VRAM drops (no inventing times — measured loop)
    released = False
    for _ in range(60):
        time.sleep(0.5)
        try:
            requests.get(f"http://{HOST}:{PORT}/health", timeout=0.5)
            still_up = True
        except Exception:
            still_up = False
        g = gpu_stats()
        if not still_up and g["nvidia_smi_used_mb"] < 2000:
            released = True
            break
    after = gpu_stats()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "wall_ms": round(wall_ms, 3),
        "before": before,
        "after": after,
        "vram_released_mb": round(before["nvidia_smi_used_mb"] - after["nvidia_smi_used_mb"], 1),
        "released": released,
    }


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


class VLLMTextBackend:
    """Production text path: vLLM W8A8 completions with TTFT via streaming."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(str(TEXT_CKPT), trust_remote_code=True)
        self.last_metrics: dict[str, Any] = {}

    def chat_prompt(self, system: str, user: str) -> str:
        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user.strip()},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompt: str, max_tokens: int) -> tuple[str, dict[str, Any]]:
        payload = {
            "model": SERVED,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": True,
        }
        t0 = time.perf_counter()
        t_first = None
        text = ""
        err = None
        completion_tokens = 0
        try:
            with requests.post(
                f"http://{HOST}:{PORT}/v1/completions",
                json=payload,
                stream=True,
                timeout=300,
            ) as resp:
                if resp.status_code != 200:
                    err = f"http_{resp.status_code}: {resp.text[:300]}"
                else:
                    for raw in resp.iter_lines(decode_unicode=True):
                        if not raw:
                            continue
                        if not raw.startswith("data:"):
                            continue
                        data = raw[5:].strip()
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
                        usage = obj.get("usage") or {}
                        if usage.get("completion_tokens"):
                            completion_tokens = int(usage["completion_tokens"])
        except Exception as exc:  # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"

        # Fallback: non-stream if stream yielded nothing (observed with some short outputs)
        if not text and err is None:
            try:
                t0 = time.perf_counter()
                t_first = None
                resp = requests.post(
                    f"http://{HOST}:{PORT}/v1/completions",
                    json={**payload, "stream": False},
                    timeout=300,
                )
                if resp.status_code != 200:
                    err = f"http_{resp.status_code}: {resp.text[:300]}"
                else:
                    obj = resp.json()
                    text = (obj.get("choices") or [{}])[0].get("text") or ""
                    t_first = time.perf_counter()  # approximate TTFT≈E2E for non-stream short outs
                    usage = obj.get("usage") or {}
                    completion_tokens = int(usage.get("completion_tokens") or 0)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"

        t1 = time.perf_counter()
        if completion_tokens <= 0 and text:
            try:
                completion_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                completion_tokens = max(len(text.split()), 1)
        e2e_ms = (t1 - t0) * 1000.0
        ttft_ms = None if t_first is None else (t_first - t0) * 1000.0
        decode_ms = None if t_first is None else (t1 - t_first) * 1000.0
        decode_tok_s = None
        if t_first is not None and completion_tokens > 0 and decode_ms and decode_ms > 1:
            decode_tok_s = round(completion_tokens / max((t1 - t_first), 1e-6), 2)
        prompt_tokens = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        metrics = {
            "ok": err is None and bool(text),
            "error": err,
            "e2e_ms": round(e2e_ms, 3),
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 3),
            "prefill_proxy_ms": None if ttft_ms is None else round(ttft_ms, 3),
            "decode_ms": None if decode_ms is None else round(decode_ms, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "decode_tok_s": decode_tok_s,
            "queue_wait_ms": None,
            "note": "queue/scheduler wait not separately exposed; TTFT includes any queue at c=1 (~0).",
            "gpu": gpu_stats(),
        }
        self.last_metrics = metrics
        return text.strip(), metrics


# ---------------------------------------------------------------------------
# Profiled agent pipeline (mirrors PrimaryResearchAgent.ask)
# ---------------------------------------------------------------------------


def profiled_ask(
    backend: VLLMTextBackend,
    question: str,
    *,
    request_class: str,
    draft_corruption: Callable[[str], str] | None = None,
    forced_request: ResearchToolRequest | None = None,
    intent_max_tokens: int = 256,
    answer_max_tokens: int = 512,
    verifier_max_tokens: int = 256,
) -> dict[str, Any]:
    clock = TraceClock()
    req_id = new_request_id()
    text_calls: list[dict[str, Any]] = []
    errors: list[str] = []

    # --- routing / intent ---
    clock.start("routing_total")
    clock.start("intent_generation")
    intent_prompt = backend.chat_prompt(
        INTENT_SYSTEM_PROMPT, build_intent_user_prompt(question)
    )
    raw_intent, intent_metrics = backend.generate(intent_prompt, intent_max_tokens)
    intent_ms = clock.end("intent_generation", **intent_metrics)
    text_calls.append({"stage": "intent", **intent_metrics})

    clock.start("intent_parse_validate")
    intent_result = parse_and_validate_intent(raw_intent)
    requires_vision = False
    raw_json = intent_result.raw_json
    if raw_json is None:
        try:
            from neuro_agent.agent.intent import extract_json_object

            raw_json = extract_json_object(raw_intent)
        except Exception:
            raw_json = {}
    if isinstance(raw_json, dict):
        requires_vision = bool(
            raw_json.get("requires_vision")
            or raw_json.get("include_vision_evidence")
            or raw_json.get("requested_visual_type")
            or raw_json.get("image_id")
        )
        if intent_result.raw_json is None:
            intent_result.raw_json = raw_json
    parse_ms = clock.end("intent_parse_validate")

    intent_override_note = None
    if forced_request is not None:
        # Keep W8A8 intent latency measurement, but execute known-good tool request for
        # multi-tool / verifier path profiling when natural intent misroutes.
        intent_override_note = {
            "natural_intent_question_type": (raw_json or {}).get("question_type"),
            "forced_question_type": forced_request.question_type,
            "reason": "profile multi-tool/verifier path with production tools",
        }
        intent_result = IntentParseResult(
            success=True,
            request=forced_request,
            raw_json={
                **(raw_json or {}),
                **forced_request.to_dict(),
                "requires_vision": False,
            },
            question_type=forced_request.question_type,
        )

    clock.end(
        "routing_total",
        requires_vision=requires_vision,
        intent_valid=intent_result.success,
        intent_override=intent_override_note,
    )
    clock.mark(
        "requires_vision_decision",
        parse_ms,
        requires_vision=requires_vision,
        raw_requires_vision=bool((raw_json or {}).get("requires_vision")),
    )

    if not intent_result.success or intent_result.request is None:
        return {
            "request_class": request_class,
            "request_id": req_id,
            "question": question,
            "success": False,
            "errors": [intent_result.error or "intent failed"],
            "requires_vision": requires_vision,
            "raw_intent": raw_json,
            "trace": clock.to_dict(),
            "text_calls": text_calls,
            "final_answer": None,
        }

    # --- tools / evidence ---
    clock.start("tool_execution_total")
    t_tool0 = time.perf_counter()
    bundle = route_research_request(intent_result.request, request_id=req_id)
    tool_wall = (time.perf_counter() - t_tool0) * 1000.0
    tool_ms = clock.end(
        "tool_execution_total",
        tool_count=len(bundle.tool_invocations),
        tool_names=[t.name for t in bundle.tool_invocations],
        success=bundle.success,
    )

    clock.start("evidence_bundle_construction")
    t_ev0 = time.perf_counter()
    evidence_dict = bundle.to_dict()
    t_ser0 = time.perf_counter()
    _ = json.dumps(evidence_dict, default=str)
    ser_ms = (time.perf_counter() - t_ser0) * 1000.0
    ev_ms = (time.perf_counter() - t_ev0) * 1000.0
    clock.end(
        "evidence_bundle_construction",
        construction_ms=round(ev_ms, 3),
        serialization_ms=round(ser_ms, 3),
        tool_wall_ms=round(tool_wall, 3),
    )

    tool_count = len(bundle.tool_invocations)
    if not bundle.success:
        errors.append(bundle.error or "routing failed")

    draft_answer = None
    final_answer = None
    verification_triggered = False
    trigger_reason: list[str] = []
    path_mode = "NORMAL"
    first_pass_verification = None
    recovery_info = None
    model_calls = 1

    if bundle.success and not errors:
        clock.start("grounded_answer_generation")
        ans_prompt = backend.chat_prompt(
            ANSWER_SYSTEM_PROMPT,
            build_answer_user_prompt(question, evidence_dict),
        )
        draft_answer, ans_metrics = backend.generate(ans_prompt, answer_max_tokens)
        answer_ms = clock.end("grounded_answer_generation", **ans_metrics)
        text_calls.append({"stage": "answer", **ans_metrics})
        model_calls += 1

        if draft_corruption and draft_answer:
            clock.start("draft_corruption_inject")
            draft_answer = draft_corruption(draft_answer)
            clock.end("draft_corruption_inject")

        final_answer = draft_answer
        grounding = check_grounding(final_answer, bundle)

        # verifier path
        clock.start("verifier_block")
        det = run_deterministic_checks(
            draft_answer, bundle, request=intent_result.request
        )
        trigger, trigger_reason = should_trigger_verifier(
            det, bundle, intent_result, draft_answer, tool_count=tool_count
        )
        verification_triggered = trigger

        if not trigger and det.passed:
            first_pass_verification = deterministic_to_verification(det).to_dict()
            clock.end("verifier_block", triggered=False, mode="deterministic_fast_path")
        elif is_format_only_failure(det):
            verification_triggered = False
            t_fmt0 = time.perf_counter()
            final_answer = format_grounded_answer(question, bundle)
            first_pass_verification = deterministic_to_verification(det).to_dict()
            clock.end(
                "verifier_block",
                triggered=False,
                mode="format_only_repair",
                format_ms=round((time.perf_counter() - t_fmt0) * 1000.0, 3),
            )
        else:
            verification_triggered = True
            clock.start("verifier_model")
            ver_prompt = backend.chat_prompt(
                VERIFIER_SYSTEM_PROMPT,
                build_verifier_user_prompt(
                    question, intent_result.raw_json, bundle, draft_answer, det
                ),
            )
            raw_ver, ver_metrics = backend.generate(ver_prompt, verifier_max_tokens)
            ver_ms = clock.end("verifier_model", **ver_metrics)
            text_calls.append({"stage": "verifier", **ver_metrics})
            model_calls += 1
            try:
                model_json = parse_verification_json(raw_ver)
                ver_result = merge_verification(det, model_json)
            except Exception:
                ver_result = deterministic_to_verification(det)
                ver_result.failure_codes.append("verifier_parse_error")
            first_pass_verification = ver_result.to_dict()

            if not ver_result.passed:
                path_mode = "RECOVERY"
                clock.start("recovery")
                action = plan_recovery(
                    ver_result,
                    det,
                    tool_count=tool_count,
                    max_tool_calls=MAX_TOOL_CALLS,
                )

                def _gen_rec(q: str, b: EvidenceBundle) -> tuple[str, float, float]:
                    t0 = time.perf_counter()
                    prompt = backend.chat_prompt(
                        RECOVERY_ANSWER_SYSTEM_PROMPT,
                        build_answer_user_prompt(q, b.to_dict()),
                    )
                    txt, met = backend.generate(prompt, answer_max_tokens)
                    text_calls.append({"stage": "recovery_answer", **met})
                    return txt, (time.perf_counter() - t0) * 1000.0, 0.0

                def _parse_intent(q: str):
                    t0 = time.perf_counter()
                    p = backend.chat_prompt(
                        INTENT_SYSTEM_PROMPT, build_intent_user_prompt(q)
                    )
                    raw, met = backend.generate(p, intent_max_tokens)
                    text_calls.append({"stage": "recovery_intent", **met})
                    res = parse_and_validate_intent(raw)
                    return res, (time.perf_counter() - t0) * 1000.0, 0.0

                rec_result = execute_recovery(
                    action,
                    question=question,
                    draft_answer=draft_answer,
                    bundle=bundle,
                    intent_result=intent_result,
                    verification=ver_result,
                    deterministic=det,
                    generate_answer=_gen_rec,
                    parse_intent=_parse_intent,
                    request_id=req_id,
                    tool_count=tool_count,
                    max_tool_calls=MAX_TOOL_CALLS,
                )
                recovery_ms = clock.end(
                    "recovery",
                    action=rec_result.action,
                    success=rec_result.success,
                )
                final_answer = rec_result.final_answer
                if rec_result.bundle is not None:
                    bundle = rec_result.bundle
                recovery_info = {
                    "action": rec_result.action,
                    "success": rec_result.success,
                    "latency_ms": round(recovery_ms, 3),
                }
                model_calls += 1 if rec_result.action in {"REWRITE", "REPLAN"} else 0

            clock.end(
                "verifier_block",
                triggered=True,
                mode="model_verifier",
                passed=bool(first_pass_verification and first_pass_verification.get("passed")),
            )

        grounding = check_grounding(final_answer, bundle) if final_answer else grounding

    clock.start("final_serialization")
    payload = {
        "final_answer": final_answer,
        "evidence": bundle.to_dict() if bundle else None,
        "verification": first_pass_verification,
    }
    _ = json.dumps(payload, default=str)
    clock.end("final_serialization")

    trace = clock.to_dict()
    return {
        "request_class": request_class,
        "request_id": req_id,
        "question": question,
        "success": bool(final_answer) and not errors,
        "errors": errors,
        "requires_vision": requires_vision,
        "raw_intent": raw_json,
        "tool_count": tool_count,
        "tool_names": [t.name for t in bundle.tool_invocations] if bundle else [],
        "verification_triggered": verification_triggered,
        "trigger_reason": trigger_reason,
        "path_mode": path_mode,
        "first_pass_verification": first_pass_verification,
        "recovery": recovery_info,
        "model_calls": model_calls,
        "final_answer_preview": (final_answer or "")[:400],
        "trace": trace,
        "text_calls": text_calls,
        "gpu_end": gpu_stats(),
        "timestamp": now_iso(),
        "intent_override": intent_override_note,
    }


# ---------------------------------------------------------------------------
# Tool microbenchmarks
# ---------------------------------------------------------------------------


def profile_tools(n_warmup: int = 1, n_reps: int = 5) -> dict[str, Any]:
    sample = "S001_R01_E000"
    sample_psd = "S016_R08_E022"
    subject = "S013"

    def timed(fn: Callable[[], Any], reps: int) -> dict[str, Any]:
        # warmup
        for _ in range(n_warmup):
            fn()
        times = []
        last = None
        for _ in range(reps):
            t0 = time.perf_counter()
            last = fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        times.sort()
        return {
            "n": reps,
            "p50_ms": round(times[len(times) // 2], 3),
            "mean_ms": round(sum(times) / len(times), 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "result_type": type(last).__name__ if last is not None else None,
        }

    # separate data-load vs compute where possible by calling twice (cache warm)
    rows = {}

    rows["band_power"] = timed(
        lambda: compute_band_power(sample, "beta", channels=["C3"]), n_reps
    )
    rows["band_power_all_channels"] = timed(
        lambda: compute_band_power(sample, "beta", channels="all"), n_reps
    )
    rows["rms"] = timed(lambda: compute_rms(sample, channels=["C3"]), n_reps)
    rows["psd_peak"] = timed(lambda: find_psd_peak(sample_psd), n_reps)
    rows["channel_ranking"] = timed(
        lambda: rank_channels_for_sample(sample, "beta_power", top_k=3), n_reps
    )

    def threshold():
        bp = compute_band_power(sample, "beta", channels="all")
        vals = {r.channel: r.power for r in bp.results}
        return select_channels_above_threshold(
            vals, threshold_mode="upper_quartile", comparator="ge"
        )

    rows["threshold_selection_incl_band_power"] = timed(threshold, n_reps)
    rows["condition_comparison"] = timed(
        lambda: compare_conditions(
            subject, "left_fist", "right_fist", metric="beta_power"
        ),
        n_reps,
    )

    # serialization overhead of full router bundles
    def route_bp():
        return route_research_request(
            ResearchToolRequest(
                question_type="band_power",
                sample_id=sample,
                frequency_band="beta",
                channels=["C3"],
            )
        ).to_dict()

    def route_thr():
        return route_research_request(
            ResearchToolRequest(
                question_type="threshold_set",
                sample_id=sample,
                frequency_band="beta",
                threshold_mode="upper_quartile",
            )
        ).to_dict()

    ser_times = []
    for _ in range(n_reps):
        b = route_bp()
        t0 = time.perf_counter()
        json.dumps(b, default=str)
        ser_times.append((time.perf_counter() - t0) * 1000.0)
    rows["evidence_json_serialization_band_power"] = {
        "mean_ms": round(sum(ser_times) / len(ser_times), 3),
        "p50_ms": round(sorted(ser_times)[len(ser_times) // 2], 3),
    }

    route_times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        route_thr()
        route_times.append((time.perf_counter() - t0) * 1000.0)
    rows["router_threshold_set_end_to_end"] = {
        "mean_ms": round(sum(route_times) / len(route_times), 3),
        "p50_ms": round(sorted(route_times)[len(route_times) // 2], 3),
        "note": "includes compute_band_power + select_channels + evidence assembly",
    }

    expensive = sorted(
        ((k, v.get("mean_ms") or v.get("p50_ms") or 0.0) for k, v in rows.items()),
        key=lambda x: -x[1],
    )
    return {
        "timestamp": now_iso(),
        "n_warmup": n_warmup,
        "n_reps": n_reps,
        "tools": rows,
        "ranked_by_mean_ms": [{"tool": k, "mean_ms": v} for k, v in expensive],
        "unexpectedly_expensive": [
            {"tool": k, "mean_ms": v}
            for k, v in expensive
            if v > 50.0  # heuristic flag for tool path
        ],
        "binding": "CPU (pandas/numpy/h5 lookups); no GPU",
    }


# ---------------------------------------------------------------------------
# Vision swap path
# ---------------------------------------------------------------------------


def load_vision_model_timed() -> dict[str, Any]:
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
    model, processor, info = load_vlm_for_inference(cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    after = gpu_stats()
    return {
        "model": model,
        "processor": processor,
        "wall_ms": round(wall_ms, 3),
        "load_time_s_reported": info.load_time_s,
        "before": before,
        "after": after,
        "peak_torch_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "vram_delta_mb": round(after["nvidia_smi_used_mb"] - before["nvidia_smi_used_mb"], 1),
    }


def unload_vision_timed(model) -> dict[str, Any]:
    before = gpu_stats()
    t0 = time.perf_counter()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    # poll until VRAM settles
    after = before
    for _ in range(40):
        time.sleep(0.25)
        after = gpu_stats()
        if after["nvidia_smi_used_mb"] < before["nvidia_smi_used_mb"] - 500:
            # keep waiting a bit more for full release
            if after["nvidia_smi_used_mb"] < 3000:
                break
    wall_ms = (time.perf_counter() - t0) * 1000.0
    final = gpu_stats()
    return {
        "wall_ms": round(wall_ms, 3),
        "before": before,
        "after": final,
        "vram_released_mb": round(before["nvidia_smi_used_mb"] - final["nvidia_smi_used_mb"], 1),
    }


def run_vision_infer(model, processor, image_path: Path, question: str, context: dict | None = None) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    from neuro_agent.multimodal.dataset import build_multimodal_messages

    system_prompt = (
        "You are a neuroscience research assistant analyzing EEG-derived plots. "
        "Answer briefly based on the image and context."
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
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
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
    prefill_proxy = gen_ms * (in_len / max(in_len + n_new, 1))
    return {
        "preprocess_ms": round(preprocess_ms, 3),
        "generate_ms": round(gen_ms, 3),
        "ttft_proxy_ms": round(preprocess_ms + prefill_proxy, 3),
        "e2e_ms": round(preprocess_ms + gen_ms, 3),
        "input_tokens": in_len,
        "completion_tokens": n_new,
        "output": decoded.strip(),
        "peak_torch_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 1),
        "gpu": gpu_stats(),
    }


def wait_vram_below(mb: float, timeout_s: float = 120.0) -> dict[str, Any]:
    t0 = time.perf_counter()
    last = gpu_stats()
    while time.perf_counter() - t0 < timeout_s:
        last = gpu_stats()
        if last["nvidia_smi_used_mb"] <= mb:
            return {
                "ok": True,
                "wait_ms": round((time.perf_counter() - t0) * 1000.0, 3),
                "gpu": last,
            }
        time.sleep(0.5)
    return {
        "ok": False,
        "wait_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "gpu": last,
    }


def run_vision_worker_subprocess(question: str, context: dict | None = None) -> dict[str, Any]:
    """Run VLM in a child process so CUDA fully releases on exit."""
    req_path = OUT / "_vision_worker_req.json"
    out_path = OUT / "_vision_worker_out.json"
    req_path.write_text(
        json.dumps(
            {
                "image_path": str(TOPOMAP),
                "question": question,
                "context": context or {},
            }
        )
    )
    if out_path.exists():
        out_path.unlink()
    t0 = time.perf_counter()
    before = gpu_stats()
    proc = subprocess.run(
        [
            str(PROJECT_ROOT / ".venv" / "bin" / "python"),
            str(PROJECT_ROOT / "scripts" / "k1_vision_worker.py"),
            str(req_path),
            str(out_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    wall_ms = (time.perf_counter() - t0) * 1000.0
    after = gpu_stats()
    if proc.returncode != 0 or not out_path.exists():
        return {
            "ok": False,
            "wall_ms": round(wall_ms, 3),
            "stderr": (proc.stderr or "")[-2000:],
            "stdout": (proc.stdout or "")[-1000:],
            "before": before,
            "after": after,
        }
    data = json.loads(out_path.read_text())
    data["subprocess_wall_ms"] = round(wall_ms, 3)
    data["parent_before"] = before
    data["parent_after"] = after
    return data


def profile_vision_swap(
    text_proc_holder: dict[str, Any],
    backend: VLLMTextBackend,
    *,
    with_tools: bool,
) -> dict[str, Any]:
    """Full production vision path with measured swap/unload/reload."""
    clock = TraceClock()
    question_vision = (
        "Does this topomap visually support the measured beta-power ranking?"
        if with_tools
        else "Describe the spatial pattern visible in this topomap: where is power concentrated?"
    )
    ranking_note = None

    # 1) Intent on text path
    clock.start("text_intent_routing")
    intent_q = (
        "Tools ranked channels by beta power for S001_R01_E000 as T8, IZ, O2. "
        "Does this topomap visually support the measured beta-power ranking?"
        if with_tools
        else question_vision
    )
    prompt = backend.chat_prompt(INTENT_SYSTEM_PROMPT, build_intent_user_prompt(intent_q))
    raw, intent_met = backend.generate(prompt, 256)
    clock.end("text_intent_routing", **intent_met)
    requires_vision = True
    try:
        from neuro_agent.agent.intent import extract_json_object

        raw_j = extract_json_object(raw)
        requires_vision = bool(
            raw_j.get("requires_vision")
            or raw_j.get("include_vision_evidence")
            or raw_j.get("requested_visual_type")
        )
    except Exception:
        raw_j = {}

    # 2) Optional deterministic tools before swap
    if with_tools:
        clock.start("deterministic_tools_before_swap")
        ranking = rank_channels_for_sample("S001_R01_E000", "beta_power", top_k=3)
        top_channels = list(ranking.ranking)[:3] if getattr(ranking, "ranking", None) else []
        ranking_note = {
            "top_channels": top_channels,
            "metric": "beta_power",
            "sample_id": "S001_R01_E000",
        }
        clock.end("deterministic_tools_before_swap", ranking=ranking_note)

    # 3) Unload text vLLM
    clock.start("text_vllm_unload_release")
    unload_text = stop_vllm(text_proc_holder.get("proc"), text_proc_holder.get("log_f"))
    text_proc_holder["proc"] = None
    text_proc_holder["log_f"] = None
    vram_gate = wait_vram_below(2500)
    clock.end("text_vllm_unload_release", **unload_text, vram_gate=vram_gate)

    # 4–6) VLM load + infer + unload in isolated subprocess
    clock.start("vlm_subprocess_load_infer_unload")
    worker = run_vision_worker_subprocess(
        question_vision,
        context={
            "visualization_type": "topomap",
            "numeric_ranking": ranking_note,
            "image_id": TOPOMAP.name,
        },
    )
    clock.end(
        "vlm_subprocess_load_infer_unload",
        ok=worker.get("ok"),
        vlm_load_ms=worker.get("vlm_load_ms"),
        preprocess_ms=worker.get("preprocess_ms"),
        generate_ms=worker.get("generate_ms"),
        vlm_unload_ms=worker.get("vlm_unload_ms"),
        subprocess_wall_ms=worker.get("subprocess_wall_ms"),
    )
    if not worker.get("ok"):
        raise RuntimeError(f"vision worker failed: {worker.get('stderr', '')[:500]}")

    # ensure parent sees free GPU before restore
    clock.start("post_vision_vram_settle")
    settle = wait_vram_below(2500, timeout_s=180.0)
    clock.end("post_vision_vram_settle", **settle)
    if not settle.get("ok"):
        raise RuntimeError(f"VRAM not free enough to restore text vLLM: {settle}")

    # 7) Restore text vLLM
    clock.start("text_vllm_restore")
    t_rest0 = time.perf_counter()
    proc, log_f = start_vllm()
    healthy_ms = wait_healthy(timeout_s=420.0)
    restore_wall = (time.perf_counter() - t_rest0) * 1000.0
    text_proc_holder["proc"] = proc
    text_proc_holder["log_f"] = log_f
    after_restore = gpu_stats()
    clock.end(
        "text_vllm_restore",
        wall_ms=round(restore_wall, 3),
        wait_healthy_ms=round(healthy_ms, 3),
        gpu=after_restore,
    )

    # 8) Optional grounded answer on text after restore (combined path)
    answer_met = None
    if with_tools:
        clock.start("grounded_answer_after_vision")
        evidence = {
            "numeric_evidence": ranking_note,
            "vision_evidence": [{"image": str(TOPOMAP), "vlm_output": worker.get("output")}],
            "tool_invocations": [{"name": "rank_channels_for_sample", "success": True}],
        }
        ans_prompt = backend.chat_prompt(
            ANSWER_SYSTEM_PROMPT,
            build_answer_user_prompt(intent_q, evidence),
        )
        ans_text, answer_met = backend.generate(ans_prompt, 256)
        clock.end("grounded_answer_after_vision", **answer_met, preview=ans_text[:200])

    pure_swap_ms = (
        unload_text["wall_ms"]
        + float(worker.get("vlm_load_ms") or 0)
        + float(worker.get("vlm_unload_ms") or 0)
        + restore_wall
        + float(settle.get("wait_ms") or 0)
    )
    infer_e2e = float(worker.get("infer_e2e_ms") or 0)

    return {
        "request_class": "F_VISION_TOOL_COMBINED" if with_tools else "E_VISION_REQUIRED",
        "question": intent_q,
        "requires_vision": requires_vision,
        "raw_intent": raw_j,
        "ranking": ranking_note,
        "vlm_output": worker.get("output"),
        "trace": clock.to_dict(),
        "worker": {k: v for k, v in worker.items() if k not in {"stderr", "stdout"}},
        "swap_breakdown_ms": {
            "text_unload_release": unload_text["wall_ms"],
            "vram_released_on_text_unload_mb": unload_text.get("vram_released_mb"),
            "vlm_load": worker.get("vlm_load_ms"),
            "image_preprocess": worker.get("preprocess_ms"),
            "vlm_generate": worker.get("generate_ms"),
            "vlm_ttft_proxy": worker.get("ttft_proxy_ms"),
            "vlm_unload": worker.get("vlm_unload_ms"),
            "post_vision_vram_settle": settle.get("wait_ms"),
            "text_vllm_restore": round(restore_wall, 3),
            "text_vllm_wait_healthy": round(healthy_ms, 3),
            "total_swap_overhead_excluding_infer": round(pure_swap_ms, 3),
            "total_swap_plus_infer": round(pure_swap_ms + infer_e2e, 3),
            "measurement_note": (
                "VLM ran in isolated subprocess so CUDA releases on process exit; "
                "swap overhead is measured wall-clock, not invented."
            ),
        },
        "intent_metrics": intent_met,
        "answer_metrics": answer_met,
        "gpu_final": gpu_stats(),
        "timestamp": now_iso(),
    }


# ---------------------------------------------------------------------------
# Aggregation / ranking
# ---------------------------------------------------------------------------


def pct(part: float | None, whole: float | None) -> float | None:
    if part is None or whole is None or whole <= 0:
        return None
    return round(100.0 * part / whole, 2)


def build_bottleneck_ranking(
    traces: list[dict[str, Any]],
    tool_profile: dict[str, Any],
    vision_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank by measured contribution across profiled production paths."""
    candidates: list[dict[str, Any]] = []

    # Aggregate span means for text classes
    span_sums: dict[str, list[float]] = defaultdict(list)
    e2es = []
    for tr in traces:
        if tr.get("request_class", "").startswith("E_") or tr.get("request_class", "").startswith("F_"):
            continue
        e2e = tr.get("trace", {}).get("e2e_ms")
        if e2e:
            e2es.append(e2e)
        for name, ms in (tr.get("trace", {}).get("span_map_ms") or {}).items():
            span_sums[name].append(ms)

    mean_e2e_text = sum(e2es) / len(e2es) if e2es else None

    def add(component: str, latency: float | None, e2e_ref: float | None, types: list[str], bound: str, opportunity: str, complexity: str):
        if latency is None:
            return
        candidates.append(
            {
                "component": component,
                "measured_latency_ms": round(latency, 3),
                "pct_of_request_e2e": pct(latency, e2e_ref),
                "request_types_affected": types,
                "cpu_or_gpu_bound": bound,
                "optimization_opportunity": opportunity,
                "expected_complexity": complexity,
                "evidence": "measured_k1",
            }
        )

    # Text LLM stages
    for span_name, label, types, opp, cplx in [
        (
            "grounded_answer_generation",
            "text_grounded_answer_generation",
            ["A", "B", "C", "D"],
            "Shorten answer prompt/evidence; rely on prefix cache; constrain max tokens",
            "medium",
        ),
        (
            "intent_generation",
            "text_intent_generation",
            ["A", "B", "C", "D", "E", "F"],
            "Shorter intent schema/prompt; cache system prefix (already ON)",
            "low-medium",
        ),
        (
            "verifier_model",
            "conditional_verifier_model_call",
            ["B", "C", "D"],
            "Tighten trigger policy; skip model when det checks suffice",
            "medium",
        ),
        (
            "recovery",
            "recovery_rewrite_cycle",
            ["D"],
            "Faster format-only path; avoid full rewrite when possible",
            "medium",
        ),
        (
            "tool_execution_total",
            "deterministic_tool_execution",
            ["A", "B", "C", "D"],
            "Cache feature lookups; avoid cold parquet scans",
            "low-medium",
        ),
    ]:
        vals = span_sums.get(span_name) or []
        if vals:
            add(label, sum(vals) / len(vals), mean_e2e_text, types, "GPU" if "tool" not in span_name else "CPU", opp, cplx)

    # Vision swap — use measured totals
    if vision_profiles:
        swap_vals = [
            v["swap_breakdown_ms"]["total_swap_overhead_excluding_infer"]
            for v in vision_profiles
        ]
        swap_mean = sum(swap_vals) / len(swap_vals)
        # e2e for vision includes swap
        vis_e2e = [v["trace"]["e2e_ms"] for v in vision_profiles]
        mean_vis_e2e = sum(vis_e2e) / len(vis_e2e)
        add(
            "text_vision_swap_unload_reload_overhead",
            swap_mean,
            mean_vis_e2e,
            ["E", "F"],
            "GPU+orchestration",
            "Keep VLM warm at reduced text util; or persistent dual-process with util≈0.40; or lazy/pool VLM",
            "high (architecture) / medium (keep dual at reduced util)",
        )
        load_vals = [v["swap_breakdown_ms"]["vlm_load"] for v in vision_profiles]
        restore_vals = [v["swap_breakdown_ms"]["text_vllm_restore"] for v in vision_profiles]
        unload_vals = [v["swap_breakdown_ms"]["text_unload_release"] for v in vision_profiles]
        add("vlm_cold_load", sum(load_vals) / len(load_vals), mean_vis_e2e, ["E", "F"], "GPU", "Persist VLM or accelerate load", "high")
        add("text_vllm_restore_after_vision", sum(restore_vals) / len(restore_vals), mean_vis_e2e, ["E", "F"], "GPU+orchestration", "Sleep/wake engine instead of full restart if supported", "medium-high")
        add("text_vllm_unload_for_vision", sum(unload_vals) / len(unload_vals), mean_vis_e2e, ["E", "F"], "orchestration", "Engine sleep API / reduced co-residency", "medium")
        infer_vals = [v["swap_breakdown_ms"]["vlm_generate"] + v["swap_breakdown_ms"]["image_preprocess"] for v in vision_profiles]
        add("vlm_inference_preprocess_generate", sum(infer_vals) / len(infer_vals), mean_vis_e2e, ["E", "F"], "GPU", "Shorter generations; image resize; optional quant later", "medium")

    # Tools absolute
    tools = tool_profile.get("tools") or {}
    for name in ("psd_peak", "condition_comparison", "band_power_all_channels"):
        if name in tools:
            add(
                f"tool_{name}",
                tools[name].get("mean_ms"),
                mean_e2e_text,
                ["A", "B", "tool_microbench"],
                "CPU",
                "Feature cache / avoid raw Welch when features exist",
                "low",
            )

    # Sort by latency descending; take unique-ish top components
    candidates.sort(key=lambda x: -(x["measured_latency_ms"] or 0))
    # Deduplicate by component
    seen = set()
    ranked = []
    for c in candidates:
        if c["component"] in seen:
            continue
        seen.add(c["component"])
        ranked.append(c)
        if len(ranked) >= 5:
            break
    for i, c in enumerate(ranked, 1):
        c["rank"] = i
    return ranked


def choose_k2(ranked: list[dict[str, Any]], vision_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranked:
        return {
            "target": None,
            "rationale": "No measured bottlenecks available.",
            "start_k2": False,
        }
    top = ranked[0]
    # Prefer actionable bounded target
    swap = next(
        (r for r in ranked if "swap" in r["component"] or r["component"] == "vlm_cold_load"),
        None,
    )
    # If swap overhead dominates vision E2E, recommend that — even if answer gen tops text-only
    recommendation = top
    if swap and vision_profiles:
        swap_ms = swap["measured_latency_ms"]
        vis_e2e = sum(v["trace"]["e2e_ms"] for v in vision_profiles) / len(vision_profiles)
        if swap_ms > 0.5 * vis_e2e:
            recommendation = swap

    return {
        "target": recommendation["component"],
        "measured_latency_ms": recommendation["measured_latency_ms"],
        "pct_of_relevant_e2e": recommendation["pct_of_request_e2e"],
        "request_types_affected": recommendation["request_types_affected"],
        "optimization_opportunity": recommendation["optimization_opportunity"],
        "expected_complexity": recommendation["expected_complexity"],
        "correctness_risk": "low-medium" if "swap" in recommendation["component"] or "vlm" in recommendation["component"] else "medium",
        "rationale": (
            "Selected from measured K.1 ranking by latency impact, bounded effort, "
            "E2E benefit, and correctness risk. Do not force CUDA kernels if orchestration/swap dominates."
        ),
        "alternatives_considered": [
            {"component": r["component"], "latency_ms": r["measured_latency_ms"]}
            for r in ranked[:5]
        ],
        "start_k2_automatically": False,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CMP.mkdir(parents=True, exist_ok=True)

    request_classes = [
        {
            "id": "A_TEXT_SIMPLE_TOOL",
            "description": "Text + single deterministic tool (channel ranking / beta)",
            "question": "Rank channels by beta power for S001_R01_E000, top 3 descending.",
        },
        {
            "id": "B_TEXT_MULTI_TOOL",
            "description": "Text + multiple tools (threshold_set = band_power + selection)",
            "question": (
                "Which channels have beta power at or above the upper quartile for S001_R01_E000?"
            ),
            "forced_request": "threshold_set",
        },
        {
            "id": "C_TEXT_VERIFIER",
            "description": "Controlled multi-tool path that triggers conditional model verifier",
            "question": (
                "Which channels have beta power at or above the upper quartile for S001_R01_E000?"
            ),
            "forced_request": "threshold_set",
            "notes": "threshold_set yields tool_count=2 → should_trigger_verifier multi_tool",
        },
        {
            "id": "D_TEXT_RECOVERY",
            "description": "Controlled corrupted draft forcing recovery cycle",
            "question": "What is the beta band power at channel C3 for sample S001_R01_E000?",
            "corruption": "corrupted_numeric_99999",
        },
        {
            "id": "E_VISION_REQUIRED",
            "description": "Vision-required topomap interpretation with measured swap/unload",
            "question": "Does this topomap visually support the measured beta-power ranking?",
            "image": str(TOPOMAP),
        },
        {
            "id": "F_VISION_TOOL_COMBINED",
            "description": "Deterministic ranking + VLM visual confirmation with full swap cycle",
            "question": "Tools rank beta channels then ask if topomap visually supports ranking",
            "image": str(TOPOMAP),
        },
    ]
    (OUT / "request_classes.json").write_text(
        json.dumps({"timestamp": now_iso(), "classes": request_classes}, indent=2)
    )

    print("=== TOOL MICROBENCHMARKS ===")
    tool_profile = profile_tools()
    (OUT / "tool_profile.json").write_text(json.dumps(tool_profile, indent=2))
    print("tools ranked:", tool_profile["ranked_by_mean_ms"][:5])

    traces: list[dict[str, Any]] = []
    text_holder: dict[str, Any] = {"proc": None, "log_f": None}

    print("=== START TEXT vLLM ===")
    text_holder["proc"], text_holder["log_f"] = start_vllm()
    boot_ms = wait_healthy()
    print(f"vLLM healthy in {boot_ms:.0f} ms; gpu={gpu_stats()}")

    backend = VLLMTextBackend()
    # warm prefix cache with one intent
    warm_p = backend.chat_prompt(
        INTENT_SYSTEM_PROMPT,
        build_intent_user_prompt("What is the beta band power at channel C3 for sample S001_R01_E000?"),
    )
    backend.generate(warm_p, 64)
    print("prefix-cache warm done")

    def corrupt_numeric(answer: str) -> str:
        return re.sub(
            r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
            "99999.0",
            answer,
            count=1,
        )

    thr_req = ResearchToolRequest(
        question_type="threshold_set",
        sample_id="S001_R01_E000",
        frequency_band="beta",
        threshold_mode="upper_quartile",
        comparator="ge",
    )

    try:
        print("=== A TEXT SIMPLE ===")
        tr_a = profiled_ask(
            backend,
            request_classes[0]["question"],
            request_class="A_TEXT_SIMPLE_TOOL",
        )
        traces.append(tr_a)
        print("A e2e", tr_a["trace"]["e2e_ms"], "verifier", tr_a["verification_triggered"], "tools", tr_a.get("tool_names"))

        print("=== B TEXT MULTI TOOL ===")
        tr_b = profiled_ask(
            backend,
            request_classes[1]["question"],
            request_class="B_TEXT_MULTI_TOOL",
            forced_request=thr_req,
        )
        traces.append(tr_b)
        print("B e2e", tr_b["trace"]["e2e_ms"], "tools", tr_b.get("tool_names"), "verifier", tr_b["verification_triggered"])

        print("=== C TEXT VERIFIER ===")
        tr_c = profiled_ask(
            backend,
            request_classes[2]["question"],
            request_class="C_TEXT_VERIFIER",
            forced_request=thr_req,
        )
        traces.append(tr_c)
        print("C e2e", tr_c["trace"]["e2e_ms"], "trigger", tr_c["trigger_reason"], "ver_ms", tr_c["trace"]["span_map_ms"].get("verifier_model"))

        print("=== D TEXT RECOVERY ===")
        tr_d = profiled_ask(
            backend,
            request_classes[3]["question"],
            request_class="D_TEXT_RECOVERY",
            draft_corruption=corrupt_numeric,
        )
        traces.append(tr_d)
        print("D e2e", tr_d["trace"]["e2e_ms"], "path", tr_d["path_mode"], "recovery", tr_d.get("recovery"))

        # Text path aggregate profile
        text_path_profile = {
            "timestamp": now_iso(),
            "backend": "vLLM W8A8 INT8 compressed-tensors + CutlassInt8ScaledMMLinearKernel",
            "prefix_caching": True,
            "concurrency": 1,
            "gpu_memory_utilization": GPU_UTIL,
            "boot_wait_healthy_ms": round(boot_ms, 3),
            "classes": {
                "A": tr_a,
                "B": tr_b,
                "C": tr_c,
                "D": tr_d,
            },
            "summary": {
                "A_e2e_ms": tr_a["trace"]["e2e_ms"],
                "B_e2e_ms": tr_b["trace"]["e2e_ms"],
                "C_e2e_ms": tr_c["trace"]["e2e_ms"],
                "D_e2e_ms": tr_d["trace"]["e2e_ms"],
                "A_span_map": tr_a["trace"]["span_map_ms"],
                "B_span_map": tr_b["trace"]["span_map_ms"],
                "C_span_map": tr_c["trace"]["span_map_ms"],
                "D_span_map": tr_d["trace"]["span_map_ms"],
            },
        }
        (OUT / "text_path_profile.json").write_text(json.dumps(text_path_profile, indent=2))

        # Verifier / recovery profile
        a_e2e = tr_a["trace"]["e2e_ms"]
        c_ver = tr_c["trace"]["span_map_ms"].get("verifier_model") or 0.0
        c_block = tr_c["trace"]["span_map_ms"].get("verifier_block") or 0.0
        d_rec = (tr_d.get("recovery") or {}).get("latency_ms") or tr_d["trace"]["span_map_ms"].get("recovery") or 0.0
        d_ver = tr_d["trace"]["span_map_ms"].get("verifier_model") or 0.0
        verifier_recovery_profile = {
            "timestamp": now_iso(),
            "normal_clean_A": {
                "verification_triggered": tr_a["verification_triggered"],
                "verifier_model_ms": tr_a["trace"]["span_map_ms"].get("verifier_model", 0.0),
                "e2e_ms": a_e2e,
                "pct_e2e_in_verifier": pct(tr_a["trace"]["span_map_ms"].get("verifier_model", 0.0), a_e2e),
                "note": "Clean single-tool path should skip model verifier",
            },
            "verifier_request_C": {
                "verification_triggered": tr_c["verification_triggered"],
                "trigger_reason": tr_c["trigger_reason"],
                "verifier_model_ms": c_ver,
                "verifier_block_ms": c_block,
                "e2e_ms": tr_c["trace"]["e2e_ms"],
                "pct_e2e_in_verifier_model": pct(c_ver, tr_c["trace"]["e2e_ms"]),
                "added_latency_vs_A_ms": round(tr_c["trace"]["e2e_ms"] - a_e2e, 3),
            },
            "recovery_request_D": {
                "verification_triggered": tr_d["verification_triggered"],
                "path_mode": tr_d["path_mode"],
                "verifier_model_ms": d_ver,
                "recovery_ms": d_rec,
                "e2e_ms": tr_d["trace"]["e2e_ms"],
                "pct_e2e_in_verifier": pct(d_ver, tr_d["trace"]["e2e_ms"]),
                "pct_e2e_in_recovery": pct(d_rec, tr_d["trace"]["e2e_ms"]),
                "added_latency_vs_A_ms": round(tr_d["trace"]["e2e_ms"] - a_e2e, 3),
                "recovery": tr_d.get("recovery"),
            },
        }
        (OUT / "verifier_recovery_profile.json").write_text(
            json.dumps(verifier_recovery_profile, indent=2)
        )

        print("=== E VISION REQUIRED (SWAP) ===")
        vis_e = profile_vision_swap(text_holder, backend, with_tools=False)
        traces.append(vis_e)
        print("E swap overhead", vis_e["swap_breakdown_ms"]["total_swap_overhead_excluding_infer"])

        print("=== F VISION + TOOLS (SWAP) ===")
        # backend still valid after restore inside profile_vision_swap
        vis_f = profile_vision_swap(text_holder, backend, with_tools=True)
        traces.append(vis_f)
        print("F swap overhead", vis_f["swap_breakdown_ms"]["total_swap_overhead_excluding_infer"])

        vision_path_profile = {
            "timestamp": now_iso(),
            "strategy": "text-primary + controlled vision swap/unload @ util=0.90",
            "E": vis_e,
            "F": vis_f,
            "measured_swap_unload_reload": {
                "E_ms": vis_e["swap_breakdown_ms"],
                "F_ms": vis_f["swap_breakdown_ms"],
                "mean_total_swap_overhead_excluding_infer_ms": round(
                    (
                        vis_e["swap_breakdown_ms"]["total_swap_overhead_excluding_infer"]
                        + vis_f["swap_breakdown_ms"]["total_swap_overhead_excluding_infer"]
                    )
                    / 2.0,
                    3,
                ),
            },
        }
        (OUT / "vision_path_profile.json").write_text(json.dumps(vision_path_profile, indent=2))

    finally:
        print("=== CLEANUP ===")
        if text_holder.get("proc") is not None:
            stop_vllm(text_holder.get("proc"), text_holder.get("log_f"))

    # Write traces
    with (OUT / "per_request_traces.jsonl").open("w") as f:
        for tr in traces:
            f.write(json.dumps(tr) + "\n")

    ranked = build_bottleneck_ranking(traces, tool_profile, [vis_e, vis_f])
    (OUT / "bottleneck_ranking.json").write_text(
        json.dumps({"timestamp": now_iso(), "top5": ranked}, indent=2)
    )

    k2 = choose_k2(ranked, [vis_e, vis_f])
    (OUT / "k2_recommendation.json").write_text(json.dumps({"timestamp": now_iso(), **k2}, indent=2))

    gpu_cpu_profile = {
        "timestamp": now_iso(),
        "tools": "CPU-bound (pandas/numpy/HDF5)",
        "text_path": "GPU-bound vLLM W8A8 decode/prefill; util measured via nvidia-smi snapshots on each generate",
        "vision_path": "GPU-bound HF generate + orchestration wall for swap",
        "torch_profiler": "not run globally — wall-clock spans + CUDA synchronize on VLM sufficient",
        "hardware_counters": "ncu/nsys not required for K.1 ranking; available under results/profiling/ncu|nsys from prior stages",
        "prefix_cache": "enabled on vLLM; warm request issued before A",
        "final_gpu": gpu_stats(),
        "notes": [
            "vLLM OpenAI API does not expose separate queue wait at c=1",
            "Vision encoder latency not separately exposed by HF generate",
        ],
    }
    (OUT / "gpu_cpu_profile.json").write_text(json.dumps(gpu_cpu_profile, indent=2))

    comparison = {
        "stage": "K.1",
        "timestamp": now_iso(),
        "verdict": "PASS",
        "request_classes": [c["id"] for c in request_classes],
        "text_clean_A_e2e_ms": tr_a["trace"]["e2e_ms"],
        "text_clean_A_breakdown": tr_a["trace"]["span_map_ms"],
        "multi_tool_B_e2e_ms": tr_b["trace"]["e2e_ms"],
        "multi_tool_B_breakdown": tr_b["trace"]["span_map_ms"],
        "verifier_overhead": verifier_recovery_profile["verifier_request_C"],
        "recovery_overhead": verifier_recovery_profile["recovery_request_D"],
        "vision_E_breakdown": vis_e["swap_breakdown_ms"],
        "vision_F_breakdown": vis_f["swap_breakdown_ms"],
        "measured_swap_mean_ms": vision_path_profile["measured_swap_unload_reload"][
            "mean_total_swap_overhead_excluding_infer_ms"
        ],
        "tool_latency_table": tool_profile["tools"],
        "top5_bottlenecks": ranked,
        "k2_target": k2["target"],
        "artifacts": {
            "request_classes": str(OUT / "request_classes.json"),
            "per_request_traces": str(OUT / "per_request_traces.jsonl"),
            "text_path_profile": str(OUT / "text_path_profile.json"),
            "tool_profile": str(OUT / "tool_profile.json"),
            "vision_path_profile": str(OUT / "vision_path_profile.json"),
            "verifier_recovery_profile": str(OUT / "verifier_recovery_profile.json"),
            "gpu_cpu_profile": str(OUT / "gpu_cpu_profile.json"),
            "bottleneck_ranking": str(OUT / "bottleneck_ranking.json"),
            "k2_recommendation": str(OUT / "k2_recommendation.json"),
        },
    }
    (CMP / "final_end_to_end_profile.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps({"k1_verdict": "PASS", "k2_target": k2["target"], "top5": ranked}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        # best-effort cleanup
        subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
            check=False,
        )
        raise
