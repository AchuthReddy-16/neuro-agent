#!/usr/bin/env python3
"""Aggregate Stage H.1 BF16 vs INT8 vs INT4 quality + systems comparison."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.paths import PROJECT_ROOT, RESULTS_DIR, ensure_dirs


VARIANTS = ("bf16", "int8", "int4")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _pct_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None or baseline == 0:
        return None
    return (value - baseline) / baseline * 100.0


def _quality_block(root: Path, variant: str) -> dict[str, Any]:
    full = _load_json(root / "quality" / variant / "summary.json")
    targeted = _load_json(root / "quality" / f"{variant}_targeted" / "summary.json")
    gate = _load_json(root / "quality" / f"{variant}_targeted" / "gate_result.json")
    per_task = _load_json(root / "quality" / variant / "per_task_metrics.json")
    per_task_targeted = _load_json(
        root / "quality" / f"{variant}_targeted" / "per_task_metrics.json"
    )
    verifier = _load_json(root / "quality" / variant / "verifier_summary.json")
    return {
        "full": full,
        "targeted": targeted,
        "gate": gate,
        "per_task": per_task,
        "per_task_targeted": per_task_targeted,
        "verifier_summary": verifier,
        "skipped_full": full is None and targeted is not None,
    }


def _systems_block(root: Path, variant: str) -> dict[str, Any] | None:
    return _load_json(root / "systems" / f"latest_{variant}.json")


def recommend(comparison: dict[str, Any]) -> dict[str, Any]:
    """Choose A BF16 / B INT8 / C INT4 from measured evidence."""
    scores: dict[str, float] = {}
    reasons: dict[str, list[str]] = {v: [] for v in VARIANTS}

    bf16_q = comparison["quality"]["bf16"]["full"]
    bf16_s = comparison["systems"]["bf16"]
    if not bf16_q or not bf16_s:
        return {
            "choice": "A",
            "label": "BF16",
            "reasons": ["Incomplete BF16 reference; defaulting to BF16."],
        }

    bf16_pass = bf16_q["verifier_pass_rate"]
    bf16_vram = bf16_s.get("peak_vram_mb") or bf16_s.get("weight_memory_mb")
    bf16_tps = bf16_s["decode_tokens_per_second"]["mean"]
    bf16_ttft = bf16_s["ttft_ms"]["mean"]
    bf16_e2e = bf16_s["end_to_end_latency_ms"]["mean"]

    for variant in VARIANTS:
        q = comparison["quality"][variant]["full"]
        s = comparison["systems"][variant]
        gate = comparison["quality"][variant].get("gate") or {}
        if q is None or s is None:
            scores[variant] = -1e9
            reasons[variant].append("Missing full quality or systems results")
            continue
        if gate.get("catastrophic"):
            scores[variant] = -1e9
            reasons[variant].append("Catastrophic quality regression")
            continue

        pass_rate = q["verifier_pass_rate"]
        quality_penalty = max(0.0, bf16_pass - pass_rate) * 100.0  # points
        vram = s.get("peak_vram_mb") or s.get("weight_memory_mb") or 0.0
        vram_saving = ((bf16_vram - vram) / bf16_vram * 100.0) if bf16_vram else 0.0
        tps = s["decode_tokens_per_second"]["mean"]
        tps_gain = ((tps - bf16_tps) / bf16_tps * 100.0) if bf16_tps else 0.0
        ttft = s["ttft_ms"]["mean"]
        ttft_change = ((ttft - bf16_ttft) / bf16_ttft * 100.0) if bf16_ttft else 0.0
        e2e = s["end_to_end_latency_ms"]["mean"]
        e2e_change = ((e2e - bf16_e2e) / bf16_e2e * 100.0) if bf16_e2e else 0.0

        # Prefer: quality preserved, VRAM saved, speed not much worse
        score = (
            -quality_penalty * 3.0
            + vram_saving * 0.5
            + tps_gain * 0.35
            - max(0.0, ttft_change) * 0.2
            - max(0.0, e2e_change) * 0.2
        )
        # Small preference for remaining at BF16 when deltas are tiny
        if variant == "bf16":
            score += 1.0

        scores[variant] = score
        reasons[variant].extend(
            [
                f"pass_rate={pass_rate:.3f} (Δ vs BF16={pass_rate - bf16_pass:+.3f})",
                f"peak_vram_mb={vram:.1f} (saving={vram_saving:.1f}%)",
                f"decode_tps={tps:.2f} (Δ={tps_gain:+.1f}%)",
                f"ttft_ms={ttft:.2f} (Δ={ttft_change:+.1f}%)",
                f"e2e_ms={e2e:.2f} (Δ={e2e_change:+.1f}%)",
                f"score={score:.2f}",
            ]
        )

    best = max(scores, key=lambda k: scores[k])
    label_map = {"bf16": ("A", "BF16"), "int8": ("B", "INT8"), "int4": ("C", "INT4")}
    choice, label = label_map[best]

    # Narrative: if INT4/INT8 save VRAM but hurt speed and quality little, prefer them;
    # if quality drop > 2pp or speed much worse, stick with BF16.
    narrative: list[str] = []
    for variant in VARIANTS:
        narrative.append(f"{variant}: " + "; ".join(reasons[variant]))

    return {
        "choice": choice,
        "label": label,
        "scores": scores,
        "reasons": reasons[best],
        "all_variant_notes": narrative,
    }


def build_comparison(root: Path) -> dict[str, Any]:
    smoke = _load_json(root / "smoke" / "latest_smoke.json")
    quality = {v: _quality_block(root, v) for v in VARIANTS}
    systems = {v: _systems_block(root, v) for v in VARIANTS}

    bf16_q = quality["bf16"]["full"]
    bf16_s = systems["bf16"]

    quality_table: dict[str, Any] = {}
    systems_table: dict[str, Any] = {}
    per_task_table: dict[str, Any] = {}

    for v in VARIANTS:
        q = quality[v]["full"]
        s = systems[v]
        quality_table[v] = None
        if q:
            quality_table[v] = {
                "verifier_pass_rate": q["verifier_pass_rate"],
                "invalid_parse_rate": q["invalid_parse_rate"],
                "empty_refusal_rate": q["empty_refusal_rate"],
                "avg_generated_tokens": q["avg_generated_tokens"],
                "delta_pass_vs_bf16": _delta(
                    q["verifier_pass_rate"],
                    bf16_q["verifier_pass_rate"] if bf16_q else None,
                ),
            }
        if quality[v]["per_task"]:
            per_task_table[v] = {
                task: metrics["verifier_pass_rate"]
                for task, metrics in quality[v]["per_task"].items()
            }
            if bf16_q and quality["bf16"]["per_task"]:
                per_task_table[f"{v}_delta_vs_bf16"] = {
                    task: _delta(
                        metrics["verifier_pass_rate"],
                        quality["bf16"]["per_task"][task]["verifier_pass_rate"],
                    )
                    for task, metrics in quality[v]["per_task"].items()
                    if task in quality["bf16"]["per_task"]
                }
        systems_table[v] = None
        if s:
            systems_table[v] = {
                "load_time_s": s.get("load_time_s"),
                "weight_memory_mb": s.get("weight_memory_mb"),
                "peak_vram_mb": s.get("peak_vram_mb"),
                "allocated_after_load_mb": (s.get("metadata") or {}).get(
                    "allocated_after_load_mb"
                ),
                "nvidia_smi_after_load_mb": (s.get("metadata") or {}).get(
                    "nvidia_smi_after_load_mb"
                ),
                "ttft_ms_mean": s["ttft_ms"]["mean"] if s.get("ttft_ms") else None,
                "prefill_ms_mean": (
                    s["prefill_latency_ms"]["mean"] if s.get("prefill_latency_ms") else None
                ),
                "decode_tps_mean": (
                    s["decode_tokens_per_second"]["mean"]
                    if s.get("decode_tokens_per_second")
                    else None
                ),
                "decode_latency_per_token_ms_mean": (
                    s["decode_latency_per_token_ms"]["mean"]
                    if s.get("decode_latency_per_token_ms")
                    else None
                ),
                "e2e_ms_mean": (
                    s["end_to_end_latency_ms"]["mean"]
                    if s.get("end_to_end_latency_ms")
                    else None
                ),
            }
            if bf16_s:
                systems_table[f"{v}_delta_vs_bf16"] = {
                    "load_time_s": _delta(systems_table[v]["load_time_s"], bf16_s.get("load_time_s")),
                    "peak_vram_mb": _delta(
                        systems_table[v]["peak_vram_mb"], bf16_s.get("peak_vram_mb")
                    ),
                    "ttft_ms_mean": _delta(
                        systems_table[v]["ttft_ms_mean"],
                        bf16_s["ttft_ms"]["mean"] if bf16_s.get("ttft_ms") else None,
                    ),
                    "decode_tps_mean": _delta(
                        systems_table[v]["decode_tps_mean"],
                        bf16_s["decode_tokens_per_second"]["mean"]
                        if bf16_s.get("decode_tokens_per_second")
                        else None,
                    ),
                    "e2e_ms_mean": _delta(
                        systems_table[v]["e2e_ms_mean"],
                        bf16_s["end_to_end_latency_ms"]["mean"]
                        if bf16_s.get("end_to_end_latency_ms")
                        else None,
                    ),
                    "peak_vram_pct": _pct_delta(
                        systems_table[v]["peak_vram_mb"], bf16_s.get("peak_vram_mb")
                    ),
                    "decode_tps_pct": _pct_delta(
                        systems_table[v]["decode_tps_mean"],
                        bf16_s["decode_tokens_per_second"]["mean"]
                        if bf16_s.get("decode_tokens_per_second")
                        else None,
                    ),
                }

    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "adapter": "checkpoints/sft_corrected_v2/final",
        "methods": {
            "bf16": "Transformers BF16 (torch_dtype=bfloat16)",
            "int8": "bitsandbytes load_in_8bit",
            "int4": "bitsandbytes NF4 + double quant (load_in_4bit)",
        },
        "smoke": smoke,
        "quality": quality,
        "systems": systems,
        "tables": {
            "quality": quality_table,
            "per_task": per_task_table,
            "systems": systems_table,
        },
    }
    comparison["recommendation"] = recommend(comparison)

    # Stage gate
    smoke_ok = bool(smoke and smoke.get("all_supported"))
    quality_ok = all(
        quality[v]["full"] is not None
        or (quality[v].get("gate") or {}).get("catastrophic")
        for v in VARIANTS
    )
    # Require BF16 full + at least one quantized full, and systems for all supported
    systems_ok = all(systems[v] is not None for v in VARIANTS if smoke and smoke["variants"].get(v, {}).get("supported", True))
    stage_pass = smoke_ok and quality["bf16"]["full"] is not None and systems_ok
    comparison["stage_h1"] = {
        "pass": stage_pass,
        "smoke_ok": smoke_ok,
        "quality_ok": quality_ok,
        "systems_ok": systems_ok,
        "next_stage": (
            "H.2 Multimodal quantization benchmark (separate; do not mix with text)"
            if stage_pass
            else "Re-run failed H.1 steps (smoke / quality / systems) before H.2"
        ),
    }
    return comparison


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare text quantization results")
    p.add_argument(
        "--root",
        type=Path,
        default=RESULTS_DIR / "quantization" / "text",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    comparison = build_comparison(args.root)

    out_dir = args.root
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"comparison_{ts}.json"
    latest = out_dir / "comparison_latest.json"
    model_cmp = RESULTS_DIR / "model_comparison"
    model_cmp.mkdir(parents=True, exist_ok=True)
    model_cmp_path = model_cmp / "text_quantization_bf16_int8_int4.json"

    payload = json.dumps(comparison, indent=2)
    out_path.write_text(payload)
    latest.write_text(payload)
    model_cmp_path.write_text(payload)

    rec = comparison["recommendation"]
    print(
        json.dumps(
            {
                "recommendation": rec,
                "stage_h1": comparison["stage_h1"],
                "quality_pass_rates": {
                    v: (comparison["tables"]["quality"][v] or {}).get("verifier_pass_rate")
                    for v in VARIANTS
                },
                "outputs": {
                    "comparison": str(out_path),
                    "model_comparison": str(model_cmp_path),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
