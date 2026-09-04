"""Typed analysis / vision result contracts for API responses.

Frontend maps these into independent explorer slots. An uploaded figure must
never appear as waveform/psd/spectrogram/band_power/comparison.
"""

from __future__ import annotations

from typing import Any


RESULT_TYPES = (
    "waveform",
    "spectrogram",
    "psd",
    "band_power",
    "topomap",
    "comparison",
    "vision_interpretation",
    "uploaded_figure",
    "generated_visualization",
)


def _slot(
    result_type: str,
    *,
    status: str = "idle",
    payload: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "type": result_type,
        "status": status,
        "payload": payload,
        "provenance": provenance or {},
        "error": error,
    }


def empty_analysis_results() -> dict[str, Any]:
    return {
        "waveform": _slot("waveform"),
        "spectrogram": _slot("spectrogram"),
        "psd": _slot("psd"),
        "band_power": _slot("band_power"),
        "topomap": _slot("topomap"),
        "comparison": _slot("comparison"),
        "vision_interpretation": _slot("vision_interpretation"),
    }


def build_analysis_results_payload(
    *,
    question: str,
    route: str,
    tools_used: list[str],
    computed_evidence: list[Any],
    visual_evidence: list[Any],
    answer: str,
    sample_id: str | None,
    experiment_id: str | None,
    task_plan: dict[str, Any] | None,
    vlm_text: str | None,
    image_id: str | None,
) -> dict[str, Any]:
    """Populate only the slots that this request actually produced."""
    out = empty_analysis_results()
    plan = task_plan or {}
    if plan.get("text_only") and not plan.get("use_tools") and not plan.get("use_vision"):
        return out

    base_prov = {
        "experiment_id": experiment_id,
        "sample_id": sample_id,
        "source": "analyze",
    }

    tool_blob = ",".join(tools_used).lower()
    if any(k in tool_blob for k in ("rank", "band_power", "band-power")) or any(
        "rank" in str(getattr(c, "label", c)).lower()
        or "beta" in str(getattr(c, "label", c)).lower()
        for c in computed_evidence
    ):
        rows = []
        for c in computed_evidence:
            if hasattr(c, "model_dump"):
                rows.append(c.model_dump(by_alias=True))
            elif isinstance(c, dict):
                rows.append(c)
            else:
                rows.append({"label": str(c), "value": ""})
        out["band_power"] = _slot(
            "band_power",
            status="ready",
            payload={"kind": "band_power_table", "rows": rows},
            provenance={**base_prov, "metric": "band_power", "tool": tools_used[0] if tools_used else None},
        )

    if "compar" in tool_blob:
        rows = []
        for c in computed_evidence:
            if hasattr(c, "model_dump"):
                rows.append(c.model_dump(by_alias=True))
            elif isinstance(c, dict):
                rows.append(c)
        out["comparison"] = _slot(
            "comparison",
            status="ready",
            payload={
                "kind": "comparison",
                "summary": (answer or "")[:400],
                "rows": rows,
            },
            provenance={
                **base_prov,
                "sample_id_a": sample_id,
                "sample_id_b": sample_id,
                "metric": "condition_comparison",
                "tool": tools_used[0] if tools_used else None,
            },
        )

    for ve in visual_evidence:
        data = ve.model_dump(by_alias=True) if hasattr(ve, "model_dump") else dict(ve)
        tab = str(data.get("tab") or "").lower()
        url = data.get("imageUrl") or data.get("image_url")
        if not url:
            continue
        payload = {
            "kind": "plot_image",
            "imageUrl": url,
            "title": data.get("label"),
            "visualizationId": data.get("id"),
        }
        prov = {
            **base_prov,
            "visualization_id": data.get("id"),
            "image_id": data.get("id"),
        }
        if tab == "psd":
            out["psd"] = _slot("psd", status="ready", payload=payload, provenance=prov)
        elif tab == "spectrogram":
            out["spectrogram"] = _slot(
                "spectrogram", status="ready", payload=payload, provenance=prov
            )
        elif tab == "topomap":
            out["topomap"] = _slot("topomap", status="ready", payload=payload, provenance=prov)
        elif tab == "waveform":
            out["waveform"] = _slot(
                "waveform",
                status="ready",
                payload={"kind": "static_plot", "imageUrl": url},
                provenance=prov,
            )
        elif tab == "comparison":
            out["comparison"] = _slot(
                "comparison",
                status="ready",
                payload={"kind": "comparison", "imageUrl": url, "summary": (answer or "")[:400]},
                provenance={
                    **prov,
                    "sample_id_a": sample_id,
                    "sample_id_b": sample_id,
                },
            )
        # figure / unknown → vision only

    if route == "VISION" or plan.get("use_vision"):
        out["vision_interpretation"] = _slot(
            "vision_interpretation",
            status="ready" if (vlm_text or answer) else "idle",
            payload={
                "kind": "vision",
                "imageId": image_id,
                "interpretation": vlm_text or answer,
            },
            provenance={**base_prov, "image_id": image_id},
        )

    return out
