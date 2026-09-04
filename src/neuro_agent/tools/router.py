"""Deterministic research tool router and evidence assembly."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Callable

from neuro_agent.tools.comparison import compare_conditions
from neuro_agent.tools.eeg_signal import compute_band_power, compute_rms, find_psd_peak
from neuro_agent.tools.evidence import (
    BAND_POWER_RECOMPUTE_NOTE,
    EvidenceBundle,
    ProvenanceRecord,
    ResearchToolRequest,
    ToolInvocation,
    new_request_id,
)
from neuro_agent.tools.metadata import lookup_sample_metadata
from neuro_agent.tools.ranking import rank_channels_for_sample, select_channels_above_threshold
from neuro_agent.tools.schemas import (
    BandPowerOutput,
    ChannelNotFoundError,
    CompareConditionsOutput,
    PsdPeakResult,
    RankChannelsOutput,
    RmsOutput,
    SampleNotFoundError,
    ThresholdSelectionOutput,
    ToolError,
    amplitude_units,
    frequency_units,
    power_units,
    resolve_metric_column,
)
from neuro_agent.tools.vision_evidence import resolve_vision_evidence

ROUTER_VERSION = "g.2.v1"

QUESTION_TYPE_ALIASES: dict[str, str] = {
    "band_power": "band_power",
    "rms": "rms",
    "psd_peak": "psd_peak",
    "channel_ranking": "channel_ranking",
    "threshold_set": "threshold_set",
    "condition_comparison": "condition_comparison",
}


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _band_power_provenance(output: BandPowerOutput) -> list[ProvenanceRecord]:
    records: list[ProvenanceRecord] = []
    for result in output.results:
        if output.source == "features":
            records.append(
                ProvenanceRecord(
                    field=f"band_power.{result.channel}",
                    source="stored_feature",
                    detail="features.parquet band column lookup",
                    sample_id=output.sample_id,
                    artifact_path="data/processed/features.parquet",
                    method=result.method,
                )
            )
        else:
            records.append(
                ProvenanceRecord(
                    field=f"band_power.{result.channel}",
                    source="welch_psd",
                    detail="Welch PSD trapezoid integration from raw epoch",
                    sample_id=output.sample_id,
                    method=result.method,
                    note=BAND_POWER_RECOMPUTE_NOTE,
                )
            )
    return records


def _rms_provenance(output: RmsOutput) -> list[ProvenanceRecord]:
    source = "stored_feature" if output.method == "features_parquet_lookup" else "raw_eeg"
    detail = (
        "features.parquet rms column"
        if source == "stored_feature"
        else "sqrt(mean(x^2)) on epoch waveform"
    )
    return [
        ProvenanceRecord(
            field=f"rms.{item.channel}",
            source=source,
            detail=detail,
            sample_id=output.sample_id,
            artifact_path="data/processed/features.parquet" if source == "stored_feature" else None,
            method=output.method,
        )
        for item in output.results
    ]


def _psd_provenance(result: PsdPeakResult, sample_id: str) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            field="peak_frequency_hz",
            source="welch_psd",
            detail="Argmax of group-mean Welch PSD in search band",
            sample_id=sample_id,
            method=result.method,
        )
    ]


def _ranking_provenance(
    output: RankChannelsOutput,
    *,
    sample_id: str,
    metric: str,
) -> list[ProvenanceRecord]:
    column = resolve_metric_column(metric)
    source: str = "stored_feature" if column.endswith("_power") or column == "rms" else "metadata"
    return [
        ProvenanceRecord(
            field="channel_ranking",
            source=source,  # type: ignore[arg-type]
            detail=f"Ranked by {column} from features.parquet",
            sample_id=sample_id,
            artifact_path="data/processed/features.parquet",
            method="rank_channels_descending_tiebreak_channel_name",
        )
    ]


def _threshold_provenance(
    output: ThresholdSelectionOutput,
    *,
    sample_id: str | None,
    metric: str,
) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            field="threshold_set",
            source="stored_feature",
            detail=f"Threshold policy {output.policy} on {metric}",
            sample_id=sample_id,
            artifact_path="data/processed/features.parquet",
            method=output.policy,
        )
    ]


def _condition_provenance(output: CompareConditionsOutput) -> list[ProvenanceRecord]:
    return [
        ProvenanceRecord(
            field="condition_comparison",
            source="comparison",
            detail="Mean across epochs per channel from features.parquet",
            sample_id=None,
            artifact_path="data/processed/features.parquet",
            method=output.aggregation,
        )
    ]


def _invoke(
    name: str,
    func: Callable[..., Any],
    inputs: dict[str, Any],
    provenance_builder: Callable[[Any], list[ProvenanceRecord]] | None = None,
) -> tuple[ToolInvocation, Any, list[ProvenanceRecord]]:
    start = time.perf_counter()
    result = func(**inputs)
    provenance = provenance_builder(result) if provenance_builder else []
    invocation = ToolInvocation(
        name=name,
        inputs=inputs,
        outputs=_serialize_output(result),
        runtime_ms=_elapsed_ms(start),
        provenance=provenance,
        success=True,
    )
    return invocation, result, provenance


def _serialize_output(output: Any) -> dict[str, Any]:
    if hasattr(output, "__dataclass_fields__"):
        return asdict(output)
    if isinstance(output, dict):
        return output
    return {"value": output}


def _resolve_sample_id(request: ResearchToolRequest) -> str:
    if request.sample_id:
        return request.sample_id
    meta = lookup_sample_metadata(
        subject_id=request.subject_id,
        run_id=request.run_id,
        epoch=request.epoch,
    )
    return meta.sample_id


def _resolve_subject_id(request: ResearchToolRequest) -> str:
    if request.subject_id:
        return request.subject_id
    if request.sample_id:
        return lookup_sample_metadata(sample_id=request.sample_id).subject_id
    raise ValueError("subject_id or sample_id required for condition comparison")


def _metric_from_request(request: ResearchToolRequest) -> str:
    if request.metric:
        return request.metric
    if request.frequency_band:
        return f"{request.frequency_band}_power"
    return "beta_power"


def _channels_from_request(request: ResearchToolRequest, default: list[str] | str = "all") -> list[str] | str:
    if request.channels is not None:
        return request.channels
    return default


def _route_band_power(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    sample_id = _resolve_sample_id(request)
    band = request.frequency_band or "beta"
    channels = _channels_from_request(request)

    invocation, output, provenance = _invoke(
        "compute_band_power",
        compute_band_power,
        {
            "sample_id": sample_id,
            "band": band,
            "channels": channels,
            "source": "features",
        },
        _band_power_provenance,
    )
    bundle.tool_invocations.append(invocation)
    bundle.provenance.extend(provenance)
    bundle.units = power_units()

    values = {item.channel: item.power for item in output.results}
    if len(values) == 1:
        channel = next(iter(values))
        bundle.numeric_evidence = {
            "channel": channel,
            "band": band,
            "value": values[channel],
        }
    else:
        bundle.numeric_evidence = {"band": band, "values": values}

    bundle.warnings.append(BAND_POWER_RECOMPUTE_NOTE)


def _route_rms(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    sample_id = _resolve_sample_id(request)
    channels = _channels_from_request(request, default=["C3"])

    invocation, output, provenance = _invoke(
        "compute_rms",
        compute_rms,
        {
            "sample_id": sample_id,
            "channels": channels,
            "source": "epoch",
        },
        _rms_provenance,
    )
    bundle.tool_invocations.append(invocation)
    bundle.provenance.extend(provenance)
    bundle.units = amplitude_units()

    values = {item.channel: item.rms for item in output.results}
    if len(values) == 1:
        channel = next(iter(values))
        bundle.numeric_evidence = {"channel": channel, "value": values[channel]}
    else:
        bundle.numeric_evidence = {"values": values}
    if output.highest_rms_channel:
        bundle.numeric_evidence["highest_rms_channel"] = output.highest_rms_channel


def _route_psd_peak(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    sample_id = _resolve_sample_id(request)
    kwargs: dict[str, Any] = {"sample_id": sample_id}
    if request.frequency_range:
        kwargs["freq_range_hz"] = request.frequency_range
    if request.channels and request.channels != "all":
        ch_list = request.channels if isinstance(request.channels, list) else [request.channels]
        kwargs["channel"] = ch_list[0]

    invocation, result, provenance = _invoke(
        "find_psd_peak",
        find_psd_peak,
        kwargs,
        lambda r: _psd_provenance(r, sample_id),
    )
    bundle.tool_invocations.append(invocation)
    bundle.provenance.extend(provenance)
    bundle.units = frequency_units()
    bundle.numeric_evidence = {
        "peak_frequency_hz": result.peak_frequency_hz,
        "channel": result.channel,
        "search_range_hz": result.search_range_hz,
        "psd_peaks_per_channel": result.psd_peaks_per_channel,
    }


def _route_channel_ranking(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    sample_id = _resolve_sample_id(request)
    metric = _metric_from_request(request)
    channels = _channels_from_request(request)

    invocation, output, provenance = _invoke(
        "rank_channels_for_sample",
        rank_channels_for_sample,
        {
            "sample_id": sample_id,
            "metric": metric,
            "channels": channels,
            "order": request.sort_direction,
            "top_k": request.top_k,
        },
        lambda r: _ranking_provenance(r, sample_id=sample_id, metric=metric),
    )
    bundle.tool_invocations.append(invocation)
    bundle.provenance.extend(provenance)
    bundle.units = output.units
    bundle.ranked_evidence = {
        "ranking": output.ranking,
        "values": output.values,
        "order": output.order,
        "top_k": output.top_k,
        "metric": metric,
    }
    bundle.numeric_evidence = {"top_channel": output.ranking[0] if output.ranking else None}


def _route_threshold_set(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    sample_id = _resolve_sample_id(request)
    metric = _metric_from_request(request)
    channels = _channels_from_request(request)

    band_output = compute_band_power(
        sample_id,
        request.frequency_band or "beta",
        channels=channels,
        source="features",
    )
    values = {item.channel: item.power for item in band_output.results}
    bundle.provenance.extend(_band_power_provenance(band_output))

    threshold_mode = request.threshold_mode
    comparator = request.comparator
    if threshold_mode == "upper_quartile" and comparator is None:
        comparator = "ge"
    if threshold_mode == "median" and comparator is None:
        comparator = "gt"

    start = time.perf_counter()
    threshold_output = select_channels_above_threshold(
        values,
        threshold=request.threshold,
        comparator=comparator,
        threshold_mode=threshold_mode,
        units=power_units(),
    )
    threshold_invocation = ToolInvocation(
        name="select_channels_above_threshold",
        inputs={
            "values": values,
            "threshold": request.threshold,
            "comparator": comparator,
            "threshold_mode": threshold_mode,
        },
        outputs=_serialize_output(threshold_output),
        runtime_ms=_elapsed_ms(start),
        provenance=_threshold_provenance(threshold_output, sample_id=sample_id, metric=metric),
        success=True,
    )
    bundle.tool_invocations.extend(
        [
            ToolInvocation(
                name="compute_band_power",
                inputs={
                    "sample_id": sample_id,
                    "band": request.frequency_band or "beta",
                    "channels": channels,
                    "source": "features",
                },
                outputs=_serialize_output(band_output),
                runtime_ms=0.0,
                provenance=_band_power_provenance(band_output),
                success=True,
            ),
            threshold_invocation,
        ]
    )
    bundle.provenance.extend(threshold_invocation.provenance)
    bundle.units = power_units()
    bundle.set_evidence = {
        "channels": threshold_output.channels,
        "threshold_used": threshold_output.threshold_used,
        "comparator": threshold_output.comparator,
        "threshold_mode": threshold_output.threshold_mode,
        "n_selected": threshold_output.n_selected,
        "metric": metric,
        "values": values,
    }
    bundle.numeric_evidence = {
        "n_selected": threshold_output.n_selected,
        "threshold_used": threshold_output.threshold_used,
    }
    bundle.warnings.append(BAND_POWER_RECOMPUTE_NOTE)


def _route_condition_comparison(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    subject_id = _resolve_subject_id(request)
    if not request.condition_a or not request.condition_b:
        raise ValueError("condition_a and condition_b are required for condition_comparison")

    metric = request.metric or "alpha_mu_power"
    invocation, output, provenance = _invoke(
        "compare_conditions",
        compare_conditions,
        {
            "subject_id": subject_id,
            "condition_a": request.condition_a,
            "condition_b": request.condition_b,
            "metric": metric,
        },
        _condition_provenance,
    )
    bundle.tool_invocations.append(invocation)
    bundle.provenance.extend(provenance)
    bundle.units = output.units
    bundle.condition_evidence = _serialize_output(output)
    bundle.numeric_evidence = {
        "mean_a": output.mean_a,
        "mean_b": output.mean_b,
        "higher_condition": output.higher_condition,
        "largest_absolute_difference_channel": output.largest_absolute_difference_channel,
        "signed_difference": output.signed_difference,
    }


def _attach_metadata(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    if request.question_type == "condition_comparison":
        subject_id = _resolve_subject_id(request)
        bundle.metadata = {
            "subject_id": subject_id,
            "router_version": ROUTER_VERSION,
        }
        return

    try:
        meta = lookup_sample_metadata(
            sample_id=request.sample_id,
            subject_id=request.subject_id,
            run_id=request.run_id,
            epoch=request.epoch,
        )
        bundle.metadata = meta.to_dict()
        bundle.metadata["router_version"] = ROUTER_VERSION
    except (SampleNotFoundError, ValueError) as exc:
        bundle.warnings.append(f"metadata lookup skipped: {exc}")


def _attach_vision_evidence(request: ResearchToolRequest, bundle: EvidenceBundle) -> None:
    if not request.include_vision_evidence and not request.image_id and not request.requested_visual_type:
        return

    try:
        sample_id = request.sample_id
        if sample_id is None and request.question_type != "condition_comparison":
            sample_id = _resolve_sample_id(request)
        refs = resolve_vision_evidence(
            sample_id=sample_id,
            image_id=request.image_id,
            visual_type=request.requested_visual_type,
            subject_id=request.subject_id,
            run_id=request.run_id,
            epoch=request.epoch,
        )
        bundle.vision_evidence = [ref.to_dict() for ref in refs]
        for ref in refs:
            bundle.provenance.append(
                ProvenanceRecord(
                    field="vision_evidence",
                    source="vision_metadata",
                    detail=f"Resolved {ref.family} image sidecar values",
                    sample_id=ref.source_sample_id,
                    artifact_path=ref.image_path,
                    method="images.jsonl_lookup",
                )
            )
    except (SampleNotFoundError, ValueError) as exc:
        bundle.warnings.append(f"vision evidence unavailable: {exc}")


_ROUTE_HANDLERS: dict[str, Callable[[ResearchToolRequest, EvidenceBundle], None]] = {
    "band_power": _route_band_power,
    "rms": _route_rms,
    "psd_peak": _route_psd_peak,
    "channel_ranking": _route_channel_ranking,
    "threshold_set": _route_threshold_set,
    "condition_comparison": _route_condition_comparison,
}


def route_research_request(
    request: ResearchToolRequest,
    *,
    request_id: str | None = None,
) -> EvidenceBundle:
    """Route a structured research request to tool chains and assemble evidence."""
    qtype = QUESTION_TYPE_ALIASES.get(request.question_type, request.question_type)
    if qtype not in _ROUTE_HANDLERS:
        return EvidenceBundle(
            request_id=request_id or new_request_id(),
            question_type=request.question_type,
            success=False,
            error=f"Unsupported question_type: {request.question_type!r}",
        )

    bundle = EvidenceBundle(
        request_id=request_id or new_request_id(),
        question_type=qtype,  # type: ignore[arg-type]
    )

    try:
        _attach_metadata(request, bundle)
        _ROUTE_HANDLERS[qtype](request, bundle)
        _attach_vision_evidence(request, bundle)
    except (ToolError, ValueError) as exc:
        bundle.success = False
        bundle.error = str(exc)
        bundle.tool_invocations.append(
            ToolInvocation(
                name=f"route_{qtype}",
                inputs=request.to_dict(),
                outputs={},
                runtime_ms=0.0,
                success=False,
                error=str(exc),
            )
        )
    except Exception as exc:
        bundle.success = False
        bundle.error = f"Unexpected error: {exc}"
        bundle.tool_invocations.append(
            ToolInvocation(
                name=f"route_{qtype}",
                inputs=request.to_dict(),
                outputs={},
                runtime_ms=0.0,
                success=False,
                error=str(exc),
            )
        )

    return bundle
