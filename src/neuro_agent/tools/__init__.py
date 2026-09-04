"""Deterministic neuroscience tool layer."""

from neuro_agent.tools.comparison import compare_conditions, lookup_condition_summary
from neuro_agent.tools.eeg_signal import compute_band_power, compute_rms, find_psd_peak
from neuro_agent.tools.evidence import EvidenceBundle, ResearchToolRequest, new_request_id
from neuro_agent.tools.metadata import lookup_sample_metadata
from neuro_agent.tools.ranking import (
    THRESHOLD_POLICIES,
    rank_channels,
    rank_channels_for_sample,
    select_channels_above_threshold,
    select_channels_for_multimodal_source_values,
    select_channels_for_rlvr_context,
)
from neuro_agent.tools.router import route_research_request
from neuro_agent.tools.schemas import (
    BAND_DEFINITIONS,
    BandPowerOutput,
    CompareConditionsOutput,
    PsdPeakResult,
    RankChannelsOutput,
    RmsOutput,
    ThresholdSelectionOutput,
)
from neuro_agent.tools.vision_evidence import resolve_vision_evidence

__all__ = [
    "BAND_DEFINITIONS",
    "THRESHOLD_POLICIES",
    "BandPowerOutput",
    "CompareConditionsOutput",
    "EvidenceBundle",
    "PsdPeakResult",
    "RankChannelsOutput",
    "ResearchToolRequest",
    "RmsOutput",
    "ThresholdSelectionOutput",
    "compare_conditions",
    "compute_band_power",
    "compute_rms",
    "find_psd_peak",
    "lookup_condition_summary",
    "lookup_sample_metadata",
    "new_request_id",
    "rank_channels",
    "rank_channels_for_sample",
    "resolve_vision_evidence",
    "route_research_request",
    "select_channels_above_threshold",
    "select_channels_for_multimodal_source_values",
    "select_channels_for_rlvr_context",
]
