"""Pydantic request/response schemas for the FastAPI backend.

Contract notes (see docs/api_contract.md):
- AnalyzeResponse evidence fields use snake_case (matches web AnalyzeResponse).
- Nested timing / system / verification / timeline / metrics / upload ids use
  camelCase aliases so the Next.js workstation can consume them directly.
- AnalyzeRequest accepts both experiment_id and experimentId.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, AliasChoices


class APIErrorBody(BaseModel):
    error: str
    detail: str | None = None
    code: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: Literal["ok", "degraded", "unavailable"]
    version: str = "0.1.0"
    backend: str = "fastapi"
    text_model: str | None = Field(default=None, serialization_alias="textModel")
    vision_model: str | None = Field(default=None, serialization_alias="visionModel")
    serving_mode: str | None = Field(default=None, serialization_alias="servingMode")
    agent_loaded: bool = Field(default=False, serialization_alias="agentLoaded")
    vision_loaded: bool = Field(default=False, serialization_alias="visionLoaded")
    # Process is up vs model readiness (distinct from HTTP liveness).
    text_status: Literal["disabled", "unloaded", "loading", "ready", "error"] = Field(
        default="unloaded", serialization_alias="textStatus"
    )
    vision_status: Literal[
        "disabled", "unloaded", "loading", "ready", "error"
    ] = Field(default="disabled", serialization_alias="visionStatus")
    vision_enabled: bool = Field(default=False, serialization_alias="visionEnabled")
    text_backend: str | None = Field(default=None, serialization_alias="textBackend")
    vision_backend: str | None = Field(default=None, serialization_alias="visionBackend")
    precision: str | None = None
    text_error: str | None = Field(default=None, serialization_alias="textError")
    vision_error: str | None = Field(default=None, serialization_alias="visionError")
    # Release identification (additive; null when unavailable — never invent).
    git_commit: str | None = Field(default=None, serialization_alias="gitCommit")
    text_checkpoint: str | None = Field(
        default=None, serialization_alias="textCheckpoint"
    )
    vision_checkpoint: str | None = Field(
        default=None, serialization_alias="visionCheckpoint"
    )
    runtime: str | None = None
    package_version: str | None = Field(
        default=None, serialization_alias="packageVersion"
    )
    frontend_build_id: str | None = Field(
        default=None, serialization_alias="frontendBuildId"
    )


class SystemMetricsResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    model: str
    vision_model: str = Field(serialization_alias="visionModel")
    post_training: str = Field(serialization_alias="postTraining")
    serving: str
    precision: str = "BF16"
    # Live / last-request fields (null when not yet measured)
    ttft_ms: float | None = Field(default=None, serialization_alias="ttftMs")
    tokens_per_sec: float | None = Field(default=None, serialization_alias="tokensPerSec")
    p95_latency_ms: float | None = Field(default=None, serialization_alias="p95LatencyMs")
    gpu_utilization_pct: float | None = Field(
        default=None, serialization_alias="gpuUtilizationPct"
    )
    gpu_memory_used_mb: float | None = Field(
        default=None, serialization_alias="gpuMemoryUsedMb"
    )
    gpu_memory_total_mb: float | None = Field(
        default=None, serialization_alias="gpuMemoryTotalMb"
    )
    last_request_latency_ms: float | None = Field(
        default=None, serialization_alias="lastRequestLatencyMs"
    )
    route: Literal["TEXT", "VISION"] | None = None
    verifier_status: str | None = Field(default=None, serialization_alias="verifierStatus")
    serving_mode: str | None = Field(default=None, serialization_alias="servingMode")


class EEGMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    filename: str | None = None
    format: Literal["edf", "csv", "npy", "hdf5", "json"] | None = None
    sampling_rate_hz: float | None = Field(
        default=None,
        validation_alias=AliasChoices("sampling_rate_hz", "samplingRateHz"),
        serialization_alias="samplingRateHz",
    )
    channels: int | None = None
    channel_labels: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("channel_labels", "channelLabels"),
        serialization_alias="channelLabels",
    )
    duration_sec: float | None = Field(
        default=None,
        validation_alias=AliasChoices("duration_sec", "durationSec"),
        serialization_alias="durationSec",
    )
    sample_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sample_id", "sampleId"),
        serialization_alias="sampleId",
    )
    auto_detected: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("auto_detected", "autoDetected"),
        serialization_alias="autoDetected",
    )


class ExperimentMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str | None = None
    run: str | None = None
    task_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("task_type", "taskType"),
        serialization_alias="taskType",
    )
    movement_condition: str | None = Field(
        default=None,
        validation_alias=AliasChoices("movement_condition", "movementCondition"),
        serialization_alias="movementCondition",
    )
    sampling_rate_hz: float | None = Field(
        default=None,
        validation_alias=AliasChoices("sampling_rate_hz", "samplingRateHz"),
        serialization_alias="samplingRateHz",
    )
    channels: int | None = None
    recording_duration_sec: float | None = Field(
        default=None,
        validation_alias=AliasChoices("recording_duration_sec", "recordingDurationSec"),
        serialization_alias="recordingDurationSec",
    )
    sample_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sample_id", "sampleId"),
        serialization_alias="sampleId",
    )


class VisualizationInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    tab: str
    title: str
    image_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("image_url", "imageUrl"),
        serialization_alias="imageUrl",
    )
    index: int = 0
    channel: str | None = None
    band: str | None = None
    condition: str | None = None
    sample_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("sample_id", "sampleId"),
        serialization_alias="sampleId",
    )
    image_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("image_path", "imagePath"),
        serialization_alias="imagePath",
    )


class UploadedArtifact(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    kind: Literal["eeg", "figure", "metadata"]
    size_bytes: int = Field(
        validation_alias=AliasChoices("size_bytes", "sizeBytes"),
        serialization_alias="sizeBytes",
    )
    content_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("content_type", "contentType"),
        serialization_alias="contentType",
    )
    stored_path: str | None = Field(
        default=None,
        validation_alias=AliasChoices("stored_path", "storedPath"),
        serialization_alias="storedPath",
    )
    image_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("image_id", "imageId"),
        serialization_alias="imageId",
    )


class UploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    experiment_id: str = Field(
        validation_alias=AliasChoices("experiment_id", "experimentId"),
        serialization_alias="experimentId",
    )
    asset_id: str = Field(
        validation_alias=AliasChoices("asset_id", "assetId"),
        serialization_alias="assetId",
    )
    uploaded_artifacts: list[UploadedArtifact] = Field(default_factory=list)
    detected_input_types: list[str] = Field(default_factory=list)
    available_visualizations: list[VisualizationInfo] = Field(default_factory=list)
    metadata: ExperimentMetadata | None = None
    eeg: EEGMetadata | None = None
    status: Literal["idle", "uploading", "ready", "error"] = "ready"
    error: str | None = None


class ExperimentResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    experiment_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("experiment_id", "experimentId"),
        serialization_alias="experimentId",
    )
    eeg: EEGMetadata | None = None
    figure: dict[str, Any] | None = None
    metadata: ExperimentMetadata | None = None
    visualizations: list[VisualizationInfo] = Field(default_factory=list)
    modalities: dict[str, bool] = Field(default_factory=dict)
    files: list[UploadedArtifact] = Field(default_factory=list)
    status: Literal["empty", "ready", "processing", "error"] = "ready"
    is_demo: bool = Field(default=False, serialization_alias="isDemo")
    error_message: str | None = Field(
        default=None, serialization_alias="errorMessage"
    )
    analysis_history: list[dict[str, Any]] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    experiment_id: str = Field(
        validation_alias=AliasChoices("experiment_id", "experimentId")
    )
    question: str
    image_id: str | None = Field(
        default=None, validation_alias=AliasChoices("image_id", "imageId")
    )
    visualization_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("visualization_id", "visualizationId"),
    )
    context: dict[str, Any] | None = None
    tools: list[str] | None = None
    settings: dict[str, Any] | None = None
    conversation_history: list[dict[str, Any]] | None = Field(
        default=None,
        validation_alias=AliasChoices("conversation_history", "conversationHistory"),
    )


class RouteInfo(BaseModel):
    """Detailed routing metadata. Top-level route remains TEXT|VISION for the UI."""

    intent: dict[str, Any] | None = None
    requires_vision: bool = False
    requested_visual_type: str | None = None
    question_type: str | None = None
    components: list[str] | None = None
    task_plan: dict[str, Any] | None = None
    text_only: bool | None = None
    needs_input: bool | None = None
    need_kind: str | None = None
    reason: str | None = None


class ComputedEvidenceItem(BaseModel):
    label: str
    value: str
    unit: str | None = None
    tool: str | None = None
    highlight: bool | None = None
    metric: str | None = None
    channel: str | None = None
    band: str | None = None
    condition: str | None = None
    provenance: str | None = None


class VisualEvidenceItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    label: str
    tab: str
    observation: str | None = None
    image_url: str | None = Field(
        default=None,
        validation_alias=AliasChoices("image_url", "imageUrl"),
        serialization_alias="imageUrl",
    )
    image_type: str | None = None
    vlm_interpretation: str | None = None
    provenance: str | None = None


class VerificationInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str
    message: str | None = None
    recovery_performed: bool | None = Field(
        default=None,
        validation_alias=AliasChoices("recovery_performed", "recoveryPerformed"),
        serialization_alias="recoveryPerformed",
    )
    triggered: bool | None = None
    result: dict[str, Any] | None = None
    recovery_triggered: bool | None = None


class TimingInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("total_ms", "totalMs"),
        serialization_alias="totalMs",
    )
    routing_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("routing_ms", "routingMs"),
        serialization_alias="routingMs",
    )
    tools_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("tools_ms", "toolsMs"),
        serialization_alias="toolsMs",
    )
    vision_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("vision_ms", "visionMs"),
        serialization_alias="visionMs",
    )
    generation_ms: float | None = Field(default=None)
    synthesis_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("synthesis_ms", "synthesisMs"),
        serialization_alias="synthesisMs",
    )
    verification_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("verification_ms", "verificationMs"),
        serialization_alias="verificationMs",
    )
    verifier_ms: float | None = None
    recovery_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("recovery_ms", "recoveryMs"),
        serialization_alias="recoveryMs",
    )


class SystemInfo(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text_model: str = Field(serialization_alias="textModel")
    vision_model: str = Field(serialization_alias="visionModel")
    precision: str
    serving: str
    route: Literal["TEXT", "VISION"]
    verifier_status: str | None = Field(
        default=None, serialization_alias="verifierStatus"
    )
    text_backend: str | None = None
    vision_backend: str | None = None
    serving_mode: str | None = None


class TimelineStage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    status: str
    latency_ms: float | None = Field(
        default=None,
        validation_alias=AliasChoices("latency_ms", "latencyMs"),
        serialization_alias="latencyMs",
    )
    summary: str | None = None


class AnalyzeResponse(BaseModel):
    """Top-level evidence keys stay snake_case (web AnalyzeResponse)."""

    model_config = ConfigDict(populate_by_name=True)

    answer: str
    route: Literal["TEXT", "VISION"]
    computed_evidence: list[ComputedEvidenceItem] = Field(default_factory=list)
    visual_evidence: list[VisualEvidenceItem] = Field(default_factory=list)
    model_interpretation: str = ""
    tools_used: list[str] = Field(default_factory=list)
    verification: VerificationInfo
    uncertainty: str = ""
    timing: TimingInfo
    system: SystemInfo
    timeline: list[TimelineStage] = Field(default_factory=list)
    question: str | None = None
    id: str | None = None
    raw_tool_output: str | None = None
    route_detail: RouteInfo | None = None
    experiment_id: str | None = None
    analysis_results: dict[str, Any] | None = None
    # Strict vision provenance — exact asset analyzed (never a silent substitute)
    source_image_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_image_id", "sourceImageId"),
        serialization_alias="sourceImageId",
    )
    source_image_name: str | None = Field(
        default=None,
        validation_alias=AliasChoices("source_image_name", "sourceImageName"),
        serialization_alias="sourceImageName",
    )
    vision_used: bool = Field(
        default=False,
        validation_alias=AliasChoices("vision_used", "visionUsed"),
        serialization_alias="visionUsed",
    )
    vision_asset_origin: str | None = Field(
        default=None,
        validation_alias=AliasChoices("vision_asset_origin", "visionAssetOrigin"),
        serialization_alias="visionAssetOrigin",
    )
    vision_content_type: str | None = Field(
        default=None,
        validation_alias=AliasChoices("vision_content_type", "visionContentType"),
        serialization_alias="visionContentType",
    )
