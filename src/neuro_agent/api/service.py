"""Research analysis service — wires PrimaryResearchAgent + tools + optional VLM."""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from neuro_agent.agent.task_plan import (
    ArtifactContext,
    history_from_payload,
    plan_task,
)
from neuro_agent.api.result_contracts import (
    build_analysis_results_payload,
)
from neuro_agent.api.experiment_store import ExperimentRecord, ExperimentStore
from neuro_agent.api.schemas import (
    AnalyzeResponse,
    ComputedEvidenceItem,
    RouteInfo,
    SystemInfo,
    TimingInfo,
    TimelineStage,
    VerificationInfo,
    VisualEvidenceItem,
    VisualizationInfo,
)
from neuro_agent.paths import PROJECT_ROOT
from neuro_agent.tools.evidence import ResearchToolRequest
from neuro_agent.tools.metadata import (
    default_vision_asset_index,
    lookup_sample_metadata,
)
from neuro_agent.tools.router import route_research_request
from neuro_agent.tools.schemas import SampleNotFoundError

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

FAMILY_TO_TAB = {
    "topomap_multi_band": "topomap",
    "power_spectral_density": "psd",
    "spectrogram": "spectrogram",
    "channel_band_power": "band_power",
    "waveform": "waveform",
    "condition_comparison": "comparison",
}

# Truthful labels for the API path actually used (HF Transformers + LoRA, not vLLM W8A8).
TEXT_MODEL_LABEL = "Qwen3-4B corrected SFT (HF+LoRA)"
VISION_MODEL_LABEL = "Qwen2.5-VL-3B corrected SFT (HF+PEFT)"
PRECISION = "BF16"
TEXT_BACKEND_DEFAULT = "PrimaryResearchAgent HF Transformers + LoRA (sft_corrected_v2)"
VISION_BACKEND_DEFAULT = "HF Transformers + PEFT (lazy)"

# Shown under Limitations for open-ended topomap/spectrogram VLM answers only.
UNRELIABLE_VISION_FIGURE_NOTE = (
    "Experimental visual reading for topomap/spectrogram figures — "
    "not reliable for research conclusions."
)

_UNRELIABLE_VISION_FIGURE_RE = re.compile(
    r"(topomap|topo[\s_-]?map|scalp[\s_-]*map|spectrogram|time[\s_-]?freq(?:uency)?)",
    re.IGNORECASE,
)
_WAVEFORM_FIGURE_RE = re.compile(
    r"(waveform|eeg[\s_-]*trace|time[\s_-]?series|(?:^|[^A-Za-z0-9])waves?(?:[^A-Za-z0-9]|$))",
    re.IGNORECASE,
)

# Visual-language cues: vision only when the question needs image interpretation.
_VISUAL_LANGUAGE_RE = re.compile(
    r"\b("
    r"image|figure|plot|topomap|spectrogram|waveform|heatmap|"
    r"visual(?:ly|s|ization)?|look(?:ing)?\s+at|depict|illustrat|"
    r"show(?:s|ing)?\s+(?:this|the)\s+(?:figure|plot|image|topomap)|"
    r"what\s+does\s+this\s+(?:figure|plot|image|topomap)"
    r")\b",
    re.IGNORECASE,
)

_TOOL_QUESTION_TYPES = frozenset(
    {
        "channel_ranking",
        "band_power",
        "channel_threshold",
        "condition_comparison",
        "psd_peak",
        "rms",
        "metadata",
        "select_channels",
    }
)


class AgentRunner(Protocol):
    def ask(self, question: str, *, request_id: str | None = None) -> Any: ...


class VisionRuntimeError(RuntimeError):
    """VLM load/generate failed or timed out — mapped to a structured API error."""


@dataclass
class RuntimeState:
    serving_mode: str = "hybrid"
    agent_loaded: bool = False
    vision_loaded: bool = False
    text_status: str = "unloaded"  # disabled|unloaded|loading|ready|error
    vision_status: str = "disabled"  # disabled|unloaded|loading|ready|error
    vision_enabled: bool = False
    text_error: str | None = None
    vision_error: str | None = None
    precision: str = PRECISION
    last_request_latency_ms: float | None = None
    last_route: str | None = None
    last_verifier_status: str | None = None
    last_ttft_ms: float | None = None
    text_backend: str = TEXT_BACKEND_DEFAULT
    vision_backend: str = VISION_BACKEND_DEFAULT
    agent: Any | None = None
    vision_model: Any | None = None
    vision_processor: Any | None = None


@dataclass
class AnalysisService:
    store: ExperimentStore
    state: RuntimeState = field(default_factory=RuntimeState)
    agent_factory: Callable[[], Any] | None = None
    enable_vlm: bool = False
    # Injected runner for tests (bypasses GPU)
    mock_runner: AgentRunner | None = None
    vlm_timeout_s: float = field(
        default_factory=lambda: float(os.environ.get("NEURO_API_VLM_TIMEOUT_S", "180"))
    )

    def __post_init__(self) -> None:
        self.state.vision_enabled = bool(self.enable_vlm)
        if self.enable_vlm:
            if self.state.vision_status == "disabled":
                self.state.vision_status = "unloaded"
        else:
            self.state.vision_status = "disabled"
        if self.mock_runner is not None:
            self.state.agent_loaded = True
            self.state.text_status = "ready"
            self.state.text_backend = "mock_runner"
            self.state.precision = PRECISION

    # --- visualizations ---
    def visualizations_for_sample(self, sample_id: str) -> list[VisualizationInfo]:
        try:
            meta = lookup_sample_metadata(sample_id=sample_id)
        except SampleNotFoundError:
            return []
        out: list[VisualizationInfo] = []
        for i, va in enumerate(meta.vision_assets):
            tab = FAMILY_TO_TAB.get(va.visualization_type, "figure")
            url = f"/api/visualization/{va.image_id}"
            out.append(
                VisualizationInfo(
                    id=va.image_id,
                    tab=tab,
                    title=f"{tab} — {va.image_id}",
                    image_url=url,
                    index=i,
                    band=va.frequency_band,
                    condition=meta.condition,
                    sample_id=sample_id,
                    image_path=va.image_path,
                )
            )
        return out

    def resolve_visualization(self, visualization_id: str) -> VisualizationInfo | None:
        idx = default_vision_asset_index()
        rec = idx.get_image_record(visualization_id)
        if rec is None:
            # also search experiment uploads
            return None
        fam = str(rec.get("visualization_type", "figure"))
        tab = FAMILY_TO_TAB.get(fam, "figure")
        return VisualizationInfo(
            id=str(rec["image_id"]),
            tab=tab,
            title=f"{tab} — {rec['image_id']}",
            image_url=f"/api/visualization/{rec['image_id']}",
            index=0,
            band=rec.get("frequency_band"),
            condition=rec.get("condition") or rec.get("movement_condition"),
            sample_id=rec.get("epoch_sample_id"),
            image_path=str(rec.get("image_path")),
        )

    def visualization_file_path(self, visualization_id: str) -> Path | None:
        info = self.resolve_visualization(visualization_id)
        if info and info.image_path:
            p = Path(info.image_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return p
        # uploaded assets named as visualization ids
        for exp_dir in self.store.root.glob("exp_*"):
            meta_path = exp_dir / "experiment.json"
            if not meta_path.exists():
                continue
            data = json.loads(meta_path.read_text())
            for art in data.get("artifacts", []):
                if art.get("id") == visualization_id or art.get("image_id") == visualization_id:
                    sp = art.get("stored_path")
                    if sp:
                        p = Path(sp)
                        if not p.is_absolute():
                            p = PROJECT_ROOT / p
                        if p.exists():
                            return p
        return None

    def _abspath(self, stored: str | None) -> Path | None:
        if not stored:
            return None
        p = Path(stored)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p if p.exists() else None

    def resolve_exact_vision_asset(
        self,
        *,
        experiment_id: str,
        image_id: str,
    ) -> dict[str, Any] | None:
        """Resolve the exact selected asset — no alternate-image fallback.

        Returns dict with visual (VisualEvidenceItem), path, name, origin, content_type
        or None if the ID cannot be resolved.
        """
        requested = (image_id or "").strip()
        if not requested:
            return None

        rec = self.store.get(experiment_id)
        if rec is not None:
            for art in rec.artifacts:
                aid = str(art.get("id") or "")
                iid = str(art.get("image_id") or "")
                if requested not in {aid, iid}:
                    continue
                path = self._abspath(art.get("stored_path"))
                if path is None:
                    continue
                name = str(art.get("name") or Path(path).name)
                origin = "uploaded" if art.get("kind") in {"figure", "image", "vision"} else "uploaded"
                visual = VisualEvidenceItem(
                    id=aid or requested,
                    label=name,
                    tab="figure",
                    image_url=f"/api/visualization/{aid or requested}",
                    image_type="figure",
                    # Do not expose filesystem paths in API provenance strings
                    provenance=f"asset:{aid or requested}",
                )
                return {
                    "visual": visual,
                    "path": path,
                    "name": name,
                    "origin": origin,
                    "content_type": art.get("content_type"),
                }

            for viz in rec.visualizations:
                vid = str(viz.get("id") or viz.get("image_id") or "")
                if vid != requested:
                    continue
                path = self._abspath(viz.get("image_path") or viz.get("imagePath"))
                if path is None:
                    path = self.visualization_file_path(requested)
                if path is None:
                    continue
                name = str(viz.get("title") or Path(path).name)
                tab = str(viz.get("tab") or "figure")
                origin = "generated" if tab != "figure" else "uploaded"
                visual = VisualEvidenceItem(
                    id=vid,
                    label=name,
                    tab=tab,
                    image_url=viz.get("image_url") or viz.get("imageUrl") or f"/api/visualization/{vid}",
                    image_type=tab,
                    provenance=f"visualization:{vid}",
                )
                return {
                    "visual": visual,
                    "path": path,
                    "name": name,
                    "origin": origin,
                    "content_type": None,
                }

        # Dataset / sample-linked asset by exact ID only (never "first sample image")
        info = self.resolve_visualization(requested)
        if info is not None:
            path = self._abspath(info.image_path) or self.visualization_file_path(requested)
            if path is not None:
                name = str(info.title or requested)
                visual = VisualEvidenceItem(
                    id=info.id,
                    label=name,
                    tab=info.tab or "figure",
                    image_url=info.image_url or f"/api/visualization/{info.id}",
                    image_type=info.tab or "figure",
                    provenance=f"dataset:{info.id}",
                )
                return {
                    "visual": visual,
                    "path": path,
                    "name": name,
                    "origin": "dataset",
                    "content_type": None,
                }
        return None

    # --- upload helpers ---
    def attach_sample_to_experiment(
        self, rec: ExperimentRecord, sample_id: str
    ) -> ExperimentRecord:
        meta = lookup_sample_metadata(sample_id=sample_id)
        rec.linked_sample_id = sample_id
        rec.metadata = {
            "subject": meta.subject_id,
            "run": meta.run_id,
            "task_type": meta.task_type,
            "movement_condition": meta.condition,
            "sampling_rate_hz": meta.sampling_rate_hz,
            "channels": len(meta.channels),
            "sample_id": sample_id,
        }
        rec.eeg = {
            "format": "hdf5",
            "sampling_rate_hz": meta.sampling_rate_hz,
            "channels": len(meta.channels),
            "channel_labels": list(meta.channels),
            "sample_id": sample_id,
            "auto_detected": True,
            "filename": meta.array_ref.array_path,
        }
        rec.visualizations = [v.model_dump() for v in self.visualizations_for_sample(sample_id)]
        rec.linked_image_ids = [v["id"] for v in rec.visualizations]
        rec.modalities = {
            "eeg": True,
            "metadata": True,
            "vision": bool(rec.visualizations),
            "text": True,
        }
        rec.status = "ready"
        self.store.save(rec)
        return rec

    def ensure_demo_experiment(self) -> ExperimentRecord:
        """Create or return a demo experiment bound to a known processed sample."""
        demo_id = "exp_demo_s001"
        existing = self.store.get(demo_id)
        if existing is not None:
            return existing
        # manually create with fixed id
        from neuro_agent.api.experiment_store import ExperimentRecord as ER
        import time as _t

        now = _t.time()
        rec = ER(
            id=demo_id,
            created_at=now,
            updated_at=now,
            status="ready",
            is_demo=True,
            modalities={"eeg": True, "metadata": True, "vision": True, "text": True},
        )
        d = self.store._exp_dir(demo_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "uploads").mkdir(exist_ok=True)
        self.store._write(rec)
        return self.attach_sample_to_experiment(rec, "S001_R01_E000")

    # --- runtime lifecycle ---
    def initialize_tools(self) -> None:
        """Warm deterministic tool metadata indexes (no GPU)."""
        try:
            default_vision_asset_index()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tool_index_warmup_failed: %s", exc)

    def ensure_agent(self) -> Any:
        if self.mock_runner is not None:
            self.state.agent_loaded = True
            self.state.text_status = "ready"
            return self.mock_runner
        if self.state.agent is not None and getattr(self.state.agent, "is_loaded", True):
            self.state.agent_loaded = True
            self.state.text_status = "ready"
            return self.state.agent

        self.state.text_status = "loading"
        self.state.text_error = None
        logger.info("TEXT_RUNTIME_LOADING")
        try:
            if self.agent_factory is not None:
                self.state.agent = self.agent_factory()
                self.state.agent_loaded = True
                self.state.text_status = "ready"
                self.state.text_backend = "injected agent_factory"
                self.state.precision = PRECISION
                logger.info("TEXT_RUNTIME_READY backend=%s", self.state.text_backend)
                return self.state.agent

            from neuro_agent.agent import PrimaryResearchAgent, ResearchAgentConfig

            agent = PrimaryResearchAgent(ResearchAgentConfig())
            agent.load()
            self.state.agent = agent
            self.state.agent_loaded = True
            self.state.text_status = "ready"
            self.state.text_backend = TEXT_BACKEND_DEFAULT
            self.state.precision = PRECISION
            logger.info(
                "TEXT_RUNTIME_READY backend=%s precision=%s",
                self.state.text_backend,
                self.state.precision,
            )
            return agent
        except Exception as exc:
            self.state.agent = None
            self.state.agent_loaded = False
            self.state.text_status = "error"
            self.state.text_error = str(exc)
            logger.error("TEXT_RUNTIME_ERROR: %s", exc)
            raise

    def unload_text_runtime(self) -> None:
        """Release text GPU weights before loading VLM (safe residency policy)."""
        agent = self.state.agent
        if agent is None:
            self.state.agent_loaded = False
            if self.state.text_status == "ready":
                self.state.text_status = "unloaded"
            return
        logger.info("TEXT_RUNTIME_UNLOADING for VLM residency")
        if hasattr(agent, "unload"):
            agent.unload()
        self.state.agent = None
        self.state.agent_loaded = False
        self.state.text_status = "unloaded"

    def unload_vision_runtime(self) -> None:
        self.state.vision_model = None
        self.state.vision_processor = None
        self.state.vision_loaded = False
        if self.enable_vlm:
            self.state.vision_status = "unloaded"
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass

    @staticmethod
    def question_has_visual_language(question: str) -> bool:
        return bool(_VISUAL_LANGUAGE_RE.search(question or ""))

    def decide_requires_vision(
        self,
        *,
        question: str,
        raw_intent: dict[str, Any],
        image_id: str | None,
        visualization_id: str | None,
        context: dict[str, Any] | None,
    ) -> bool:
        """Vision only for visual interpretation — not merely because a topomap exists."""
        ctx = context or {}
        if ctx.get("requires_vision"):
            return True

        explicit_image = bool(image_id or visualization_id)
        has_visual_language = self.question_has_visual_language(question)
        intent_flag = bool(raw_intent.get("requires_vision"))
        qtype = str(raw_intent.get("question_type") or "")

        # Do NOT treat include_vision_evidence / requested_visual_type / image_id alone
        # as requiring the VISION route — those fields historically over-fired for
        # ranking questions when a sample had linked topomaps.
        if intent_flag:
            if has_visual_language or explicit_image:
                return True
            # Guard false positives: tool/numeric questions without visual language.
            if qtype in _TOOL_QUESTION_TYPES or not has_visual_language:
                return False
            return True

        if has_visual_language:
            return True
        # Explicitly selected image only forces vision when the question depends on it.
        if explicit_image and has_visual_language:
            return True
        return False

    def gpu_stats(self) -> dict[str, float | None]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            used, total, util = [float(x.strip()) for x in out.split(",")]
            return {
                "gpu_memory_used_mb": used,
                "gpu_memory_total_mb": total,
                "gpu_utilization_pct": util,
            }
        except Exception:
            return {
                "gpu_memory_used_mb": None,
                "gpu_memory_total_mb": None,
                "gpu_utilization_pct": None,
            }

    # --- analyze ---
    def analyze(
        self,
        *,
        experiment_id: str,
        question: str,
        image_id: str | None = None,
        visualization_id: str | None = None,
        context: dict[str, Any] | None = None,
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> AnalyzeResponse:
        q = (question or "").strip()
        if not q:
            raise ValueError("missing_question")

        rec = self.store.get(experiment_id)
        if rec is None:
            raise KeyError("invalid_experiment_id")

        selected_image = (image_id or "").strip() or None
        # visualization_id must NOT silently substitute for a missing selected image_id
        # on vision requests — that caused wrong-figure VLM inputs.
        history = history_from_payload(conversation_history)
        artifacts = ArtifactContext(
            has_sample=bool(rec.linked_sample_id),
            has_image=bool(selected_image or visualization_id),
            sample_id=rec.linked_sample_id,
            image_id=selected_image or visualization_id,
        )
        plan = plan_task(q, history=history, artifacts=artifacts)

        # Missing required input — clean user-facing message, no fake analysis.
        if plan.needs_input:
            return self._needs_input_response(
                question=q,
                experiment_id=experiment_id,
                plan=plan,
            )

        t0 = time.perf_counter()
        agent = self.ensure_agent()

        history_snippet = None
        if history:
            bits = []
            for turn in history[-6:]:
                bits.append(f"{turn.role}: {turn.content[:400]}")
            history_snippet = "\n".join(bits)

        # TEXT-only conversational / concept / explanatory follow-up
        if plan.text_only and not plan.use_vision and not plan.use_tools:
            trace = agent.ask_text_only(
                plan.resolved_question or q,
                prior_context=plan.prior_context_for_answer,
                history_snippet=history_snippet,
            )
            total_ms = (time.perf_counter() - t0) * 1000.0
            response = self._trace_to_response(
                trace=trace,
                question=q,
                experiment_id=experiment_id,
                requires_vision=False,
                raw_intent=getattr(trace, "parsed_intent", None) or {},
                visual_items=[],
                vision_ms=0.0,
                total_ms_override=total_ms,
                vlm_text=None,
                task_plan=plan.to_dict(),
            )
            self._record_analyze(experiment_id, response, q)
            return response

        # Enrich with sample context ONLY when tools are planned.
        enriched = plan.resolved_question or q
        if plan.use_tools and rec.linked_sample_id and rec.linked_sample_id not in enriched:
            enriched = f"{enriched} (sample {rec.linked_sample_id})"
        if (
            plan.use_vision
            and selected_image
            and selected_image not in enriched
            and self.question_has_visual_language(q)
        ):
            enriched = f"{enriched} [image_id={selected_image}]"

        if plan.use_tools:
            trace = agent.ask(
                enriched,
                enable_verification=plan.use_verify,
            )
        elif plan.use_vision:
            # Vision-primary: do NOT feed prior tool evidence into synthesis.
            # Follow-up visual context is OK; tool/EEG history is not.
            vision_prior = None
            if plan.is_follow_up and plan.prior_context_for_answer:
                vision_prior = plan.prior_context_for_answer
            trace = agent.ask_text_only(
                enriched,
                prior_context=vision_prior,
                history_snippet=None,
            )
        else:
            trace = agent.ask_text_only(
                enriched,
                prior_context=plan.prior_context_for_answer,
                history_snippet=history_snippet,
            )

        total_ms = (time.perf_counter() - t0) * 1000.0

        raw_intent = getattr(trace, "parsed_intent", None) or {}
        requires_vision = bool(plan.use_vision)
        # Still consult decide_requires_vision as a soft check when client sent context,
        # but task plan is authoritative for production gating.
        if requires_vision and context and context.get("requires_vision") is False:
            requires_vision = self.decide_requires_vision(
                question=q,
                raw_intent=raw_intent if isinstance(raw_intent, dict) else {},
                image_id=image_id,
                visualization_id=visualization_id,
                context=context,
            )
            requires_vision = True  # plan wins for explicit vision tasks

        vision_ms = 0.0
        vlm_text: str | None = None
        visual_items: list[VisualEvidenceItem] = []
        source_image_id: str | None = None
        source_image_name: str | None = None
        vision_used = False
        vision_asset_origin: str | None = None
        vision_content_type: str | None = None

        evidence = getattr(trace, "evidence_bundle", None) or {}
        # Tool-sidecar vision refs are metadata only — NEVER preferred over the
        # explicitly selected image_id for VLM input.
        resolve_image = selected_image if requires_vision else None
        if requires_vision and not resolve_image and visualization_id:
            # Explicit visualization-only request (no upload selected)
            resolve_image = visualization_id

        if requires_vision:
            logger.info("VISION_REQUEST image_id=%s", resolve_image or "")
            if not resolve_image:
                raise FileNotFoundError("missing_image_for_vision")
            resolved = self.resolve_exact_vision_asset(
                experiment_id=experiment_id,
                image_id=resolve_image,
            )
            if resolved is None:
                logger.error(
                    "VISION_ASSET_RESOLVE_FAILED image_id=%s experiment_id=%s",
                    resolve_image,
                    experiment_id,
                )
                raise FileNotFoundError("missing_image_for_vision")

            visual_items = [resolved["visual"]]
            source_image_id = resolved["visual"].id
            source_image_name = resolved["name"]
            vision_asset_origin = resolved["origin"]
            vision_content_type = resolved.get("content_type")
            logger.info(
                "VISION_ASSET_RESOLVED image_id=%s name=%s origin=%s",
                source_image_id,
                source_image_name,
                vision_asset_origin,
            )

            if self.enable_vlm:
                t_v = time.perf_counter()
                try:
                    vlm_text = self._run_vlm_on_path(q, resolved["path"])
                    visual_items[0].vlm_interpretation = vlm_text
                    visual_items[0].observation = vlm_text
                    vision_used = True
                except Exception as exc:  # noqa: BLE001
                    vision_ms = (time.perf_counter() - t_v) * 1000.0
                    raise VisionRuntimeError(str(exc)) from exc
                vision_ms = (time.perf_counter() - t_v) * 1000.0
            else:
                # Mock / VLM-disabled: still bind provenance to the exact asset
                vision_used = True

            logger.info("VISION_RESPONSE source_image_id=%s", source_image_id)

        response = self._trace_to_response(
            trace=trace,
            question=q,
            experiment_id=experiment_id,
            requires_vision=requires_vision,
            raw_intent=raw_intent if isinstance(raw_intent, dict) else {},
            visual_items=visual_items,
            vision_ms=vision_ms,
            total_ms_override=total_ms,
            vlm_text=vlm_text,
            task_plan=plan.to_dict(),
            source_image_id=source_image_id,
            source_image_name=source_image_name,
            vision_used=vision_used,
            vision_asset_origin=vision_asset_origin,
            vision_content_type=vision_content_type,
        )

        self._record_analyze(experiment_id, response, q)
        return response

    def _record_analyze(
        self, experiment_id: str, response: AnalyzeResponse, question: str
    ) -> None:
        self.state.last_request_latency_ms = response.timing.total_ms
        self.state.last_route = response.route
        self.state.last_verifier_status = response.verification.status
        if response.timing.routing_ms is not None:
            self.state.last_ttft_ms = response.timing.routing_ms
        self.store.append_analysis(
            experiment_id,
            {
                "id": response.id,
                "question": question,
                "route": response.route,
                "answer_preview": (response.answer or "")[:240],
                "total_ms": response.timing.total_ms,
                "components": (response.route_detail.components if response.route_detail else None),
            },
        )

    def _needs_input_response(
        self,
        *,
        question: str,
        experiment_id: str,
        plan: Any,
    ) -> AnalyzeResponse:
        msg = plan.missing_input_message or "Additional input is required."
        return AnalyzeResponse(
            answer=msg,
            route="TEXT",
            computed_evidence=[],
            visual_evidence=[],
            model_interpretation="",
            tools_used=[],
            verification=VerificationInfo(status="skipped"),
            uncertainty="None",
            timing=TimingInfo(total_ms=0.0, routing_ms=0.0),
            system=SystemInfo(
                text_model=TEXT_MODEL_LABEL,
                vision_model=VISION_MODEL_LABEL,
                precision=PRECISION,
                serving=self.state.serving_mode,
                route="TEXT",
                verifier_status="skipped",
                text_backend=self.state.text_backend,
                vision_backend=self.state.vision_backend,
                serving_mode=self.state.serving_mode,
            ),
            timeline=[],
            question=question,
            id=None,
            raw_tool_output=None,
            route_detail=RouteInfo(
                intent=None,
                requires_vision=False,
                components=list(plan.components),
                task_plan=plan.to_dict(),
                text_only=True,
                needs_input=True,
                need_kind=plan.need_kind,
                reason=plan.reason,
            ),
            experiment_id=experiment_id,
            analysis_results=build_analysis_results_payload(
                question=question,
                route="TEXT",
                tools_used=[],
                computed_evidence=[],
                visual_evidence=[],
                answer=msg,
                sample_id=None,
                experiment_id=experiment_id,
                task_plan=plan.to_dict(),
                vlm_text=None,
                image_id=None,
            ),
        )

    def _run_vlm(self, question: str, visual: VisualEvidenceItem) -> str:
        """Load/generate with text unloaded and a hard timeout (no indefinite hang)."""
        path = self.visualization_file_path(visual.id)
        if path is None:
            # uploaded artifacts may store path on the visual provenance
            if visual.provenance:
                p = Path(visual.provenance)
                if not p.is_absolute():
                    p = PROJECT_ROOT / p
                if p.exists():
                    path = p
        if path is None or not path.exists():
            raise FileNotFoundError("vision_image_missing")
        return self._run_vlm_on_path(question, path)

    def _run_vlm_on_path(self, question: str, path: Path) -> str:
        """Invoke VLM on an already-resolved filesystem path (exact selected asset)."""
        if path is None or not path.exists():
            raise FileNotFoundError("vision_image_missing")

        timeout_s = float(self.vlm_timeout_s)
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self._run_vlm_sync, question, path)
            try:
                return fut.result(timeout=timeout_s)
            except FuturesTimeout as exc:
                self.state.vision_status = "error"
                self.state.vision_error = f"vlm_timeout after {timeout_s}s"
                logger.error("VISION_RUNTIME_ERROR: %s", self.state.vision_error)
                raise VisionRuntimeError(self.state.vision_error) from exc

    def _run_vlm_sync(self, question: str, path: Path) -> str:
        from neuro_agent.inference.config import InferenceConfig
        from neuro_agent.multimodal.dataset import build_multimodal_messages
        from neuro_agent.multimodal.model import load_vlm_for_inference
        from qwen_vl_utils import process_vision_info
        import torch

        # Known residency: text util ~0.90 + VLM co-residency fails; unload text first.
        if self.state.agent is not None and not self.state.vision_loaded:
            self.unload_text_runtime()

        if self.state.vision_model is None:
            self.state.vision_status = "loading"
            self.state.vision_error = None
            logger.info("VISION_RUNTIME_LOADING")
            try:
                from neuro_agent.api.release_manifest import VISION_CHECKPOINT_REL

                cfg = InferenceConfig(
                    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                    dtype="bfloat16",
                    trust_remote_code=True,
                    adapter_path=str(PROJECT_ROOT / VISION_CHECKPOINT_REL),
                    max_new_tokens=64,
                    do_sample=False,
                )
                model, processor, _info = load_vlm_for_inference(cfg)
                self.state.vision_model = model
                self.state.vision_processor = processor
                self.state.vision_loaded = True
                self.state.vision_status = "ready"
                self.state.vision_backend = "HF Transformers + PEFT (warm)"
                logger.info("VISION_RUNTIME_READY backend=%s", self.state.vision_backend)
            except Exception as exc:
                self.state.vision_status = "error"
                self.state.vision_error = str(exc)
                logger.error("VISION_RUNTIME_ERROR: %s", exc)
                raise

        model = self.state.vision_model
        processor = self.state.vision_processor
        messages = build_multimodal_messages(
            system_prompt=(
                "You are a neuroscience research assistant analyzing EEG-derived plots. "
                "Answer briefly based on the image."
            ),
            user_text=f"Question: {question.strip()}",
            image_uri=f"file://{path.resolve()}",
        )
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        in_len = int(inputs["input_ids"].shape[-1])
        decoded = processor.batch_decode(
            out[:, in_len:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        result = decoded.strip()
        if not result:
            raise VisionRuntimeError("vlm_empty_generation")
        return result

    def _trace_to_response(
        self,
        *,
        trace: Any,
        question: str,
        experiment_id: str,
        requires_vision: bool,
        raw_intent: dict[str, Any],
        visual_items: list[VisualEvidenceItem],
        vision_ms: float,
        total_ms_override: float | None,
        vlm_text: str | None,
        task_plan: dict[str, Any] | None = None,
        source_image_id: str | None = None,
        source_image_name: str | None = None,
        vision_used: bool = False,
        vision_asset_origin: str | None = None,
        vision_content_type: str | None = None,
    ) -> AnalyzeResponse:
        evidence = getattr(trace, "evidence_bundle", None) or {}
        tools = []
        for inv in getattr(trace, "tool_invocations", None) or evidence.get("tool_invocations") or []:
            if isinstance(inv, dict):
                name = inv.get("name") or inv.get("tool")
            else:
                name = getattr(inv, "name", None)
            if name:
                tools.append(str(name))

        plan_use_tools = bool((task_plan or {}).get("use_tools"))
        plan_use_vision = bool(requires_vision)

        # Evidence isolation: only keep evidence from components that ran THIS request.
        if plan_use_vision and not plan_use_tools:
            tools = []
            computed: list[ComputedEvidenceItem] = []
        else:
            computed = self._extract_computed_evidence(evidence, tools)

        raw_answer = getattr(trace, "final_answer", None) or ""

        if plan_use_vision and not plan_use_tools:
            # Vision-only answers come from the VLM for the selected image — never
            # from a text shell that may still carry stale EEG/tool prose.
            if vlm_text and str(vlm_text).strip():
                raw_answer = str(vlm_text).strip()
            elif visual_items and (visual_items[0].observation or visual_items[0].vlm_interpretation):
                raw_answer = str(
                    visual_items[0].observation or visual_items[0].vlm_interpretation or ""
                ).strip()
            else:
                raw_answer = ""
        elif plan_use_tools and not plan_use_vision:
            # Tool path: keep tool synthesis; drop any accidental vision rows
            visual_items = []
        elif plan_use_tools and plan_use_vision and vlm_text and str(vlm_text).strip():
            # Rare combined path: append current VLM observation to tool answer
            tool_ans = str(raw_answer).strip()
            vis_ans = str(vlm_text).strip()
            if tool_ans and vis_ans and vis_ans not in tool_ans:
                raw_answer = f"{tool_ans}\n\nVisual observation: {vis_ans}"
            elif vis_ans and not tool_ans:
                raw_answer = vis_ans

        answer, uncertainty = self._present_user_facing_answer(
            raw_answer,
            getattr(trace, "warnings", []) or [],
        )

        # Product safety: open-ended topomap/spectrogram VLM reading failed the
        # real V2/V3 gate — keep vision enabled, but mark those answers clearly.
        if plan_use_vision and not plan_use_tools:
            if self._is_unreliable_open_ended_vision_figure(
                question=question,
                vision_content_type=vision_content_type,
                source_image_name=source_image_name,
                raw_intent=raw_intent,
                visual_items=visual_items,
            ):
                if UNRELIABLE_VISION_FIGURE_NOTE.lower() not in uncertainty.lower():
                    uncertainty = (
                        f"{uncertainty}; {UNRELIABLE_VISION_FIGURE_NOTE}"
                        if uncertainty
                        else UNRELIABLE_VISION_FIGURE_NOTE
                    )

        ver_triggered = bool(getattr(trace, "verification_triggered", False))
        recovery = getattr(trace, "recovery", None)
        recovery_triggered = recovery is not None
        final_ver = getattr(trace, "final_verification", None)

        # Verifier gating: never for vision-only / text-only conversational paths
        if plan_use_vision and not plan_use_tools:
            ver_triggered = False
            recovery_triggered = False
            final_ver = None
        if bool((task_plan or {}).get("text_only")) and not plan_use_tools:
            ver_triggered = False
            recovery_triggered = False
            final_ver = None

        if recovery_triggered:
            ver_status = "recovered"
        elif ver_triggered:
            passed = bool(final_ver.get("passed")) if isinstance(final_ver, dict) else False
            ver_status = "passed" if passed else "triggered"
        else:
            ver_status = "skipped"

        route: str = "VISION" if requires_vision else "TEXT"
        timing = TimingInfo(
            total_ms=round(total_ms_override or getattr(trace, "runtime_ms", 0.0), 3),
            routing_ms=round(getattr(trace, "intent_latency_ms", 0.0), 3),
            tools_ms=None,  # not separately timed in AgentTrace
            vision_ms=round(vision_ms, 3) if vision_ms else None,
            generation_ms=round(getattr(trace, "answer_latency_ms", 0.0), 3),
            synthesis_ms=round(getattr(trace, "answer_latency_ms", 0.0), 3),
            verification_ms=round(getattr(trace, "verifier_latency_ms", 0.0), 3)
            if ver_triggered
            else None,
            verifier_ms=round(getattr(trace, "verifier_latency_ms", 0.0), 3)
            if ver_triggered
            else None,
            recovery_ms=round(getattr(trace, "recovery_latency_ms", 0.0), 3)
            if recovery_triggered
            else None,
        )

        timeline = [
            TimelineStage(
                id="routing",
                name="Intent / routing",
                status="complete",
                latency_ms=timing.routing_ms,
            ),
            TimelineStage(
                id="tools",
                name="Deterministic tools",
                status="complete" if tools else "skipped",
                summary=", ".join(tools) if tools else None,
            ),
            TimelineStage(
                id="vision",
                name="Vision",
                status="complete" if vision_ms else ("skipped" if not requires_vision else "error"),
                latency_ms=timing.vision_ms,
            ),
            TimelineStage(
                id="synthesis",
                name="Grounded answer",
                status="complete" if answer else "error",
                latency_ms=timing.generation_ms,
            ),
            TimelineStage(
                id="verifier",
                name="Verifier",
                status="complete" if ver_triggered else "skipped",
                latency_ms=timing.verifier_ms,
            ),
            TimelineStage(
                id="recovery",
                name="Recovery",
                status="complete" if recovery_triggered else "skipped",
                latency_ms=timing.recovery_ms,
            ),
        ]

        # Interpretation: only when it adds something beyond the research answer.
        if plan_use_vision and not plan_use_tools:
            interpretation = ""  # answer already is the VLM text
        elif vlm_text and vlm_text.strip() and vlm_text.strip() != answer.strip():
            interpretation = vlm_text.strip()
        elif tools and answer and not (task_plan or {}).get("text_only"):
            interpretation = ""
        else:
            interpretation = ""

        # Deduplicate visual rows with empty observations for cleaner UI
        cleaned_visuals: list[VisualEvidenceItem] = []
        seen_viz: set[str] = set()
        for item in visual_items:
            if item.id in seen_viz:
                continue
            seen_viz.add(item.id)
            if item.label and item.tab and item.label.lower() == str(item.tab).lower():
                item = item.model_copy(
                    update={"label": f"{str(item.tab).replace('_', ' ').title()} figure"}
                )
            cleaned_visuals.append(item)

        components = list((task_plan or {}).get("components") or [])
        if not components:
            components = ["TEXT"]
            if tools:
                components.append("TOOLS")
            if requires_vision:
                components.append("VISION")

        return AnalyzeResponse(
            answer=answer,
            route=route,  # type: ignore[arg-type]
            computed_evidence=computed if tools else [],
            visual_evidence=cleaned_visuals if requires_vision else [],
            model_interpretation=interpretation,
            tools_used=tools,
            verification=VerificationInfo(
                status=ver_status,
                message="; ".join(getattr(trace, "trigger_reason", []) or []) or None,
                recovery_performed=recovery_triggered,
                triggered=ver_triggered,
                result=final_ver if isinstance(final_ver, dict) else None,
                recovery_triggered=recovery_triggered,
            ),
            uncertainty=uncertainty,
            timing=timing,
            system=SystemInfo(
                text_model=TEXT_MODEL_LABEL,
                vision_model=VISION_MODEL_LABEL,
                precision=PRECISION,
                serving=self.state.serving_mode,
                route=route,  # type: ignore[arg-type]
                verifier_status=ver_status,
                text_backend=self.state.text_backend,
                vision_backend=self.state.vision_backend,
                serving_mode=self.state.serving_mode,
            ),
            timeline=timeline,
            question=question,
            id=getattr(trace, "request_id", None),
            raw_tool_output=json.dumps(evidence, default=str)[:8000] if evidence and tools else None,
            route_detail=RouteInfo(
                intent=raw_intent or None,
                requires_vision=requires_vision,
                requested_visual_type=raw_intent.get("requested_visual_type"),
                question_type=raw_intent.get("question_type"),
                components=components,
                task_plan=task_plan,
                text_only=bool((task_plan or {}).get("text_only")),
                needs_input=bool((task_plan or {}).get("needs_input")),
                need_kind=(task_plan or {}).get("need_kind"),
                reason=(task_plan or {}).get("reason"),
            ),
            experiment_id=experiment_id,
            analysis_results=build_analysis_results_payload(
                question=question,
                route=route,
                tools_used=tools,
                computed_evidence=computed if tools else [],
                visual_evidence=cleaned_visuals if requires_vision else [],
                answer=answer,
                sample_id=None,
                experiment_id=experiment_id,
                task_plan=task_plan,
                vlm_text=vlm_text,
                image_id=source_image_id,
            ),
            source_image_id=source_image_id if requires_vision else None,
            source_image_name=source_image_name if requires_vision else None,
            vision_used=bool(vision_used and requires_vision),
            vision_asset_origin=vision_asset_origin if requires_vision else None,
            vision_content_type=vision_content_type if requires_vision else None,
        )

    def _extract_computed_evidence(
        self, evidence: dict[str, Any], tools: list[str]
    ) -> list[ComputedEvidenceItem]:
        items: list[ComputedEvidenceItem] = []
        units = evidence.get("units") or {}

        for row in evidence.get("numeric_evidence") or []:
            if not isinstance(row, dict):
                continue
            channel = row.get("channel")
            band = row.get("band") or row.get("frequency_band")
            val = row.get("value") or row.get("power") or row.get("rms") or row.get("peak_frequency_hz")
            label_parts = [p for p in (channel, band, row.get("metric")) if p]
            label = " ".join(str(p) for p in label_parts) or row.get("name") or "metric"
            unit = row.get("units") or units.get(row.get("metric") or "") or None
            items.append(
                ComputedEvidenceItem(
                    label=str(label),
                    value=str(val) if val is not None else "",
                    unit=unit,
                    tool=tools[0] if tools else None,
                    metric=row.get("metric"),
                    channel=channel,
                    band=band,
                    provenance="evidence_bundle.numeric_evidence",
                )
            )

        ranked = evidence.get("ranked_evidence")
        if isinstance(ranked, dict):
            ranking = ranked.get("ranking") or ranked.get("channels") or []
            values = ranked.get("values") or {}
            unit = ranked.get("units")
            for i, ch in enumerate(list(ranking)[:5]):
                val = values.get(ch)
                items.append(
                    ComputedEvidenceItem(
                        label=f"Rank {i+1} · {ch}",
                        value=str(val) if val is not None else str(ch),
                        unit=str(unit) if unit and val is not None else None,
                        tool="rank_channels_for_sample"
                        if "rank" in ",".join(tools)
                        else (tools[0] if tools else None),
                        channel=str(ch),
                        highlight=i == 0,
                        provenance="evidence_bundle.ranked_evidence",
                    )
                )

        set_ev = evidence.get("set_evidence")
        if isinstance(set_ev, dict):
            chans = set_ev.get("channels") or []
            items.append(
                ComputedEvidenceItem(
                    label="selected channels",
                    value=", ".join(str(c) for c in chans),
                    tool="select_channels_above_threshold",
                    provenance="evidence_bundle.set_evidence",
                )
            )

        cond = evidence.get("condition_evidence")
        if isinstance(cond, dict):
            a = cond.get("condition_a") or "A"
            b = cond.get("condition_b") or "B"
            winner = cond.get("winner") or cond.get("higher_condition")
            va = cond.get("value_a")
            vb = cond.get("value_b")
            if va is not None and vb is not None:
                value = f"{a}={va}, {b}={vb}" + (f"; higher={winner}" if winner else "")
            else:
                value = str(winner or cond.get("summary") or cond)
            items.append(
                ComputedEvidenceItem(
                    label=f"{a} vs {b}",
                    value=value,
                    condition=f"{a}/{b}",
                    provenance="evidence_bundle.condition_evidence",
                )
            )

        return items

    @staticmethod
    def _is_unreliable_open_ended_vision_figure(
        *,
        question: str,
        vision_content_type: str | None,
        source_image_name: str | None,
        raw_intent: dict[str, Any],
        visual_items: list[VisualEvidenceItem],
    ) -> bool:
        """True for topomap/spectrogram open-ended vision; false for waveform-style.

        Prefer the concrete asset name over intent (intent can be a generic
        visual_inspection stub that still says requested_visual_type=topomap).
        """
        name = source_image_name or ""
        # Filename is the strongest signal for uploads.
        if name:
            if _WAVEFORM_FIGURE_RE.search(name) and not _UNRELIABLE_VISION_FIGURE_RE.search(
                name
            ):
                return False
            if _UNRELIABLE_VISION_FIGURE_RE.search(name):
                return True

        req_type = str(raw_intent.get("requested_visual_type") or "")
        tabs = " ".join(
            str(getattr(item, "tab", "") or "")
            + " "
            + str(getattr(item, "image_type", "") or "")
            + " "
            + str(getattr(item, "label", "") or "")
            for item in visual_items[:1]
        )
        blob = " ".join(
            [
                question or "",
                vision_content_type or "",
                req_type,
                tabs,
            ]
        )
        if _WAVEFORM_FIGURE_RE.search(blob) and not _UNRELIABLE_VISION_FIGURE_RE.search(blob):
            return False
        return bool(_UNRELIABLE_VISION_FIGURE_RE.search(blob))

    def _present_user_facing_answer(
        self, raw_answer: str, warnings: list[str]
    ) -> tuple[str, str]:
        """Split legacy sectioned agent text into clean answer + uncertainty."""
        text = (raw_answer or "").strip()
        uncertainty_parts: list[str] = []

        # Strip legacy section labels from the user-facing answer body.
        answer_m = re.search(
            r"(?is)^(?:Answer:\s*)?(.*?)(?:\n\s*Evidence:|\n\s*Tools used:|\n\s*Uncertainty:|\Z)",
            text,
        )
        answer = (answer_m.group(1) if answer_m else text).strip()
        answer = re.sub(r"(?im)^(Answer|Evidence|Tools used):\s*", "", answer).strip()

        unc_m = re.search(r"(?im)^Uncertainty:\s*(.+)$", text)
        if unc_m:
            uncertainty_parts.append(unc_m.group(1).strip())

        for w in warnings:
            w = str(w).strip()
            if not w:
                continue
            # Never surface internal intent/model JSON dumps to users.
            if w.startswith("raw_model_output=") or "raw_model_output={" in w:
                continue
            if w.startswith("{") and "requires_vision" in w:
                continue
            uncertainty_parts.append(w)

        # Deduplicate while preserving order
        seen: set[str] = set()
        clean_unc: list[str] = []
        for p in uncertainty_parts:
            if p.lower() in {"none", "n/a"}:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            clean_unc.append(p)

        uncertainty = "; ".join(clean_unc) if clean_unc else ""
        return answer, uncertainty

    def _extract_uncertainty(self, answer: str, warnings: list[str]) -> str:
        _, unc = self._present_user_facing_answer(answer, warnings)
        return unc
