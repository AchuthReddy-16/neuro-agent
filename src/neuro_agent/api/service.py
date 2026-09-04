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
from neuro_agent.tools.vision_evidence import resolve_vision_evidence

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
    ) -> AnalyzeResponse:
        q = (question or "").strip()
        if not q:
            raise ValueError("missing_question")

        rec = self.store.get(experiment_id)
        if rec is None:
            raise KeyError("invalid_experiment_id")

        selected_image = image_id or visualization_id
        # Do NOT auto-pick a linked topomap/figure merely because the experiment has one.
        # That previously injected [image_id=…] into the prompt and forced a false VISION route.

        t0 = time.perf_counter()
        agent = self.ensure_agent()

        # Enrich with sample context only — never inject an image_id unless the client
        # explicitly selected one (and even then, routing still requires visual intent).
        enriched = q
        if rec.linked_sample_id and rec.linked_sample_id not in q:
            enriched = f"{q} (sample {rec.linked_sample_id})"
        if selected_image and selected_image not in enriched and self.question_has_visual_language(q):
            enriched = f"{enriched} [image_id={selected_image}]"

        trace = agent.ask(enriched)
        total_ms = (time.perf_counter() - t0) * 1000.0

        raw_intent = getattr(trace, "parsed_intent", None) or {}
        requires_vision = self.decide_requires_vision(
            question=q,
            raw_intent=raw_intent if isinstance(raw_intent, dict) else {},
            image_id=image_id,
            visualization_id=visualization_id,
            context=context,
        )

        vision_ms = 0.0
        vlm_text: str | None = None
        visual_items: list[VisualEvidenceItem] = []

        # Sidecar vision evidence from tools / index
        evidence = getattr(trace, "evidence_bundle", None) or {}
        for ve in evidence.get("vision_evidence") or []:
            fam = ve.get("family") or "figure"
            tab = FAMILY_TO_TAB.get(fam, "figure")
            iid = ve.get("image_id") or "unknown"
            visual_items.append(
                VisualEvidenceItem(
                    id=iid,
                    label=tab,
                    tab=tab,
                    observation=None,
                    image_url=f"/api/visualization/{iid}",
                    image_type=tab,
                    vlm_interpretation=None,
                    provenance=ve.get("image_path"),
                )
            )

        # Resolve selected / sample visuals when vision is required or for linked display.
        resolve_image = selected_image
        if requires_vision and resolve_image is None and rec.linked_image_ids:
            # Prefer an uploaded figure, else first linked sample visualization.
            for art in rec.artifacts:
                if art.get("kind") == "figure" and art.get("image_id"):
                    resolve_image = art["image_id"]
                    break
            if resolve_image is None:
                resolve_image = rec.linked_image_ids[0]

        if not visual_items and (resolve_image or (requires_vision and rec.linked_sample_id)):
            try:
                refs = resolve_vision_evidence(
                    sample_id=rec.linked_sample_id,
                    image_id=resolve_image,
                    visual_type=raw_intent.get("requested_visual_type")
                    if requires_vision
                    else None,
                )
            except (SampleNotFoundError, ValueError):
                refs = []
                # Fall back: any linked experiment visualization (no fabrication)
                for vid in rec.linked_image_ids:
                    info = self.resolve_visualization(vid)
                    if info is None and rec.visualizations:
                        # try matching stored viz dicts
                        for v in rec.visualizations:
                            if v.get("id") == vid:
                                info = VisualizationInfo.model_validate(v)
                                break
                    if info is not None:
                        visual_items.append(
                            VisualEvidenceItem(
                                id=info.id,
                                label=info.tab,
                                tab=info.tab,
                                image_url=info.image_url or f"/api/visualization/{info.id}",
                                image_type=info.tab,
                                provenance=info.image_path,
                            )
                        )
                if not visual_items:
                    for v in rec.visualizations:
                        info = VisualizationInfo.model_validate(v)
                        visual_items.append(
                            VisualEvidenceItem(
                                id=info.id,
                                label=info.tab,
                                tab=info.tab,
                                image_url=info.image_url or f"/api/visualization/{info.id}",
                                image_type=info.tab,
                                provenance=info.image_path,
                            )
                        )
            else:
                for ref in refs:
                    tab = FAMILY_TO_TAB.get(ref.family, "figure")
                    visual_items.append(
                        VisualEvidenceItem(
                            id=ref.image_id,
                            label=tab,
                            tab=tab,
                            image_url=f"/api/visualization/{ref.image_id}",
                            image_type=tab,
                            provenance=ref.image_path,
                        )
                    )

        if requires_vision and self.enable_vlm and visual_items:
            t_v = time.perf_counter()
            try:
                vlm_text = self._run_vlm(q, visual_items[0])
                visual_items[0].vlm_interpretation = vlm_text
                visual_items[0].observation = vlm_text
            except Exception as exc:  # noqa: BLE001
                vision_ms = (time.perf_counter() - t_v) * 1000.0
                # Hard fail when VLM is enabled — do not invent interpretation text.
                raise VisionRuntimeError(str(exc)) from exc
            vision_ms = (time.perf_counter() - t_v) * 1000.0
        elif requires_vision and self.enable_vlm and not visual_items:
            raise FileNotFoundError("missing_image_for_vision")
        elif requires_vision and not visual_items:
            raise FileNotFoundError("missing_image_for_vision")

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
        )

        self.state.last_request_latency_ms = response.timing.total_ms
        self.state.last_route = response.route
        self.state.last_verifier_status = response.verification.status
        if response.timing.routing_ms is not None:
            self.state.last_ttft_ms = response.timing.routing_ms

        self.store.append_analysis(
            experiment_id,
            {
                "id": response.id,
                "question": q,
                "route": response.route,
                "answer_preview": (response.answer or "")[:240],
                "total_ms": response.timing.total_ms,
            },
        )
        return response

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
                cfg = InferenceConfig(
                    model_name="Qwen/Qwen2.5-VL-3B-Instruct",
                    dtype="bfloat16",
                    trust_remote_code=True,
                    adapter_path=str(
                        PROJECT_ROOT / "checkpoints" / "multimodal_sft_corrected" / "final"
                    ),
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

        computed = self._extract_computed_evidence(evidence, tools)
        answer = getattr(trace, "final_answer", None) or ""
        # When the VISION path produced a real VLM interpretation but the text
        # synthesizer returned nothing, surface the VLM text as the answer
        # (do not invent unrelated prose).
        if requires_vision and vlm_text and not str(answer).strip():
            answer = vlm_text
        uncertainty = self._extract_uncertainty(answer, getattr(trace, "warnings", []) or [])

        ver_triggered = bool(getattr(trace, "verification_triggered", False))
        recovery = getattr(trace, "recovery", None)
        recovery_triggered = recovery is not None
        final_ver = getattr(trace, "final_verification", None)
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

        interpretation = answer
        if vlm_text:
            interpretation = f"{answer}\n\nVision model: {vlm_text}".strip()

        return AnalyzeResponse(
            answer=answer,
            route=route,  # type: ignore[arg-type]
            computed_evidence=computed,
            visual_evidence=visual_items,
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
            raw_tool_output=json.dumps(evidence, default=str)[:8000] if evidence else None,
            route_detail=RouteInfo(
                intent=raw_intent or None,
                requires_vision=requires_vision,
                requested_visual_type=raw_intent.get("requested_visual_type"),
                question_type=raw_intent.get("question_type"),
            ),
            experiment_id=experiment_id,
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
            for i, ch in enumerate(list(ranking)[:5]):
                items.append(
                    ComputedEvidenceItem(
                        label=f"rank {i+1}",
                        value=str(ch),
                        unit=None,
                        tool="rank_channels_for_sample" if "rank" in ",".join(tools) else (tools[0] if tools else None),
                        channel=str(ch),
                        highlight=i == 0,
                        provenance="evidence_bundle.ranked_evidence",
                    )
                )
                if ch in values:
                    items.append(
                        ComputedEvidenceItem(
                            label=f"{ch} value",
                            value=str(values[ch]),
                            unit=ranked.get("units"),
                            channel=str(ch),
                            provenance="evidence_bundle.ranked_evidence.values",
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
            items.append(
                ComputedEvidenceItem(
                    label="condition comparison",
                    value=str(cond.get("winner") or cond.get("summary") or cond),
                    condition=str(cond.get("condition_a") or ""),
                    provenance="evidence_bundle.condition_evidence",
                )
            )

        return items

    def _extract_uncertainty(self, answer: str, warnings: list[str]) -> str:
        m = re.search(r"Uncertainty:\s*(.+)$", answer, flags=re.IGNORECASE | re.MULTILINE)
        parts = []
        if m:
            parts.append(m.group(1).strip())
        parts.extend(warnings)
        return "; ".join(p for p in parts if p) or "None"
