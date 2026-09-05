"""FastAPI application — production research-agent backend."""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from neuro_agent.api import dependencies as api_deps
from neuro_agent.api.schemas import (
    APIErrorBody,
    AnalyzeRequest,
    AnalyzeResponse,
    EEGMetadata,
    ExperimentMetadata,
    ExperimentResponse,
    HealthResponse,
    SystemMetricsResponse,
    UploadResponse,
    UploadedArtifact,
    VisualizationInfo,
)
from neuro_agent.api.service import (
    PRECISION,
    TEXT_MODEL_LABEL,
    VISION_MODEL_LABEL,
    VisionRuntimeError,
)
from neuro_agent.tools.schemas import SampleNotFoundError

ALLOWED_FIGURE_EXT = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_META_EXT = {".json"}
# EEG: only JSON metadata that references an existing processed sample_id.
# Raw EDF/CSV/NPY uploads are rejected — the project tools read processed HDF5/parquet.


def create_app() -> FastAPI:
    settings = api_deps.get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        svc = api_deps.get_service()
        try:
            svc.initialize_tools()
            svc.ensure_demo_experiment()
        except Exception:
            pass
        if settings["load_agent_on_startup"]:
            try:
                svc.ensure_agent()
            except Exception as exc:
                svc.state.agent_loaded = False
                svc.state.text_status = "error"
                svc.state.text_error = str(exc)
        yield

    app = FastAPI(
        title="Neuro-Agent Research API",
        version="0.1.0",
        description="FastAPI backend for the neuroscience research agent .",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(HTTPException)
    async def http_exc_handler(_request: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            body = APIErrorBody(
                error=str(detail.get("error") or detail.get("code") or "error"),
                detail=detail.get("detail"),
                code=detail.get("code"),
            )
        else:
            body = APIErrorBody(error=str(detail), detail=None)
        return JSONResponse(status_code=exc.status_code, content=body.model_dump())

    @app.exception_handler(Exception)
    async def unhandled_exc_handler(_request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content=APIErrorBody(
                error="internal_error",
                detail="An internal error occurred",
                code="internal_error",
            ).model_dump(),
        )

    # ----- health / metrics -----
    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        from neuro_agent.api.release_manifest import build_release_fields

        svc = api_deps.get_service()
        st = svc.state
        mock = svc.mock_runner is not None
        text_status = "ready" if mock else st.text_status
        vision_status = st.vision_status
        if mock:
            overall = "ok"
        elif text_status == "ready":
            overall = "ok"
        elif text_status == "error":
            overall = "unavailable"
        else:
            # Process is up; text model not yet ready (lazy / loading).
            overall = "degraded"
        release = build_release_fields(agent=None if mock else st.agent)
        return HealthResponse(
            status=overall,  # type: ignore[arg-type]
            text_model=TEXT_MODEL_LABEL,
            vision_model=VISION_MODEL_LABEL,
            serving_mode=st.serving_mode,
            agent_loaded=bool(st.agent_loaded or mock),
            vision_loaded=bool(st.vision_loaded),
            text_status=text_status,  # type: ignore[arg-type]
            vision_status=vision_status,  # type: ignore[arg-type]
            vision_enabled=bool(svc.enable_vlm),
            text_backend=st.text_backend,
            vision_backend=st.vision_backend,
            precision=st.precision or PRECISION,
            text_error=st.text_error,
            vision_error=st.vision_error,
            git_commit=release["git_commit"],
            text_checkpoint=release["text_checkpoint"],
            vision_checkpoint=release["vision_checkpoint"],
            runtime=release["runtime"],
            package_version=release["package_version"],
            frontend_build_id=release["frontend_build_id"],
        )

    @app.get("/api/system/metrics", response_model=SystemMetricsResponse)
    def system_metrics() -> SystemMetricsResponse:
        svc = api_deps.get_service()
        gpu = svc.gpu_stats()
        return SystemMetricsResponse(
            model=TEXT_MODEL_LABEL,
            vision_model=VISION_MODEL_LABEL,
            post_training="corrected SFT (+ optional RLVR lineage)",
            serving=svc.state.serving_mode,
            precision=svc.state.precision or PRECISION,
            ttft_ms=svc.state.last_ttft_ms,
            tokens_per_sec=None,
            p95_latency_ms=None,
            gpu_utilization_pct=gpu.get("gpu_utilization_pct"),
            gpu_memory_used_mb=gpu.get("gpu_memory_used_mb"),
            gpu_memory_total_mb=gpu.get("gpu_memory_total_mb"),
            last_request_latency_ms=svc.state.last_request_latency_ms,
            route=svc.state.last_route,  # type: ignore[arg-type]
            verifier_status=svc.state.last_verifier_status,
            serving_mode=svc.state.serving_mode,
        )

    # ----- upload -----
    @app.post("/api/upload", response_model=UploadResponse)
    async def upload(
        file: UploadFile = File(...),
        fileType: str = Form(...),
        filename: str | None = Form(None),
        experiment_id: str | None = Form(None),
    ) -> UploadResponse:
        settings = api_deps.get_settings()
        store = api_deps.get_store()
        svc = api_deps.get_service()

        name = filename or file.filename
        if not name or not str(name).strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "missing_filename", "code": "missing_filename"},
            )
        name = Path(str(name)).name
        ftype = (fileType or "").strip().lower()
        if ftype not in {"eeg", "figure", "metadata"}:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "unsupported_content",
                    "detail": "fileType must be eeg|figure|metadata",
                    "code": "unsupported_content",
                },
            )

        content = await file.read()
        if not content:
            raise HTTPException(
                status_code=400,
                detail={"error": "empty_file", "code": "empty_file"},
            )
        if len(content) > settings["max_upload_bytes"]:
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "file_too_large",
                    "detail": f"max {settings['max_upload_bytes']} bytes",
                    "code": "file_too_large",
                },
            )

        ext = Path(name).suffix.lower()
        detected: list[str] = []
        eeg_meta = None
        exp_meta = None
        visuals: list[VisualizationInfo] = []

        if ftype == "figure":
            if ext not in ALLOWED_FIGURE_EXT:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "unsupported_file",
                        "detail": f"figure must be one of {sorted(ALLOWED_FIGURE_EXT)}",
                        "code": "unsupported_file",
                    },
                )
            detected.append("vision")
        elif ftype in {"eeg", "metadata"}:
            # Only JSON that references an existing processed sample is accepted.
            if ext not in ALLOWED_META_EXT:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "unsupported_file",
                        "detail": (
                            "EEG/raw EDF/CSV/NPY uploads are not parsed by this backend. "
                            "Upload a JSON metadata file with sample_id referencing "
                            "processed data (e.g. S001_R01_E000), or use the demo experiment."
                        ),
                        "code": "unsupported_file",
                    },
                )
            try:
                payload = json.loads(content.decode("utf-8"))
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "malformed_input", "code": "malformed_input"},
                )
            sample_id = payload.get("sample_id")
            if not sample_id:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "malformed_input",
                        "detail": "JSON must include sample_id",
                        "code": "malformed_input",
                    },
                )
            try:
                from neuro_agent.tools.metadata import lookup_sample_metadata

                meta = lookup_sample_metadata(sample_id=str(sample_id))
            except SampleNotFoundError:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "unknown_sample_id",
                        "detail": f"sample_id {sample_id} not in processed registry",
                        "code": "unknown_sample_id",
                    },
                )
            detected.extend(["eeg", "metadata"])
            eeg_meta = EEGMetadata(
                filename=name,
                format="json",
                sampling_rate_hz=meta.sampling_rate_hz,
                channels=len(meta.channels),
                channel_labels=list(meta.channels),
                sample_id=meta.sample_id,
                auto_detected=True,
            )
            exp_meta = ExperimentMetadata(
                subject=meta.subject_id,
                run=meta.run_id,
                task_type=meta.task_type,
                movement_condition=meta.condition,
                sampling_rate_hz=meta.sampling_rate_hz,
                channels=len(meta.channels),
                sample_id=meta.sample_id,
            )
            visuals = svc.visualizations_for_sample(meta.sample_id)
            if visuals:
                detected.append("vision")

        if experiment_id:
            rec = store.get(experiment_id)
            if rec is None:
                raise HTTPException(
                    status_code=404,
                    detail={"error": "invalid_experiment_id", "code": "invalid_experiment_id"},
                )
        else:
            rec = store.create()

        artifact = store.store_upload(
            rec.id,
            filename=name,
            content=content,
            kind=ftype if ftype != "metadata" else "metadata",
            content_type=file.content_type,
        )

        # refresh record
        rec = store.get(rec.id)
        assert rec is not None

        if ftype in {"eeg", "metadata"} and eeg_meta and eeg_meta.sample_id:
            rec = svc.attach_sample_to_experiment(rec, eeg_meta.sample_id)
            visuals = [VisualizationInfo.model_validate(v) for v in rec.visualizations]
        elif ftype == "figure":
            rec = store.get(rec.id)
            assert rec is not None
            rec.modalities["vision"] = True
            vid = artifact["id"]
            for a in rec.artifacts:
                if a["id"] == vid:
                    a["image_id"] = vid
                    artifact = a
                    break
            vis = VisualizationInfo(
                id=vid,
                tab="figure",
                title=name,
                image_url=f"/api/visualization/{vid}",
                index=len(rec.visualizations),
                image_path=artifact.get("stored_path"),
            )
            rec.visualizations.append(vis.model_dump(by_alias=False))
            if vid not in rec.linked_image_ids:
                rec.linked_image_ids.append(vid)
            rec.status = "ready"
            store.save(rec)
            visuals = [VisualizationInfo.model_validate(v) for v in rec.visualizations]

        return UploadResponse(
            experiment_id=rec.id,
            asset_id=artifact["id"],
            uploaded_artifacts=[UploadedArtifact.model_validate(a) for a in rec.artifacts],
            detected_input_types=sorted(set(detected)),
            available_visualizations=visuals,
            metadata=ExperimentMetadata.model_validate(rec.metadata) if rec.metadata else exp_meta,
            eeg=EEGMetadata.model_validate(rec.eeg) if rec.eeg else eeg_meta,
            status="ready",
        )

    # ----- experiment / visualization -----
    @app.post("/api/experiment", response_model=ExperimentResponse)
    def create_experiment() -> ExperimentResponse:
        """Create an empty live experiment session (Chat / Workspace)."""
        store = api_deps.get_store()
        rec = store.create()
        return ExperimentResponse(
            id=rec.id,
            experiment_id=rec.id,
            status=rec.status or "empty",
            is_demo=False,
            files=[],
            visualizations=[],
            modalities={"eeg": False, "metadata": False, "vision": False, "text": True},
            analysis_history=[],
        )

    @app.get("/api/experiment/{experiment_id}", response_model=ExperimentResponse)
    def get_experiment(experiment_id: str) -> ExperimentResponse:
        store = api_deps.get_store()
        # alias demo
        if experiment_id in {"demo", "exp_demo"}:
            experiment_id = "exp_demo_s001"
        rec = store.get(experiment_id)
        if rec is None:
            raise HTTPException(
                status_code=404,
                detail={"error": "invalid_experiment_id", "code": "invalid_experiment_id"},
            )
        figure = None
        for art in rec.artifacts:
            if art.get("kind") == "figure":
                figure = {
                    "id": art.get("image_id") or art["id"],
                    "filename": art["name"],
                    "url": f"/api/visualization/{art.get('image_id') or art['id']}",
                    "type": "figure",
                    "label": art["name"],
                }
                break
        return ExperimentResponse(
            id=rec.id,
            eeg=EEGMetadata.model_validate(rec.eeg) if rec.eeg else None,
            figure=figure,
            metadata=ExperimentMetadata.model_validate(rec.metadata or {}),
            visualizations=[VisualizationInfo.model_validate(v) for v in rec.visualizations],
            modalities=rec.modalities,
            files=[UploadedArtifact.model_validate(a) for a in rec.artifacts],
            status=rec.status,  # type: ignore[arg-type]
            is_demo=rec.is_demo,
            error_message=rec.error_message,
            analysis_history=rec.analysis_history,
        )

    @app.get("/api/visualization/{visualization_id}")
    def get_visualization(visualization_id: str, request: Request):
        svc = api_deps.get_service()
        accept = request.headers.get("accept", "")
        info = svc.resolve_visualization(visualization_id)
        path = svc.visualization_file_path(visualization_id)

        # Prefer JSON metadata when client asks for JSON; otherwise stream image bytes
        wants_json = "application/json" in accept and "image/" not in accept
        if wants_json or path is None:
            if info is None and path is None:
                raise HTTPException(
                    status_code=404,
                    detail={"error": "visualization_not_found", "code": "visualization_not_found"},
                )
            if info is None:
                # uploaded file without registry record
                info = VisualizationInfo(
                    id=visualization_id,
                    tab="figure",
                    title=visualization_id,
                    image_url=f"/api/visualization/{visualization_id}",
                    index=0,
                    image_path=str(path) if path else None,
                )
            return info

        media = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        return FileResponse(path, media_type=media, filename=path.name)

    # ----- analyze -----
    @app.post("/api/analyze", response_model=AnalyzeResponse)
    def analyze(body: AnalyzeRequest) -> AnalyzeResponse:
        svc = api_deps.get_service()
        if not body.question or not body.question.strip():
            raise HTTPException(
                status_code=400,
                detail={"error": "missing_question", "code": "missing_question"},
            )
        exp_id = body.experiment_id
        if exp_id in {"demo", "exp_demo"}:
            exp_id = "exp_demo_s001"
        try:
            return svc.analyze(
                experiment_id=exp_id,
                question=body.question,
                image_id=body.image_id,
                visualization_id=body.visualization_id,
                context=body.context,
                conversation_history=body.conversation_history,
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={"error": "invalid_experiment_id", "code": "invalid_experiment_id"},
            )
        except FileNotFoundError as exc:
            if "missing_image" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "missing_image_for_vision",
                        "detail": "Vision-required question but no image is linked to this experiment",
                        "code": "missing_image_for_vision",
                    },
                )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "vlm_unavailable",
                    "detail": str(exc),
                    "code": "vlm_unavailable",
                },
            )
        except VisionRuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "vlm_unavailable",
                    "detail": str(exc),
                    "code": "vlm_unavailable",
                },
            )
        except TimeoutError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "vlm_timeout",
                    "detail": str(exc),
                    "code": "vlm_timeout",
                },
            )
        except ValueError as exc:
            if "missing_question" in str(exc):
                raise HTTPException(
                    status_code=400,
                    detail={"error": "missing_question", "code": "missing_question"},
                )
            raise
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "cuda" in msg or "out of memory" in msg or "vlm" in msg:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "error": "model_backend_unavailable",
                        "detail": str(exc),
                        "code": "model_backend_unavailable",
                    },
                )
            raise HTTPException(
                status_code=500,
                detail={"error": "tool_execution_failure", "detail": "analysis failed", "code": "tool_execution_failure"},
            )

    return app


# ASGI entry
app = create_app()
