"""FastAPI dependencies and shared application state."""

from __future__ import annotations

import os
from functools import lru_cache

from neuro_agent.api.experiment_store import ExperimentStore
from neuro_agent.api.service import AnalysisService, RuntimeState
from neuro_agent.paths import RESULTS_DIR


@lru_cache(maxsize=1)
def get_settings() -> dict:
    # Prefer NEURO_ALLOWED_ORIGINS; keep NEURO_API_CORS_ORIGINS as alias.
    cors_raw = os.environ.get("NEURO_ALLOWED_ORIGINS") or os.environ.get(
        "NEURO_API_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001",
    )
    return {
        "max_upload_bytes": int(os.environ.get("NEURO_API_MAX_UPLOAD_MB", "25")) * 1024 * 1024,
        "cors_origins": [o.strip() for o in cors_raw.split(",") if o.strip()],
        "serving_mode": os.environ.get("NEURO_SERVING_MODE", "hybrid"),
        "enable_vlm": os.environ.get("NEURO_API_ENABLE_VLM", "0") == "1",
        "load_agent_on_startup": os.environ.get("NEURO_API_LOAD_AGENT", "0") == "1",
        "store_root": os.environ.get(
            "NEURO_API_STORE_ROOT", str(RESULTS_DIR / "api_experiments")
        ),
    }


@lru_cache(maxsize=1)
def get_store() -> ExperimentStore:
    from pathlib import Path

    return ExperimentStore(root=Path(get_settings()["store_root"]))


@lru_cache(maxsize=1)
def get_service() -> AnalysisService:
    settings = get_settings()
    state = RuntimeState(
        serving_mode=settings["serving_mode"],
        vision_enabled=bool(settings["enable_vlm"]),
        vision_status="unloaded" if settings["enable_vlm"] else "disabled",
        precision="BF16",
        text_backend="PrimaryResearchAgent HF Transformers + LoRA (sft_corrected_v2)",
        vision_backend="HF Transformers + PEFT (lazy)",
    )
    svc = AnalysisService(
        store=get_store(),
        state=state,
        enable_vlm=bool(settings["enable_vlm"]),
    )
    return svc


def reset_singletons() -> None:
    """Test helper to clear cached store/service."""
    get_settings.cache_clear()
    get_store.cache_clear()
    get_service.cache_clear()
