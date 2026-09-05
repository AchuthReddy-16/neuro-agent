"""Runtime release identification for /api/health (no secrets).

Checkpoint paths are the same relative locations used by model loaders.
Git commit is resolved from the live checkout or known CI/deploy env vars.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from neuro_agent import __version__ as PACKAGE_VERSION
from neuro_agent.paths import PROJECT_ROOT

# Vision path used by AnalysisService VLM loader (single source of truth).
VISION_CHECKPOINT_REL = "checkpoints/multimodal_sft_corrected/final"

# Text path mirrors ResearchAgentConfig.adapter_path (resolved at call time).
TEXT_CHECKPOINT_REL = "checkpoints/sft_corrected_v2/final"

_GIT_ENV_KEYS = (
    "NEURO_GIT_COMMIT",
    "GIT_COMMIT",
    "SOURCE_COMMIT",
    "VERCEL_GIT_COMMIT_SHA",
    "GITHUB_SHA",
    "COMMIT_SHA",
)

_FRONTEND_BUILD_ENV_KEYS = (
    "VERCEL_DEPLOYMENT_ID",
    "NEXT_PUBLIC_VERCEL_GIT_COMMIT_SHA",
    "NEXT_PUBLIC_BUILD_ID",
    "NEURO_FRONTEND_BUILD_ID",
)


def _git_commit_from_repo() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode == 0:
            sha = (proc.stdout or "").strip()
            if sha and all(c in "0123456789abcdef" for c in sha.lower()):
                return sha
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def resolve_git_commit() -> str | None:
    for key in _GIT_ENV_KEYS:
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return _git_commit_from_repo()


def resolve_frontend_build_id() -> str | None:
    for key in _FRONTEND_BUILD_ENV_KEYS:
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return None


def resolve_text_checkpoint_path(*, agent: Any | None = None) -> str:
    """Return the adapter path the text runtime is configured to use."""
    if agent is not None:
        cfg = getattr(agent, "config", None)
        path = getattr(cfg, "adapter_path", None) if cfg is not None else None
        if path:
            return str(path)
    # Lazy import avoids cycles; ResearchAgentConfig is the loader source of truth.
    from neuro_agent.agent.research_agent import ResearchAgentConfig

    return str(ResearchAgentConfig().adapter_path)


def resolve_vision_checkpoint_path() -> str:
    """Return the adapter path the vision runtime loader uses (service.py)."""
    return VISION_CHECKPOINT_REL


def checkpoint_exists(rel_or_abs: str) -> bool:
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.exists()


def build_release_fields(*, agent: Any | None = None) -> dict[str, Any]:
    text_ckpt = resolve_text_checkpoint_path(agent=agent)
    vision_ckpt = resolve_vision_checkpoint_path()
    return {
        "git_commit": resolve_git_commit(),
        "text_checkpoint": text_ckpt,
        "vision_checkpoint": vision_ckpt,
        "runtime": "HF Transformers + LoRA/PEFT",
        "package_version": PACKAGE_VERSION,
        "frontend_build_id": resolve_frontend_build_id(),
    }
