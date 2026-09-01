#!/usr/bin/env python3
"""Hardware verification entry point."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running before editable install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuro_agent.paths import configure_hf_cache, ensure_dirs
from neuro_agent.profiling.hardware import main

if __name__ == "__main__":
    configure_hf_cache()
    ensure_dirs()
    main()
