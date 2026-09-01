# Neuro Agent

Post-training and systems optimization of a multimodal neuroscience research agent.

## Goal

Build a capable neuroscience research assistant on **Qwen3-4B-Instruct-2507**, optimized for a single **RTX 4090 (24 GB)** GPU. The project focuses on efficient post-training (QLoRA/SFT, RLVR) and rigorous systems evaluation across precision levels — without making clinical or diagnostic claims.

## Architecture

```
configs/          # YAML configuration (base.yaml)
scripts/          # Environment setup and hardware verification
src/neuro_agent/
  ├── agents/     # Research agent orchestration (stub)
  ├── data/       # Dataset schemas (stub)
  ├── evaluation/ # Systems benchmarking (stub)
  ├── inference/  # Model inference engine (stub)
  ├── profiling/  # Hardware verification and profiling
  ├── quantization/ # PTQ pipeline (stub)
  ├── tools/      # Agent tool interfaces (stub)
  └── training/   # QLoRA/SFT training (stub)
benchmarks/       # Benchmark suites (future)
checkpoints/      # Model checkpoints (persistent)
results/          # Reports and profiling output
```

All Hugging Face caches and checkpoints are stored under `/workspace` (persistent volume), not the ephemeral overlay disk.

## Planned Stages

| Stage | Description |
|-------|-------------|
| **Base** | Scaffold, hardware verification, environment setup |
| **QLoRA / SFT** | 4-bit QLoRA fine-tuning on neuroscience instruction data |
| **RLVR** | Reinforcement learning with verifiable rewards |
| **PTQ** | Post-training quantization (INT8 / INT4) for deployment |

## Systems Evaluation

The project will benchmark across precisions and measure:

- **BF16 / INT8 / INT4** inference latency and throughput
- **KV cache** memory footprint
- **TTFT** (time to first token)
- **Tokens/s** generation speed
- **VRAM** peak usage
- **Profiling** via PyTorch profiler and custom timers

## Quick Start

```bash
# 1. Setup environment (creates venv, installs minimal deps + torch)
bash scripts/setup_env.sh

# 2. Verify GPU hardware
python scripts/verify_hardware.py
# Report saved to results/hardware_verify.json
```

### Minimal install (without full setup script)

```bash
pip install -e .
pip install -e ".[hardware]"
python scripts/verify_hardware.py
```

## Hardware

- **GPU:** 1× NVIDIA RTX 4090 (24 GB VRAM)
- **Default training:** QLoRA (4-bit base + LoRA adapters) via PEFT + bitsandbytes
- **Default dtype:** bfloat16

## Configuration

See `configs/base.yaml` for model, training, inference, and profiling defaults.

## License

MIT
