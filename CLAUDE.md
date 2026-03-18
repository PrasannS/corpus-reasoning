# CLAUDE.md

## Repository Scope

This repository implements a long-context training and evaluation pipeline for language models. The current focus is fine-tuning models on retrieval-intensive tasks (e.g., needle-in-a-haystack) using multi-GPU training, with plans to expand to a broader variety of long-context reasoning tasks.

## Architecture

- **Training**: Axolotl + DeepSpeed ZeRO for multi-GPU LoRA fine-tuning
- **Evaluation**: vLLM for fast batched inference, comparing base vs. finetuned models
- **Two separate conda environments** to avoid dependency conflicts:
  - `corpus-reasoning` — training (axolotl, deepspeed, flash-attn)
  - `corpus-reasoning-eval` — evaluation (vllm)

## Key Paths

- `configs/` — Axolotl YAML configs and DeepSpeed JSON configs
- `scripts/` — Data generation, training launcher, evaluation
- `data/` — Generated datasets (gitignored)
- `outputs/` — Checkpoints and eval results (gitignored)

## Common Commands

```bash
# Training (use corpus-reasoning env)
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH
bash scripts/run_training.sh

# Evaluation (use corpus-reasoning-eval env)
python scripts/evaluate_niah.py
```

## Conventions

- Training configs go in `configs/` as YAML (axolotl format)
- Data generation scripts go in `scripts/` and write to `data/`
- Dataset format: JSONL with alpaca-style fields (`instruction`, `input`, `output`)
- Local data paths in axolotl configs must point to actual file paths (not HF repo IDs) for local datasets

## Known Issues

- Axolotl 0.15.0 is missing `telemetry/whitelist.yaml` in its package — must be manually created (see README)
- DeepSpeed requires `CUDA_HOME` to be set explicitly on this machine
