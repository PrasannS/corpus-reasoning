# CLAUDE.md

## Repository Scope

Long-context training and evaluation pipeline for language models. Fine-tunes models on retrieval and reasoning tasks using multi-GPU training, evaluates with vLLM.

## Architecture

- **Training**: Axolotl + DeepSpeed ZeRO for multi-GPU LoRA fine-tuning
- **Evaluation**: vLLM for fast batched inference, comparing base vs. finetuned models
- **Two separate conda environments** to avoid dependency conflicts:
  - `corpus-reasoning` — training (axolotl, deepspeed, flash-attn)
  - `corpus-reasoning-eval` — evaluation and data generation (vllm, datasets)

## Key Paths

- `configs/` — Axolotl YAML configs and DeepSpeed JSON configs
- `scripts/` — Data generation, training, evaluation
  - `scripts/lib/` — Shared utilities (I/O, vLLM, metrics, shell helpers)
- `data/` — Generated datasets (gitignored)
- `outputs/` — Checkpoints and eval results (gitignored)
- `results/` — Experiment results and reproduction instructions (checked in)

## Common Commands

```bash
# Generate data (use corpus-reasoning-eval env)
python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20
python scripts/generate_contradiction_data.py --num-claims 100 --num-contradictions 3

# Train (use corpus-reasoning env)
bash scripts/train.sh configs/nq_rag_lora_multigpu.yml

# Evaluate (use corpus-reasoning-eval env)
python scripts/evaluate_helmet_rag.py --datasets nq --num-docs 20
python scripts/evaluate_contradiction.py --eval-data data/contradiction_eval_n100_k3.jsonl
```

## Conventions

- Training configs go in `configs/` as YAML (axolotl format). Task-specific params at top, common LoRA/optimizer settings below.
- Data generation scripts go in `scripts/` and write to `data/`
- Dataset format: JSONL with alpaca-style fields (`instruction`, `input`, `output`)
- Shared code goes in `scripts/lib/` (io.py, vllm_utils.py, metrics.py, common.sh)
- **Experiment tracking**: All experiment results and reproduction instructions go in `results/` as markdown files (one per task). Each file should include: task description, dataset details, config parameters, exact commands to reproduce, and results tables. Update these files whenever a new experiment is run.

## User Preferences

- **Always log progress for long-running processes.** When running downloads, extractions, training, or any task that takes more than ~30 seconds, ensure there is a way for the user to monitor progress (e.g., `--progress`, `pv`, `tqdm`, writing to a log file with `tee`, or periodic status lines). Provide the user a `tail -f` command or similar to watch the output. Avoid approaches that flood Claude's context with huge outputs — prefer writing to a file the user can `tail`.

## Known Issues

- Axolotl 0.15.0 is missing `telemetry/whitelist.yaml` in its package — must be manually created (see README)
- DeepSpeed requires `CUDA_HOME` to be set explicitly on this machine
