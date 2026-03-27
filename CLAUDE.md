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
- `examples/` — Sample prompts showing exactly what the model sees for each task variant (checked in). Regenerate with `python scripts/generate_examples.py`

## Common Commands

```bash
# Generate data (use corpus-reasoning-eval env)
# Data generation defaults to retrieval mode (output doc IDs). Use --no-retrieval for QA mode.
python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20
python scripts/generate_hotpotqa_data.py --num-examples 5000 --num-docs 20 --question-type bridge
python scripts/generate_multi_hotpotqa_data.py --num-examples 1000 --num-queries 10
python scripts/convert_to_qboth.py data/input.jsonl data/output_qboth.jsonl
python scripts/convert_to_qbefore.py data/input.jsonl data/output_qbefore.jsonl

# Train - standard attention (use corpus-reasoning env)
bash scripts/train.sh configs/nq_rag_lora_multigpu.yml

# Train - chunked attention (use corpus-reasoning env)
accelerate launch --num_processes 4 scripts/train_chunked_fast.py configs/nq_rag_chunked_qboth_fast.yml

# Evaluate - standard attention (use corpus-reasoning-eval env)
python scripts/evaluate_helmet_rag.py --datasets nq --num-docs 20
python scripts/evaluate_helmet_rag.py --datasets hotpotqa --num-docs 20 --query-position both --lora-path ./outputs/model

# Evaluate - retrieval tasks (use corpus-reasoning-eval env)
python scripts/evaluate_retrieval.py --eval-data data/nq_train_k20_random_500_retrieval.jsonl --lora-path ./outputs/model
python scripts/evaluate_retrieval.py --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl --lora-path ./outputs/model

# Evaluate - chunked attention (use corpus-reasoning-eval env)
python scripts/evaluate_chunked.py --datasets nq --num-docs 20 --query-position both --lora-path ./outputs/model
```

## SLURM (Batch Jobs)

The cluster has 8x A100 GPUs per node on the `lambda` partition. Use sbatch for batch experiments.

```bash
# Submit a batch job
sbatch scripts/run_batch_part1.sh

# Check job status
squeue -u $(whoami)
sacct -j JOBID --format=JobID,State,ExitCode,Elapsed

# Monitor logs
tail -f outputs/batch_JOBID.log
```

**Key sbatch conventions:**
- Use absolute paths in sbatch scripts (SLURM copies scripts to temp dirs, so `BASH_SOURCE` won't resolve correctly). Set `PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"` explicitly.
- Use `eval "$(conda shell.bash hook)"` before `conda activate` in sbatch scripts.
- Request 4 GPUs per job to run 2 jobs in parallel (8 GPU total limit).
- Standard training uses `accelerate launch -m axolotl.cli.train`; chunked uses `accelerate launch scripts/train_chunked_fast.py`.
- Batch scripts go in `scripts/` (e.g., `run_batch_part1.sh`, `run_hotpotqa_k50.sh`).

## Conventions

- Training configs go in `configs/` as YAML (axolotl format). Task-specific params at top, common LoRA/optimizer settings below.
- Data generation scripts go in `scripts/` and write to `data/`
- Dataset format: JSONL with alpaca-style fields (`instruction`, `input`, `output`)
- Shared code goes in `scripts/lib/` (io.py, vllm_utils.py, metrics.py, common.sh)
- **Experiment tracking**: All experiment results and reproduction instructions go in `results/` as markdown files (one per task). Each file should include: task description, dataset details, config parameters, exact commands to reproduce, and results tables. Update these files whenever a new experiment is run.
- **Eval prompt must exactly match training prompt.** This means:
  - Trained models (LoRA/full FT) use alpaca prompt format + 0 few-shot demos (training data has no demos)
  - Base models use HELMET prompt format + 2 few-shot demos (standard HELMET eval)
  - Eval scripts auto-enforce this: when `use_alpaca=True`, shots is set to 0
  - Any new features added to eval prompts must also be added to training data generation, and vice versa
- **No intermediate checkpoints by default.** Training configs should use `saves_per_epoch: 0` and `save_strategy: "no"` to avoid slow checkpoint saves during training. Axolotl saves the final HF model weights to `output_dir` at the end automatically. Only enable intermediate checkpoints if explicitly requested.
- **Example prompts**: When adding a new task variant or changing prompt format, add an entry to `scripts/generate_examples.py` and regenerate `examples/`. Each example file shows the full alpaca-wrapped prompt the model sees during training/eval, plus the expected output. This makes it easy to visually verify prompt format correctness.

## User Preferences

- **Commit to git frequently.** After completing a meaningful unit of work (new script, config, experiment results, bug fix), commit the changes. Don't let work accumulate uncommitted across long sessions.
- **Always log progress for long-running processes.** When running downloads, extractions, training, or any task that takes more than ~30 seconds, ensure there is a way for the user to monitor progress (e.g., `--progress`, `pv`, `tqdm`, writing to a log file with `tee`, or periodic status lines). Provide the user a `tail -f` command or similar to watch the output. Avoid approaches that flood Claude's context with huge outputs — prefer writing to a file the user can `tail`.
- **Never suppress command output with `| tail -N` for background commands.** When running commands in the background, always use `tee` to write to a log file so partial output is visible. Piping through `tail` hides all output until the command finishes, making it impossible to monitor progress or see errors early.

## Known Issues

- Axolotl 0.15.0 is missing `telemetry/whitelist.yaml` in its package — must be manually created (see README)
- DeepSpeed requires `CUDA_HOME` to be set explicitly on this machine
