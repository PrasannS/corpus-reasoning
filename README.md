# Corpus Reasoning: Long-Context Training

Multi-GPU fine-tuning pipeline for long-context reasoning tasks using [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) with DeepSpeed.

## Hardware Requirements

- NVIDIA GPUs with Ampere architecture or newer (tested on 4x A100-80GB)
- CUDA 12.8+

## Environment Setup

### 1. Create the conda environment

```bash
conda create -n corpus-reasoning python=3.11 -y
conda activate corpus-reasoning
```

### 2. Install PyTorch (CUDA 12.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install build prerequisites

```bash
pip install -U packaging setuptools wheel ninja psutil
```

### 4. Install Axolotl with DeepSpeed

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
pip install --no-build-isolation "axolotl[deepspeed]"
```

### 5. Install Flash Attention

```bash
pip install --no-build-isolation flash-attn
```

### 6. Fix telemetry whitelist (axolotl 0.15.0 packaging bug)

```bash
AXOLOTL_PKG=$(python -c "import axolotl; import os; print(os.path.dirname(axolotl.__file__))")
echo 'organizations: []' > "$AXOLOTL_PKG/telemetry/whitelist.yaml"
```

### 7. Verify installation

```bash
export CUDA_HOME=/usr/local/cuda-12.8
python -c "
import axolotl; print(f'axolotl={axolotl.__version__}')
import torch; print(f'torch={torch.__version__}, gpus={torch.cuda.device_count()}')
import deepspeed; print(f'deepspeed={deepspeed.__version__}')
import flash_attn; print(f'flash_attn={flash_attn.__version__}')
"
```

Expected output (versions may vary):
```
axolotl=0.15.0
torch=2.10.0+cu128, gpus=4
deepspeed=0.18.2
flash_attn=2.8.3
```

## Project Structure

```
├── configs/
│   ├── niah_lora_multigpu.yml     # Axolotl training config (LoRA + DeepSpeed ZeRO-1)
│   └── deepspeed_zero1.json       # DeepSpeed ZeRO Stage 1 config
├── scripts/
│   ├── generate_niah_data.py      # Synthetic NIAH data generator
│   ├── run_training.sh            # Multi-GPU training launcher
│   ├── evaluate_niah.py           # vLLM-based NIAH evaluation (base vs. LoRA)
│   └── evaluate_helmet_rag.py     # vLLM-based HELMET RAG evaluation
├── data/                          # Generated training data (gitignored)
│   └── data/kilt/                 # HELMET KILT RAG data (downloaded separately)
└── outputs/                       # Model checkpoints & eval results (gitignored)
```

## Quick Start: Needle-in-a-Haystack Task

This repo includes a toy needle-in-a-haystack (NIAH) task where a factual "needle" is hidden among filler paragraphs, and the model must retrieve it.

### 1. Generate training data

```bash
python scripts/generate_niah_data.py --output-dir data
```

Options:
- `--num-train 500` — number of training examples
- `--num-val 50` — number of validation examples
- `--min-paragraphs 5` / `--max-paragraphs 15` — filler paragraph range
- `--seed 42` — random seed

### 2. Run multi-GPU training

```bash
# Set environment
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Run on all available GPUs
bash scripts/run_training.sh

# Or run directly with a specific GPU count
accelerate launch --num_processes 4 \
    -m axolotl.cli.train configs/niah_lora_multigpu.yml
```

### 3. Training output

Checkpoints and the final LoRA adapter are saved to `outputs/niah-lora/`.

### 4. Evaluate with vLLM

Evaluation uses a **separate conda environment** to avoid dependency conflicts with the training env.

```bash
# One-time setup
conda create -n corpus-reasoning-eval python=3.11 -y
conda activate corpus-reasoning-eval
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install vllm
```

```bash
# Run evaluation (compares base model vs. LoRA-finetuned)
conda activate corpus-reasoning-eval
python scripts/evaluate_niah.py
```

Options:
- `--base-model NousResearch/Llama-3.2-1B` — base model
- `--lora-path outputs/niah-lora` — path to LoRA adapter (omit to eval base only)
- `--eval-data data/niah_val.jsonl` — evaluation data
- `--tensor-parallel-size 1` — number of GPUs for inference
- `--output-file outputs/eval_results.json` — where to save results

Example output:
```
=== Comparison ===
Metric                     Base       LoRA      Delta
----------------------------------------------------
exact_match              10.0%     98.0%    +88.0%
substring_match          86.0%     98.0%    +12.0%
```

## HELMET RAG Evaluation

Evaluate models on the four KILT-based RAG benchmarks from [HELMET](https://github.com/princeton-nlp/HELMET): Natural Questions, TriviaQA, HotpotQA, and PopQA. Uses HELMET's data and replicates their prompt format and metrics without depending on their codebase.

### 1. Download HELMET data

```bash
wget -c https://huggingface.co/datasets/princeton-nlp/HELMET/resolve/main/data.tar.gz -O data/helmet_data.tar.gz
cd data && tar xzf helmet_data.tar.gz 'data/kilt/'
```

This extracts ~22GB of KILT RAG data with varying document counts (20, 50, 105, 220, 440, 500, 1000 docs per question).

### Context length statistics

Average context tokens per question (Llama-3 tokenizer, measured over 50 samples):

| Docs | NQ | TriviaQA | HotpotQA | PopQA |
|---:|---:|---:|---:|---:|
| 20 | 2,304 | 2,453 | 2,356 | 2,087 |
| 50 | 5,820 | 6,167 | 5,820 | 5,365 |
| 105 | 12,346 | 12,913 | 12,294 | 11,705 |
| 220 | 25,950 | 26,963 | 25,995 | 24,838 |
| 440 | 52,094 | 53,950 | 52,328 | 50,129 |
| 1000 | 118,467 | 122,502 | 119,703 | 115,674 |

### 2. Run evaluation

```bash
conda activate corpus-reasoning-eval
python scripts/evaluate_helmet_rag.py \
    --base-model NousResearch/Llama-3.2-1B \
    --datasets nq,triviaqa,hotpotqa,popqa \
    --max-test-samples 100 \
    --num-docs 20 \
    --max-model-len 4096 \
    --shots 2
```

Options:
- `--datasets nq,triviaqa,hotpotqa,popqa` — comma-separated list of benchmarks
- `--num-docs 20` — number of retrieved documents per question (available: 20, 50, 105, 220, 440, 500, 1000)
- `--max-test-samples 100` — cap on evaluation examples per dataset
- `--shots 2` — number of few-shot demos
- `--max-model-len 4096` — model context window (increase for more docs)
- `--lora-path outputs/niah-lora` — optional LoRA adapter to evaluate
- `--output-file outputs/helmet_rag_results.json` — where to save results

### 3. Metrics

Follows HELMET's evaluation protocol:
- **Exact Match (EM)** — normalized prediction matches a gold answer exactly
- **Substring Exact Match** — a gold answer appears as a substring of the prediction
- **F1** — token-level F1 between prediction and best-matching gold answer

Baseline results (Llama-3.2-1B, 2-shot):

| Dataset | Docs | ~Context Tokens | EM | Substring EM | F1 |
|---|---:|---:|---|---|---|
| NQ | 20 | 2.3k | 30.0% | 32.0% | 38.8% |
| NQ | 220 | 26k | 28.0% | 31.0% | 35.5% |
| TriviaQA | 20 | 2.5k | 48.0% | 48.0% | 53.8% |
| HotpotQA | 20 | 2.4k | 38.0% | 40.0% | 52.4% |
| PopQA | 20 | 2.1k | 42.0% | 46.0% | 48.9% |

## Training Configuration

The default config (`configs/niah_lora_multigpu.yml`) trains a LoRA adapter on Llama-3.2-1B:

| Parameter | Value |
|---|---|
| Base model | `NousResearch/Llama-3.2-1B` |
| Adapter | LoRA (r=16, alpha=32) |
| Sequence length | 4096 |
| Micro batch size | 2 |
| Gradient accumulation | 4 |
| Epochs | 2 |
| Optimizer | AdamW 8-bit |
| Learning rate | 2e-4 |
| Sharding | DeepSpeed ZeRO Stage 1 |
| Flash attention | enabled |
| Sample packing | enabled |

## Customization

### Switching sharding strategies

Edit `configs/niah_lora_multigpu.yml`:

```yaml
# DeepSpeed ZeRO-1 (default, least memory overhead)
deepspeed: configs/deepspeed_zero1.json

# Or use FSDP2 instead (remove deepspeed line, add):
# fsdp_version: 2
# fsdp_config:
#   offload_params: false
#   auto_wrap_policy: TRANSFORMER_BASED_WRAP
```

### Scaling to larger models / longer contexts

- Increase `sequence_len` for longer contexts
- Switch to DeepSpeed ZeRO-2 or ZeRO-3 for larger models
- Reduce `micro_batch_size` if running out of VRAM
- Use `axolotl fetch deepspeed_configs` to get all available DeepSpeed configs


# Yichuan's Suggestions
- Have separate shells, use git worktree to have multiple agents
- Have a TODO list / progress / record of all experimentso