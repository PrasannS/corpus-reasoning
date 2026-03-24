# NQ RAG Experiments

## Task Description

Fine-tune Llama-3.2-1B on Natural Questions with retrieval-augmented contexts (HELMET format). Each training example has 1 gold document among N-1 distractors. Evaluated on HELMET's KILT NQ eval set.

## Datasets

- **Training data source**: `tilyupo/nq_cqa` (HuggingFace)
- **Eval data source**: HELMET KILT NQ (`data/data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl`)
- **Format**: Axolotl alpaca (instruction/input/output) with HELMET-compatible prompts

## Summary (NQ, 20 docs, 100 eval samples, 2-shot)

### Query position × Attention type ablation (2.5k training examples)

Base model uses HELMET prompt format; LoRA models use alpaca prompt format (matching training).

| Attention | Query Position | Base EM | Base F1 | LoRA EM | LoRA F1 |
|---|---|---|---|---|---|
| Standard | after (default) | 29.0% | 39.9% | 33.0% | 44.3% |
| Standard | before | 1.0% | 6.5% | 34.0% | 41.0% |
| Standard | both | 28.0% | 38.6% | **41.0%** | **51.9%** |
| Chunked | after | 9.0% | 22.4% | 30.0% | 38.9% |
| Chunked | before | 3.0% | 7.7% | **39.0%** | **46.4%** |
| Chunked | both | 12.0% | 25.4% | 32.0% | 45.2% |

**Key findings**:
- **Query-both is the best for standard attention**: Standard+both LoRA achieves 41% EM / 51.9% F1, the best overall.
- **Query-before works well with chunked LoRA** (39% EM), nearly matching standard+both. This makes sense: with chunked attention, documents can't attend to each other anyway, so placing the query before them gives each document direct access to the question during its own processing.
- **Query-before fails for base models** (1-3% EM) because the generation position is too far from the question without fine-tuning.
- **Standard attention outperforms chunked for base model** (29% vs 9% EM for after, 28% vs 12% for both), but after fine-tuning the gap narrows (41% vs 32% for both).
- **Fine-tuning provides large gains across all configurations**, with the biggest delta for chunked+before (+36% EM).

### Prior results (pre-prompt-fix, deprecated)

These results used a non-alpaca prompt format at eval time and are **no longer comparable** to the corrected numbers above. Kept for historical reference only.

| Model | EM | Substring EM | F1 |
|---|---|---|---|
| No context (parametric only) | 3.0% | 5.0% | 5.8% |
| Base + full attn | 30.0% | 32.0% | 38.8% |
| Base + chunked attn | 3.0% | 13.0% | 11.7% |
| LoRA + full attn (1k ex) | 32.0% | 34.0% | 44.1% |
| LoRA + chunked attn (2.5k ex) | 32.0% | 37.0% | 43.1% |

**No-context baseline** (parametric knowledge only):
```bash
python scripts/evaluate_helmet_rag.py --datasets nq --num-docs 0 --max-test-samples 100
```

## Experiment 1: 1k examples, 2 epochs, lr=2e-4

**Config**: `configs/nq_rag_lora_multigpu.yml` (original)

| Parameter | Value |
|---|---|
| Train examples | 1,000 |
| Num docs | 20 |
| Gold position | random |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Sequence length | 8192 |
| Batch size | 1 x 8 grad_accum x 4 GPUs |

**Generate data**:
```bash
conda activate corpus-reasoning-eval
python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20 --gold-position random --output-dir data
```

**Train**:
```bash
conda activate corpus-reasoning
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH
accelerate launch --num_processes 4 -m axolotl.cli.train configs/nq_rag_lora_multigpu.yml
```

**Evaluate**:
```bash
conda activate corpus-reasoning-eval
python scripts/evaluate_helmet_rag.py --base-model NousResearch/Llama-3.2-1B --lora-path outputs/nq-rag-lora --datasets nq --max-test-samples 100 --num-docs 20 --max-model-len 4096 --shots 2 --output-file outputs/nq_rag_lora_eval_results.json
```

**Results** (NQ, 20 docs, 100 eval samples, 2-shot):

*Note: These results used a non-alpaca eval prompt format. See Experiment 4 for corrected numbers with alpaca-format prompts matching the training format.*

| Model | EM | Substring EM | F1 |
|---|---|---|---|
| Base (Llama-3.2-1B) | 30.0% | 32.0% | 38.8% |
| LoRA (1k ex, 2 ep) | 32.0% | 34.0% | 44.1% |
| **Delta** | **+2.0%** | **+2.0%** | **+5.3%** |

**Training stats**: 62 steps, ~5 min, final loss ~0.13-0.22 (avg 0.49)

## Experiment 2: 10k examples, 1 epoch, lr=5e-4

**Config**: `configs/nq_rag_lora_multigpu.yml` (updated)

| Parameter | Value |
|---|---|
| Train examples | 10,000 |
| Num docs | 20 |
| Gold position | random |
| Epochs | 1 |
| Learning rate | 5e-4 |
| Sequence length | 8192 |
| Batch size | 1 x 8 grad_accum x 4 GPUs |

**Generate data**:
```bash
conda activate corpus-reasoning-eval
python scripts/generate_nq_training_data.py --num-examples 10000 --num-docs 20 --gold-position random --output-dir data
```

**Train**:
```bash
conda activate corpus-reasoning
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH
accelerate launch --num_processes 4 -m axolotl.cli.train configs/nq_rag_lora_multigpu.yml
```

**Training stats**: 312 steps, ~25 min, loss ~0.35-0.54 at midpoint

**Results**: Pending evaluation

## Experiment 3: Chunked Document Attention, 2.5k examples, 1 epoch, lr=5e-4

Documents attend only within themselves (block-diagonal attention), while query/answer tokens attend to everything. Uses custom HF Trainer with SDPA for 4D attention masks.

**Config**: `configs/nq_rag_chunked_lora.yml`

| Parameter | Value |
|---|---|
| Train examples | 2,500 |
| Num docs | 20 |
| Gold position | random |
| Epochs | 1 |
| Learning rate | 5e-4 |
| Sequence length | 8192 |
| Batch size | 1 x 8 grad_accum x 4 GPUs |
| Attention | Chunked (SDPA, block-diagonal for docs) |

**Train**:
```bash
conda activate corpus-reasoning
export CUDA_HOME=/usr/local/cuda-12.8
accelerate launch --num_processes 4 scripts/train_chunked.py configs/nq_rag_chunked_lora.yml
```

**Evaluate**:
```bash
conda activate corpus-reasoning-eval
# Base model (chunked attention, no LoRA)
python scripts/evaluate_chunked.py --datasets nq --num-docs 20 --max-test-samples 100 --output-file outputs/chunked_eval_nq_k20_base.json
# LoRA fine-tuned (chunked attention)
python scripts/evaluate_chunked.py --datasets nq --num-docs 20 --max-test-samples 100 --lora-path outputs/nq-rag-chunked-lora --output-file outputs/chunked_eval_nq_k20_lora.json
```

**Training stats**: 79 steps, ~12 min, final loss 0.075, avg loss 0.725

**Results** (NQ, 20 docs, 100 eval samples, 2-shot, chunked attention):

| Model | EM | Substring EM | F1 |
|---|---|---|---|
| Base (chunked attn) | 3.0% | 13.0% | 11.7% |
| LoRA (chunked attn, 2.5k ex) | 32.0% | 37.0% | 43.1% |
| **Delta** | **+29.0%** | **+24.0%** | **+31.4%** |

**Comparison with standard (full) attention** (Experiment 1):

| Model | EM | Substring EM | F1 |
|---|---|---|---|
| Standard attn LoRA (1k ex) | 32.0% | 34.0% | 44.1% |
| Chunked attn LoRA (2.5k ex) | 32.0% | 37.0% | 43.1% |

Chunked attention performs comparably to standard full attention despite documents being isolated during prefill.

## Experiment 4: Query position × Attention type ablation

Six configurations: {standard, chunked} × {query_before, query_after, query_both}, all trained on 2,500 examples.

**Configs**:
- `configs/nq_rag_std_qafter.yml` — axolotl, standard attention, query after
- `configs/nq_rag_std_qbefore.yml` — axolotl, standard attention, query before
- `configs/nq_rag_std_qboth.yml` — axolotl, standard attention, query both
- `configs/nq_rag_chunked_qafter.yml` — custom trainer, chunked attention, query after
- `configs/nq_rag_chunked_qbefore.yml` — custom trainer, chunked attention, query before
- `configs/nq_rag_chunked_qboth.yml` — custom trainer, chunked attention, query both

**Train**:
```bash
conda activate corpus-reasoning
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH

# Standard attention (axolotl)
accelerate launch --num_processes 4 -m axolotl.cli.train configs/nq_rag_std_qafter.yml
accelerate launch --num_processes 4 -m axolotl.cli.train configs/nq_rag_std_qbefore.yml
accelerate launch --num_processes 4 -m axolotl.cli.train configs/nq_rag_std_qboth.yml

# Chunked attention (custom trainer)
accelerate launch --num_processes 4 scripts/train_chunked.py configs/nq_rag_chunked_qafter.yml
accelerate launch --num_processes 4 scripts/train_chunked.py configs/nq_rag_chunked_qbefore.yml
accelerate launch --num_processes 4 scripts/train_chunked.py configs/nq_rag_chunked_qboth.yml
```

**Evaluate**:
```bash
conda activate corpus-reasoning-eval

# Standard attention evals (vLLM)
python scripts/evaluate_helmet_rag.py --datasets nq --num-docs 20 --max-test-samples 100 --output-file outputs/eval_std_qafter_base.json
python scripts/evaluate_helmet_rag.py --datasets nq --num-docs 20 --max-test-samples 100 --lora-path outputs/nq-rag-std-qafter --output-file outputs/eval_std_qafter_lora.json
python scripts/evaluate_helmet_rag.py --datasets nq --num-docs 20 --max-test-samples 100 --query-position before --output-file outputs/eval_std_qbefore_base.json
python scripts/evaluate_helmet_rag.py --datasets nq --num-docs 20 --max-test-samples 100 --query-position before --lora-path outputs/nq-rag-std-qbefore --output-file outputs/eval_std_qbefore_lora.json

# Chunked attention evals (HF generate with 4D masks)
python scripts/evaluate_chunked.py --datasets nq --num-docs 20 --max-test-samples 100 --output-file outputs/eval_chunked_qafter_base.json
python scripts/evaluate_chunked.py --datasets nq --num-docs 20 --max-test-samples 100 --lora-path outputs/nq-rag-chunked-qafter --output-file outputs/eval_chunked_qafter_lora.json
python scripts/evaluate_chunked.py --datasets nq --num-docs 20 --max-test-samples 100 --query-position before --output-file outputs/eval_chunked_qbefore_base.json
python scripts/evaluate_chunked.py --datasets nq --num-docs 20 --max-test-samples 100 --query-position before --lora-path outputs/nq-rag-chunked-qbefore --output-file outputs/eval_chunked_qbefore_lora.json
```

**Training stats** (all ~79 steps, 1 epoch):
| Config | Time | Final loss |
|---|---|---|
| std_qafter (axolotl) | ~6 min | 0.42 |
| std_qbefore (axolotl) | ~6 min | 0.42 |
| chunked_qafter | ~11 min | 0.36 |
| chunked_qbefore | ~11 min | 0.35 |

**Results** (NQ, 20 docs, 100 eval samples, 2-shot):

Base model uses HELMET prompt format; LoRA models use alpaca prompt format (matching training).

| Attention | Query Position | Base EM | Base F1 | LoRA EM | LoRA F1 |
|---|---|---|---|---|---|
| Standard | after (default) | 29.0% | 39.9% | 33.0% | 44.3% |
| Standard | before | 1.0% | 6.5% | 34.0% | 41.0% |
| Standard | both | 28.0% | 38.6% | **41.0%** | **51.9%** |
| Chunked | after | 9.0% | 22.4% | 30.0% | 38.9% |
| Chunked | before | 3.0% | 7.7% | **39.0%** | **46.4%** |
| Chunked | both | 12.0% | 25.4% | 32.0% | 45.2% |

**Analysis**: Query-both is the best for standard attention LoRA (41% EM), but query-before works surprisingly well with chunked attention LoRA (39% EM). With chunked attention, documents can't attend to each other, so placing the query before gives each document direct access to the question — nearly as effective as having it on both sides. Standard+both LoRA still leads overall. Fine-tuning provides the biggest gains for chunked+before (+36% EM).

## Experiment 5: Scaled-up ablation (50k NQ, 500 eval samples)

Same query position × attention type ablation as Experiment 4, but with 50k training examples and evaluated on 500 samples for more reliable estimates.

*Results pending — SLURM jobs in progress (job 107789).*

**Configs**: `configs/nq_{std,chunked}_q{before,after,both}_lora.yml`

```bash
# Run via SLURM
sbatch scripts/run_batch_part2.sh
```
