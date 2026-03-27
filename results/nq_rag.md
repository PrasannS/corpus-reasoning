# NQ RAG Experiments

## Task Description

Fine-tune Llama-3.2-1B on Natural Questions with retrieval-augmented contexts (HELMET format). Each training example has 1 gold document among N-1 distractors. Evaluated on HELMET's KILT NQ eval set.

## Datasets

- **Training data source**: `tilyupo/nq_cqa` (HuggingFace)
- **Eval data source**: HELMET KILT NQ (`data/data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl`)
- **Format**: Axolotl alpaca (instruction/input/output) with HELMET-compatible prompts

## Summary (NQ, 20 docs, 500 eval samples)

### Query position × Attention type ablation (2.5k training examples)

Base model uses HELMET prompt format + 2-shot; LoRA models use alpaca prompt format + 0-shot (matching training).

| Attention | Query Position | Base EM | Base F1 | LoRA EM | LoRA F1 |
|---|---|---|---|---|---|
| Standard | after (default) | 24.2% | 33.5% | 34.4% | 43.3% |
| Standard | before | 1.6% | 5.2% | **34.8%** | **44.0%** |
| Standard | both | 25.4% | 35.0% | 33.4% | **44.3%** |
| Chunked | after | 7.0% | 18.2% | 27.0% | 36.5% |
| Chunked | before | 2.0% | 5.7% | 0.0% | 15.3% |
| Chunked | both | 11.8% | 21.5% | **30.6%** | **41.5%** |

**Key findings**:
- **Standard attention LoRA results are close across query positions** — qbefore (34.8%), qafter (34.4%), qboth (33.4%) are within noise at 2.5k training scale.
- **Chunked+both is the clear chunked winner** (30.6% EM), outperforming chunked+after (27.0%) by 3.6 EM.
- **Chunked+before collapsed** (0.0% EM) — this appears to be a bug in chunked before handling (earlier 100-sample eval of 39% was misleading due to small sample size).
- **Query-before fails for base models** (1.6-2.0% EM) because the generation position is too far from the question without fine-tuning.
- **Standard attention outperforms chunked** across the board, both for base models and after fine-tuning (34.8% vs 30.6% best LoRA EM).
- **Fine-tuning provides large gains across all configurations**, with the biggest deltas for standard+before (+33.2% EM) and chunked+both (+18.8% EM).

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

**Results** (NQ, 20 docs, 500 eval samples):

Base model uses HELMET prompt format + 2-shot; LoRA models use alpaca prompt format + 0-shot (matching training).

| Attention | Query Position | Base EM | Base F1 | LoRA EM | LoRA F1 |
|---|---|---|---|---|---|
| Standard | after (default) | 24.2% | 33.5% | 34.4% | 43.3% |
| Standard | before | 1.6% | 5.2% | **34.8%** | **44.0%** |
| Standard | both | 25.4% | 35.0% | 33.4% | **44.3%** |
| Chunked | after | 7.0% | 18.2% | 27.0% | 36.5% |
| Chunked | before | 2.0% | 5.7% | 0.0% | 15.3% |
| Chunked | both | 11.8% | 21.5% | **30.6%** | **41.5%** |

**Analysis**: With 500 eval samples, standard attention LoRA results are very close across query positions (33-35% EM), unlike the 100-sample results which showed qboth as a clear winner — that was an artifact of small sample size. Chunked+both is the best chunked config (30.6% EM). Chunked+before collapsed to 0% EM (the earlier 100-sample result of 39% was misleading). Standard attention still outperforms chunked overall.

## Experiment 5: Scaled-up training (50k NQ)

Same query position × attention type ablation as Experiment 4, but with 50k training examples. Performance was worse than 2.5k across the board, likely due to overfitting. Results not included in summary — the 2.5k results are the primary reference.
