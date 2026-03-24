# Multi-hop QA (HotpotQA)

## Task Description

Multi-hop question answering using HotpotQA bridge questions. Each question requires reasoning across 2 supporting documents to find the answer, with distractor documents mixed in.

## Dataset

- **Source**: `hotpotqa/hotpot_qa` (distractor config) from HuggingFace
- **Question type**: Bridge only (filtered from full dataset)
- **Training data**: Various sizes (2.5k-73k examples), 20/50/105 docs, shuffled order
- **Eval data**: HELMET's `hotpotqa-dev-multikilt_1000_k{20,50,105}_dep3.jsonl`

### Data Generation

```bash
# Generate base training data (corpus-reasoning-eval env)
python scripts/generate_hotpotqa_data.py --num-examples 5000 --num-docs 20 --question-type bridge --doc-order shuffled

# Convert to query-before / query-both formats
python scripts/convert_to_qbefore.py data/input.jsonl data/output_qbefore.jsonl
python scripts/convert_to_qboth.py data/input.jsonl data/output_qboth.jsonl
```

## Training

```bash
# Standard attention (corpus-reasoning env)
bash scripts/train.sh configs/hotpotqa_std_qboth_lora.yml

# Chunked attention
accelerate launch --num_processes 4 scripts/train_chunked_fast.py configs/hotpotqa_chunked_qboth_lora.yml

# Batch runs via SLURM (see scripts/run_batch_part1.sh, run_hotpotqa_k50.sh, etc.)
sbatch scripts/run_batch_part1.sh
```

## Evaluation

```bash
# Standard attention (vLLM)
python scripts/evaluate_helmet_rag.py --datasets hotpotqa --num-docs 20 --query-position both --max-test-samples 500 --lora-path ./outputs/hotpotqa-std-qboth-lora

# Chunked attention (HF generate with 4D masks)
python scripts/evaluate_chunked.py --datasets hotpotqa --num-docs 20 --query-position both --max-test-samples 500 --lora-path ./outputs/hotpotqa-chunked-qboth-lora
```

## Results

### Early experiments (20 docs, query-both only, 100 eval samples)

| Setting | EM | SubEM | F1 |
|---------|-----|-------|-----|
| Closed-book (no context) | 6.0% | 13.0% | 9.9% |
| Base model, 0-shot | 23.0% | 25.0% | 32.3% |
| LoRA finetuned (2.5k ex, 1ep) | 47.0% | 52.0% | 59.1% |
| LoRA finetuned (73k ex, 1ep) | **52.0%** | **58.0%** | **66.7%** |

### Query position × Attention type ablation (20 docs, 73k/72991 train examples, 500 eval samples)

| Attention | Query Position | EM | SubEM | F1 |
|-----------|---------------|-----|-------|-----|
| Standard | before | 47.2% | 51.6% | 59.8% |
| Standard | after | 44.0% | 49.4% | 57.7% |
| Standard | both | **47.4%** | **53.4%** | **61.3%** |
| Chunked | before | 0.8% | 18.6% | 9.5% |
| Chunked | after | 37.2% | 44.8% | 51.0% |
| Chunked | both | 38.2% | 42.8% | 49.7% |

### Longer context experiments (50/105 docs, 5k train examples, 500 eval samples)

*Results pending — jobs in progress.*

- **k=50** (50 docs): `configs/hotpotqa_k50_std_q{before,after,both}_lora.yml`, seq_len=8192 (2% truncated)
- **k=105** (105 docs): `configs/hotpotqa_k105_std_q{before,after,both}_lora.yml`, seq_len=16384 (5% truncated)

### No-context (closed-book) finetuning baseline (100 eval samples)

| Setting | EM | SubEM | F1 |
|---------|-----|-------|-----|
| Base model (no context) | 6.0% | 13.0% | 9.9% |
| No-context LoRA (2.5k ex) | 2.0% | 13.0% | 8.7% |

### Notes

- Full fine-tuning (lr=1e-5, 1 epoch) collapsed to 1% EM — LoRA is much more stable for this data size.
- **Query-both is best for standard attention** (47.4% EM), with query-before close behind (47.2%).
- **Chunked attention underperforms standard** by ~10% EM across the board.
- **Chunked + before collapsed** (0.8% EM) — likely a bug in chunked before handling.
- Scaling from 2.5k → 73k examples gives +5 EM / +7.6 F1. Most gains come from the first 2.5k examples.
- No-context finetuning does not improve closed-book performance — gains come from learning to use the documents.
- For k=105 docs, `sequence_len` must be increased to 16384 to avoid massive truncation (100% at 8192 vs 5% at 16384).
