# Multi-hop QA (HotpotQA)

## Task Description

Multi-hop question answering using HotpotQA bridge questions. Each question requires reasoning across 2 supporting documents to find the answer, with distractor documents mixed in.

## Dataset

- **Source**: `hotpotqa/hotpot_qa` (distractor config) from HuggingFace
- **Question type**: Bridge only (filtered from full dataset)
- **Training data**: 2500 or 72991 (full bridge subset) examples, 20 docs each, shuffled order, query-both format
- **Eval data**: HELMET's `hotpotqa-dev-multikilt_1000_k20_dep3.jsonl` (100 samples)

### Data Generation

```bash
# Generate base training data (corpus-reasoning-eval env)
python scripts/generate_hotpotqa_data.py --num-examples 2500 --num-docs 20 --question-type bridge --doc-order shuffled

# Convert to query-both format
python scripts/convert_to_qboth.py data/hotpotqa_train_k20_shuffled_bridge_2500.jsonl data/hotpotqa_train_k20_shuffled_bridge_2500_qboth.jsonl
```

## Training

```bash
# LoRA - 2.5k examples (corpus-reasoning env)
bash scripts/train.sh configs/hotpotqa_std_qboth_lora.yml

# LoRA - full 73k examples
bash scripts/train.sh configs/hotpotqa_std_qboth_lora_full.yml
```

## Evaluation

```bash
# Base model (0-shot, HELMET format)
python scripts/evaluate_helmet_rag.py --datasets hotpotqa --num-docs 20 --query-position both --max-test-samples 100

# Base model (closed-book)
python scripts/evaluate_helmet_rag.py --datasets hotpotqa --num-docs 0 --max-test-samples 100

# LoRA finetuned
python scripts/evaluate_helmet_rag.py --datasets hotpotqa --num-docs 20 --query-position both --max-test-samples 100 --lora-path ./outputs/hotpotqa-std-qboth-lora
```

## Results (Llama-3.2-1B, 20 docs, query-both, 100 eval samples)

| Setting | EM | SubEM | F1 |
|---------|-----|-------|-----|
| Closed-book (no context) | 6.0% | 13.0% | 9.9% |
| Base model, 0-shot | 23.0% | 25.0% | 32.3% |
| Base model, 2-shot | 23.0% | 25.0% | 32.3% |
| LoRA finetuned (2.5k ex, 1ep) | 47.0% | 52.0% | 59.1% |
| LoRA finetuned (73k ex, 1ep) | **52.0%** | **58.0%** | **66.7%** |

### No-context (closed-book) finetuning baseline

| Setting | EM | SubEM | F1 |
|---------|-----|-------|-----|
| Base model (no context) | 6.0% | 13.0% | 9.9% |
| No-context LoRA (2.5k ex) | 2.0% | 13.0% | 8.7% |

### Notes

- Full fine-tuning (lr=1e-5, 1 epoch) collapsed to 1% EM — LoRA is much more stable for this data size.
- Few-shot demos provided no benefit for the base model on this task.
- LoRA roughly doubles base model performance (23% → 47% EM).
- Scaling from 2.5k → 73k examples gives +5 EM / +7.6 F1. Most gains come from the first 2.5k examples (diminishing returns).
- No-context finetuning does not improve closed-book performance — gains from context-based training come from learning to use the documents, not memorizing answers.
