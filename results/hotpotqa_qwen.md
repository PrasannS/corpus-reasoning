# HotpotQA — Qwen3.5-0.8B-Base Experiments

## Summary

Fine-tuning Qwen3.5-0.8B-Base on HotpotQA with 2.5k examples. LoRA shows a strong improvement over base (+9.6 EM), while full fine-tuning slightly underperforms the base.

## Setup

- **Base model**: Qwen/Qwen3.5-0.8B-Base (multimodal model, text-only usage)
- **Task**: HotpotQA multi-hop QA with 20 retrieved documents, query position "both"
- **Training data**: 2.5k examples, bridge-type questions
- **Eval**: 500 samples, 0-shot alpaca format (LoRA/FT), 2-shot HELMET format (base)
- **Inference**: vLLM with `--enforce-eager` (required for Qwen3.5 GDN architecture)

## Results (500-sample eval)

| Model | EM | SubEM | F1 |
|---|---|---|---|
| Base (Qwen3.5-0.8B-Base) | 39.8% | 44.0% | 51.2% |
| **LoRA** (2.5k, r=16) | **49.4%** | **52.6%** | **62.0%** |
| Full FT (2.5k) | 37.4% | 43.2% | 50.0% |

## Context Size Ablation: k=4 (LoRA vs Full FT)

To investigate whether full FT degradation is related to long contexts, we trained both LoRA and Full FT on a short-context variant (k=4: 2 supporting + 2 distractor docs, 5k examples) and evaluated on both k=4 (in-distribution) and k=20 (out-of-distribution).

### Retrieval Task (500-sample eval)

| Model | k=4 Retrieval EM | k=4 Retrieval F1 | k=20 Retrieval EM | k=20 Retrieval F1 |
|---|---|---|---|---|
| k=4 LoRA (5k, r=16) | 89.0% | 94.5% | 10.0% | 29.1% |
| k=4 Full FT (5k) | 87.4% | 93.5% | 3.8% | 21.3% |

### QA Task — k=20 eval (500-sample eval)

| Model | k=20 QA EM | k=20 QA F1 |
|---|---|---|
| Base (Qwen3.5-0.8B-Base) | 39.8% | 51.2% |
| k=4 LoRA (5k) | 0.0% | 0.2% |
| k=4 Full FT (5k) | 0.0% | 0.2% |

### Also tested: k=2 Full FT

With k=2 (only the 2 supporting docs, no distractors), the answer is always `[1], [2]`. Full FT memorized this trivial shortcut: 100% on k=2 eval, 0.8% on k=20 eval.

### Key Findings from Context Size Ablation

1. **Both methods learn k=4 retrieval equally well** (~88% EM), showing full FT is not inherently broken — it can learn the task when context is short.
2. **Neither generalizes from k=4 to k=20 retrieval** (LoRA: 10%, Full FT: 3.8%). The retrieval task requires seeing longer contexts during training to generalize.
3. **Both k=4-trained models lose QA ability** (0% EM on k=20 QA) — they always output document IDs instead of answers, since they were only trained on retrieval. This is expected: the training task (retrieval) replaces the QA behavior.
4. **Full FT is slightly more prone to overfitting**: lower k=20 retrieval (3.8% vs 10%) and more degenerate outputs, consistent with the k=20 QA results above where full FT underperforms LoRA.

## Key Findings (Original k=20 Experiments)

1. **LoRA is effective**: +9.6 EM over base, consistent with Llama-3.2-1B results on NQ.
2. **Full FT underperforms**: Similar to the Llama full FT investigation — sparse supervision signal (short answers, long contexts) makes full FT unstable for this task.
3. **Qwen3.5 requires vLLM patch for LoRA**: vLLM 0.18.0 has a bug where LoRA on Qwen3.5 fails with an IndexError in the GDN `in_proj_qkvz` fused layer. See `results/vllm_qwen35_lora_fix.md` for the fix.

## Reproduction

### Training

```bash
# Generate data (corpus-reasoning-eval env)
python scripts/generate_hotpotqa_data.py --num-examples 5000 --num-docs 20 --question-type bridge
python scripts/convert_to_qboth.py data/hotpotqa_train_k20_shuffled_bridge_5000.jsonl data/hotpotqa_train_k20_shuffled_qboth_bridge_5000.jsonl

# LoRA training (corpus-reasoning env)
bash scripts/train.sh configs/hotpotqa_std_qboth_qwen_lora.yml

# Full FT training (corpus-reasoning env)
bash scripts/train.sh configs/hotpotqa_std_qboth_qwen_fullft.yml
```

### Evaluation

### k=4 Context Size Ablation

```bash
# Generate data (corpus-reasoning-eval env)
python scripts/generate_hotpotqa_data.py --num-examples 5000 --num-docs 4 --question-type bridge --split both --num-eval 500
python scripts/convert_to_qboth.py data/hotpotqa_train_k4_shuffled_retrieval_bridge_5000.jsonl data/hotpotqa_train_k4_shuffled_retrieval_bridge_5000_qboth.jsonl

# LoRA training (corpus-reasoning env)
bash scripts/train.sh configs/hotpotqa_k4_std_qboth_qwen_lora.yml

# Full FT training (corpus-reasoning env)
bash scripts/train.sh configs/hotpotqa_k4_std_qboth_qwen_fullft.yml

# Eval — k=4 retrieval (corpus-reasoning-eval env)
python scripts/evaluate_retrieval.py --eval-data data/hotpotqa_eval_k4_shuffled_retrieval_bridge_500.jsonl \
    --base-model Qwen/Qwen3.5-0.8B-Base --lora-path ./outputs/hotpotqa-k4-std-qboth-qwen-lora --enforce-eager
python scripts/evaluate_retrieval.py --eval-data data/hotpotqa_eval_k4_shuffled_retrieval_bridge_500.jsonl \
    --base-model ./outputs/hotpotqa-k4-std-qboth-qwen-fullft --enforce-eager

# Eval — k=20 retrieval (corpus-reasoning-eval env)
python scripts/evaluate_retrieval.py --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl \
    --base-model Qwen/Qwen3.5-0.8B-Base --lora-path ./outputs/hotpotqa-k4-std-qboth-qwen-lora --enforce-eager
python scripts/evaluate_retrieval.py --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl \
    --base-model ./outputs/hotpotqa-k4-std-qboth-qwen-fullft --enforce-eager

# Eval — k=20 QA (corpus-reasoning-eval env)
python scripts/evaluate_helmet_rag.py --datasets hotpotqa --num-docs 20 --query-position both \
    --base-model Qwen/Qwen3.5-0.8B-Base --lora-path ./outputs/hotpotqa-k4-std-qboth-qwen-lora \
    --max-test-samples 500 --enforce-eager
python scripts/evaluate_helmet_rag.py --datasets hotpotqa --num-docs 20 --query-position both \
    --base-model ./outputs/hotpotqa-k4-std-qboth-qwen-fullft --max-test-samples 500 --enforce-eager
```

### k=20 Evaluation

```bash
# Apply vLLM fix first (see results/vllm_qwen35_lora_fix.md)

# Base model (corpus-reasoning-eval env)
python scripts/evaluate_helmet_rag.py \
    --datasets hotpotqa --num-docs 20 --query-position both \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --max-test-samples 500 --enforce-eager \
    --output-file outputs/hotpotqa_qwen_std_qboth_base_500eval.json

# LoRA (corpus-reasoning-eval env)
python scripts/evaluate_helmet_rag.py \
    --datasets hotpotqa --num-docs 20 --query-position both \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path ./outputs/hotpotqa-std-qboth-qwen-lora \
    --max-test-samples 500 --enforce-eager \
    --output-file outputs/hotpotqa_qwen_std_qboth_lora2500_500eval.json

# Full FT (corpus-reasoning-eval env)
python scripts/evaluate_helmet_rag.py \
    --datasets hotpotqa --num-docs 20 --query-position both \
    --base-model ./outputs/hotpotqa-std-qboth-qwen-fullft \
    --max-test-samples 500 --enforce-eager \
    --output-file outputs/hotpotqa_qwen_std_qboth_fullft2500_500eval.json
```
