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

## Key Findings

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
