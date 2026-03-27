# Full Fine-Tuning Investigation

## Summary

Full fine-tuning (no LoRA) on NQ RAG consistently fails (~0-1% EM) despite LoRA achieving 41-48% EM on the same data. Investigated across multiple hyperparameter settings, training frameworks, and loss configurations.

## Setup

- **Base model**: NousResearch/Llama-3.2-1B (1.2B parameters)
- **Task**: NQ open-domain QA with 20 retrieved documents, query position "both"
- **Training data**: 2.5k examples, no-titles, sentence-split docs (matched to eval distribution)
- **Eval**: 0-shot (matching training format), no titles, vLLM inference
- **LoRA baseline**: r=16, ~20M trainable params → 41% EM (2.5k), 48% EM (10k)

## Experiments

### Axolotl-based training

| Run | LR | Epochs | Packing | Optimizer | Weight Decay | EM | Notes |
|-----|-----|--------|---------|-----------|-------------|-----|-------|
| v1 | 1e-5 | 1 | yes | adamw_torch | 0.01 | 0% | Single-token outputs + EOS |
| v2 | 5e-5 | 3 | yes | adamw_torch | 0.01 | — | Loss went up (1.70 avg), tokenizer error on eval |
| v3 | 2e-5 | 3 | no | adamw_torch | 0.01 | 1% | Multi-token outputs but wrong answers, "aerial" repeated |
| v4 | 2e-6 | 1 | no | adamw_torch | 0.01 | 0% | Loss barely moved, model unchanged |
| v5 | 5e-5 | 1 | yes | adamw_8bit | 0.0 | 0% | Degenerate repetitive text (catastrophic forgetting) |
| 10k | 5e-6 | 1 | yes | adamw_8bit | 0.0 | 1% | 10k data, more stable training, still wrong answers |

### Custom training script (bypasses axolotl)

| Run | LR | Grad Accum | Train on Inputs | EM | Notes |
|-----|-----|-----------|----------------|-----|-------|
| chunked v1 | 1e-5 | 8 | no | 0% | Same behavior as axolotl — rules out axolotl bug |
| chunked v2 | 1e-5 | 8 | yes | 0% | Grad norms dropped to 0.65 (very stable), but model outputs "Question:" continuations |
| chunked v3 | 2e-6 | 2 | no | 1% | 4x more steps, loss dropped to 0.1-0.4, still wrong answers |
| chunked v4 | 5e-7 | 8 | no | 0% | Very low LR, model barely adapts, base-model-like outputs |

## Key Findings

### 1. Not an axolotl bug
Custom HuggingFace Trainer script produces identical results. The issue is fundamental to full FT on this task.

### 2. Root cause: sparse supervision signal
- Training examples are ~2000 tokens but only ~5 tokens are supervised (the answer after `### Response:`)
- This means 99.3% of each sequence is masked from the loss
- LoRA: 5 supervised tokens updating ~20M params → sufficient signal
- Full FT: 5 supervised tokens updating 1.2B params → extremely noisy gradients

### 3. Gradient instability
- Full FT grad norms: 10-30 (response-only loss)
- LoRA grad norms: 1.0-1.3
- Training on all tokens: grad norms drop to 0.65 (stable but wrong objective)

### 4. Training on all tokens doesn't help
When `train_on_inputs=true`, gradients stabilize but the model learns to predict the next context token rather than answer questions. The response signal is drowned out by ~2000 context tokens.

### 5. Failure modes by learning rate
- **LR ≥ 5e-5**: Catastrophic forgetting — degenerate/repetitive text
- **LR 1e-5 to 2e-5**: Model partially adapts but generates wrong answers
- **LR ≤ 2e-6**: Model barely changes from base, doesn't learn the task

## Other issues discovered and fixed during investigation

### Eval prompt mismatches (fixed)
- **Few-shot demos**: Eval used 2-shot demos but training had none → auto-set shots=0 for trained models
- **Prompt format**: Trained models need alpaca format, base models use HELMET format → auto-detected
- **Titles**: Training titles were heuristically generated (often malformed) vs eval titles from dataset → added `--no-titles` flag
- **Document lengths**: Training docs were 2.5x longer than eval docs → added sentence-level splitting

### Impact of fixes on LoRA results

| Fix | 2.5k LoRA EM | 10k LoRA EM |
|-----|-------------|-------------|
| Before fixes (wrong prompt + 2-shot) | 37% | 43% |
| After fixes (correct prompt + 0-shot) | 41% | 48% |

## Potential paths forward (not yet tried)

1. **Much larger dataset** (50k-100k): More supervised tokens per step with sample packing
2. **Longer answers**: Restructure training data to require multi-sentence answers
3. **Higher LoRA rank**: Approximate full FT via high-rank LoRA (e.g., r=256) which constrains the update space
4. **Curriculum**: Start with train_on_inputs, then switch to response-only
5. **Different architecture**: Use a model with more pre-training on QA-style tasks

## Reproduction

```bash
# LoRA baseline (works)
bash scripts/train.sh configs/nq_rag_std_qboth_notitle.yml
conda run -n corpus-reasoning-eval python scripts/evaluate_helmet_rag.py \
    --lora-path outputs/nq-rag-std-qboth-notitle --datasets nq \
    --num-docs 20 --query-position both --no-titles

# Full FT with custom script (doesn't work)
conda activate corpus-reasoning
accelerate launch --num_processes 4 scripts/train_chunked_fast.py \
    configs/nq_rag_std_qboth_notitle_fullft_chunked.yml
conda run -n corpus-reasoning-eval python scripts/evaluate_helmet_rag.py \
    --base-model outputs/nq-rag-std-qboth-notitle-fullft-chunked --datasets nq \
    --num-docs 20 --query-position both --no-titles
```
