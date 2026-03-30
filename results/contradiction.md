# Contradiction Detection Experiments

## Task Description

Long-context contradiction detection: given a corpus of N numbered claims, identify all pairs that contradict each other. Contradiction pairs are sourced from SNLI (label=2). The model must output a JSON list of pair IDs (e.g., `[[2, 7], [4, 9]]`).

## Datasets

- **Source**: `stanfordnlp/snli` (HuggingFace)
- **Train split**: SNLI train (183k contradiction pairs, 150k filler claims)
- **Eval split**: SNLI validation+test (6.5k contradiction pairs)
- **Format**: Axolotl alpaca (instruction/input/output)

### Generated datasets

| File | Claims | Contradictions | Examples | Avg input chars |
|---|---|---|---|---|
| `contradiction_train_n100_k3.jsonl` | 100 | 3 pairs | 5,000 | 7,299 |
| `contradiction_eval_n100_k3.jsonl` | 100 | 3 pairs | 500 | 8,182 |
| `contradiction_eval_n1000_k3.jsonl` | 1,000 | 3 pairs | 10 | 83,794 |
| `contradiction_eval_n10000_k3.jsonl` | 10,000 | 3 pairs | 10 | 849,539 |

## Experiment 1: Base model baseline (100 claims, 3 contradictions)

**Evaluate**:
```bash
conda activate corpus-reasoning-eval
python scripts/eval/evaluate_contradiction.py \
    --eval-data data/contradiction_eval_n100_k3.jsonl \
    --max-test-samples 200 \
    --max-model-len 8192 \
    --output-file outputs/contradiction_base_n100_k3.json
```

**Results** (200 eval samples):

| Metric | Base (Llama-3.2-1B) |
|---|---|
| Parse rate | 98.5% |
| Pair Precision | 0.3% |
| Pair Recall | 0.3% |
| Pair F1 | 0.2% |
| Exact Match | 0.0% |

**Notes**: Base model outputs the same hallucinated pairs (`[[1, 4], [3, 7]]`) for every input — it doesn't actually read the claims.

## Experiment 2: LoRA fine-tuned (100 claims, 3 contradictions)

**Config**: `configs/contradiction_lora_multigpu.yml`

| Parameter | Value |
|---|---|
| Train examples | 5,000 |
| Claims per example | 100 |
| Contradiction pairs | 3 |
| Epochs | 1 |
| Learning rate | 5e-4 |
| Sequence length | 8192 |
| Batch size | 1 x 8 grad_accum x 4 GPUs |

**Generate data**:
```bash
conda activate corpus-reasoning-eval
python scripts/data/generate_contradiction_data.py --num-claims 100 --num-contradictions 3 --num-train 5000 --num-eval 500 --output-dir data
```

**Train**:
```bash
conda activate corpus-reasoning
export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH
accelerate launch --num_processes 4 -m axolotl.cli.train configs/contradiction_lora_multigpu.yml
```

**Evaluate**:
```bash
conda activate corpus-reasoning-eval
python scripts/eval/evaluate_contradiction.py \
    --eval-data data/contradiction_eval_n100_k3.jsonl \
    --lora-path outputs/contradiction-lora \
    --max-test-samples 200 \
    --max-model-len 8192 \
    --output-file outputs/contradiction_lora_n100_k3.json
```

**Training stats**: 39 steps, ~3 min, loss flat at ~1.25-1.26 (did not converge)

**Results** (200 eval samples):

| Metric | Base | LoRA |
|---|---|---|
| Parse rate | 98.5% | 100.0% |
| Pair Precision | 0.3% | 0.0% |
| Pair Recall | 0.3% | 0.0% |
| Pair F1 | 0.2% | 0.0% |
| Exact Match | 0.0% | 0.0% |

**Notes**: The LoRA model learned the output format (always produces exactly 3 valid JSON pairs) but outputs the same memorized pairs (`[[2, 5], [3, 8], [4, 10]]`) for every input — it did not learn to read the claims. The loss plateau at ~1.26 confirms the model never learned the mapping. This task likely requires a much larger model or different training approach (e.g., more epochs, chain-of-thought, or a model with stronger reasoning capabilities).

## Experiment 3: API models (Gemini 2.5 Pro / Flash)

Evaluate frontier API models on contradiction detection using `scripts/eval/evaluate_contradiction_api.py`. Uses `llm_request_client.py` for parallel async requests with SQLite caching and cost tracking.

**Ablation flags**:
- `--instruction-after`: place task instruction after the corpus (vs before)
- `--hint`: tell the model exactly how many contradicting pairs exist (value of K)
- `--ablate-all`: run all 4 combinations

### 100 claims, 3 contradictions (10 examples)

```bash
conda activate corpus-reasoning-eval
python scripts/eval/evaluate_contradiction_api.py \
    --eval-data data/contradiction_eval_n100_k3.jsonl \
    --models gemini-2.5-flash,gemini-2.5-pro \
    --max-examples 10 --ablate-all
```

| Config | Flash Prec | Flash Rec | Flash F1 | Pro Prec | Pro Rec | Pro F1 |
|---|---|---|---|---|---|---|
| instr_before | 9.3% | 23.3% | 10.5% | 40.1% | 26.7% | 27.4% |
| instr_after | 21.1% | 13.3% | 11.8% | 53.3% | 20.0% | 28.3% |
| instr_before+hint | 20.0% | 20.0% | 20.0% | 30.0% | 30.0% | 30.0% |
| instr_after+hint | 13.3% | 13.3% | 13.3% | 23.3% | 23.3% | 23.3% |

**Notes**: Pro significantly outperforms Flash. Hint calibrates output count (precision ≈ recall) but doesn't improve F1. Instruction placement has minimal effect. Flash tends to over-predict without hint (up to 63 pairs for one example).

### 1,000 claims, 3 contradictions (10 examples)

```bash
conda activate corpus-reasoning-eval
python scripts/eval/evaluate_contradiction_api.py \
    --eval-data data/contradiction_eval_n1000_k3.jsonl \
    --models gemini-2.5-flash,gemini-2.5-pro \
    --max-examples 10
python scripts/eval/evaluate_contradiction_api.py \
    --eval-data data/contradiction_eval_n1000_k3.jsonl \
    --models gemini-2.5-flash,gemini-2.5-pro \
    --max-examples 10 --instruction-after
```

| Config | Flash Prec | Flash Rec | Flash F1 | Pro Prec | Pro Rec | Pro F1 |
|---|---|---|---|---|---|---|
| instr_before | 0.2% | 3.3% | 0.4% | 2.9% | 13.3% | 4.4% |
| instr_after | 1.4% | 6.7% | 2.2% | 1.1% | 3.3% | 1.7% |

**Notes**: Performance drops drastically at 1000 claims (~21k tokens). Both models massively over-predict (Flash sometimes outputs [1,2], [1,3], [1,4]... sequentially). Pro's advantage narrows — it still has higher precision with instr_before but both models are near zero. The needle-in-a-haystack aspect (3 pairs among ~500k possible) makes this extremely challenging at scale.

**Cost**: ~$0.77 total for n1000 runs ($0.40 instr_before + $0.37 instr_after).

## Large-scale eval sets (for future use)

Generated but not yet evaluated:

- `data/contradiction_eval_n10000_k3.jsonl` — 10,000 claims, 10 examples (~850k chars/example)
