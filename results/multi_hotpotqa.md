# Multi-Query HotpotQA

## Task Description

Multi-query HotpotQA bundles N independent bridge-type HotpotQA questions into a single context. The corpus contains the 2 supporting documents for each of the N queries (shuffled together), padded with distractors to reach a fixed total of 50 documents. The model must answer all N questions as a comma-separated list.

This tests the model's ability to handle multiple independent retrieval tasks within a single long context, and how per-query accuracy degrades as the number of queries increases.

## Dataset Details

- **Source**: HotpotQA distractor split (bridge questions only)
- **Training examples**: 5,000 per N value
- **Eval examples**: 200 per N value (generated on-the-fly from validation split)
- **Total documents**: 50 (constant across all N)
  - N=1: 2 supporting + 48 distractors
  - N=5: 10 supporting + 40 distractors
  - N=10: 20 supporting + 30 distractors
- **Model**: Llama-3.2-1B (NousResearch/Llama-3.2-1B)
- **Training**: LoRA (r=16, alpha=32), 1 epoch, lr=5e-4, standard attention

## Results

### Overall Metrics (200 eval examples each)

| N (queries) | Model   | EM     | SubEM  | F1     | All-Correct |
|-------------|---------|--------|--------|--------|-------------|
| 1           | Trained | 47.5%  | 51.0%  | 61.8%  | 47.5%       |
| 1           | Base    | 11.5%  | 15.5%  | 18.5%  | 11.5%       |
| 5           | Trained | 39.1%  | 43.0%  | 54.7%  | 2.5%        |
| 5           | Base    | 0.0%   | 1.9%   | 1.1%   | 0.0%        |
| 10          | Trained | 30.3%  | 32.8%  | 39.7%  | 0.0%        |
| 10          | Base    | 0.0%   | 0.7%   | 0.5%   | 0.0%        |

### Per-Position EM (Trained Models)

| Position | N=5   | N=10  |
|----------|-------|-------|
| Q1       | 43.0% | 47.0% |
| Q2       | 41.5% | 47.5% |
| Q3       | 36.5% | 37.5% |
| Q4       | 37.0% | 31.5% |
| Q5       | 37.5% | 32.5% |
| Q6       | —     | 29.5% |
| Q7       | —     | 22.5% |
| Q8       | —     | 21.5% |
| Q9       | —     | 17.0% |
| Q10      | —     | 16.0% |

### Key Observations

1. **Training provides massive gains over base**: The base model completely fails at the multi-query format for N>=5 (outputs literal `[answer1], [answer2], ...` placeholders). Training teaches the format and retrieval skill.
2. **Per-query EM degrades with more queries**: For N=10, Q1-Q2 match single-query performance (~47% EM), but later positions degrade sharply to ~16% EM at Q10.
3. **N=1 with 50 docs is comparable to standard HotpotQA**: The 47.5% EM for N=1/k=50 is similar to the 52% EM for standard HotpotQA at k=20 (see multihop.md), showing expected degradation from more distractors.
4. **All-correct rate drops fast**: 47.5% for N=1, 2.5% for N=5, 0% for N=10 — getting every query right in a bundle is very hard.
5. **Positional degradation**: Later queries consistently score lower, suggesting the model struggles with longer generation and maintaining accuracy across multiple answers.

## Reproduction

### 1. Generate training data (corpus-reasoning-eval env)

```bash
# N=1
python scripts/generate_multi_hotpotqa_data.py \
    --num-examples 5000 --num-queries 1 --total-docs 50 --question-type bridge

# N=5
python scripts/generate_multi_hotpotqa_data.py \
    --num-examples 5000 --num-queries 5 --total-docs 50 --question-type bridge

# N=10
python scripts/generate_multi_hotpotqa_data.py \
    --num-examples 5000 --num-queries 10 --total-docs 50 --question-type bridge
```

### 2. Train LoRA models (corpus-reasoning env)

```bash
bash scripts/train.sh configs/multi_hotpotqa_n1_k50_std_lora.yml
bash scripts/train.sh configs/multi_hotpotqa_n5_k50_std_lora.yml
bash scripts/train.sh configs/multi_hotpotqa_n10_k50_std_lora.yml
```

### 3. Evaluate (corpus-reasoning-eval env)

```bash
# Trained models
python scripts/evaluate_multi_hotpotqa.py --num-queries 1 --total-docs 50 \
    --question-type bridge --max-test-samples 200 \
    --lora-path outputs/multi-hotpotqa-n1-k50-std-lora \
    --output-file outputs/multi_hotpotqa_n1_k50_eval.json

python scripts/evaluate_multi_hotpotqa.py --num-queries 5 --total-docs 50 \
    --question-type bridge --max-test-samples 200 \
    --lora-path outputs/multi-hotpotqa-n5-k50-std-lora \
    --output-file outputs/multi_hotpotqa_n5_k50_eval.json

python scripts/evaluate_multi_hotpotqa.py --num-queries 10 --total-docs 50 \
    --question-type bridge --max-test-samples 200 \
    --lora-path outputs/multi-hotpotqa-n10-k50-std-lora \
    --output-file outputs/multi_hotpotqa_n10_k50_eval.json

# Base model
python scripts/evaluate_multi_hotpotqa.py --num-queries 1 --total-docs 50 \
    --question-type bridge --max-test-samples 200 \
    --output-file outputs/multi_hotpotqa_n1_k50_base_eval.json

python scripts/evaluate_multi_hotpotqa.py --num-queries 5 --total-docs 50 \
    --question-type bridge --max-test-samples 200 \
    --output-file outputs/multi_hotpotqa_n5_k50_base_eval.json

python scripts/evaluate_multi_hotpotqa.py --num-queries 10 --total-docs 50 \
    --question-type bridge --max-test-samples 200 \
    --output-file outputs/multi_hotpotqa_n10_k50_base_eval.json
```

### Or run all at once via SLURM

```bash
sbatch scripts/run_multi_hotpotqa_experiments.sh
```
