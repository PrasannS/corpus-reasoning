#!/bin/bash
#SBATCH --job-name=contra-eval
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --qos=normal
#SBATCH --output=outputs/contra_fever_eval_%j.log
#SBATCH --error=outputs/contra_fever_eval_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-eval

LORA_PATH="./outputs/contradiction-fever-qwen-lora"
EVAL_DATA="data/contradiction_eval_fever_n100_k3.jsonl"

echo "Evaluating contradiction (LoRA, standard backend)..."
python scripts/eval/evaluate.py \
    --backend standard \
    --task contradiction \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 200 \
    --output-file outputs/contra_fever_qwen_eval.json \
    2>&1 | tee outputs/eval_contra_fever_qwen.log

echo ""
echo "Evaluating contradiction (base model, standard backend)..."
python scripts/eval/evaluate.py \
    --backend standard \
    --task contradiction \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 200 \
    --output-file outputs/contra_fever_qwen_base_eval.json \
    2>&1 | tee outputs/eval_contra_fever_qwen_base.log

echo ""
echo "Done!"
