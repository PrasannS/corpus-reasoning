#!/bin/bash
#SBATCH --job-name=contra-n1000-qwen
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --qos=normal
#SBATCH --output=outputs/contra_fever_n1000_qwen_%j.log
#SBATCH --error=outputs/contra_fever_n1000_qwen_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

echo "============================================================"
echo "Contradiction (FEVER): n=1000, Qwen LoRA, standard attention"
echo "============================================================"

# ============================================================
# Phase 1: Train
# ============================================================
echo ""
echo "PHASE 1: Training"
echo "============================================================"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

CUDA_HOME=/usr/local/cuda-12.8 accelerate launch --num_processes 2 \
    scripts/train/train_chunked_fast.py \
    configs/contradiction_fever_n1000_qwen_lora.yml \
    2>&1 | tee outputs/train_contra_fever_n1000_qwen.log

echo "Training complete."

# ============================================================
# Phase 2: Evaluate
# ============================================================
echo ""
echo "PHASE 2: Evaluation"
echo "============================================================"

conda activate corpus-reasoning-eval

LORA_PATH="./outputs/contradiction-fever-n1000-qwen-lora"
EVAL_DATA="data/contradiction_eval_fever_n1000_k3.jsonl"

echo "Evaluating contradiction (LoRA, standard backend)..."
python scripts/eval/evaluate.py \
    --backend standard \
    --task contradiction \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 100 \
    --output-file outputs/contra_fever_n1000_qwen_eval.json \
    2>&1 | tee outputs/eval_contra_fever_n1000_qwen.log

echo ""
echo "Evaluating contradiction (base model, standard backend)..."
python scripts/eval/evaluate.py \
    --backend standard \
    --task contradiction \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 100 \
    --output-file outputs/contra_fever_n1000_qwen_base_eval.json \
    2>&1 | tee outputs/eval_contra_fever_n1000_qwen_base.log

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
