#!/bin/bash
#SBATCH --job-name=std-qboth-qwen
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/std_qboth_qwen_%j.log
#SBATCH --error=outputs/std_qboth_qwen_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

echo "============================================================"
echo "Standard attention: HotpotQA qboth Qwen LoRA (2.5k examples)"
echo "============================================================"

# ============================================================
# Phase 1: Train (standard attention via train_chunked_fast.py)
# ============================================================
echo ""
echo "PHASE 1: Training (standard attention)"
echo "============================================================"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

CUDA_HOME=/usr/local/cuda-12.8 accelerate launch --num_processes 4 \
    scripts/train/train_chunked_fast.py \
    configs/hotpotqa_k20_std_qboth_qwen_lora.yml \
    2>&1 | tee outputs/train_std_qboth_qwen.log

echo "Training complete."

# ============================================================
# Phase 2: Evaluate (standard backend — matches training type)
# ============================================================
echo ""
echo "PHASE 2: Evaluation"
echo "============================================================"

conda activate corpus-reasoning-eval

LORA_PATH="./outputs/hotpotqa-std-qboth-qwen-lora"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_bridge_500.jsonl"

echo "Evaluating retrieval (standard attention)..."
python scripts/eval/evaluate.py \
    --backend standard \
    --task retrieval \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --query-position both \
    --max-test-samples 100 \
    --output-file outputs/std_qboth_qwen_retrieval_eval.json \
    2>&1 | tee outputs/eval_std_qboth_qwen_retrieval.log

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
