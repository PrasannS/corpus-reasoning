#!/bin/bash
#SBATCH --job-name=ad100-rerun
#SBATCH --partition=lambda
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/ad100_rerun_%j.log
#SBATCH --error=outputs/ad100_rerun_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"
OUTPUT_DIR="./outputs/hotpotqa-k20-chunked-qbefore-qwen-lora-ad100"

echo "=== ad100 rerun — Job $SLURM_JOB_ID at $(date) ==="

# Remove old output to force retrain
rm -rf "$OUTPUT_DIR"

accelerate launch --num_processes 4 \
    scripts/train/train_chunked_fast.py \
    configs/hotpotqa_k20_chunked_qbefore_qwen_lora_ad100.yml \
    2>&1 | tee outputs/chunked_qbefore_ad100_train.log

echo "  Training complete at $(date)"

python scripts/eval/evaluate_chunked.py \
    --base-model "$BASE_MODEL" \
    --lora-path "$OUTPUT_DIR" \
    --eval-data "$EVAL_DATA" \
    --query-position before \
    --after-dummy 100 \
    --max-test-samples 500 \
    --max-tokens 50 \
    --output-file outputs/hotpotqa_k20_chunked_qbefore_ad100_eval.json \
    2>&1 | tee outputs/chunked_qbefore_ad100_eval.log

echo "=== Done at $(date) ==="
