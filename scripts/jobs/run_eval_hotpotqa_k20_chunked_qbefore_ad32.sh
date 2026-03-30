#!/bin/bash
#SBATCH --job-name=eval-chunked-ad32
#SBATCH --partition=lambda
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/eval_chunked_ad32_%j.log
#SBATCH --error=outputs/eval_chunked_ad32_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

LORA_PATH="./outputs/hotpotqa-k20-chunked-qbefore-qwen-lora-ad32"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"
BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"

echo "=== Eval Job $SLURM_JOB_ID started at $(date) ==="
echo "  LoRA: $LORA_PATH"
echo "  Eval data: $EVAL_DATA"
echo "  Chunked attention, query-position=before, after-dummy=32"

# Eval with chunked attention + matching prompt format
python scripts/eval/evaluate_chunked.py \
    --base-model "$BASE_MODEL" \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --query-position before \
    --after-dummy 32 \
    --max-test-samples 500 \
    --max-tokens 50 \
    --output-file outputs/hotpotqa_k20_chunked_qbefore_ad32_eval.json \
    2>&1 | tee outputs/chunked_qbefore_ad32_eval.log

echo "=== Eval complete at $(date) ==="
