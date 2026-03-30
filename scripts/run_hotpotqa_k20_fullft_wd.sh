#!/bin/bash
#SBATCH --job-name=hpqa-k20-fullft-wd
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/hotpotqa_k20_fullft_wd_%j.log
#SBATCH --error=outputs/hotpotqa_k20_fullft_wd_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=4
eval "$(conda shell.bash hook)"

echo "============================================================"
echo "Full FT k=20 retrieval with weight_decay=0.01"
echo "============================================================"

FULLFT_CONFIG="configs/hotpotqa_k20_std_qboth_qwen_fullft_wd.yml"
FULLFT_OUTPUT="./outputs/hotpotqa-k20-std-qboth-qwen-fullft-wd"

if [ -f "$FULLFT_OUTPUT/model.safetensors" ]; then
    echo "  Model already exists at $FULLFT_OUTPUT, skipping training"
else
    conda activate corpus-reasoning
    accelerate launch --num_processes "$NUM_GPUS" \
        -m axolotl.cli.train "$FULLFT_CONFIG" \
        2>&1 | tee "$LOG_DIR/train_hotpotqa_k20_fullft_wd.log"
    echo "  Training complete: $FULLFT_OUTPUT"
fi

conda activate corpus-reasoning-eval
echo "  Evaluating Full FT on k=20 retrieval..."
python scripts/evaluate_retrieval.py \
    --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl \
    --base-model "$FULLFT_OUTPUT" \
    --enforce-eager \
    --output-file outputs/hotpotqa_k20_qwen_fullft_wd_retrieval_k20eval.json \
    2>&1 | tee "$LOG_DIR/eval_hotpotqa_k20_fullft_wd_retrieval.log"

echo ""
echo "--- Full FT (weight_decay=0.01) Results ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_hotpotqa_k20_fullft_wd_retrieval.log" 2>/dev/null || echo "  (no results)"
