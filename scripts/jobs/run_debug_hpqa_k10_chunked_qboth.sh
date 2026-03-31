#!/bin/bash
#SBATCH --job-name=dbg-k10-qboth
#SBATCH --output=outputs/debug_hpqa_k10_chunked_qboth_%j.log
#SBATCH --error=outputs/debug_hpqa_k10_chunked_qboth_%j.log
#SBATCH --partition=lambda
#SBATCH --qos=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=4:00:00

# =============================================================================
# Debug run: HotpotQA k10 chunked qboth (1k train examples)
# Investigating why chunked qboth gets 2.4% F1 on k100
# =============================================================================

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

set -eo pipefail

CONFIG="configs/hotpotqa_k10_chunked_qboth_qwen_lora.yml"
EVAL_DATA="data/hotpotqa_eval_k10_shuffled_bridge_500.jsonl"
OUTPUT_DIR="./outputs/hotpotqa-k10-chunked-qboth-qwen-lora"

echo "=============================================="
echo "Debug: HotpotQA k10 chunked qboth"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=============================================="

# --- Train ---
echo "[1/2] Training..."
conda activate corpus-reasoning
accelerate launch --num_processes 4 \
    scripts/train/train_chunked_fast.py "$CONFIG"
echo "[1/2] Training done: $(date)"

# --- Eval ---
echo "[2/2] Evaluating..."
conda activate corpus-reasoning-eval

python scripts/eval/evaluate.py \
    --backend chunked-sdpa \
    --task retrieval \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --eval-data "$EVAL_DATA" \
    --lora-path "$OUTPUT_DIR" \
    --query-position both \
    --max-test-samples 500 \
    --output-file "outputs/eval_hotpotqa_k10_chunked_qboth.json"

echo "[2/2] Eval done: $(date)"

echo ""
echo "=============================================="
echo "Debug run complete: $(date)"
echo "=============================================="
