#!/bin/bash
#SBATCH --job-name=k100-hpqa-std
#SBATCH --output=outputs/k100_hpqa_std_%j.log
#SBATCH --error=outputs/k100_hpqa_std_%j.log
#SBATCH --partition=lambda
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00

# =============================================================================
# HotpotQA k100 Standard Attention: train + eval for 2 query positions
#   1. qafter
#   2. qboth
#
# Monitor: tail -f outputs/k100_hpqa_std_<JOBID>.log
# =============================================================================

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

set -eo pipefail

HPQA_EVAL="data/hotpotqa_eval_k100_shuffled_bridge_500.jsonl"

CONFIGS=(
    "configs/hotpotqa_k100_std_qafter_qwen_lora.yml"
    "configs/hotpotqa_k100_std_qboth_qwen_lora.yml"
)
QUERY_POSITIONS=("after" "both")
AFTER_DUMMIES=(0 0)

echo "=============================================="
echo "HotpotQA k100 Standard Attention — 2 configs"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=============================================="

for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    QP="${QUERY_POSITIONS[$i]}"
    AD="${AFTER_DUMMIES[$i]}"
    NAME=$(basename "$CONFIG" .yml)
    OUTPUT_DIR=$(grep "^output_dir:" "$CONFIG" | sed 's/^output_dir: *//' | sed 's/^[[:space:]]*//')

    echo ""
    echo "======================================================"
    echo "[$(($i+1))/${#CONFIGS[@]}] Training: $NAME"
    echo "Started: $(date)"
    echo "======================================================"

    conda activate corpus-reasoning
    launch_training 4 "$CONFIG"

    echo "[$(($i+1))/${#CONFIGS[@]}] Training done: $(date)"

    # --- Eval ---
    echo "--- Eval: $NAME ---"
    conda activate corpus-reasoning-eval

    AD_FLAG=""
    if [ "$AD" -gt 0 ]; then
        AD_FLAG="--after-dummy $AD"
    fi

    python scripts/eval/evaluate.py \
        --backend vllm \
        --task retrieval \
        --base-model Qwen/Qwen3.5-0.8B-Base \
        --eval-data "$HPQA_EVAL" \
        --lora-path "$OUTPUT_DIR" \
        --query-position "$QP" \
        $AD_FLAG \
        --max-test-samples 500 \
        --output-file "outputs/eval_${NAME}.json"

    echo "[$(($i+1))/${#CONFIGS[@]}] Eval done: $(date)"
done

echo ""
echo "=============================================="
echo "All HotpotQA k100 standard experiments done: $(date)"
echo "=============================================="
