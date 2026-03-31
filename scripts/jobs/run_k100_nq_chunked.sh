#!/bin/bash
#SBATCH --job-name=k100-nq-chk
#SBATCH --output=outputs/k100_nq_chunked_%j.log
#SBATCH --error=outputs/k100_nq_chunked_%j.log
#SBATCH --partition=lambda
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=48:00:00

# =============================================================================
# NQ k100 Chunked Attention: train + eval
#   Main experiments: qbefore_ad1, qafter, qboth
#   Dummy ablation:   qbefore_ad100, qbefore_ad500
#
# Monitor: tail -f outputs/k100_nq_chunked_<JOBID>.log
# =============================================================================

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

set -eo pipefail

NQ_EVAL="data/nq_validation_k60_random_500.jsonl"

CONFIGS=(
    "configs/nq_k60_chunked_qafter_qwen_lora.yml"
    "configs/nq_k60_chunked_qboth_qwen_lora.yml"
)
QUERY_POSITIONS=("after" "both")
AFTER_DUMMIES=(0 0)

echo "=============================================="
echo "NQ k100 Chunked Attention — 2 configs"
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
    accelerate launch --num_processes 4 \
        scripts/train/train_chunked_fast.py "$CONFIG"

    echo "[$(($i+1))/${#CONFIGS[@]}] Training done: $(date)"

    # --- Eval ---
    echo "--- Eval: $NAME ---"
    conda activate corpus-reasoning-eval

    AD_FLAG=""
    if [ "$AD" -gt 0 ]; then
        AD_FLAG="--after-dummy $AD"
    fi

    python scripts/eval/evaluate.py \
        --backend chunked-sdpa \
        --task retrieval \
        --base-model Qwen/Qwen3.5-0.8B-Base \
        --eval-data "$NQ_EVAL" \
        --lora-path "$OUTPUT_DIR" \
        --query-position "$QP" \
        $AD_FLAG \
        --max-test-samples 500 \
        --output-file "outputs/eval_${NAME}.json"

    echo "[$(($i+1))/${#CONFIGS[@]}] Eval done: $(date)"
done

echo ""
echo "=============================================="
echo "All NQ k100 chunked experiments done: $(date)"
echo "=============================================="
