#!/bin/bash
#SBATCH --job-name=ad-ablation
#SBATCH --partition=lambda
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/ad_ablation_%j.log
#SBATCH --error=outputs/ad_ablation_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

echo "=== After-dummy ablation: ad1, ad100, ad500 ==="
echo "=== Job $SLURM_JOB_ID started at $(date) ==="

for N in 1 100 500; do
    CONFIG="configs/hotpotqa_k20_chunked_qbefore_qwen_lora_ad${N}.yml"
    OUTPUT_DIR="./outputs/hotpotqa-k20-chunked-qbefore-qwen-lora-ad${N}"

    echo ""
    echo "============================================================"
    echo "  Training: after_dummy=${N}"
    echo "  Config: ${CONFIG}"
    echo "============================================================"

    accelerate launch --num_processes 4 \
        scripts/train/train_chunked_fast.py "$CONFIG" \
        2>&1 | tee "outputs/chunked_qbefore_ad${N}_train.log"

    echo "  Training ad${N} complete at $(date)"

    echo "  Evaluating ad${N}..."
    python scripts/eval/evaluate_chunked.py \
        --base-model "$BASE_MODEL" \
        --lora-path "$OUTPUT_DIR" \
        --eval-data "$EVAL_DATA" \
        --query-position before \
        --after-dummy "$N" \
        --max-test-samples 500 \
        --max-tokens 50 \
        --output-file "outputs/hotpotqa_k20_chunked_qbefore_ad${N}_eval.json" \
        2>&1 | tee "outputs/chunked_qbefore_ad${N}_eval.log"

    echo "  Eval ad${N} complete at $(date)"
done

echo ""
echo "=== All runs complete at $(date) ==="
echo ""
echo "=== SUMMARY ==="
for N in 1 32 100 500; do
    echo "--- ad${N} ---"
    grep -E "EM:|complete" "outputs/chunked_qbefore_ad${N}_eval.log" 2>/dev/null || echo "  (no results)"
done
