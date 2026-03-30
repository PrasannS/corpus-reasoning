#!/bin/bash
#SBATCH --job-name=std-ad-ablation
#SBATCH --partition=lambda
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/std_ad_ablation_%j.log
#SBATCH --error=outputs/std_ad_ablation_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

eval "$(conda shell.bash hook)"

BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

echo "=== Standard attention after-dummy ablation: ad1, ad32, ad100, ad500 ==="
echo "=== Job $SLURM_JOB_ID started at $(date) ==="

for N in 1 32 100 500; do
    CONFIG="configs/hotpotqa_k20_std_qbefore_qwen_lora_ad${N}.yml"
    OUTPUT_DIR="./outputs/hotpotqa-k20-std-qbefore-qwen-lora-ad${N}"

    echo ""
    echo "============================================================"
    echo "  Training: standard attention, after_dummy=${N}"
    echo "  Config: ${CONFIG}"
    echo "============================================================"

    # Standard attention uses axolotl
    conda activate corpus-reasoning
    accelerate launch --num_processes 4 \
        -m axolotl.cli.train "$CONFIG" \
        2>&1 | tee "outputs/std_qbefore_ad${N}_train.log"

    echo "  Training std ad${N} complete at $(date)"

    echo "  Evaluating std ad${N}..."
    # Use evaluate_chunked.py with --standard-attention for HF-based eval
    python scripts/evaluate_chunked.py \
        --base-model "$BASE_MODEL" \
        --lora-path "$OUTPUT_DIR" \
        --eval-data "$EVAL_DATA" \
        --query-position before \
        --after-dummy "$N" \
        --standard-attention \
        --max-test-samples 500 \
        --max-tokens 50 \
        --output-file "outputs/hotpotqa_k20_std_qbefore_ad${N}_eval.json" \
        2>&1 | tee "outputs/std_qbefore_ad${N}_eval.log"

    echo "  Eval std ad${N} complete at $(date)"
done

echo ""
echo "=== All runs complete at $(date) ==="
echo ""
echo "=== SUMMARY ==="
for N in 1 32 100 500; do
    echo "--- std ad${N} ---"
    grep -E "EM:" "outputs/std_qbefore_ad${N}_eval.log" 2>/dev/null || echo "  (no results)"
done
