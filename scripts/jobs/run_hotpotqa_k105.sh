#!/bin/bash
#SBATCH --job-name=hpqa-k105
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/hotpotqa_k105_%j.log
#SBATCH --error=outputs/hotpotqa_k105_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=4
EVAL_SAMPLES=500
K=105

eval "$(conda shell.bash hook)"

for QPOS in before after both; do
    CONFIG="configs/hotpotqa_k${K}_std_q${QPOS}_lora.yml"
    OUTPUT_DIR="./outputs/hotpotqa-k${K}-std-q${QPOS}-lora"
    TRAIN_LOG="$LOG_DIR/train_hotpotqa_k${K}_std_q${QPOS}.log"
    EVAL_LOG="$LOG_DIR/eval_hotpotqa_k${K}_std_q${QPOS}.log"
    EVAL_OUTPUT="outputs/hotpotqa_k${K}_std_q${QPOS}_eval.json"

    echo ""
    echo "============================================================"
    echo "hotpotqa k${K} / std / q${QPOS}"
    echo "============================================================"

    # Training
    conda activate corpus-reasoning
    accelerate launch --num_processes "$NUM_GPUS" \
        -m axolotl.cli.train "$CONFIG" \
        2>&1 | tee "$TRAIN_LOG"
    echo "  Training complete: $OUTPUT_DIR"

    # Evaluation
    conda activate corpus-reasoning-eval
    python scripts/eval/evaluate_helmet_rag.py \
        --datasets hotpotqa \
        --num-docs "$K" \
        --query-position "$QPOS" \
        --lora-path "$OUTPUT_DIR" \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "$EVAL_OUTPUT" \
        2>&1 | tee "$EVAL_LOG"
    echo "  Eval complete: $EVAL_OUTPUT"
done

echo ""
echo "============================================================"
echo "ALL k${K} RUNS COMPLETE"
echo "============================================================"
for QPOS in before after both; do
    EVAL_LOG="$LOG_DIR/eval_hotpotqa_k${K}_std_q${QPOS}.log"
    echo "--- hotpotqa k${K} / std / q${QPOS} ---"
    grep -A 4 "^SUMMARY" "$EVAL_LOG" 2>/dev/null || grep "EM:" "$EVAL_LOG" 2>/dev/null || echo "  (no results)"
done
