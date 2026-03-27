#!/bin/bash
#SBATCH --job-name=remaining
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/remaining_%j.log
#SBATCH --error=outputs/remaining_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=4
EVAL_SAMPLES=500

eval "$(conda shell.bash hook)"

# ============================================================
# 1) 2.5k NQ std qboth eval on 500 samples (sanity check)
# ============================================================
echo ""
echo "============================================================"
echo "NQ 2.5k std qboth — 500 sample eval (sanity check)"
echo "============================================================"
conda activate corpus-reasoning-eval
python scripts/evaluate_helmet_rag.py \
    --datasets nq \
    --num-docs 20 \
    --query-position both \
    --lora-path ./outputs/nq-rag-std-qboth \
    --max-test-samples 500 \
    --output-file outputs/nq_2.5k_std_qboth_500eval.json \
    2>&1 | tee "$LOG_DIR/eval_nq_2.5k_std_qboth_500.log"

# ============================================================
# 2) HotpotQA k=50 (qbefore model exists, skip its training)
# ============================================================
K=50
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

    if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        echo "  Model already exists at $OUTPUT_DIR, skipping training"
    else
        conda activate corpus-reasoning
        accelerate launch --num_processes "$NUM_GPUS" \
            -m axolotl.cli.train "$CONFIG" \
            2>&1 | tee "$TRAIN_LOG"
        echo "  Training complete: $OUTPUT_DIR"
    fi

    conda activate corpus-reasoning-eval
    python scripts/evaluate_helmet_rag.py \
        --datasets hotpotqa \
        --num-docs "$K" \
        --query-position "$QPOS" \
        --lora-path "$OUTPUT_DIR" \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "$EVAL_OUTPUT" \
        2>&1 | tee "$EVAL_LOG"
    echo "  Eval complete: $EVAL_OUTPUT"
done

# ============================================================
# 3) HotpotQA k=105
# ============================================================
K=105
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

    if [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        echo "  Model already exists at $OUTPUT_DIR, skipping training"
    else
        conda activate corpus-reasoning
        accelerate launch --num_processes "$NUM_GPUS" \
            -m axolotl.cli.train "$CONFIG" \
            2>&1 | tee "$TRAIN_LOG"
        echo "  Training complete: $OUTPUT_DIR"
    fi

    conda activate corpus-reasoning-eval
    python scripts/evaluate_helmet_rag.py \
        --datasets hotpotqa \
        --num-docs "$K" \
        --query-position "$QPOS" \
        --lora-path "$OUTPUT_DIR" \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "$EVAL_OUTPUT" \
        2>&1 | tee "$EVAL_LOG"
    echo "  Eval complete: $EVAL_OUTPUT"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "ALL REMAINING RUNS COMPLETE"
echo "============================================================"
echo "--- NQ 2.5k sanity check ---"
grep "EM:" "$LOG_DIR/eval_nq_2.5k_std_qboth_500.log" 2>/dev/null || echo "  (no results)"
for K in 50 105; do
    for QPOS in before after both; do
        EVAL_LOG="$LOG_DIR/eval_hotpotqa_k${K}_std_q${QPOS}.log"
        echo "--- hotpotqa k${K} / std / q${QPOS} ---"
        grep "EM:" "$EVAL_LOG" 2>/dev/null || echo "  (no results)"
    done
done
