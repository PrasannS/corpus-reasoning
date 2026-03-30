#!/bin/bash
#SBATCH --job-name=multi-hotpotqa-nquery
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/multi_hotpotqa_%j.log
#SBATCH --error=outputs/multi_hotpotqa_%j.log
set -eo pipefail

# Multi-query HotpotQA experiment: vary N (queries per context) at fixed 50 docs.
# N=1, 5, 10 | 5k training examples | LoRA | standard attention
#
# Three phases per N:
#   1. Generate training data (corpus-reasoning-eval env)
#   2. Train LoRA (corpus-reasoning env)
#   3. Evaluate trained model + base model (corpus-reasoning-eval env)

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/multi_hotpotqa_logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=4
EVAL_SAMPLES=200
TOTAL_DOCS=50
TRAIN_SIZE=5000
QUERY_COUNTS=(1 5 10)

echo "============================================================"
echo "Multi-query HotpotQA experiment"
echo "N values: ${QUERY_COUNTS[*]}"
echo "Total docs: $TOTAL_DOCS | Train size: $TRAIN_SIZE | GPUs: $NUM_GPUS"
echo "============================================================"

# ============================================================
# Phase 1: Generate all training data
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 1: Data generation"
echo "============================================================"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-eval

for N in "${QUERY_COUNTS[@]}"; do
    DATA_FILE="data/multi_hotpotqa_train_n${N}_k${TOTAL_DOCS}_bridge_${TRAIN_SIZE}.jsonl"
    DATA_LOG="$LOG_DIR/datagen_n${N}.log"

    if [ -f "$DATA_FILE" ]; then
        echo "  [n=$N] Data already exists: $DATA_FILE — skipping"
        continue
    fi

    echo "  [n=$N] Generating $TRAIN_SIZE examples..."
    python scripts/data/generate_hotpotqa_data.py \
        --num-examples "$TRAIN_SIZE" \
        --num-queries "$N" \
        --total-docs "$TOTAL_DOCS" \
        --question-type bridge \
        --split train \
        2>&1 | tee "$DATA_LOG"
    echo "  [n=$N] Data generation complete"
done

# ============================================================
# Phase 2: Train all models
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 2: Training"
echo "============================================================"

conda activate corpus-reasoning

for N in "${QUERY_COUNTS[@]}"; do
    CONFIG="configs/multi_hotpotqa_n${N}_k${TOTAL_DOCS}_std_lora.yml"
    OUTPUT_DIR="./outputs/multi-hotpotqa-n${N}-k${TOTAL_DOCS}-std-lora"
    TRAIN_LOG="$LOG_DIR/train_n${N}.log"

    if [ -d "$OUTPUT_DIR" ] && [ -f "$OUTPUT_DIR/adapter_config.json" ]; then
        echo "  [n=$N] Model already trained: $OUTPUT_DIR — skipping"
        continue
    fi

    echo "  [n=$N] Training: $CONFIG"
    accelerate launch --num_processes "$NUM_GPUS" \
        -m axolotl.cli.train "$CONFIG" \
        2>&1 | tee "$TRAIN_LOG"
    echo "  [n=$N] Training complete: $OUTPUT_DIR"
done

# ============================================================
# Phase 3: Evaluate all models
# ============================================================
echo ""
echo "============================================================"
echo "PHASE 3: Evaluation"
echo "============================================================"

conda activate corpus-reasoning-eval

for N in "${QUERY_COUNTS[@]}"; do
    OUTPUT_DIR="./outputs/multi-hotpotqa-n${N}-k${TOTAL_DOCS}-std-lora"
    EVAL_OUTPUT="outputs/multi_hotpotqa_n${N}_k${TOTAL_DOCS}_eval.json"
    EVAL_LOG="$LOG_DIR/eval_n${N}.log"
    BASE_EVAL_OUTPUT="outputs/multi_hotpotqa_n${N}_k${TOTAL_DOCS}_base_eval.json"
    BASE_EVAL_LOG="$LOG_DIR/eval_n${N}_base.log"

    echo ""
    echo "  [n=$N] Evaluating trained model: $OUTPUT_DIR"
    python scripts/eval/evaluate_multi_hotpotqa.py \
        --num-queries "$N" \
        --total-docs "$TOTAL_DOCS" \
        --question-type bridge \
        --max-test-samples "$EVAL_SAMPLES" \
        --lora-path "$OUTPUT_DIR" \
        --output-file "$EVAL_OUTPUT" \
        2>&1 | tee "$EVAL_LOG"

    echo "  [n=$N] Evaluating base model"
    python scripts/eval/evaluate_multi_hotpotqa.py \
        --num-queries "$N" \
        --total-docs "$TOTAL_DOCS" \
        --question-type bridge \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "$BASE_EVAL_OUTPUT" \
        2>&1 | tee "$BASE_EVAL_LOG"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "============================================================"
echo ""
echo "Results summary:"
echo ""
printf "%-6s %-10s %8s %8s %8s %10s\n" "N" "Model" "EM" "SubEM" "F1" "AllCorrect"
echo "--------------------------------------------------------------"

for N in "${QUERY_COUNTS[@]}"; do
    for label in "" "_base"; do
        EVAL_LOG="$LOG_DIR/eval_n${N}${label}.log"
        if [ "$label" = "_base" ]; then
            model_name="base"
        else
            model_name="trained"
        fi
        # Extract overall metrics from eval log
        metrics=$(grep -A 4 "^RESULTS" "$EVAL_LOG" 2>/dev/null | tail -4 || echo "")
        if [ -n "$metrics" ]; then
            printf "%-6s %-10s " "$N" "$model_name"
            echo "$metrics" | grep "Overall EM:" | awk '{printf "%8s", $NF}'
            echo "$metrics" | grep "Overall SubEM:" | awk '{printf "%8s", $NF}'
            echo "$metrics" | grep "Overall F1:" | awk '{printf "%8s", $NF}'
            echo "$metrics" | grep "All-correct:" | awk '{printf "%10s\n", $NF}'
        else
            printf "%-6s %-10s %8s\n" "$N" "$model_name" "(no results)"
        fi
    done
done
