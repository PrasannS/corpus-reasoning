#!/bin/bash
#SBATCH --job-name=batch-part2
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/batch_part2_%j.log
#SBATCH --error=outputs/batch_part2_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=4
EVAL_SAMPLES=500

# Part 2: All NQ runs (6 train+eval)
RUNS=(
    "nq std before"
    "nq std after"
    "nq std both"
    "nq chunked before"
    "nq chunked after"
    "nq chunked both"
)

echo "============================================================"
echo "PART 2: NQ experiments (${#RUNS[@]} runs)"
echo "Node: $(hostname) | GPUs: $NUM_GPUS | Eval samples: $EVAL_SAMPLES"
echo "============================================================"

eval "$(conda shell.bash hook)"
run_count=0
total_runs=${#RUNS[@]}

for run_spec in "${RUNS[@]}"; do
    read -r DATASET ATTN QPOS <<< "$run_spec"
    run_count=$((run_count + 1))

    CONFIG="configs/${DATASET}_${ATTN}_q${QPOS}_lora.yml"
    OUTPUT_DIR="./outputs/${DATASET}-${ATTN}-q${QPOS}-lora"
    TRAIN_LOG="$LOG_DIR/train_${DATASET}_${ATTN}_q${QPOS}.log"
    EVAL_LOG="$LOG_DIR/eval_${DATASET}_${ATTN}_q${QPOS}.log"
    EVAL_OUTPUT="outputs/${DATASET}_${ATTN}_q${QPOS}_eval.json"

    echo ""
    echo "============================================================"
    echo "[$run_count/$total_runs] $DATASET / $ATTN / q$QPOS"
    echo "============================================================"

    # --- TRAINING ---
    echo "  Training: $CONFIG"
    conda activate corpus-reasoning

    if [ "$ATTN" = "chunked" ]; then
        accelerate launch --num_processes "$NUM_GPUS" \
            scripts/train_chunked_fast.py "$CONFIG" \
            2>&1 | tee "$TRAIN_LOG"
    else
        accelerate launch --num_processes "$NUM_GPUS" \
            -m axolotl.cli.train "$CONFIG" \
            2>&1 | tee "$TRAIN_LOG"
    fi
    echo "  Training complete: $OUTPUT_DIR"

    # --- EVALUATION ---
    echo "  Evaluating: $OUTPUT_DIR"
    conda activate corpus-reasoning-eval

    if [ "$ATTN" = "chunked" ]; then
        python scripts/evaluate_chunked.py \
            --datasets "$DATASET" \
            --num-docs 20 \
            --query-position "$QPOS" \
            --lora-path "$OUTPUT_DIR" \
            --max-test-samples "$EVAL_SAMPLES" \
            --output-file "$EVAL_OUTPUT" \
            2>&1 | tee "$EVAL_LOG"
    else
        python scripts/evaluate_helmet_rag.py \
            --datasets "$DATASET" \
            --num-docs 20 \
            --query-position "$QPOS" \
            --lora-path "$OUTPUT_DIR" \
            --max-test-samples "$EVAL_SAMPLES" \
            --output-file "$EVAL_OUTPUT" \
            2>&1 | tee "$EVAL_LOG"
    fi
    echo "  Eval complete: $EVAL_OUTPUT"
done

echo ""
echo "============================================================"
echo "PART 2 COMPLETE"
echo "============================================================"
echo "Results:"
for run_spec in "${RUNS[@]}"; do
    read -r DATASET ATTN QPOS <<< "$run_spec"
    EVAL_LOG="$LOG_DIR/eval_${DATASET}_${ATTN}_q${QPOS}.log"
    echo "--- $DATASET / $ATTN / q$QPOS ---"
    grep -A 4 "^SUMMARY" "$EVAL_LOG" 2>/dev/null || echo "  (no results found)"
done
