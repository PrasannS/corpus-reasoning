#!/bin/bash
#SBATCH --job-name=cot-retr
#SBATCH --output=outputs/cot_retr_%j.log
#SBATCH --error=outputs/cot_retr_%j.log
#SBATCH --partition=lambda
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00

# =============================================================================
# CoT Retrieval: Generate CoTs, prepare data, train, and evaluate
# =============================================================================
# This script runs the full CoT retrieval pipeline:
#   1. Generate chain-of-thought reasoning with API LLM (uses corpus-reasoning-eval env)
#   2. Convert to alpaca format for Axolotl training
#   3. Train with standard attention (uses corpus-reasoning env)
#   4. Evaluate on retrieval task (uses corpus-reasoning-eval env)
#
# Prerequisites:
#   - Data files exist: data/nq_train_k20_random_1000.jsonl,
#     data/hotpotqa_train_k20_shuffled_bridge_2500.jsonl
#   - Eval files exist: data/nq_eval_k20_random_500.jsonl (or similar),
#     data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl
#   - Google Vertex AI credentials configured for Gemini API
#
# Usage:
#   sbatch scripts/jobs/run_cot_retrieval.sh
#   # Or run specific steps:
#   sbatch scripts/jobs/run_cot_retrieval.sh --skip-cot-gen   # skip API calls
#   sbatch scripts/jobs/run_cot_retrieval.sh --train-only      # only training
#   sbatch scripts/jobs/run_cot_retrieval.sh --eval-only       # only evaluation
# =============================================================================

set -euo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

# Parse arguments
SKIP_COT_GEN=false
TRAIN_ONLY=false
EVAL_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-cot-gen) SKIP_COT_GEN=true ;;
        --train-only)   TRAIN_ONLY=true ;;
        --eval-only)    EVAL_ONLY=true ;;
    esac
done

# --- Configuration ---
COT_MODEL="gemini-2.5-flash-lite"
NQ_TRAIN_DATA="data/nq_train_k20_random_1000.jsonl"
HPQA_TRAIN_DATA="data/hotpotqa_train_k20_shuffled_bridge_2500.jsonl"
NQ_EVAL_DATA="data/nq_eval_k20_random_500.jsonl"
HPQA_EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

NQ_CONFIG="configs/nq_k20_std_qboth_qwen_lora_cot_retr.yml"
HPQA_CONFIG="configs/hotpotqa_k20_std_qboth_qwen_lora_cot_retr.yml"

# Conda setup
eval "$(conda shell.bash hook)"

echo "=============================================="
echo "CoT Retrieval Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "=============================================="

# =============================================================================
# Step 1: Generate CoTs with API LLM
# =============================================================================
if [ "$EVAL_ONLY" = false ] && [ "$TRAIN_ONLY" = false ] && [ "$SKIP_COT_GEN" = false ]; then
    echo ""
    echo "=== Step 1: Generating CoTs ==="
    conda activate corpus-reasoning-eval

    echo "--- NQ CoT generation ---"
    python scripts/data/generate_cot.py \
        --input-data "$NQ_TRAIN_DATA" \
        --model "$COT_MODEL" \
        --max-concurrent 25 \
        --temperature 0.7

    echo "--- HotpotQA CoT generation ---"
    python scripts/data/generate_cot.py \
        --input-data "$HPQA_TRAIN_DATA" \
        --model "$COT_MODEL" \
        --max-concurrent 25 \
        --temperature 0.7

    echo "CoT generation complete."
fi

# =============================================================================
# Step 2: Prepare alpaca-format training data
# =============================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "=== Step 2: Preparing training data ==="
    conda activate corpus-reasoning-eval

    NQ_COT_DATA="${NQ_TRAIN_DATA%.jsonl}_cot.jsonl"
    HPQA_COT_DATA="${HPQA_TRAIN_DATA%.jsonl}_cot.jsonl"

    echo "--- NQ alpaca conversion ---"
    python scripts/data/prepare_training_data.py \
        --input-data "$NQ_COT_DATA" \
        --task cot_retrieval \
        --query-position both

    echo "--- HotpotQA alpaca conversion ---"
    python scripts/data/prepare_training_data.py \
        --input-data "$HPQA_COT_DATA" \
        --task cot_retrieval \
        --query-position both

    echo "Data preparation complete."
fi

# =============================================================================
# Step 3: Train with standard attention
# =============================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "=== Step 3: Training ==="
    conda activate corpus-reasoning
    source scripts/lib/common.sh

    echo "--- NQ CoT Retrieval Training ---"
    launch_training "$NQ_CONFIG"

    echo "--- HotpotQA CoT Retrieval Training ---"
    launch_training "$HPQA_CONFIG"

    echo "Training complete."
fi

# =============================================================================
# Step 4: Evaluate
# =============================================================================
if [ "$TRAIN_ONLY" = false ]; then
    echo ""
    echo "=== Step 4: Evaluation ==="
    conda activate corpus-reasoning-eval

    NQ_OUTPUT_DIR="./outputs/nq-k20-std-qboth-qwen-lora-cot-retr"
    HPQA_OUTPUT_DIR="./outputs/hotpotqa-k20-std-qboth-qwen-lora-cot-retr"

    if [ -d "$NQ_OUTPUT_DIR" ] && [ -f "$NQ_EVAL_DATA" ]; then
        echo "--- NQ Retrieval Eval (CoT model) ---"
        python scripts/eval/evaluate.py \
            --backend vllm \
            --task cot_retrieval \
            --eval-data "$NQ_EVAL_DATA" \
            --lora-path "$NQ_OUTPUT_DIR" \
            --query-position both \
            --max-tokens 512 \
            --max-test-samples 500 \
            --output-file "outputs/eval_nq_cot_retr.json"
    else
        echo "Skipping NQ eval (model or eval data not found)"
    fi

    if [ -d "$HPQA_OUTPUT_DIR" ] && [ -f "$HPQA_EVAL_DATA" ]; then
        echo "--- HotpotQA Retrieval Eval (CoT model) ---"
        python scripts/eval/evaluate.py \
            --backend vllm \
            --task cot_retrieval \
            --eval-data "$HPQA_EVAL_DATA" \
            --lora-path "$HPQA_OUTPUT_DIR" \
            --query-position both \
            --max-tokens 512 \
            --max-test-samples 500 \
            --output-file "outputs/eval_hotpotqa_cot_retr.json"
    else
        echo "Skipping HotpotQA eval (model or eval data not found)"
    fi

    echo "Evaluation complete."
fi

echo ""
echo "=============================================="
echo "Pipeline finished: $(date)"
echo "=============================================="
