#!/bin/bash
#SBATCH --job-name=hpqa-k20-wd
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/hotpotqa_k20_wd_%j.log
#SBATCH --error=outputs/hotpotqa_k20_wd_%j.log
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
# 1. Full FT with weight decay (k=20 retrieval)
# ============================================================
echo ""
echo "============================================================"
echo "Full FT k=20 retrieval with weight_decay=0.01"
echo "============================================================"

FULLFT_CONFIG="configs/hotpotqa_k20_std_qboth_qwen_fullft_wd.yml"
FULLFT_OUTPUT="./outputs/hotpotqa-k20-std-qboth-qwen-fullft-wd"
FULLFT_TRAIN_LOG="$LOG_DIR/train_hotpotqa_k20_fullft_wd.log"

if [ -f "$FULLFT_OUTPUT/model.safetensors" ]; then
    echo "  Model already exists at $FULLFT_OUTPUT, skipping training"
else
    conda activate corpus-reasoning
    accelerate launch --num_processes "$NUM_GPUS" \
        -m axolotl.cli.train "$FULLFT_CONFIG" \
        2>&1 | tee "$FULLFT_TRAIN_LOG"
    echo "  Training complete: $FULLFT_OUTPUT"
fi

# Eval Full FT on k=20 retrieval
conda activate corpus-reasoning-eval
echo "  Evaluating Full FT on k=20 retrieval..."
python scripts/evaluate_retrieval.py \
    --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl \
    --base-model "$FULLFT_OUTPUT" \
    --enforce-eager \
    --output-file outputs/hotpotqa_k20_qwen_fullft_wd_retrieval_k20eval.json \
    2>&1 | tee "$LOG_DIR/eval_hotpotqa_k20_fullft_wd_retrieval.log"

# ============================================================
# 2. LoRA (k=20 retrieval)
# ============================================================
echo ""
echo "============================================================"
echo "LoRA k=20 retrieval"
echo "============================================================"

LORA_CONFIG="configs/hotpotqa_k20_std_qboth_qwen_lora_retr.yml"
LORA_OUTPUT="./outputs/hotpotqa-k20-std-qboth-qwen-lora-retr"
LORA_TRAIN_LOG="$LOG_DIR/train_hotpotqa_k20_lora_retr.log"

if [ -f "$LORA_OUTPUT/adapter_config.json" ]; then
    echo "  Model already exists at $LORA_OUTPUT, skipping training"
else
    conda activate corpus-reasoning
    accelerate launch --num_processes "$NUM_GPUS" \
        -m axolotl.cli.train "$LORA_CONFIG" \
        2>&1 | tee "$LORA_TRAIN_LOG"
    echo "  Training complete: $LORA_OUTPUT"
fi

# Eval LoRA on k=20 retrieval
conda activate corpus-reasoning-eval
echo "  Evaluating LoRA on k=20 retrieval..."
python scripts/evaluate_retrieval.py \
    --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_OUTPUT" \
    --enforce-eager \
    --output-file outputs/hotpotqa_k20_qwen_lora_retr_retrieval_k20eval.json \
    2>&1 | tee "$LOG_DIR/eval_hotpotqa_k20_lora_retr_retrieval.log"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "ALL k=20 RETRIEVAL RUNS COMPLETE"
echo "============================================================"
echo "--- Full FT (weight_decay=0.01) ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_hotpotqa_k20_fullft_wd_retrieval.log" 2>/dev/null || echo "  (no results)"
echo "--- LoRA ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_hotpotqa_k20_lora_retr_retrieval.log" 2>/dev/null || echo "  (no results)"
