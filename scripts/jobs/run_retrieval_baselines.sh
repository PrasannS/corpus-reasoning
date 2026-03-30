#!/bin/bash
#SBATCH --job-name=retrieval-baselines
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/retrieval_baselines_%j.log
#SBATCH --error=outputs/retrieval_baselines_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=4
eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-retrieval

TRAIN_DATA="data/hotpotqa_train_k20_retrieval_triplets_52k.jsonl"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

# ============================================================
# 1. Train DPR (single-vector) baseline
# ============================================================
echo "============================================================"
echo "Training DPR baseline (ModernBERT-base)"
echo "============================================================"

DPR_OUTPUT="outputs/ModernBERT-base-dpr-hotpotqa-k20-lr3e-05"
if [ -d "$DPR_OUTPUT/final" ]; then
    echo "  DPR model already exists at $DPR_OUTPUT/final, skipping training"
else
    torchrun --nproc_per_node="$NUM_GPUS" scripts/train/train_retrieval_baseline.py \
        --mode dpr \
        --train-data "$TRAIN_DATA" \
        --model-name answerdotai/ModernBERT-base \
        --lr 3e-5 \
        --epochs 1 \
        --batch-size 256 \
        --mini-batch-size 32 \
        --dataset-tag hotpotqa-k20 \
        2>&1 | tee "$LOG_DIR/train_dpr.log"
    echo "  DPR training complete"
fi

# ============================================================
# 2. Train ColBERT (multi-vector) baseline
# ============================================================
echo ""
echo "============================================================"
echo "Training ColBERT baseline (ModernBERT-base)"
echo "============================================================"

COLBERT_OUTPUT="outputs/ModernBERT-base-colbert-hotpotqa-k20-lr3e-05-emb128"
if [ -d "$COLBERT_OUTPUT/final" ]; then
    echo "  ColBERT model already exists at $COLBERT_OUTPUT/final, skipping training"
else
    torchrun --nproc_per_node="$NUM_GPUS" scripts/train/train_retrieval_baseline.py \
        --mode colbert \
        --train-data "$TRAIN_DATA" \
        --model-name answerdotai/ModernBERT-base \
        --lr 3e-5 \
        --epochs 1 \
        --batch-size 256 \
        --mini-batch-size 32 \
        --embedding-dim 128 \
        --dataset-tag hotpotqa-k20 \
        2>&1 | tee "$LOG_DIR/train_colbert.log"
    echo "  ColBERT training complete"
fi

# ============================================================
# 3. Evaluate DPR — base (untrained) model
# ============================================================
echo ""
echo "============================================================"
echo "Evaluating base DPR (untrained ModernBERT-base)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python scripts/eval/evaluate_retrieval_baseline.py \
    --mode dpr \
    --model-path answerdotai/ModernBERT-base \
    --eval-data "$EVAL_DATA" \
    --output-file outputs/hotpotqa_k20_dpr_base_retrieval.json \
    2>&1 | tee "$LOG_DIR/eval_dpr_base.log"

# ============================================================
# 4. Evaluate DPR — trained model
# ============================================================
echo ""
echo "============================================================"
echo "Evaluating trained DPR"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python scripts/eval/evaluate_retrieval_baseline.py \
    --mode dpr \
    --model-path "$DPR_OUTPUT/final" \
    --eval-data "$EVAL_DATA" \
    --output-file outputs/hotpotqa_k20_dpr_trained_retrieval.json \
    2>&1 | tee "$LOG_DIR/eval_dpr_trained.log"

# ============================================================
# 5. Evaluate ColBERT — base (untrained) model
# ============================================================
echo ""
echo "============================================================"
echo "Evaluating base ColBERT (untrained ModernBERT-base)"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python scripts/eval/evaluate_retrieval_baseline.py \
    --mode colbert \
    --model-path answerdotai/ModernBERT-base \
    --eval-data "$EVAL_DATA" \
    --output-file outputs/hotpotqa_k20_colbert_base_retrieval.json \
    2>&1 | tee "$LOG_DIR/eval_colbert_base.log"

# ============================================================
# 6. Evaluate ColBERT — trained model
# ============================================================
echo ""
echo "============================================================"
echo "Evaluating trained ColBERT"
echo "============================================================"

CUDA_VISIBLE_DEVICES=0 python scripts/eval/evaluate_retrieval_baseline.py \
    --mode colbert \
    --model-path "$COLBERT_OUTPUT/final" \
    --eval-data "$EVAL_DATA" \
    --output-file outputs/hotpotqa_k20_colbert_trained_retrieval.json \
    2>&1 | tee "$LOG_DIR/eval_colbert_trained.log"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo ""
echo "--- Base DPR ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_dpr_base.log" 2>/dev/null || echo "  (no results)"
echo ""
echo "--- Trained DPR ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_dpr_trained.log" 2>/dev/null || echo "  (no results)"
echo ""
echo "--- Base ColBERT ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_colbert_base.log" 2>/dev/null || echo "  (no results)"
echo ""
echo "--- Trained ColBERT ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_colbert_trained.log" 2>/dev/null || echo "  (no results)"
