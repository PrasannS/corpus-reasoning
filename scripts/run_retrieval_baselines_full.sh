#!/bin/bash
#SBATCH --job-name=retr-full
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/retrieval_baselines_full_%j.log
#SBATCH --error=outputs/retrieval_baselines_full_%j.log
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

# Full 180k triplets (all positive-negative pairs, ~3.5x token-matched budget)
TRAIN_DATA_FULL="data/hotpotqa_train_k20_retrieval_triplets_all.jsonl"
# Token-matched 52k triplets
TRAIN_DATA_MATCHED="data/hotpotqa_train_k20_retrieval_triplets_52k.jsonl"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

# ============================================================
# 1. Train DPR — token-matched (52k triplets, 1 epoch)
# ============================================================
echo "============================================================"
echo "Training DPR (token-matched, 52k triplets)"
echo "============================================================"

DPR_MATCHED="outputs/ModernBERT-base-dpr-hotpotqa-k20-lr3e-05"
if [ -d "$DPR_MATCHED/final" ]; then
    echo "  Already exists, skipping"
else
    torchrun --nproc_per_node="$NUM_GPUS" scripts/train_retrieval_baseline.py \
        --mode dpr \
        --train-data "$TRAIN_DATA_MATCHED" \
        --model-name answerdotai/ModernBERT-base \
        --lr 3e-5 --epochs 1 --batch-size 256 --mini-batch-size 32 \
        --dataset-tag hotpotqa-k20 \
        2>&1 | tee "$LOG_DIR/train_dpr_matched.log"
fi

# ============================================================
# 2. Train DPR — full coverage (180k triplets, 1 epoch)
# ============================================================
echo ""
echo "============================================================"
echo "Training DPR (full, 180k triplets)"
echo "============================================================"

DPR_FULL="outputs/ModernBERT-base-dpr-hotpotqa-k20-full-lr3e-05"
if [ -d "$DPR_FULL/final" ]; then
    echo "  Already exists, skipping"
else
    torchrun --nproc_per_node="$NUM_GPUS" --master_port=29501 scripts/train_retrieval_baseline.py \
        --mode dpr \
        --train-data "$TRAIN_DATA_FULL" \
        --model-name answerdotai/ModernBERT-base \
        --lr 3e-5 --epochs 1 --batch-size 256 --mini-batch-size 32 \
        --dataset-tag hotpotqa-k20-full \
        2>&1 | tee "$LOG_DIR/train_dpr_full.log"
fi

# ============================================================
# 3. Train ColBERT — token-matched (52k triplets, 1 epoch)
# ============================================================
echo ""
echo "============================================================"
echo "Training ColBERT (token-matched, 52k triplets)"
echo "============================================================"

COLBERT_MATCHED="outputs/ModernBERT-base-colbert-hotpotqa-k20-lr3e-05-emb128"
if [ -d "$COLBERT_MATCHED/final" ]; then
    echo "  Already exists, skipping"
else
    torchrun --nproc_per_node="$NUM_GPUS" --master_port=29502 scripts/train_retrieval_baseline.py \
        --mode colbert \
        --train-data "$TRAIN_DATA_MATCHED" \
        --model-name answerdotai/ModernBERT-base \
        --lr 3e-5 --epochs 1 --batch-size 256 --mini-batch-size 32 \
        --embedding-dim 128 --dataset-tag hotpotqa-k20 \
        2>&1 | tee "$LOG_DIR/train_colbert_matched.log"
fi

# ============================================================
# 4. Train ColBERT — full coverage (180k triplets, 1 epoch)
# ============================================================
echo ""
echo "============================================================"
echo "Training ColBERT (full, 180k triplets)"
echo "============================================================"

COLBERT_FULL="outputs/ModernBERT-base-colbert-hotpotqa-k20-full-lr3e-05-emb128"
if [ -d "$COLBERT_FULL/final" ]; then
    echo "  Already exists, skipping"
else
    torchrun --nproc_per_node="$NUM_GPUS" --master_port=29503 scripts/train_retrieval_baseline.py \
        --mode colbert \
        --train-data "$TRAIN_DATA_FULL" \
        --model-name answerdotai/ModernBERT-base \
        --lr 3e-5 --epochs 1 --batch-size 256 --mini-batch-size 32 \
        --embedding-dim 128 --dataset-tag hotpotqa-k20-full \
        2>&1 | tee "$LOG_DIR/train_colbert_full.log"
fi

# ============================================================
# 5. Evaluate all models
# ============================================================
echo ""
echo "============================================================"
echo "Running evaluations"
echo "============================================================"

for MODE_NAME in \
    "dpr|answerdotai/ModernBERT-base|hotpotqa_k20_dpr_base" \
    "dpr|$DPR_MATCHED/final|hotpotqa_k20_dpr_matched" \
    "dpr|$DPR_FULL/final|hotpotqa_k20_dpr_full" \
    "colbert|answerdotai/ModernBERT-base|hotpotqa_k20_colbert_base" \
    "colbert|$COLBERT_MATCHED/final|hotpotqa_k20_colbert_matched" \
    "colbert|$COLBERT_FULL/final|hotpotqa_k20_colbert_full"
do
    IFS='|' read -r MODE MODEL_PATH OUTPUT_TAG <<< "$MODE_NAME"
    echo ""
    echo "  Evaluating $OUTPUT_TAG ($MODE, $MODEL_PATH)"

    if [ ! -d "$MODEL_PATH" ] && [[ "$MODEL_PATH" != *"/"*"/"* ]]; then
        # HF model name — always exists
        :
    elif [ ! -d "$MODEL_PATH" ]; then
        echo "    Model not found at $MODEL_PATH, skipping"
        continue
    fi

    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_retrieval_baseline.py \
        --mode "$MODE" \
        --model-path "$MODEL_PATH" \
        --eval-data "$EVAL_DATA" \
        --output-file "outputs/${OUTPUT_TAG}_retrieval.json" \
        2>&1 | tee "$LOG_DIR/eval_${OUTPUT_TAG}.log"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
for TAG in dpr_base dpr_matched dpr_full colbert_base colbert_matched colbert_full; do
    echo ""
    echo "--- $TAG ---"
    grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_hotpotqa_k20_${TAG}.log" 2>/dev/null || echo "  (no results)"
done
