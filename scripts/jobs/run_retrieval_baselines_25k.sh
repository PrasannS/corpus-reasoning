#!/bin/bash
#SBATCH --job-name=retrieval-25k
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=outputs/retrieval_baselines_25k_%j.log
#SBATCH --error=outputs/retrieval_baselines_25k_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-retrieval

EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

# ============================================================
# Phase 1: Generate triplets (25k examples → ~50k triplets)
# ============================================================
echo "PHASE 1: Generating triplets (25k examples)"
echo "============================================================"
python scripts/data/generate_retrieval_triplets.py \
    --dataset hotpotqa \
    --num-examples 25000 \
    --num-docs 20 \
    --question-type bridge \
    2>&1 | tee outputs/gen_triplets_hotpotqa_25k.log

# 25k examples × 2 gold docs = ~50k triplets; find the largest triplet file just generated
TRIPLET_FILE=$(ls -S data/hotpotqa_train_triplets_bridge_*.jsonl 2>/dev/null | head -1)
echo "Triplet file: $TRIPLET_FILE"

# ============================================================
# Phase 2: Train DPR
# ============================================================
echo ""
echo "PHASE 2: Training DPR (25k)"
echo "============================================================"
python scripts/train/train_retrieval_baseline.py \
    --mode dpr \
    --train-data "$TRIPLET_FILE" \
    --dataset-tag hotpotqa-25k \
    --epochs 1 \
    --batch-size 16 \
    2>&1 | tee outputs/train_dpr_hotpotqa_25k.log

DPR_MODEL=$(ls -dt outputs/ModernBERT-base-dpr-hotpotqa-25k-*/final 2>/dev/null | head -1)

echo ""
echo "Evaluating DPR (25k)..."
python scripts/eval/evaluate_retrieval_baseline.py \
    --mode dpr \
    --model-path "$DPR_MODEL" \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 100 \
    --output-file outputs/dpr_hotpotqa25k_trained_k20_eval.json \
    2>&1 | tee outputs/eval_dpr_hotpotqa25k_trained.log

# ============================================================
# Phase 3: Train ColBERT
# ============================================================
echo ""
echo "PHASE 3: Training ColBERT (25k)"
echo "============================================================"
python scripts/train/train_retrieval_baseline.py \
    --mode colbert \
    --train-data "$TRIPLET_FILE" \
    --dataset-tag hotpotqa-25k \
    --epochs 1 \
    --batch-size 16 \
    2>&1 | tee outputs/train_colbert_hotpotqa_25k.log

COLBERT_MODEL=$(ls -dt outputs/ModernBERT-base-colbert-hotpotqa-25k-*/final 2>/dev/null | head -1)

echo ""
echo "Evaluating ColBERT (25k)..."
python scripts/eval/evaluate_retrieval_baseline.py \
    --mode colbert \
    --model-path "$COLBERT_MODEL" \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 100 \
    --output-file outputs/colbert_hotpotqa25k_trained_k20_eval.json \
    2>&1 | tee outputs/eval_colbert_hotpotqa25k_trained.log

echo ""
echo "============================================================"
echo "All retrieval baselines (25k) complete!"
echo "============================================================"
