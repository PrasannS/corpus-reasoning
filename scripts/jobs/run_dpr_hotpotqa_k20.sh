#!/bin/bash
#SBATCH --job-name=dpr-hotpotqa
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/dpr_hotpotqa_k20_%j.log
#SBATCH --error=outputs/dpr_hotpotqa_k20_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-retrieval

# ============================================================
# Phase 1: Generate triplets (2.5k examples → ~5k triplets)
# ============================================================
echo "PHASE 1: Generating triplets"
echo "============================================================"
python scripts/data/generate_retrieval_triplets.py \
    --dataset hotpotqa \
    --num-examples 2500 \
    --num-docs 20 \
    --question-type bridge \
    2>&1 | tee outputs/gen_triplets_hotpotqa_2500.log

TRIPLET_FILE=$(ls -t data/hotpotqa_train_triplets_bridge_*.jsonl 2>/dev/null | head -1)
echo "Triplet file: $TRIPLET_FILE"

# ============================================================
# Phase 2: Train DPR
# ============================================================
echo ""
echo "PHASE 2: Training DPR"
echo "============================================================"
python scripts/train/train_retrieval_baseline.py \
    --mode dpr \
    --train-data "$TRIPLET_FILE" \
    --dataset-tag hotpotqa-2500 \
    --epochs 1 \
    --batch-size 16 \
    2>&1 | tee outputs/train_dpr_hotpotqa_2500.log

DPR_MODEL=$(ls -dt outputs/ModernBERT-base-dpr-hotpotqa-2500-*/final 2>/dev/null | head -1)
echo "DPR model: $DPR_MODEL"

# ============================================================
# Phase 3: Evaluate
# ============================================================
echo ""
echo "PHASE 3: Evaluation"
echo "============================================================"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

echo "Evaluating trained DPR..."
python scripts/eval/evaluate_retrieval_baseline.py \
    --mode dpr \
    --model-path "$DPR_MODEL" \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 100 \
    --output-file outputs/dpr_hotpotqa2500_trained_k20_eval.json \
    2>&1 | tee outputs/eval_dpr_hotpotqa2500_trained.log

echo ""
echo "Evaluating base ModernBERT (untrained baseline)..."
python scripts/eval/evaluate_retrieval_baseline.py \
    --mode dpr \
    --model-path answerdotai/ModernBERT-base \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 100 \
    --output-file outputs/dpr_hotpotqa2500_base_k20_eval.json \
    2>&1 | tee outputs/eval_dpr_hotpotqa2500_base.log

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
