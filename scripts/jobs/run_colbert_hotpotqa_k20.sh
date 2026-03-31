#!/bin/bash
#SBATCH --job-name=colbert-hotpotqa
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/colbert_hotpotqa_k20_%j.log
#SBATCH --error=outputs/colbert_hotpotqa_k20_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-retrieval

# Reuse triplets from DPR run
TRIPLET_FILE=$(ls -t data/hotpotqa_train_triplets_bridge_*.jsonl 2>/dev/null | head -1)
echo "Triplet file: $TRIPLET_FILE"

if [ -z "$TRIPLET_FILE" ]; then
    echo "No triplet file found, generating..."
    python scripts/data/generate_retrieval_triplets.py \
        --dataset hotpotqa \
        --num-examples 2500 \
        --num-docs 20 \
        --question-type bridge \
        2>&1 | tee outputs/gen_triplets_hotpotqa_2500.log
    TRIPLET_FILE=$(ls -t data/hotpotqa_train_triplets_bridge_*.jsonl 2>/dev/null | head -1)
fi

# ============================================================
# Train ColBERT
# ============================================================
echo ""
echo "Training ColBERT"
echo "============================================================"
python scripts/train/train_retrieval_baseline.py \
    --mode colbert \
    --train-data "$TRIPLET_FILE" \
    --dataset-tag hotpotqa-2500 \
    --lr 3e-4 \
    --epochs 1 \
    --batch-size 16 \
    2>&1 | tee outputs/train_colbert_hotpotqa_2500.log

COLBERT_MODEL=$(ls -dt outputs/ModernBERT-base-colbert-hotpotqa-2500-*/final 2>/dev/null | head -1)
echo "ColBERT model: $COLBERT_MODEL"

# ============================================================
# Evaluate
# ============================================================
echo ""
echo "Evaluation"
echo "============================================================"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl"

echo "Evaluating trained ColBERT..."
python scripts/eval/evaluate_retrieval_baseline.py \
    --mode colbert \
    --model-path "$COLBERT_MODEL" \
    --eval-data "$EVAL_DATA" \
    --max-test-samples 100 \
    --output-file outputs/colbert_hotpotqa2500_trained_k20_eval.json \
    2>&1 | tee outputs/eval_colbert_hotpotqa2500_trained.log

echo ""
echo "Pipeline complete!"
