#!/bin/bash
#SBATCH --job-name=eval-unified-qwen
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/eval_unified_qwen_%j.log
#SBATCH --error=outputs/eval_unified_qwen_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-eval

LORA_PATH="./outputs/hotpotqa-chunked-qboth-qwen-lora"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_bridge_500.jsonl"

echo "============================================================"
echo "Unified eval: chunked-sdpa retrieval"
echo "============================================================"
python scripts/eval/evaluate.py \
    --backend chunked-sdpa \
    --task retrieval \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --query-position both \
    --max-test-samples 100 \
    --output-file outputs/unified_chunked_retrieval_qboth_qwen_eval.json \
    2>&1 | tee outputs/eval_unified_chunked_retrieval_qboth_qwen.log

echo ""
echo "============================================================"
echo "Unified eval: standard retrieval (baseline)"
echo "============================================================"
python scripts/eval/evaluate.py \
    --backend standard \
    --task retrieval \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --query-position both \
    --max-test-samples 100 \
    --output-file outputs/unified_standard_retrieval_qboth_qwen_eval.json \
    2>&1 | tee outputs/eval_unified_standard_retrieval_qboth_qwen.log

echo ""
echo "Eval complete!"
