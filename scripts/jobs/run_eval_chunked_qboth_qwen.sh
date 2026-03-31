#!/bin/bash
#SBATCH --job-name=eval-chunked-qwen
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=outputs/eval_chunked_qwen_%j.log
#SBATCH --error=outputs/eval_chunked_qwen_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-eval

LORA_PATH="./outputs/hotpotqa-chunked-qboth-qwen-lora"
EVAL_DATA="data/hotpotqa_eval_k20_shuffled_bridge_500.jsonl"

echo "Evaluating chunked attention (retrieval task)..."
python scripts/eval/evaluate_chunked.py \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --query-position both \
    --max-test-samples 100 \
    --output-file outputs/chunked_qboth_qwen_eval_results.json \
    2>&1 | tee outputs/eval_chunked_qboth_qwen.log

echo ""
echo "Evaluating retrieval (vLLM)..."
python scripts/eval/evaluate_retrieval.py \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_PATH" \
    --eval-data "$EVAL_DATA" \
    --query-position both \
    --max-test-samples 100 \
    --output-file outputs/retrieval_qboth_qwen_eval_results.json \
    2>&1 | tee outputs/eval_retrieval_qboth_qwen.log

echo ""
echo "Eval complete!"
