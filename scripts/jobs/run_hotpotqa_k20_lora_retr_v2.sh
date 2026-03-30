#!/bin/bash
#SBATCH --job-name=hpqa-k20-lora-v2
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=150G
#SBATCH --time=12:00:00
#SBATCH --output=outputs/hotpotqa_k20_lora_retr_v2_%j.log
#SBATCH --error=outputs/hotpotqa_k20_lora_retr_v2_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

NUM_GPUS=4
eval "$(conda shell.bash hook)"

echo "============================================================"
echo "LoRA k=20 retrieval (fixed cosine scheduler)"
echo "============================================================"

LORA_CONFIG="configs/hotpotqa_k20_std_qboth_qwen_lora_retr.yml"
LORA_OUTPUT="./outputs/hotpotqa-k20-std-qboth-qwen-lora-retr-v2"

# Use a fresh output dir so we don't skip due to existing model
sed "s|output_dir:.*|output_dir: $LORA_OUTPUT|" "$LORA_CONFIG" > /tmp/hotpotqa_k20_lora_retr_v2.yml
sed -i "s|wandb_name:.*|wandb_name: std_qboth_hotpotqa_k20_5k_qwen_lora_retr_v2|" /tmp/hotpotqa_k20_lora_retr_v2.yml

conda activate corpus-reasoning
accelerate launch --num_processes "$NUM_GPUS" \
    -m axolotl.cli.train /tmp/hotpotqa_k20_lora_retr_v2.yml \
    2>&1 | tee "$LOG_DIR/train_hotpotqa_k20_lora_retr_v2.log"
echo "  Training complete: $LORA_OUTPUT"

conda activate corpus-reasoning-eval
echo "  Evaluating LoRA on k=20 retrieval..."
python scripts/eval/evaluate_retrieval.py \
    --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl \
    --base-model Qwen/Qwen3.5-0.8B-Base \
    --lora-path "$LORA_OUTPUT" \
    --enforce-eager \
    --output-file outputs/hotpotqa_k20_qwen_lora_retr_v2_retrieval_k20eval.json \
    2>&1 | tee "$LOG_DIR/eval_hotpotqa_k20_lora_retr_v2_retrieval.log"

echo ""
echo "--- LoRA v2 (fixed cosine) Results ---"
grep -E "EM:|Recall:|F1:" "$LOG_DIR/eval_hotpotqa_k20_lora_retr_v2_retrieval.log" 2>/dev/null || echo "  (no results)"
