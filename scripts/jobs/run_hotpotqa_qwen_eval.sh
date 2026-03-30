#!/bin/bash
#SBATCH --job-name=hpqa-qwen-eval
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/hotpotqa_qwen_eval_%j.log
#SBATCH --error=outputs/hotpotqa_qwen_eval_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

EVAL_SAMPLES=500
BASE_MODEL="Qwen/Qwen3.5-0.8B-Base"
LORA_PATH="./outputs/hotpotqa-std-qboth-qwen-lora"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-eval

# ============================================================
# HotpotQA eval — Qwen base model
# ============================================================
echo ""
echo "============================================================"
echo "HotpotQA std qboth — Qwen base model (${EVAL_SAMPLES} eval)"
echo "============================================================"
python scripts/eval/evaluate_helmet_rag.py \
    --datasets hotpotqa \
    --num-docs 20 \
    --query-position both \
    --base-model "$BASE_MODEL" \
    --max-test-samples "$EVAL_SAMPLES" \
    --output-file "outputs/hotpotqa_qwen_std_qboth_base_${EVAL_SAMPLES}eval.json" \
    2>&1 | tee "$LOG_DIR/eval_hotpotqa_qwen_std_qboth_base.log"

# ============================================================
# HotpotQA eval — Qwen LoRA (2.5k)
# ============================================================
echo ""
echo "============================================================"
echo "HotpotQA std qboth — Qwen LoRA 2.5k (${EVAL_SAMPLES} eval)"
echo "============================================================"
python scripts/eval/evaluate_helmet_rag.py \
    --datasets hotpotqa \
    --num-docs 20 \
    --query-position both \
    --base-model "$BASE_MODEL" \
    --lora-path "$LORA_PATH" \
    --max-test-samples "$EVAL_SAMPLES" \
    --output-file "outputs/hotpotqa_qwen_std_qboth_lora2500_${EVAL_SAMPLES}eval.json" \
    2>&1 | tee "$LOG_DIR/eval_hotpotqa_qwen_std_qboth_lora2500.log"

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "HOTPOTQA QWEN EVAL COMPLETE"
echo "============================================================"
echo -n "  Base: "; grep "EM:" "$LOG_DIR/eval_hotpotqa_qwen_std_qboth_base.log" 2>/dev/null || echo "(no results)"
echo -n "  LoRA: "; grep "EM:" "$LOG_DIR/eval_hotpotqa_qwen_std_qboth_lora2500.log" 2>/dev/null || echo "(no results)"
