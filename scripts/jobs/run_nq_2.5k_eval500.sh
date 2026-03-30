#!/bin/bash
#SBATCH --job-name=nq-eval500
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=300G
#SBATCH --time=6:00:00
#SBATCH --output=outputs/nq_2.5k_eval500_%j.log
#SBATCH --error=outputs/nq_2.5k_eval500_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

LOG_DIR="outputs/batch_logs"
mkdir -p "$LOG_DIR"

EVAL_SAMPLES=500

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-eval

# ============================================================
# Standard attention evals (vLLM) — base + 2.5k LoRA
# ============================================================
for QPOS in after before both; do
    echo ""
    echo "============================================================"
    echo "NQ std q${QPOS} — base model (500 eval)"
    echo "============================================================"
    python scripts/eval/evaluate_helmet_rag.py \
        --datasets nq \
        --num-docs 20 \
        --query-position "$QPOS" \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "outputs/nq_2.5k_std_q${QPOS}_base_500eval.json" \
        2>&1 | tee "$LOG_DIR/eval_nq_2.5k_std_q${QPOS}_base_500.log"

    echo ""
    echo "============================================================"
    echo "NQ std q${QPOS} — 2.5k LoRA (500 eval)"
    echo "============================================================"
    python scripts/eval/evaluate_helmet_rag.py \
        --datasets nq \
        --num-docs 20 \
        --query-position "$QPOS" \
        --lora-path "./outputs/nq-rag-std-q${QPOS}" \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "outputs/nq_2.5k_std_q${QPOS}_lora_500eval.json" \
        2>&1 | tee "$LOG_DIR/eval_nq_2.5k_std_q${QPOS}_lora_500.log"
done

# ============================================================
# Chunked attention evals (HF generate) — base + 2.5k LoRA
# ============================================================
for QPOS in after before both; do
    echo ""
    echo "============================================================"
    echo "NQ chunked q${QPOS} — base model (500 eval)"
    echo "============================================================"
    python scripts/eval/evaluate_chunked.py \
        --datasets nq \
        --num-docs 20 \
        --query-position "$QPOS" \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "outputs/nq_2.5k_chunked_q${QPOS}_base_500eval.json" \
        2>&1 | tee "$LOG_DIR/eval_nq_2.5k_chunked_q${QPOS}_base_500.log"

    echo ""
    echo "============================================================"
    echo "NQ chunked q${QPOS} — 2.5k LoRA (500 eval)"
    echo "============================================================"
    python scripts/eval/evaluate_chunked.py \
        --datasets nq \
        --num-docs 20 \
        --query-position "$QPOS" \
        --lora-path "./outputs/nq-rag-chunked-q${QPOS}" \
        --max-test-samples "$EVAL_SAMPLES" \
        --output-file "outputs/nq_2.5k_chunked_q${QPOS}_lora_500eval.json" \
        2>&1 | tee "$LOG_DIR/eval_nq_2.5k_chunked_q${QPOS}_lora_500.log"
done

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================================"
echo "ALL NQ 2.5k EVALS (500 samples) COMPLETE"
echo "============================================================"
for ATT in std chunked; do
    for QPOS in after before both; do
        echo "--- NQ ${ATT} q${QPOS} ---"
        echo -n "  Base: "; grep "EM:" "$LOG_DIR/eval_nq_2.5k_${ATT}_q${QPOS}_base_500.log" 2>/dev/null || echo "(no results)"
        echo -n "  LoRA: "; grep "EM:" "$LOG_DIR/eval_nq_2.5k_${ATT}_q${QPOS}_lora_500.log" 2>/dev/null || echo "(no results)"
    done
done
