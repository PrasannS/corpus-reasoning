#!/bin/bash
#SBATCH --job-name=cot-gen-50k
#SBATCH --output=outputs/cot_gen_50k_%j.log
#SBATCH --error=outputs/cot_gen_50k_%j.log
#SBATCH --partition=lambda
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH --time=24:00:00

# =============================================================================
# Generate CoTs for 50k HotpotQA — 4 ablation variants (sequential)
#
# Variants:
#   1. short, gold-only
#   2. short, distractor-referencing
#   3. long, gold-only
#   4. long, distractor-referencing
#
# Monitor: tail -f outputs/cot_gen_50k_<JOBID>.log
# =============================================================================

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning-eval

set -euo pipefail

INPUT="data/hotpotqa_train_k20_shuffled_bridge_50000.jsonl"
MODEL="gemini-2.5-flash-lite"
CONCURRENT=10

echo "=============================================="
echo "CoT Generation — 50k HotpotQA × 4 variants"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Started: $(date)"
echo "Input: $INPUT"
echo "Model: $MODEL"
echo "Concurrency: $CONCURRENT"
echo "=============================================="

# --- Variant 1: short, gold-only ---
echo ""
echo "======================================================"
echo "[1/4] short, gold-only"
echo "Started: $(date)"
echo "======================================================"
python scripts/data/generate_cot.py \
    --input-data "$INPUT" \
    --model "$MODEL" \
    --max-concurrent "$CONCURRENT" \
    --cot-detail short \
    --temperature 0.7

echo "[1/4] Done: $(date)"

# --- Variant 2: short, distractor-referencing ---
echo ""
echo "======================================================"
echo "[2/4] short, distractor-referencing"
echo "Started: $(date)"
echo "======================================================"
python scripts/data/generate_cot.py \
    --input-data "$INPUT" \
    --model "$MODEL" \
    --max-concurrent "$CONCURRENT" \
    --cot-detail short \
    --reference-distractors --num-distractors 2 \
    --temperature 0.7

echo "[2/4] Done: $(date)"

# --- Variant 3: long, gold-only ---
echo ""
echo "======================================================"
echo "[3/4] long, gold-only"
echo "Started: $(date)"
echo "======================================================"
python scripts/data/generate_cot.py \
    --input-data "$INPUT" \
    --model "$MODEL" \
    --max-concurrent "$CONCURRENT" \
    --cot-detail long \
    --temperature 0.7

echo "[3/4] Done: $(date)"

# --- Variant 4: long, distractor-referencing ---
echo ""
echo "======================================================"
echo "[4/4] long, distractor-referencing"
echo "Started: $(date)"
echo "======================================================"
python scripts/data/generate_cot.py \
    --input-data "$INPUT" \
    --model "$MODEL" \
    --max-concurrent "$CONCURRENT" \
    --cot-detail long \
    --reference-distractors --num-distractors 2 \
    --temperature 0.7

echo "[4/4] Done: $(date)"

# --- Summary ---
echo ""
echo "=============================================="
echo "All 4 variants complete: $(date)"
echo "=============================================="
echo "Output files:"
ls -lh data/hotpotqa_train_k20_shuffled_bridge_50000_cot*.jsonl 2>/dev/null
echo ""
echo "Cache stats:"
python -c "
import sqlite3
conn = sqlite3.connect('cache/cot_cache.db')
c = conn.cursor()
c.execute('SELECT COUNT(*) FROM response_cache WHERE success=1')
print(f'  Cached successes: {c.fetchone()[0]}')
c.execute('SELECT SUM(cost_usd) FROM response_cache WHERE success=1')
print(f'  Total cost: \${c.fetchone()[0]:.4f}')
conn.close()
"
