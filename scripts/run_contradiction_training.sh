#!/bin/bash
# End-to-end contradiction detection: generate data, then train with Axolotl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Environment setup
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

# Configurable parameters
NUM_CLAIMS="${NUM_CLAIMS:-20}"
NUM_CONTRADICTIONS="${NUM_CONTRADICTIONS:-3}"
NUM_TRAIN="${NUM_TRAIN:-5000}"
NUM_EVAL="${NUM_EVAL:-500}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
CONFIG="${1:-configs/contradiction_lora_multigpu.yml}"

echo "=== Contradiction Detection Training Pipeline ==="
echo "  Claims: $NUM_CLAIMS, Contradictions: $NUM_CONTRADICTIONS"
echo "  Train: $NUM_TRAIN, Eval: $NUM_EVAL"
echo "  GPUs: $NUM_GPUS, Config: $CONFIG"

# Step 1: Generate training data (needs corpus-reasoning-eval env for datasets library)
TRAIN_FILE="data/contradiction_train_n${NUM_CLAIMS}_k${NUM_CONTRADICTIONS}.jsonl"
if [ ! -f "$TRAIN_FILE" ]; then
    echo ""
    echo "=== Step 1: Generating training + eval data ==="
    eval "$(conda shell.bash hook)"
    conda activate corpus-reasoning-eval
    python scripts/generate_contradiction_data.py \
        --num-claims "$NUM_CLAIMS" \
        --num-contradictions "$NUM_CONTRADICTIONS" \
        --num-train "$NUM_TRAIN" \
        --num-eval "$NUM_EVAL" \
        --output-dir data
    conda deactivate
else
    echo ""
    echo "=== Step 1: Training data already exists at $TRAIN_FILE ==="
fi

# Step 2: Train with Axolotl (needs corpus-reasoning env)
echo ""
echo "=== Step 2: Starting multi-GPU training ==="
eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

accelerate launch --num_processes "$NUM_GPUS" \
    -m axolotl.cli.train "$CONFIG"

echo ""
echo "=== Training complete! Output saved to ./outputs/contradiction-lora ==="
