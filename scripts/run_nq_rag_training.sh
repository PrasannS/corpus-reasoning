#!/bin/bash
# End-to-end NQ RAG training: generate data, then train with Axolotl
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
NUM_EXAMPLES="${NUM_EXAMPLES:-1000}"
NUM_DOCS="${NUM_DOCS:-20}"
GOLD_POSITION="${GOLD_POSITION:-random}"
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
CONFIG="${1:-configs/nq_rag_lora_multigpu.yml}"

echo "=== NQ RAG Training Pipeline ==="
echo "  Examples: $NUM_EXAMPLES, Docs: $NUM_DOCS, Gold position: $GOLD_POSITION"
echo "  GPUs: $NUM_GPUS, Config: $CONFIG"

# Step 1: Generate training data (needs corpus-reasoning-eval env for datasets library)
DATA_FILE="data/nq_train_k${NUM_DOCS}_${GOLD_POSITION}.jsonl"
if [ ! -f "$DATA_FILE" ]; then
    echo ""
    echo "=== Step 1: Generating training data ==="
    # Use eval env which has the datasets library
    eval "$(conda shell.bash hook)"
    conda activate corpus-reasoning-eval
    python scripts/generate_nq_training_data.py \
        --num-examples "$NUM_EXAMPLES" \
        --num-docs "$NUM_DOCS" \
        --gold-position "$GOLD_POSITION" \
        --output-dir data
    conda deactivate
else
    echo ""
    echo "=== Step 1: Training data already exists at $DATA_FILE ==="
fi

# Step 2: Train with Axolotl (needs corpus-reasoning env)
echo ""
echo "=== Step 2: Starting multi-GPU training ==="
eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

accelerate launch --num_processes "$NUM_GPUS" \
    -m axolotl.cli.train "$CONFIG"

echo ""
echo "=== Training complete! Output saved to ./outputs/nq-rag-lora ==="
