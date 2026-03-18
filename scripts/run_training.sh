#!/bin/bash
# Multi-GPU training script for NIAH task using Axolotl + DeepSpeed
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Environment setup
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH

# Number of GPUs (auto-detect)
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
echo "=== Training with $NUM_GPUS GPUs ==="

# Step 1: Generate training data if not present
if [ ! -f data/niah_train.jsonl ]; then
    echo "=== Generating NIAH training data ==="
    python scripts/generate_niah_data.py --output-dir data
fi

# Step 2: Run multi-GPU training with Axolotl
CONFIG="${1:-configs/niah_lora_multigpu.yml}"
echo "=== Starting training with config: $CONFIG ==="

accelerate launch --num_processes "$NUM_GPUS" \
    -m axolotl.cli.train "$CONFIG"

echo "=== Training complete! Output saved to ./outputs/niah-lora ==="
