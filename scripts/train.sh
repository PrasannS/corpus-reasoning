#!/bin/bash
# Unified multi-GPU training launcher.
# Usage:
#   bash scripts/train.sh configs/nq_rag_lora_multigpu.yml
#   NUM_GPUS=2 bash scripts/train.sh configs/contradiction_lora_multigpu.yml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
cd "$(dirname "$SCRIPT_DIR")"
setup_env

NUM_GPUS=$(get_num_gpus)
CONFIG="${1:?Usage: bash scripts/train.sh <config.yml>}"

echo "=== Training: $CONFIG with $NUM_GPUS GPUs ==="
eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

launch_training "$NUM_GPUS" "$CONFIG"
echo "=== Training complete ==="
