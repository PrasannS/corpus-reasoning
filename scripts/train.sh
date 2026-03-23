#!/bin/bash
# Unified multi-GPU training launcher.
# Usage:
#   bash scripts/train.sh configs/nq_rag_lora_multigpu.yml
#   NUM_GPUS=2 bash scripts/train.sh configs/contradiction_lora_multigpu.yml
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/lib/common.sh"
cd "$(dirname "$SCRIPT_DIR")"
setup_env

NUM_GPUS=$(get_num_gpus)
CONFIG="${1:?Usage: bash scripts/train.sh <config.yml>}"

echo "=== Training: $CONFIG with $NUM_GPUS GPUs ==="
eval "$(conda shell.bash hook)"
conda activate corpus-reasoning
set -u

launch_training "$NUM_GPUS" "$CONFIG"

# Fix tokenizer for full fine-tuning: axolotl saves a stripped tokenizer_config.json
# missing added_tokens_decoder, which breaks vLLM. Use the eval env's transformers
# (which matches vLLM's expected format) to save the tokenizer.
if ! grep -q "^adapter:" "$CONFIG"; then
    BASE_MODEL=$(grep "^base_model:" "$CONFIG" | awk '{print $2}')
    OUTPUT_DIR="$(grep "^output_dir:" "$CONFIG" | sed 's/^output_dir: *//' | sed 's/^[[:space:]]*//')"
    if [ -n "$BASE_MODEL" ] && [ -d "$OUTPUT_DIR" ]; then
        echo "=== Full FT: copying tokenizer from $BASE_MODEL to $OUTPUT_DIR ==="
        conda activate corpus-reasoning-eval
        python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$BASE_MODEL')
tok.save_pretrained('$OUTPUT_DIR')
print('Tokenizer saved successfully')
"
        conda activate corpus-reasoning
    fi
fi

echo "=== Training complete ==="
