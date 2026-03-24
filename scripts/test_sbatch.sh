#!/bin/bash
#SBATCH --job-name=sbatch-test
#SBATCH --partition=lambda
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=outputs/test_sbatch_%j.log
#SBATCH --error=outputs/test_sbatch_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

echo "=== SBATCH TEST ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "CUDA_HOME: $CUDA_HOME"

eval "$(conda shell.bash hook)"

# Test 1: Chunked training (2.5k examples, ~2 min)
echo ""
echo "=== Test 1: Chunked training ==="
conda activate corpus-reasoning
accelerate launch --num_processes 4 \
    scripts/train_chunked_fast.py configs/test_chunked_tiny.yml
echo "  Chunked training DONE"

# Test 2: Chunked eval (5 samples)
echo ""
echo "=== Test 2: Chunked eval ==="
conda activate corpus-reasoning-eval
python scripts/evaluate_chunked.py \
    --datasets hotpotqa \
    --num-docs 20 \
    --query-position after \
    --lora-path ./outputs/test-chunked-tiny \
    --max-test-samples 5 \
    --output-file outputs/test_chunked_eval.json
echo "  Chunked eval DONE"

# Cleanup
rm -rf outputs/test-chunked-tiny outputs/test_chunked_eval.json

echo ""
echo "=== ALL TESTS PASSED ==="
