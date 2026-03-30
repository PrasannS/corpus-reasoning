#!/bin/bash
#SBATCH --job-name=chunked-qbefore-ad32
#SBATCH --partition=lambda
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH --output=outputs/batch_%j.log
#SBATCH --error=outputs/batch_%j.log
set -eo pipefail

PROJECT_DIR="/accounts/projects/sewonm/prasann/projects/corpus-reasoning"
cd "$PROJECT_DIR"

source "$PROJECT_DIR/scripts/lib/common.sh"
setup_env

eval "$(conda shell.bash hook)"
conda activate corpus-reasoning

echo "=== Job $SLURM_JOB_ID started at $(date) ==="
echo "  Config: configs/hotpotqa_k20_chunked_qbefore_qwen_lora_ad32.yml"
echo "  GPUs: $CUDA_VISIBLE_DEVICES"

# Train with chunked attention
accelerate launch --num_processes 4 \
    scripts/train/train_chunked_fast.py \
    configs/hotpotqa_k20_chunked_qbefore_qwen_lora_ad32.yml \
    2>&1 | tee outputs/chunked_qbefore_ad32_train.log

echo "=== Training complete at $(date) ==="
