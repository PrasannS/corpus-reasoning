#!/bin/bash
# Debug evaluate_chunked.py interactively with pdb breakpoints.
# Usage: bash scripts/debug_eval_chunked.sh [--lora-path outputs/nq-rag-chunked-lora]

source activate corpus-reasoning-eval 2>/dev/null

cd /accounts/projects/sewonm/prasann/projects/corpus-reasoning

CUDA_VISIBLE_DEVICES=0 python -m pdb scripts/evaluate_chunked.py \
    --datasets nq \
    --num-docs 20 \
    --max-test-samples 1 \
    "$@"
