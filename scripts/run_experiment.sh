#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Unified experiment runner for training + evaluation.
#
# This script consolidates the common pattern shared by all experiment scripts:
#   1. Set up environment (CUDA, conda)
#   2. Train a model (skip if output already exists)
#   3. Evaluate on one or more tasks
#
# It can be called directly or sourced by lightweight wrapper scripts that
# just set environment variables and call run_experiment().
#
# Usage (direct):
#   bash scripts/run_experiment.sh \
#       --config configs/hotpotqa_std_qboth_qwen_lora.yml \
#       --train-mode axolotl \
#       --eval-script scripts/evaluate_helmet_rag.py \
#       --eval-args "--datasets hotpotqa --num-docs 20 --query-position both"
#
# Usage (as library, from a wrapper script):
#   #!/bin/bash
#   #SBATCH --job-name=my-experiment ...
#   source scripts/run_experiment.sh
#   run_single_experiment \
#       "configs/my_config.yml" \
#       "axolotl" \
#       "scripts/evaluate_helmet_rag.py" \
#       "--datasets hotpotqa --num-docs 20"
#
# Train modes:
#   axolotl  — Standard attention training via accelerate + axolotl
#   chunked  — Chunked attention training via train_chunked.py
#
# Environment variables (optional):
#   NUM_GPUS       — Number of GPUs (default: 4)
#   TRAIN_ENV      — Conda env for training (default: corpus-reasoning)
#   EVAL_ENV       — Conda env for evaluation (default: corpus-reasoning-eval)
#   PROJECT_DIR    — Project root (default: auto-detected)
#   SKIP_TRAINING  — If set to 1, skip training and only evaluate
# ═══════════════════════════════════════════════════════════════════════════════
set -eo pipefail

# Default configuration
NUM_GPUS="${NUM_GPUS:-4}"
TRAIN_ENV="${TRAIN_ENV:-corpus-reasoning}"
EVAL_ENV="${EVAL_ENV:-corpus-reasoning-eval}"
PROJECT_DIR="${PROJECT_DIR:-/accounts/projects/sewonm/prasann/projects/corpus-reasoning}"

# ── Environment setup ──

setup_experiment_env() {
    # Set up CUDA and cache directories
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export TRITON_CACHE_DIR=/tmp/triton_cache
    mkdir -p "$TRITON_CACHE_DIR"
    mkdir -p "$PROJECT_DIR/outputs/batch_logs"
    cd "$PROJECT_DIR"
    eval "$(conda shell.bash hook)"
}

# ── Core functions ──

train_model() {
    # Train a model if output doesn't already exist.
    #
    # Args:
    #   $1 — Config file path (YAML)
    #   $2 — Train mode: "axolotl" or "chunked"
    #   $3 — Output directory (checked for existing model)
    local config="$1"
    local train_mode="$2"
    local output_dir="$3"
    local log_file="outputs/batch_logs/train_$(basename "$config" .yml).log"

    # Check if model already exists (LoRA: adapter_config.json, Full FT: model.safetensors)
    if [ -f "$output_dir/adapter_config.json" ] || [ -f "$output_dir/model.safetensors" ]; then
        echo "  Model already exists at $output_dir, skipping training"
        return 0
    fi

    if [ "${SKIP_TRAINING:-0}" = "1" ]; then
        echo "  SKIP_TRAINING=1, skipping training for $config"
        return 0
    fi

    echo "  Training: $config -> $output_dir"
    conda activate "$TRAIN_ENV"

    case "$train_mode" in
        axolotl)
            accelerate launch --num_processes "$NUM_GPUS" \
                -m axolotl.cli.train "$config" \
                2>&1 | tee "$log_file"
            ;;
        chunked)
            accelerate launch --num_processes "$NUM_GPUS" \
                scripts/train_chunked.py "$config" \
                2>&1 | tee "$log_file"
            ;;
        *)
            echo "ERROR: Unknown train mode '$train_mode' (expected 'axolotl' or 'chunked')"
            return 1
            ;;
    esac

    echo "  Training complete: $output_dir"
}

eval_model() {
    # Evaluate a trained model.
    #
    # Args:
    #   $1 — Eval script path
    #   $2 — Additional eval arguments (as a single string)
    #   $3 — Log file path
    local eval_script="$1"
    local eval_args="$2"
    local log_file="$3"

    echo "  Evaluating: $eval_script $eval_args"
    conda activate "$EVAL_ENV"

    # shellcheck disable=SC2086
    python "$eval_script" $eval_args \
        2>&1 | tee "$log_file"

    echo "  Eval complete. Log: $log_file"
}

run_single_experiment() {
    # Run a complete train + eval experiment.
    #
    # Args:
    #   $1 — Config file path
    #   $2 — Train mode ("axolotl" or "chunked")
    #   $3 — Eval script path
    #   $4 — Eval arguments (as a string)
    #   $5 — (Optional) Output dir override. If not set, extracted from config.
    local config="$1"
    local train_mode="$2"
    local eval_script="$3"
    local eval_args="$4"
    local output_dir="${5:-}"

    # Extract output_dir from config if not provided
    if [ -z "$output_dir" ]; then
        output_dir=$(grep "^output_dir:" "$config" | sed 's/output_dir: *//' | tr -d "'" | tr -d '"')
    fi

    local config_name
    config_name=$(basename "$config" .yml)
    local eval_log="outputs/batch_logs/eval_${config_name}.log"

    echo ""
    echo "============================================================"
    echo "Experiment: $config_name"
    echo "  Config:     $config"
    echo "  Train mode: $train_mode"
    echo "  Output:     $output_dir"
    echo "============================================================"

    train_model "$config" "$train_mode" "$output_dir"
    eval_model "$eval_script" "$eval_args" "$eval_log"
}

# ── CLI mode (when called directly, not sourced) ──

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Parse arguments
    CONFIG=""
    TRAIN_MODE="axolotl"
    EVAL_SCRIPT=""
    EVAL_ARGS=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)      CONFIG="$2"; shift 2 ;;
            --train-mode)  TRAIN_MODE="$2"; shift 2 ;;
            --eval-script) EVAL_SCRIPT="$2"; shift 2 ;;
            --eval-args)   EVAL_ARGS="$2"; shift 2 ;;
            --num-gpus)    NUM_GPUS="$2"; shift 2 ;;
            --help)
                echo "Usage: $0 --config CONFIG --train-mode MODE --eval-script SCRIPT --eval-args ARGS"
                echo ""
                echo "Options:"
                echo "  --config      Axolotl YAML config file"
                echo "  --train-mode  'axolotl' (standard) or 'chunked' (document-isolated attention)"
                echo "  --eval-script Python eval script to run after training"
                echo "  --eval-args   Arguments for the eval script (as quoted string)"
                echo "  --num-gpus    Number of GPUs (default: 4)"
                exit 0
                ;;
            *) echo "Unknown argument: $1"; exit 1 ;;
        esac
    done

    if [ -z "$CONFIG" ]; then
        echo "Error: --config is required"
        exit 1
    fi

    setup_experiment_env
    echo "=== Experiment started at $(date) ==="
    echo "  GPUs: ${CUDA_VISIBLE_DEVICES:-all}"

    run_single_experiment "$CONFIG" "$TRAIN_MODE" "$EVAL_SCRIPT" "$EVAL_ARGS"

    echo ""
    echo "=== Experiment complete at $(date) ==="
fi
