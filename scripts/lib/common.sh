#!/bin/bash
# Shared environment setup for training scripts.

setup_env() {
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH
    export TRITON_CACHE_DIR=/tmp/triton_cache
    mkdir -p "$TRITON_CACHE_DIR"
}

get_project_dir() {
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    dirname "$script_dir"
}

get_num_gpus() {
    echo "${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
}

launch_training() {
    local num_gpus=$1
    local config=$2
    accelerate launch --num_processes "$num_gpus" \
        -m axolotl.cli.train "$config"
}
