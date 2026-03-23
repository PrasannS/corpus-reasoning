#!/bin/bash
# Hyperparameter search for standard+both NQ config
# Varies LR, LoRA rank, epochs, weight decay, dropout, batch size, warmup, scheduler
# Uses axolotl on 2500 examples, evaluates each with vLLM
set -eo pipefail

cd "$(dirname "$0")/.."
SCRIPT_DIR="$(pwd)/scripts"
source "$SCRIPT_DIR/lib/common.sh"
setup_env

LOG_DIR="outputs/hp_search_logs"
mkdir -p "$LOG_DIR" configs/hp_search

# Baseline: lr=5e-4, r=16, a=32, dropout=0.05, wd=0.0, epochs=1, ga=8, warmup=0.1, sched=cosine

generate_config() {
    local name=$1 lr=$2 rank=$3 alpha=$4 dropout=$5 wd=$6 epochs=$7 ga=$8 warmup=$9 sched=${10}
    cat > "configs/hp_search/${name}.yml" <<YAML
base_model: NousResearch/Llama-3.2-1B

datasets:
  - path: data/nq_train_k20_random_2500_qboth.jsonl
    type: alpaca
    ds_type: json

val_set_size: 0.0
dataset_prepared_path: data/prepared_${name}
output_dir: ./outputs/${name}

wandb_project: corpus-reasoning
wandb_run_id:
wandb_watch:
wandb_name: ${name}
wandb_log_model:

sequence_len: 8192
micro_batch_size: 1
gradient_accumulation_steps: ${ga}
num_epochs: ${epochs}
learning_rate: ${lr}

adapter: lora
lora_r: ${rank}
lora_alpha: ${alpha}
lora_dropout: ${dropout}
lora_target_modules: [gate_proj, down_proj, up_proj, q_proj, v_proj, k_proj, o_proj]
sample_packing: true
eval_sample_packing: true
optimizer: adamw_8bit
lr_scheduler: ${sched}
bf16: auto
tf32: false
gradient_checkpointing: true
flash_attention: true
logging_steps: 1
saves_per_epoch: 1
evals_per_epoch: 0
warmup_ratio: ${warmup}
weight_decay: ${wd}
deepspeed: configs/deepspeed_zero1.json
special_tokens:
  pad_token: "<|end_of_text|>"
YAML
}

# Run one config: generate, train, evaluate
run_config() {
    local name=$1 lr=$2 rank=$3 alpha=$4 dropout=$5 wd=$6 epochs=$7 ga=$8 warmup=$9 sched=${10}

    generate_config "$name" "$lr" "$rank" "$alpha" "$dropout" "$wd" "$epochs" "$ga" "$warmup" "$sched"

    echo "======================================"
    echo "Config: $name"
    echo "  lr=$lr rank=$rank alpha=$alpha dropout=$dropout wd=$wd epochs=$epochs ga=$ga warmup=$warmup sched=$sched"
    echo "======================================"

    # Skip if output already exists
    if [ -d "outputs/${name}" ] && [ -f "outputs/${name}/adapter_model.safetensors" ]; then
        echo "  Skipping training — checkpoint already exists"
    else
        eval "$(conda shell.bash hook 2>/dev/null)" || true
        conda activate corpus-reasoning 2>/dev/null || true
        accelerate launch --num_processes "$NUM_GPUS" \
            -m axolotl.cli.train "configs/hp_search/${name}.yml" \
            2>&1 | tee "$LOG_DIR/${name}_train.log"
    fi

    # Skip eval if results already recorded
    if grep -q "^${name} |" "$RESULTS_FILE" 2>/dev/null; then
        echo "  Skipping eval — results already recorded"
        return
    fi

    echo "  Evaluating..."
    eval "$(conda shell.bash hook 2>/dev/null)" || true
    conda activate corpus-reasoning-eval 2>/dev/null || true

    EVAL_OUTPUT=$(python scripts/evaluate_helmet_rag.py \
        --lora-path "outputs/${name}" \
        --datasets nq --num-docs 20 --query-position both \
        --max-test-samples 100 \
        2>&1 | tee "$LOG_DIR/${name}_eval.log")

    # Extract EM and F1 from summary line like "EM:  38.0%  SubEM:  40.0%  F1:  48.5%"
    EM=$(echo "$EVAL_OUTPUT" | grep -oP 'EM:\s+\K[\d.]+' | tail -1 || echo "?")
    F1=$(echo "$EVAL_OUTPUT" | grep -oP 'F1:\s+\K[\d.]+' | tail -1 || echo "?")

    echo "$name | $lr | $rank | $alpha | $dropout | $wd | $epochs | $ga | $warmup | $sched | $EM | $F1" >> "$RESULTS_FILE"
    echo "  => EM=${EM}% F1=${F1}%"
    echo ""
}

RESULTS_FILE="$LOG_DIR/hp_results.txt"
echo "config | lr | rank | alpha | dropout | wd | epochs | ga | warmup | sched | EM | F1" > "$RESULTS_FILE"

NUM_GPUS=$(get_num_gpus)

echo "=== Starting HP search ($(date)) ==="
echo "GPUs: $NUM_GPUS"
echo ""

# --- LR sweep ---
#                     name           lr     rank alpha drop   wd   ep ga warmup sched
run_config hp_lr1e-4   1e-4  16    32   0.05  0.0  1  8  0.1  cosine
run_config hp_lr2e-4   2e-4  16    32   0.05  0.0  1  8  0.1  cosine
run_config hp_lr3e-4   3e-4  16    32   0.05  0.0  1  8  0.1  cosine
run_config hp_lr5e-4   5e-4  16    32   0.05  0.0  1  8  0.1  cosine
run_config hp_lr7e-4   7e-4  16    32   0.05  0.0  1  8  0.1  cosine
run_config hp_lr1e-3   1e-3  16    32   0.05  0.0  1  8  0.1  cosine

# --- LoRA rank sweep ---
run_config hp_r8       5e-4  8     16   0.05  0.0  1  8  0.1  cosine
# hp_r16 = hp_lr5e-4 (baseline), skip
run_config hp_r32      5e-4  32    64   0.05  0.0  1  8  0.1  cosine
run_config hp_r64      5e-4  64    128  0.05  0.0  1  8  0.1  cosine

# --- Epoch sweep ---
run_config hp_ep2      5e-4  16    32   0.05  0.0  2  8  0.1  cosine
run_config hp_ep3      5e-4  16    32   0.05  0.0  3  8  0.1  cosine

# --- Weight decay ---
run_config hp_wd0.01   5e-4  16    32   0.05  0.01 1  8  0.1  cosine
run_config hp_wd0.05   5e-4  16    32   0.05  0.05 1  8  0.1  cosine
run_config hp_wd0.1    5e-4  16    32   0.05  0.1  1  8  0.1  cosine

# --- Dropout ---
run_config hp_drop0    5e-4  16    32   0.0   0.0  1  8  0.1  cosine
run_config hp_drop0.1  5e-4  16    32   0.1   0.0  1  8  0.1  cosine

# --- Gradient accumulation (effective batch size) ---
run_config hp_ga4      5e-4  16    32   0.05  0.0  1  4  0.1  cosine
run_config hp_ga16     5e-4  16    32   0.05  0.0  1  16 0.1  cosine

# --- Warmup ratio ---
run_config hp_warmup0    5e-4  16  32   0.05  0.0  1  8  0.0  cosine
run_config hp_warmup0.05 5e-4  16  32   0.05  0.0  1  8  0.05 cosine
run_config hp_warmup0.2  5e-4  16  32   0.05  0.0  1  8  0.2  cosine

# --- LR scheduler ---
run_config hp_sched_linear    5e-4  16  32  0.05  0.0  1  8  0.1  linear
run_config hp_sched_constant  5e-4  16  32  0.05  0.0  1  8  0.1  constant_with_warmup

# --- Promising combos ---
run_config hp_combo1  1e-3  16  32   0.05  0.01 2  8   0.1   cosine
run_config hp_combo2  3e-4  32  64   0.05  0.0  2  8   0.1   cosine
run_config hp_combo3  1e-4  64  128  0.0   0.0  2  8   0.1   cosine
run_config hp_combo4  7e-4  32  64   0.0   0.01 1  8   0.05  cosine
run_config hp_combo5  3e-4  16  32   0.05  0.0  2  4   0.1   cosine

echo ""
echo "=== HP Search Complete ($(date)) ==="
echo "Results:"
column -t -s'|' "$RESULTS_FILE"
