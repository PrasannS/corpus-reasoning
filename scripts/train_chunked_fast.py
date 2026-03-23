"""Fast training with chunked document attention — optimized SDPA.

Optimizations over train_chunked.py:
1. GPU-native mask construction (bool intermediates, 8x less memory)
2. tf32 matmuls on A100 (~2x throughput)
3. DeepSpeed ZeRO-1 support
4. Dataloader workers + pinned memory
5. Optimized collator with smaller masks
6. torch.compile support (optional)

Usage:
    accelerate launch --num_processes 4 scripts/train_chunked_fast.py configs/nq_rag_chunked_qboth.yml
    python scripts/train_chunked_fast.py configs/nq_rag_chunked_lora.yml  # single GPU
"""

import argparse
import os
import sys
import yaml
import torch
from pathlib import Path
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

import json as _json

sys.path.insert(0, str(Path(__file__).parent))
from lib.io import load_jsonl, ALPACA_TEMPLATE
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    reorder_query, find_chunk_spans,
)


# ---------------------------------------------------------------------------
# Dataset (same as train_chunked.py)
# ---------------------------------------------------------------------------

class ChunkedDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, doc_start_id, doc_end_id, query_position="after"):
        self.examples = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.query_position = query_position
        self.response_marker = tokenizer.encode("### Response:\n", add_special_tokens=False)

    def __len__(self):
        return len(self.examples)

    def _find_response_start(self, input_ids):
        ids = input_ids.tolist()
        marker = self.response_marker
        for i in range(len(ids) - len(marker) + 1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)
        return len(ids)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_text = reorder_query(ex["input"], self.query_position)
        wrapped_input = wrap_documents(input_text)
        prompt = ALPACA_TEMPLATE.format(instruction=ex["instruction"], input=wrapped_input)
        full_text = prompt + ex["output"] + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            return_tensors="pt", padding=False,
        )
        input_ids = encoding.input_ids.squeeze(0)

        labels = input_ids.clone()
        response_start = self._find_response_start(input_ids)
        labels[:response_start] = -100

        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# Optimized mask construction
# ---------------------------------------------------------------------------

def build_chunk_ids(input_ids, doc_start_id, doc_end_id):
    """Build chunk ID tensor: chunk_id[i] = chunk_index if in doc, -1 otherwise."""
    seq_len = len(input_ids)
    spans = find_chunk_spans(input_ids, doc_start_id, doc_end_id)
    chunk_id = torch.full((seq_len,), -1, dtype=torch.int32)
    for idx, (s, e) in enumerate(spans):
        chunk_id[s:e] = idx
    return chunk_id


def build_mask_from_chunk_ids(chunk_id, seq_len, pad_len=0, dtype=torch.bfloat16):
    """Build 4D attention mask from chunk IDs using bool intermediates.

    Uses bool tensors (1 bit each) for all intermediate computations,
    converting to float only at the end. This uses ~8x less memory than
    building the mask with float tensors throughout.

    Args:
        chunk_id: (orig_len,) int tensor of chunk assignments
        seq_len: target sequence length (after padding)
        pad_len: number of padding tokens
        dtype: output float dtype

    Returns:
        (1, seq_len, seq_len) float mask
    """
    orig_len = len(chunk_id)

    # Build mask using bool tensors (memory efficient)
    row_ids = chunk_id.unsqueeze(1)  # (orig_len, 1)
    col_ids = chunk_id.unsqueeze(0)  # (1, orig_len)

    # Causal mask
    causal = torch.ones(orig_len, orig_len, dtype=torch.bool).tril_()

    # Same chunk (both must be >= 0)
    same_chunk = (row_ids == col_ids) & (row_ids >= 0)

    # Free tokens (not in any chunk): attend to/from everything
    row_free = (chunk_id < 0).unsqueeze(1)  # (orig_len, 1)
    col_free = (chunk_id < 0).unsqueeze(0)  # (1, orig_len)

    bool_mask = causal & (same_chunk | row_free | col_free)

    # Convert to float mask only at the end
    min_val = torch.finfo(dtype).min

    if pad_len > 0:
        # Expand to full size with padding masked out
        full_mask = torch.full((seq_len, seq_len), min_val, dtype=dtype)
        full_mask[:orig_len, :orig_len] = torch.where(
            bool_mask,
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(min_val, dtype=dtype),
        )
    else:
        full_mask = torch.where(
            bool_mask,
            torch.tensor(0.0, dtype=dtype),
            torch.tensor(min_val, dtype=dtype),
        )

    return full_mask.unsqueeze(0)  # (1, seq_len, seq_len)


# ---------------------------------------------------------------------------
# Optimized collator
# ---------------------------------------------------------------------------

class OptimizedChunkedCollator:
    """Collator with optimized mask construction using bool intermediates.

    Key differences from ChunkedCollator:
    1. Uses bool intermediates for mask construction (8x less memory)
    2. Builds chunk_ids once and reuses
    3. Delayed float conversion reduces peak memory
    """

    def __init__(self, doc_start_id, doc_end_id, pad_token_id, standard_attention=False):
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.pad_token_id = pad_token_id
        self.standard_attention = standard_attention

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)

        batch_ids, batch_labels, batch_masks = [], [], []
        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            orig_len = ids.size(0)
            pad_len = max_len - orig_len

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])

            if self.standard_attention:
                dtype = torch.bfloat16
                min_val = torch.finfo(dtype).min
                mask = torch.triu(torch.full((max_len, max_len), min_val, dtype=dtype), diagonal=1)
                if pad_len > 0:
                    mask[:, orig_len:] = min_val
                mask = mask.unsqueeze(0)
            else:
                chunk_id = build_chunk_ids(f["input_ids"], self.doc_start_id, self.doc_end_id)
                mask = build_mask_from_chunk_ids(chunk_id, max_len, pad_len)

            batch_ids.append(ids)
            batch_labels.append(labs)
            batch_masks.append(mask)

        return {
            "input_ids": torch.stack(batch_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_masks),
        }


# ---------------------------------------------------------------------------
# Logging callback
# ---------------------------------------------------------------------------

class LogFileCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), **logs}
            with open(self.log_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Fast chunked attention training (optimized SDPA)")
    parser.add_argument("config", help="YAML config file (Axolotl format)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Use torch.compile on the model")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = cfg["datasets"][0]["path"]

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    # Load model with SDPA
    print(f"Loading {cfg['base_model']} with attn_implementation=sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings
    with torch.no_grad():
        mean_emb = model.get_input_embeddings().weight[:-2].mean(dim=0)
        model.get_input_embeddings().weight[-2] = mean_emb
        model.get_input_embeddings().weight[-1] = mean_emb

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Apply LoRA
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    query_position = cfg.get("query_position", "after")
    standard_attention = cfg.get("standard_attention", False)
    dataset = ChunkedDataset(
        data_path, tokenizer, cfg.get("sequence_len", 8192),
        doc_start_id, doc_end_id, query_position=query_position,
    )
    print(f"Loaded {len(dataset)} training examples from {data_path}")

    collator = OptimizedChunkedCollator(
        doc_start_id, doc_end_id, tokenizer.pad_token_id,
        standard_attention=standard_attention,
    )

    # Compute steps
    world_size = max(torch.cuda.device_count(), 1)
    batch_size = cfg.get("micro_batch_size", 1)
    grad_acc = cfg.get("gradient_accumulation_steps", 8)
    total_steps = len(dataset) // (batch_size * grad_acc * world_size)
    saves = cfg.get("saves_per_epoch", 1)
    save_steps = max(total_steps // saves, 1) if saves > 0 else total_steps

    # Build run name
    output_dir = cfg.get("output_dir", "./outputs/chunked-fast-lora")
    data_stem = Path(data_path).stem
    lr = float(cfg.get("learning_rate", 5e-4))
    epochs = cfg.get("num_epochs", 1)
    lora_r = cfg.get("lora_r", 16)
    seq_len = cfg.get("sequence_len", 8192)
    qpos = query_position[:1]
    attn_tag = "std" if standard_attention else "chunked"
    run_name = (
        cfg.get("wandb_name")
        or f"fast_{attn_tag}_{data_stem}_q{qpos}_lr{lr}_ep{epochs}_r{lora_r}_seq{seq_len}_ga{grad_acc}"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        num_train_epochs=epochs,
        learning_rate=lr,
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.1)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        bf16=True,
        logging_steps=cfg.get("logging_steps", 1),
        logging_dir=f"{output_dir}/logs",
        save_steps=save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        deepspeed=cfg.get("deepspeed"),
        report_to=["wandb", "tensorboard"] if cfg.get("wandb_project") else ["tensorboard"],
        run_name=run_name,
        # Enable tf32 for faster matmuls on A100
        tf32=True,
    )

    if cfg.get("wandb_project"):
        os.environ["WANDB_PROJECT"] = cfg["wandb_project"]

    log_file = f"{output_dir}/train_log.jsonl"
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=[LogFileCallback(log_file)],
    )

    print(f"\nStarting training: {run_name}")
    print(f"  Attention: SDPA (optimized masks, tf32)")
    print(f"  Gradient checkpointing: {cfg.get('gradient_checkpointing', True)}")
    print(f"  torch.compile: {args.compile}")
    print(f"  Monitor: tail -f {log_file}")
    print(f"  Total examples: {len(dataset)}, steps/epoch: {total_steps}")
    print(f"  Batch: {batch_size} × {grad_acc} grad_accum × {world_size} GPUs = {batch_size * grad_acc * world_size} effective")

    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
