"""Training with chunked document attention using SDPA.

Chunked attention isolates documents from each other during training: each
document can only attend to itself and to "free" tokens (instruction, query,
answer), not to other documents. This is implemented via a custom 4D attention
mask passed to PyTorch's SDPA (Scaled Dot-Product Attention).

Key features:
  - Memory-efficient mask construction using bool intermediates (8x less than float)
  - tf32 matmuls on A100 (~2x throughput)
  - DeepSpeed ZeRO-1/ZeRO-2 support
  - LoRA or full fine-tuning
  - Optional standard attention mode (standard_attention in config) for comparison
  - torch.compile support (--compile flag)

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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports
from lib.io import load_jsonl
from lib.data_format import build_prompt
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    find_chunk_spans,
)


# ---------------------------------------------------------------------------
# Dataset (same as train_chunked.py)
# ---------------------------------------------------------------------------

class ChunkedDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, doc_start_id, doc_end_id,
                 query_position="after", train_on_inputs=False, task="retrieval",
                 use_titles=True, before_dummy=0, after_dummy=0):
        self.examples = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.query_position = query_position
        self.train_on_inputs = train_on_inputs
        self.task = task
        self.use_titles = use_titles
        self.before_dummy = before_dummy
        self.after_dummy = after_dummy
        self.response_marker = tokenizer.encode("### Response:\n", add_special_tokens=False)

    def __len__(self):
        return len(self.examples)

    def _find_response_start(self, input_ids):
        """Find where the response starts in the tokenized sequence.

        Scans for the "### Response:\\n" marker in token IDs. Everything before
        this marker is the prompt (instruction + input), and we mask it to -100
        in labels so the loss is only computed on the model's output tokens.
        """
        ids = input_ids.tolist()
        marker = self.response_marker
        for i in range(len(ids) - len(marker) + 1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)
        return len(ids)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt, output = build_prompt(
            ex, task=self.task, query_position=self.query_position,
            use_titles=self.use_titles, before_dummy=self.before_dummy,
            after_dummy=self.after_dummy, use_alpaca=True,
        )

        # Insert <doc_start>/<doc_end> boundary tokens around each document
        prompt = wrap_documents(prompt)
        full_text = prompt + output + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            return_tensors="pt", padding=False,
        )
        input_ids = encoding.input_ids.squeeze(0)

        # Mask prompt tokens in labels (-100) so loss is only on the output
        labels = input_ids.clone()
        if not self.train_on_inputs:
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

    standard_attention = cfg.get("standard_attention", False)

    # Only add doc boundary tokens for chunked attention — standard attention
    # doesn't need them, and resizing embeddings makes the LoRA adapter
    # incompatible with vLLM (which can't load lm_head as a LoRA weight).
    if standard_attention:
        doc_start_id, doc_end_id = None, None
    else:
        doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    # Load model with SDPA
    print(f"Loading {cfg['base_model']} with attn_implementation=sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )

    if not standard_attention:
        model.resize_token_embeddings(len(tokenizer))

        # Initialize the new <doc_start>/<doc_end> embeddings as the mean of all
        # existing embeddings. This gives a reasonable starting point so the model
        # doesn't start with random noise for these tokens.
        with torch.no_grad():
            mean_emb = model.get_input_embeddings().weight[:-2].mean(dim=0)
            model.get_input_embeddings().weight[-2] = mean_emb
            model.get_input_embeddings().weight[-1] = mean_emb

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)

    # Apply LoRA if configured, otherwise full fine-tuning.
    # LoRA adds low-rank adapter layers (much fewer trainable params).
    # Full FT trains all parameters but may need ZeRO-2 for memory.
    if cfg.get("adapter") == "lora":
        lora_config = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", ["q_proj", "v_proj"]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Full fine-tuning: optionally freeze embedding/lm_head layers
        # to reduce memory and avoid catastrophic forgetting of token embeddings
        if cfg.get("freeze_embed", False):
            for name, param in model.named_parameters():
                if "embed_tokens" in name or "lm_head" in name:
                    param.requires_grad = False
                    print(f"  Frozen: {name} ({param.numel():,} params)")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Full fine-tuning: {trainable_params:,} / {total_params:,} parameters ({trainable_params/total_params:.1%})")

    # Dataset
    query_position = cfg.get("query_position", "after")
    train_on_inputs = cfg.get("train_on_inputs", False)
    task = cfg.get("task", "retrieval")
    use_titles = not cfg.get("no_titles", False)
    before_dummy = cfg.get("before_dummy", 0)
    after_dummy = cfg.get("after_dummy", 0)
    dataset = ChunkedDataset(
        data_path, tokenizer, cfg.get("sequence_len", 8192),
        doc_start_id, doc_end_id, query_position=query_position,
        train_on_inputs=train_on_inputs, task=task, use_titles=use_titles,
        before_dummy=before_dummy, after_dummy=after_dummy,
    )
    print(f"Loaded {len(dataset)} training examples from {data_path}")
    if train_on_inputs:
        print("  Training on ALL tokens (including inputs)")

    collator = OptimizedChunkedCollator(
        doc_start_id, doc_end_id, tokenizer.pad_token_id,
        standard_attention=standard_attention,
    )

    # Compute training steps: effective_batch = micro_batch × grad_accum × GPUs
    world_size = max(torch.cuda.device_count(), 1)
    batch_size = cfg.get("micro_batch_size", 1)
    grad_acc = cfg.get("gradient_accumulation_steps", 8)
    total_steps = len(dataset) // (batch_size * grad_acc * world_size)
    # saves_per_epoch=0 means no intermediate checkpoints (save only at end)
    saves = cfg.get("saves_per_epoch", 1)
    save_steps = max(total_steps // saves, 1) if saves > 0 else total_steps

    # Build run name
    output_dir = cfg.get("output_dir", "./outputs/chunked-lora")
    data_stem = Path(data_path).stem
    lr = float(cfg.get("learning_rate", 5e-4))
    epochs = cfg.get("num_epochs", 1)
    lora_r = cfg.get("lora_r", 16)
    seq_len = cfg.get("sequence_len", 8192)
    qpos = query_position[:1]
    attn_tag = "std" if standard_attention else "chunked"
    adapter_tag = f"r{lora_r}" if cfg.get("adapter") == "lora" else "fullft"
    run_name = (
        cfg.get("wandb_name")
        or f"fast_{attn_tag}_{data_stem}_q{qpos}_lr{lr}_ep{epochs}_{adapter_tag}_seq{seq_len}_ga{grad_acc}"
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
