"""Training with chunked document attention.

Documents attend only within themselves; query/answer tokens attend to everything.
Uses HuggingFace Trainer + PEFT LoRA + DeepSpeed, with SDPA for custom 4D masks.

Usage:
    accelerate launch --num_processes 4 scripts/train_chunked.py configs/nq_rag_chunked_lora.yml
    python scripts/train_chunked.py configs/nq_rag_chunked_lora.yml  # single GPU
"""

import argparse
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
from lib.io import load_jsonl
from lib.io import ALPACA_TEMPLATE
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    build_chunked_causal_mask, reorder_query,
)


class ChunkedDataset(Dataset):
    """Dataset that wraps documents with boundary tokens and builds labels."""

    def __init__(self, data_path, tokenizer, max_len, doc_start_id, doc_end_id, query_position="after"):
        self.examples = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.query_position = query_position

        # Precompute response marker tokens for label masking
        self.response_marker = tokenizer.encode(
            "### Response:\n", add_special_tokens=False
        )

    def __len__(self):
        return len(self.examples)

    def _find_response_start(self, input_ids):
        """Find the token index where the response/output begins."""
        ids = input_ids.tolist()
        marker = self.response_marker
        for i in range(len(ids) - len(marker) + 1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)
        return len(ids)  # fallback: no masking

    def __getitem__(self, idx):
        ex = self.examples[idx]

        # Reorder query if needed, then wrap documents with boundary tokens
        input_text = reorder_query(ex["input"], self.query_position)
        wrapped_input = wrap_documents(input_text)
        prompt = ALPACA_TEMPLATE.format(instruction=ex["instruction"], input=wrapped_input)
        full_text = prompt + ex["output"] + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text, truncation=True, max_length=self.max_len,
            return_tensors="pt", padding=False,
        )
        input_ids = encoding.input_ids.squeeze(0)

        # Labels: -100 for prompt, actual IDs for output
        labels = input_ids.clone()
        response_start = self._find_response_start(input_ids)
        labels[:response_start] = -100

        return {"input_ids": input_ids, "labels": labels}


class ChunkedCollator:
    """Collator that builds per-example 4D chunked attention masks.

    Supports batch_size > 1 by padding input_ids/labels to max length
    and building separate masks per example, then stacking.

    When standard_attention=True, uses standard causal masking (no doc isolation).
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
            pad_len = max_len - ids.size(0)

            # Right-pad input_ids and labels
            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])

            if self.standard_attention:
                # Standard causal mask — no doc isolation
                orig_len = f["input_ids"].size(0)
                dtype = torch.bfloat16
                min_val = torch.finfo(dtype).min
                mask = torch.triu(torch.full((max_len, max_len), min_val, dtype=dtype), diagonal=1)
                # Mask out padding columns
                if pad_len > 0:
                    mask[:, orig_len:] = min_val
                mask = mask.unsqueeze(0)  # (1, max_len, max_len)
            else:
                # Build chunked mask for this example, then pad to max_len
                mask = build_chunked_causal_mask(
                    f["input_ids"], self.doc_start_id, self.doc_end_id,
                )  # (1, 1, orig_len, orig_len)

                if pad_len > 0:
                    min_val = torch.finfo(mask.dtype).min
                    orig_len = f["input_ids"].size(0)
                    full_mask = torch.full((1, 1, max_len, max_len), min_val, dtype=mask.dtype)
                    full_mask[:, :, :orig_len, :orig_len] = mask
                    mask = full_mask

                mask = mask.squeeze(0)  # (1, max_len, max_len)

            batch_ids.append(ids)
            batch_labels.append(labs)
            batch_masks.append(mask)

        return {
            "input_ids": torch.stack(batch_ids),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.stack(batch_masks),  # (batch, 1, max_len, max_len)
        }


class LogFileCallback(TrainerCallback):
    """Write training logs to a JSONL file for easy monitoring via tail -f."""

    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_world_process_zero:
            entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 4), **logs}
            with open(self.log_path, "a") as f:
                f.write(_json.dumps(entry) + "\n")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train with chunked document attention")
    parser.add_argument("config", help="YAML config file (Axolotl format)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_path = cfg["datasets"][0]["path"]

    # Load tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    # Load model with SDPA (required for custom 4D masks; Flash Attention 2 doesn't support them)
    print(f"Loading {cfg['base_model']} with attn_implementation=sdpa")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new token embeddings as mean of existing
    with torch.no_grad():
        mean_emb = model.get_input_embeddings().weight[:-2].mean(dim=0)
        model.get_input_embeddings().weight[-2] = mean_emb
        model.get_input_embeddings().weight[-1] = mean_emb

    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

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
    dataset = ChunkedDataset(
        data_path, tokenizer, cfg.get("sequence_len", 8192),
        doc_start_id, doc_end_id, query_position=query_position,
    )
    print(f"Loaded {len(dataset)} training examples from {data_path}")

    standard_attention = cfg.get("standard_attention", False)
    collator = ChunkedCollator(doc_start_id, doc_end_id, tokenizer.pad_token_id,
                               standard_attention=standard_attention)

    # Compute total steps for save strategy (account for DDP world size)
    world_size = max(torch.cuda.device_count(), 1)
    total_steps = len(dataset) // (
        cfg.get("micro_batch_size", 1) * cfg.get("gradient_accumulation_steps", 8) * world_size
    )
    saves = cfg.get("saves_per_epoch", 1)
    save_steps = max(total_steps // saves, 1) if saves > 0 else total_steps

    # Build informative run name from config params
    output_dir = cfg.get("output_dir", "./outputs/chunked-lora")
    data_stem = Path(data_path).stem  # e.g. nq_train_k20_random
    lr = float(cfg.get("learning_rate", 5e-4))
    epochs = cfg.get("num_epochs", 1)
    grad_acc = cfg.get("gradient_accumulation_steps", 8)
    lora_r = cfg.get("lora_r", 16)
    seq_len = cfg.get("sequence_len", 8192)
    qpos = query_position[:1]  # "b" or "a"
    attn_tag = "std" if standard_attention else "chunked"
    run_name = (
        cfg.get("wandb_name")
        or f"{attn_tag}_{data_stem}_q{qpos}_lr{lr}_ep{epochs}_r{lora_r}_seq{seq_len}_ga{grad_acc}"
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.get("micro_batch_size", 1),
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
        dataloader_pin_memory=False,
        deepspeed=cfg.get("deepspeed"),
        report_to=["wandb", "tensorboard"] if cfg.get("wandb_project") else ["tensorboard"],
        run_name=run_name,
    )

    # wandb is initialized by HF Trainer via report_to; set env vars for project/name
    if cfg.get("wandb_project"):
        import os
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
    print(f"  Monitor: tail -f {log_file}")
    print(f"  Total examples: {len(dataset)}, steps/epoch: {total_steps}")

    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
