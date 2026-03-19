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
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent))
from lib.io import load_jsonl
from lib.io import ALPACA_TEMPLATE
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    build_chunked_causal_mask,
)


class ChunkedDataset(Dataset):
    """Dataset that wraps documents with boundary tokens and builds labels."""

    def __init__(self, data_path, tokenizer, max_len, doc_start_id, doc_end_id):
        self.examples = load_jsonl(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id

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

        # Wrap documents with boundary tokens
        wrapped_input = wrap_documents(ex["input"])
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

    Assumes batch_size=1 (required for custom per-example masks).
    """

    def __init__(self, doc_start_id, doc_end_id, pad_token_id):
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        # batch_size=1: just unsqueeze
        f = features[0]
        input_ids = f["input_ids"].unsqueeze(0)
        labels = f["labels"].unsqueeze(0)

        # Build chunked attention mask on-the-fly
        attn_mask = build_chunked_causal_mask(
            f["input_ids"], self.doc_start_id, self.doc_end_id,
        ).to(input_ids.device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attn_mask,
        }


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
    dataset = ChunkedDataset(
        data_path, tokenizer, cfg.get("sequence_len", 8192),
        doc_start_id, doc_end_id,
    )
    print(f"Loaded {len(dataset)} training examples from {data_path}")

    collator = ChunkedCollator(doc_start_id, doc_end_id, tokenizer.pad_token_id)

    # Compute total steps for save strategy
    total_steps = len(dataset) // (
        cfg.get("micro_batch_size", 1) * cfg.get("gradient_accumulation_steps", 8)
    )
    saves = cfg.get("saves_per_epoch", 1)
    save_steps = max(total_steps // saves, 1) if saves > 0 else total_steps

    # Training arguments
    output_dir = cfg.get("output_dir", "./outputs/chunked-lora")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,  # must be 1 for custom masks
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        num_train_epochs=cfg.get("num_epochs", 1),
        learning_rate=cfg.get("learning_rate", 5e-4),
        lr_scheduler_type=cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
        weight_decay=cfg.get("weight_decay", 0.0),
        bf16=True,
        logging_steps=cfg.get("logging_steps", 1),
        save_steps=save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        deepspeed=cfg.get("deepspeed"),
        report_to="wandb" if cfg.get("wandb_project") else "none",
        run_name=cfg.get("wandb_name", "chunked-train"),
    )

    if cfg.get("wandb_project"):
        import wandb
        wandb.init(project=cfg["wandb_project"], name=cfg.get("wandb_name"))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    print(f"\nStarting training. Monitor with:")
    print(f"  tail -f {output_dir}/trainer_log.jsonl")
    print(f"  Total examples: {len(dataset)}, steps/epoch: {total_steps}")

    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
