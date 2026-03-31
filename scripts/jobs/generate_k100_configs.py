"""Generate all k100 experiment configs programmatically."""

import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"


def std_config(dataset, query_position, after_dummy=0):
    """Standard attention config template."""
    ds_short = "nq" if dataset == "nq" else "hotpotqa"
    # NQ uses k60 (60 docs) to fit in 16k context; HotpotQA uses k100
    if dataset == "nq":
        ds_prefix = "nq_train_k60_random"
        k_tag = "k60"
    else:
        ds_prefix = "hotpotqa_train_k100_shuffled_bridge"
        k_tag = "k100"

    # Build alpaca data filename
    qp_tag = f"q{query_position}"
    ad_tag = f"_ad{after_dummy}" if after_dummy > 0 else ""
    data_file = f"{ds_prefix}_10000_alpaca_retrieval_{qp_tag}{ad_tag}.jsonl"

    # Build output/config names
    name = f"{ds_short}-{k_tag}-std-{qp_tag}{ad_tag}-qwen-lora"
    config_name = f"{ds_short}_{k_tag}_std_{qp_tag}{ad_tag}_qwen_lora"
    wandb_name = f"std_{qp_tag}{ad_tag}_{ds_short}_{k_tag}_10k_qwen_lora"

    seq_len = 16384
    packing = True

    config = {
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "datasets": [{"path": f"data/{data_file}", "type": "alpaca", "ds_type": "json"}],
        "val_set_size": 0.0,
        "dataset_prepared_path": "/tmp/axolotl_cache",
        "output_dir": f"./outputs/{name}",
        "wandb_project": "corpus-reasoning",
        "wandb_name": wandb_name,
        "task": "retrieval",
        "query_position": query_position,
        "standard_attention": True,
        "sequence_len": seq_len,
        "dataset_processes": 64,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1,
        "learning_rate": 5e-4,
        "adapter": "lora",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        "sample_packing": packing,
        "eval_sample_packing": packing,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "flash_attention": True,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero2.json",
        "special_tokens": {"pad_token": "<|endoftext|>"},
    }

    if after_dummy > 0:
        config["before_dummy"] = 0
        config["after_dummy"] = after_dummy

    return config_name, config


def chunked_config(dataset, query_position, after_dummy=0):
    """Chunked attention config template."""
    ds_short = "nq" if dataset == "nq" else "hotpotqa"
    if dataset == "nq":
        ds_prefix = "nq_train_k60_random"
        k_tag = "k60"
    else:
        ds_prefix = "hotpotqa_train_k100_shuffled_bridge"
        k_tag = "k100"

    # Chunked training reads unified format directly
    data_file = f"{ds_prefix}_10000.jsonl"

    qp_tag = f"q{query_position}"
    ad_tag = f"_ad{after_dummy}" if after_dummy > 0 else ""
    name = f"{ds_short}-{k_tag}-chunked-{qp_tag}{ad_tag}-qwen-lora"
    config_name = f"{ds_short}_{k_tag}_chunked_{qp_tag}{ad_tag}_qwen_lora"
    wandb_name = f"chunked_{qp_tag}{ad_tag}_{ds_short}_{k_tag}_10k_qwen_lora"

    config = {
        "base_model": "Qwen/Qwen3.5-0.8B-Base",
        "datasets": [{"path": f"data/{data_file}", "type": "alpaca", "ds_type": "json"}],
        "val_set_size": 0.0,
        "output_dir": f"./outputs/{name}",
        "wandb_project": "corpus-reasoning",
        "wandb_name": wandb_name,
        "task": "retrieval",
        "query_position": query_position,
        "standard_attention": False,
        "train_on_inputs": False,
        "sequence_len": 16384,
        "dataset_processes": 64,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_epochs": 1,
        "learning_rate": 5e-4,
        "before_dummy": 0,
        "after_dummy": after_dummy,
        "adapter": "lora",
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "lora_target_modules": ["gate_proj", "down_proj", "up_proj", "q_proj", "v_proj", "k_proj", "o_proj"],
        "sample_packing": False,
        "eval_sample_packing": False,
        "optimizer": "adamw_8bit",
        "lr_scheduler": "cosine",
        "bf16": "auto",
        "tf32": True,
        "gradient_checkpointing": True,
        "flash_attention": False,
        "logging_steps": 1,
        "saves_per_epoch": 0,
        "save_strategy": "no",
        "evals_per_epoch": 0,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "deepspeed": "configs/deepspeed_zero2.json",
        "special_tokens": {"pad_token": "<|endoftext|>"},
    }

    return config_name, config


def write_config(name, config):
    path = CONFIGS_DIR / f"{name}.yml"
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"  {path.name}")


def main():
    print("Generating k100 configs...")

    # --- Main experiments: NQ + HotpotQA × std/chunked × 3 query positions ---
    for ds in ["nq", "hotpotqa"]:
        for qp, ad in [("before", 1), ("after", 0), ("both", 0)]:
            # Standard attention
            name, cfg = std_config(ds, qp, after_dummy=ad)
            write_config(name, cfg)

            # Chunked attention
            name, cfg = chunked_config(ds, qp, after_dummy=ad)
            write_config(name, cfg)

    # --- Dummy token ablation: chunked, qbefore, ad100/ad500 ---
    for ds in ["nq", "hotpotqa"]:
        for ad in [100, 500]:
            name, cfg = chunked_config(ds, "before", after_dummy=ad)
            write_config(name, cfg)

    print(f"\nAll configs written to {CONFIGS_DIR}/")


if __name__ == "__main__":
    main()
