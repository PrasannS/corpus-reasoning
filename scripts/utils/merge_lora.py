"""Merge LoRA adapter weights into a base model and save as a full HF checkpoint.

Uses AutoModel to load the complete model (including visual encoder for multimodal
models like Qwen3.5), so the merged checkpoint is a drop-in replacement for the
original base model.

Usage:
    python scripts/merge_lora.py --base-model Qwen/Qwen3.5-0.8B-Base --lora-path ./outputs/hotpotqa-std-qboth-qwen-lora --output-dir ./outputs/hotpotqa-std-qboth-qwen-lora-merged
"""
import argparse
import os
import shutil
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import torch


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base-model", required=True, help="Base model name or path")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged model")
    args = parser.parse_args()

    # Resolve to absolute path so peft doesn't mistake it for a HF repo ID
    lora_path = os.path.abspath(args.lora_path)

    print(f"Loading base model: {args.base_model}")
    # Try loading as conditional generation model first (includes visual encoder
    # for multimodal models like Qwen3.5), fall back to causal LM
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model_cls_name = getattr(config, "architectures", [""])[0]
    if "ConditionalGeneration" in model_cls_name:
        # Multimodal model — load via AutoModelForCausalLM won't include visual encoder
        # Import the specific class dynamically
        import importlib
        mod = importlib.import_module("transformers")
        model_cls = getattr(mod, model_cls_name, None)
        if model_cls is not None:
            model = model_cls.from_pretrained(
                args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
                trust_remote_code=True,
            )
            print(f"  Loaded as {model_cls_name} (includes visual encoder)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
                trust_remote_code=True,
            )
            print(f"  {model_cls_name} not found, loaded as causal LM")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        print("  Loaded as causal LM")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print(f"Loading LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Copy processor configs from base model (needed for multimodal models)
    try:
        processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
        processor.save_pretrained(args.output_dir)
        print("Processor configs saved.")
    except Exception:
        pass  # Not a multimodal model, no processor needed

    print("Done.")


if __name__ == "__main__":
    main()
