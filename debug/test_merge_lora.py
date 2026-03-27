"""Test that merge_lora.py produces a model with identical outputs to the LoRA model.

Merges adapter weights into the base model, runs forward passes on both,
and checks that logits match exactly.

Usage:
    python debug/test_merge_lora.py --base-model Qwen/Qwen3.5-0.8B-Base --lora-path outputs/nq-rag-std-qboth-notitle-qwen
    python debug/test_merge_lora.py --base-model NousResearch/Llama-3.2-1B --lora-path outputs/hotpotqa-std-qboth-lora
"""
import argparse
import os
import subprocess
import sys
import tempfile

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_logits(model, input_ids):
    with torch.no_grad():
        return model(input_ids).logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--lora-path", required=True)
    args = parser.parse_args()

    lora_path = os.path.abspath(args.lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # Tokenize test inputs
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    test_prompts = [
        "The capital of France is",
        "In a groundbreaking study, researchers found that",
        "Document 1: The quick brown fox jumped over the lazy dog.\nQuestion: What did the fox do?\nAnswer:",
    ]
    inputs = tokenizer(test_prompts, return_tensors="pt", padding=True).to(device)

    # 1) Load base + LoRA adapter
    print("Loading base model + LoRA adapter...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=dtype, device_map=device
    )
    lora_model = PeftModel.from_pretrained(base_model, lora_path)
    lora_model.eval()
    lora_logits = get_logits(lora_model, inputs["input_ids"])
    del lora_model, base_model
    torch.cuda.empty_cache()

    # 2) Merge and save to a temp dir using the merge script
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Running merge_lora.py -> {tmpdir}")
        env = os.environ.copy()
        env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
        result = subprocess.run(
            [
                sys.executable, "scripts/merge_lora.py",
                "--base-model", args.base_model,
                "--lora-path", lora_path,
                "--output-dir", tmpdir,
            ],
            capture_output=True, text=True, env=env,
        )
        if result.returncode != 0:
            print(f"merge_lora.py failed:\n{result.stderr}")
            sys.exit(1)
        print(result.stdout)

        # 3) Load merged model
        print("Loading merged model...")
        merged_model = AutoModelForCausalLM.from_pretrained(
            tmpdir, torch_dtype=dtype, device_map=device
        )
        merged_model.eval()
        merged_logits = get_logits(merged_model, inputs["input_ids"])

    # 4) Compare
    max_diff = (lora_logits - merged_logits).abs().max().item()
    mean_diff = (lora_logits - merged_logits).abs().mean().item()
    exact_match = torch.equal(lora_logits, merged_logits)

    print(f"\n{'='*50}")
    print(f"Exact match:  {exact_match}")
    print(f"Max abs diff: {max_diff:.2e}")
    print(f"Mean abs diff: {mean_diff:.2e}")

    # bf16 has ~1e-3 precision, so allow a tiny tolerance for numerical noise
    if max_diff < 1e-2:
        print("PASS: Merged model outputs match LoRA model.")
    else:
        print("FAIL: Outputs diverge significantly.")
        sys.exit(1)


if __name__ == "__main__":
    main()
