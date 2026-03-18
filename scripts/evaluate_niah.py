"""
Evaluate a trained NIAH model using vLLM for fast inference.

Compares the base model vs. the LoRA-finetuned model on needle retrieval.
Uses substring matching and exact match metrics.
"""

import json
import argparse
import os
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_eval_data(path: str) -> list[dict]:
    """Load evaluation examples from JSONL."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_prompt(example: dict) -> str:
    """Format an example into the prompt the model was trained on (alpaca style)."""
    instruction = example["instruction"]
    inp = example["input"]
    return (
        f"Below is an instruction that describes a task, paired with an input that provides further context. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{inp}\n\n"
        f"### Response:\n"
    )


def evaluate_responses(
    examples: list[dict], responses: list[str]
) -> dict:
    """Compute evaluation metrics.

    Returns:
        Dict with exact_match, substring_match, and per-example details.
    """
    results = []
    exact_match = 0
    substring_match = 0

    for example, response in zip(examples, responses):
        gold = example["output"].strip()
        pred = response.strip()

        # Exact match (case-insensitive, strip whitespace)
        is_exact = pred.lower() == gold.lower()

        # Substring match: check if the key factual content is in the response.
        # Extract the core "needle" content from the gold answer.
        is_substring = gold.lower() in pred.lower()

        if is_exact:
            exact_match += 1
        if is_substring:
            substring_match += 1

        results.append({
            "question": example["input"].split("Question: ")[-1],
            "gold": gold,
            "predicted": pred[:200],  # truncate for display
            "exact_match": is_exact,
            "substring_match": is_substring,
        })

    n = len(examples)
    return {
        "num_examples": n,
        "exact_match": exact_match / n if n else 0,
        "substring_match": substring_match / n if n else 0,
        "details": results,
    }


def run_eval(
    llm: LLM,
    examples: list[dict],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> list[str]:
    """Run inference on all examples."""
    prompts = [format_prompt(ex) for ex in examples]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    return [o.outputs[0].text for o in outputs]


def main():
    parser = argparse.ArgumentParser(description="Evaluate NIAH model with vLLM")
    parser.add_argument(
        "--base-model",
        type=str,
        default="NousResearch/Llama-3.2-1B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/niah-lora",
        help="Path to LoRA adapter (set to empty string to skip)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="data/niah_val.jsonl",
        help="Path to evaluation JSONL",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/eval_results.json",
        help="Where to save results",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism",
    )
    args = parser.parse_args()

    examples = load_eval_data(args.eval_data)
    print(f"Loaded {len(examples)} eval examples from {args.eval_data}")

    sampling_params = SamplingParams(
        temperature=0.0,  # greedy
        max_tokens=args.max_tokens,
    )

    # Load model with LoRA support if needed
    enable_lora = bool(args.lora_path)
    print(f"Loading base model: {args.base_model} (enable_lora={enable_lora})")
    llm = LLM(
        model=args.base_model,
        enable_lora=enable_lora,
        max_lora_rank=64,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
    )

    all_results = {}

    # 1. Evaluate base model
    print("\n=== Evaluating BASE model ===")
    base_responses = run_eval(llm, examples, sampling_params)
    base_metrics = evaluate_responses(examples, base_responses)
    all_results["base"] = base_metrics
    print(f"  Exact match:     {base_metrics['exact_match']:.1%}")
    print(f"  Substring match: {base_metrics['substring_match']:.1%}")

    # 2. Evaluate LoRA-finetuned model
    if args.lora_path:
        lora_path = str(Path(args.lora_path).resolve())
        print(f"\n=== Evaluating LORA model ({lora_path}) ===")
        lora_request = LoRARequest("niah-lora", 1, lora_path)
        lora_responses = run_eval(llm, examples, sampling_params, lora_request)
        lora_metrics = evaluate_responses(examples, lora_responses)
        all_results["lora"] = lora_metrics
        print(f"  Exact match:     {lora_metrics['exact_match']:.1%}")
        print(f"  Substring match: {lora_metrics['substring_match']:.1%}")

    # Save results
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    # Print comparison table
    if args.lora_path:
        print("\n=== Comparison ===")
        print(f"{'Metric':<20} {'Base':>10} {'LoRA':>10} {'Delta':>10}")
        print("-" * 52)
        for metric in ["exact_match", "substring_match"]:
            b = base_metrics[metric]
            l = lora_metrics[metric]
            print(f"{metric:<20} {b:>9.1%} {l:>9.1%} {l - b:>+9.1%}")

    # Show a few example predictions
    print("\n=== Sample Predictions ===")
    n_show = min(5, len(examples))
    for i in range(n_show):
        print(f"\n--- Example {i+1} ---")
        print(f"  Q: {all_results['base']['details'][i]['question'][:100]}")
        print(f"  Gold: {all_results['base']['details'][i]['gold'][:100]}")
        print(f"  Base: {all_results['base']['details'][i]['predicted'][:100]}")
        if args.lora_path:
            print(f"  LoRA: {all_results['lora']['details'][i]['predicted'][:100]}")


if __name__ == "__main__":
    main()
