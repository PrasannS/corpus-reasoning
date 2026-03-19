"""Evaluate NIAH model using vLLM. Compares base vs. LoRA on needle retrieval.

Usage:
    python scripts/evaluate_niah.py
    python scripts/evaluate_niah.py --lora-path outputs/niah-lora --eval-data data/niah_val.jsonl
"""

import argparse
from vllm import SamplingParams
from lib.io import load_jsonl, save_results
from lib.vllm_utils import add_vllm_args, load_model, run_inference, format_alpaca_prompt


def evaluate(examples, responses):
    results = []
    for ex, resp in zip(examples, responses):
        gold, pred = ex["output"].strip(), resp.strip()
        results.append({
            "question": ex["input"].split("Question: ")[-1][:100],
            "gold": gold, "predicted": pred[:200],
            "exact_match": float(pred.lower() == gold.lower()),
            "substring_match": float(gold.lower() in pred.lower()),
        })
    n = len(results)
    return {
        "exact_match": sum(r["exact_match"] for r in results) / n,
        "substring_match": sum(r["substring_match"] for r in results) / n,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate NIAH with vLLM")
    add_vllm_args(parser)
    parser.add_argument("--eval-data", type=str, default="data/niah_val.jsonl")
    parser.set_defaults(lora_path="outputs/niah-lora", output_file="outputs/eval_results.json")
    args = parser.parse_args()

    examples = load_jsonl(args.eval_data)
    print(f"Loaded {len(examples)} examples from {args.eval_data}")

    llm, lora_request = load_model(args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [format_alpaca_prompt(ex["instruction"], ex["input"]) for ex in examples]

    all_results = {}

    # Base model
    print("\n=== Evaluating BASE ===")
    base = evaluate(examples, run_inference(llm, prompts, sampling_params))
    all_results["base"] = base
    print(f"  EM: {base['exact_match']:.1%}  Substring: {base['substring_match']:.1%}")

    # LoRA model
    if lora_request:
        print("\n=== Evaluating LORA ===")
        lora = evaluate(examples, run_inference(llm, prompts, sampling_params, lora_request))
        all_results["lora"] = lora
        print(f"  EM: {lora['exact_match']:.1%}  Substring: {lora['substring_match']:.1%}")
        print(f"\n  Delta EM: {lora['exact_match'] - base['exact_match']:+.1%}  "
              f"Substring: {lora['substring_match'] - base['substring_match']:+.1%}")

    save_results(args.output_file, all_results)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
