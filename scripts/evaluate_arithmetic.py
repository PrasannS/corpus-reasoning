"""Evaluate arithmetic task using vLLM.

Usage:
    python scripts/evaluate_arithmetic.py --eval-data data/arithmetic_eval_add_sub_200.jsonl
    python scripts/evaluate_arithmetic.py --eval-data data/arithmetic_eval_add_sub_200.jsonl --lora-path outputs/arithmetic-lora
"""

import argparse
from vllm import SamplingParams
from lib.io import load_jsonl, format_alpaca_prompt, save_results
from lib.vllm_utils import add_vllm_args, load_model, run_inference


def main():
    parser = argparse.ArgumentParser(description="Arithmetic evaluation")
    add_vllm_args(parser)
    parser.add_argument("--eval-data", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.set_defaults(max_tokens=20, output_file="outputs/arithmetic_results.json")
    args = parser.parse_args()

    llm, lora_request = load_model(args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["\n"])

    data = load_jsonl(args.eval_data)
    if args.max_samples:
        data = data[:args.max_samples]

    use_alpaca = bool(args.lora_path) or args.base_model != "NousResearch/Llama-3.2-1B"

    prompts = []
    for ex in data:
        if use_alpaca:
            prompts.append(format_alpaca_prompt(ex["instruction"], ex["input"]))
        else:
            prompts.append(f"{ex['instruction']}\n\n{ex['input']}\nAnswer:")

    print(f"Running inference on {len(prompts)} examples...")
    outputs = run_inference(llm, prompts, sampling_params, lora_request=lora_request)

    correct = 0
    details = []
    for ex, output in zip(data, outputs):
        pred = output.strip()
        gold = ex["output"].strip()
        is_correct = pred == gold
        if is_correct:
            correct += 1
        details.append({
            "input": ex["input"],
            "gold": gold,
            "prediction": pred,
            "correct": is_correct,
        })

    acc = correct / len(data) * 100
    print(f"\nAccuracy: {correct}/{len(data)} = {acc:.1f}%")

    # Show some errors
    errors = [d for d in details if not d["correct"]]
    if errors:
        print(f"\nSample errors ({min(10, len(errors))} of {len(errors)}):")
        for d in errors[:10]:
            print(f"  {d['input']}  gold={d['gold']}  pred={d['prediction']}")

    save_results(args.output_file, {
        "accuracy": acc,
        "correct": correct,
        "total": len(data),
        "details": details,
    })
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
