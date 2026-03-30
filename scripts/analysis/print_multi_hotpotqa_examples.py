"""Print full training/eval prompts for multi-query HotpotQA to a file.

Generates a few examples and writes the exact text the model would see
(alpaca-wrapped for training, raw for base eval) so you can inspect them.

Usage:
    python scripts/print_multi_hotpotqa_examples.py
    python scripts/print_multi_hotpotqa_examples.py --num-queries 5 --total-docs 30
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports

from datasets import load_dataset
from lib.io import ALPACA_TEMPLATE, format_alpaca_prompt
from data.generate_multi_hotpotqa_data import (
    INSTRUCTION, build_multi_example, paragraphs_from_context, get_supporting_titles,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-queries", type=int, default=3,
                        help="Queries per example (small default for readability)")
    parser.add_argument("--total-docs", type=int, default=0,
                        help="Total docs (0 = supporting only)")
    parser.add_argument("--num-examples", type=int, default=2,
                        help="Number of examples to print")
    parser.add_argument("--question-type", type=str, default="bridge")
    parser.add_argument("--output", type=str, default="data/multi_hotpotqa_example_prompts.txt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Loading HotpotQA...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    if args.question_type != "all":
        ds = ds.filter(lambda ex: ex["type"] == args.question_type)

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    needed = args.num_examples * args.num_queries
    selected = indices[:needed]
    pool_indices = indices[needed:needed * 3]

    # Build distractor pool
    distractor_pool = []
    for idx in pool_indices:
        ex = ds[idx]
        sup_titles = get_supporting_titles(ex)
        for doc in paragraphs_from_context(ex["context"]):
            if doc["title"] not in sup_titles:
                distractor_pool.append(doc)
    rng.shuffle(distractor_pool)

    # Generate examples
    groups = []
    for i in range(0, needed, args.num_queries):
        group = [ds[idx] for idx in selected[i:i + args.num_queries]]
        groups.append(group)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for i, group in enumerate(groups):
            raw = build_multi_example(group, distractor_pool, args.total_docs, rng)

            # === Training prompt (alpaca-wrapped) ===
            training_prompt = format_alpaca_prompt(raw["instruction"], raw["input"])
            training_full = training_prompt + raw["output"]

            f.write("=" * 80 + "\n")
            f.write(f"EXAMPLE {i+1} — TRAINING (alpaca format, what the model sees during fine-tuning)\n")
            f.write("=" * 80 + "\n")
            f.write(training_full + "\n")

            f.write("\n" + "-" * 80 + "\n")
            f.write(f"EXAMPLE {i+1} — EVAL/BASE MODEL (plain format, no alpaca wrapper)\n")
            f.write("-" * 80 + "\n")
            eval_prompt = f"{raw['instruction']}\n\n{raw['input']}\nAnswers:"
            f.write(eval_prompt + "\n")
            f.write(f"\n[Expected output]: {raw['output']}\n")
            f.write("\n\n")

    print(f"Wrote {args.num_examples} examples to {args.output}")
    print(f"  Settings: num_queries={args.num_queries}, total_docs={args.total_docs}")


if __name__ == "__main__":
    main()
