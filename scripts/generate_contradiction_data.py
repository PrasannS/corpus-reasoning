"""
Generate long-context contradiction detection data from SNLI.

Each example is a corpus of N numbered claims. Some pairs of claims are
contradictory (sourced from SNLI premise/hypothesis pairs with label=2).
The remaining claims are non-contradictory fillers. The model must identify
all contradicting pairs by their claim IDs.

Format (Axolotl alpaca):
    instruction: Given the following corpus of numbered claims, identify all
                 pairs of claims that contradict each other...
    input:       Claim 1: ... \n Claim 2: ... \n ... Claim N: ...
    output:      [[2, 7], [4, 9]]

Usage:
    python scripts/generate_contradiction_data.py --num-claims 20 --num-contradictions 3
    python scripts/generate_contradiction_data.py --num-claims 50 --num-contradictions 5 --num-train 5000
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


INSTRUCTION = (
    "Given the following corpus of numbered claims, identify all pairs of claims "
    "that contradict each other. A pair of claims is contradictory if they cannot "
    "both be true at the same time.\n\n"
    "Output your answer as a JSON list of pairs, where each pair is a list of two "
    "claim IDs. For example: [[1, 4], [3, 7]]\n"
    "If there are no contradicting pairs, output: []"
)


def build_example(
    contradiction_pairs: list[tuple[str, str]],
    filler_claims: list[str],
    num_claims: int,
    rng: random.Random,
) -> dict:
    """Build a single training/eval example.

    Args:
        contradiction_pairs: list of (premise, hypothesis) that contradict.
        filler_claims: list of neutral/entailment sentences to use as fillers.
        num_claims: total number of claims in the corpus.
        rng: random number generator.

    Returns:
        Dict with 'instruction', 'input', 'output' fields.
    """
    num_contradiction_claims = len(contradiction_pairs) * 2
    num_fillers = num_claims - num_contradiction_claims

    # Collect all claims with metadata
    claims = []
    pair_tracker = []  # (claim_index_a, claim_index_b) for contradiction pairs

    # Add contradiction pairs
    for premise, hypothesis in contradiction_pairs:
        idx_a = len(claims)
        claims.append(premise)
        idx_b = len(claims)
        claims.append(hypothesis)
        pair_tracker.append((idx_a, idx_b))

    # Add fillers
    selected_fillers = rng.sample(filler_claims, min(num_fillers, len(filler_claims)))
    while len(selected_fillers) < num_fillers:
        selected_fillers.append(rng.choice(filler_claims))
    claims.extend(selected_fillers)

    # Shuffle claims and build ID mapping
    indices = list(range(len(claims)))
    rng.shuffle(indices)
    # old_index -> new_position (1-indexed)
    old_to_new = {old_idx: new_pos + 1 for new_pos, old_idx in enumerate(indices)}

    shuffled_claims = [claims[i] for i in indices]

    # Build the answer: pairs of 1-indexed claim IDs, sorted
    answer_pairs = []
    for idx_a, idx_b in pair_tracker:
        pair = sorted([old_to_new[idx_a], old_to_new[idx_b]])
        answer_pairs.append(pair)
    # Sort pairs by first element for consistency
    answer_pairs.sort(key=lambda p: (p[0], p[1]))

    # Format input
    claim_lines = [f"Claim {i+1}: {claim}" for i, claim in enumerate(shuffled_claims)]
    input_text = "\n".join(claim_lines)

    output_text = json.dumps(answer_pairs)

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": output_text,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate contradiction detection data from SNLI"
    )
    parser.add_argument(
        "--num-claims", type=int, default=20,
        help="Total number of claims per example"
    )
    parser.add_argument(
        "--num-contradictions", type=int, default=3,
        help="Number of contradicting pairs per example"
    )
    parser.add_argument(
        "--num-train", type=int, default=5000,
        help="Number of training examples to generate"
    )
    parser.add_argument(
        "--num-eval", type=int, default=500,
        help="Number of eval examples to generate"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data",
        help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    assert args.num_contradictions * 2 <= args.num_claims, (
        f"Need at least {args.num_contradictions * 2} claims for "
        f"{args.num_contradictions} contradiction pairs, but num_claims={args.num_claims}"
    )

    rng = random.Random(args.seed)

    print(f"Loading stanfordnlp/snli...")
    ds = load_dataset("stanfordnlp/snli")

    # Separate contradiction pairs and filler sentences from train split
    print("Filtering contradiction pairs and filler claims...")
    train_data = ds["train"]
    contradiction_pairs_pool = []
    filler_pool = []

    for ex in train_data:
        if ex["label"] == 2:
            contradiction_pairs_pool.append((ex["premise"], ex["hypothesis"]))
        elif ex["label"] in (0, 1):
            # Use premises from entailment/neutral as fillers
            filler_pool.append(ex["premise"])

    # Deduplicate fillers
    filler_pool = list(set(filler_pool))

    print(f"  Contradiction pairs: {len(contradiction_pairs_pool)}")
    print(f"  Unique filler claims: {len(filler_pool)}")

    # Also gather from validation/test for eval pool
    eval_contradiction_pool = []
    eval_filler_pool = []
    for split in ["validation", "test"]:
        for ex in ds[split]:
            if ex["label"] == 2:
                eval_contradiction_pool.append((ex["premise"], ex["hypothesis"]))
            elif ex["label"] in (0, 1):
                eval_filler_pool.append(ex["premise"])
    eval_filler_pool = list(set(eval_filler_pool))

    print(f"  Eval contradiction pairs: {len(eval_contradiction_pool)}")
    print(f"  Eval unique filler claims: {len(eval_filler_pool)}")

    # Shuffle pools
    rng.shuffle(contradiction_pairs_pool)
    rng.shuffle(filler_pool)
    rng.shuffle(eval_contradiction_pool)
    rng.shuffle(eval_filler_pool)

    # Generate training examples
    k = args.num_contradictions
    total_pairs_needed_train = args.num_train * k
    print(f"\nGenerating {args.num_train} training examples "
          f"({args.num_claims} claims, {k} contradiction pairs each)...")

    if total_pairs_needed_train > len(contradiction_pairs_pool):
        print(f"  WARNING: Need {total_pairs_needed_train} pairs but only have "
              f"{len(contradiction_pairs_pool)}. Will reuse pairs across examples.")

    train_examples = []
    for i in range(args.num_train):
        # Sample contradiction pairs for this example (with replacement across examples)
        pairs = rng.sample(
            contradiction_pairs_pool,
            min(k, len(contradiction_pairs_pool))
        )
        example = build_example(pairs, filler_pool, args.num_claims, rng)
        train_examples.append(example)

    # Generate eval examples (using held-out pool)
    total_pairs_needed_eval = args.num_eval * k
    print(f"Generating {args.num_eval} eval examples...")

    eval_examples = []
    for i in range(args.num_eval):
        pairs = rng.sample(
            eval_contradiction_pool,
            min(k, len(eval_contradiction_pool))
        )
        example = build_example(pairs, eval_filler_pool, args.num_claims, rng)
        eval_examples.append(example)

    # Write outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"n{args.num_claims}_k{args.num_contradictions}"

    train_path = output_dir / f"contradiction_train_{tag}.jsonl"
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    eval_path = output_dir / f"contradiction_eval_{tag}.jsonl"
    with open(eval_path, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    # Stats
    for name, examples, path in [
        ("Train", train_examples, train_path),
        ("Eval", eval_examples, eval_path),
    ]:
        if not examples:
            print(f"\n{name}: 0 examples (skipped)")
            continue
        input_lens = [len(ex["input"]) for ex in examples]
        output_lens = [len(ex["output"]) for ex in examples]
        avg_input = sum(input_lens) / len(input_lens)
        avg_output = sum(output_lens) / len(output_lens)
        print(f"\n{name}: {len(examples)} examples -> {path}")
        print(f"  Avg input length:  {avg_input:,.0f} chars")
        print(f"  Avg output length: {avg_output:,.0f} chars")
        print(f"  File size: {path.stat().st_size / 1024 / 1024:.1f} MB")

    # Show a sample
    all_examples = train_examples or eval_examples
    if all_examples:
        label = "training" if train_examples else "eval"
        print(f"\n=== Sample {label} example ===")
        ex = all_examples[0]
        print(f"  instruction: {ex['instruction'][:120]}...")
        lines = ex["input"].split("\n")
        print(f"  input ({len(lines)} claims):")
        for line in lines[:5]:
            print(f"    {line[:100]}")
        if len(lines) > 5:
            print(f"    ... ({len(lines) - 5} more claims)")
        print(f"  output: {ex['output']}")


if __name__ == "__main__":
    main()
