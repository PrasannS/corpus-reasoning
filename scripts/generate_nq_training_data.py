"""
Generate NQ training data for Axolotl in HELMET-compatible prompt format.

Loads questions and gold contexts from the tilyupo/nq_cqa HuggingFace dataset,
constructs examples with one gold document among distractor documents, and formats
the prompt to match what HELMET's RAG evaluation feeds to the model:

    Instruction: Use the given documents to write a concise and short answer...
    Input: Document (Title: ...): ... \n\n Question: {question}
    Output: {answer}

This uses Axolotl's alpaca format so it can be trained directly with Axolotl.

Usage:
    python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20
    python scripts/generate_nq_training_data.py --num-examples 500 --num-docs 100 --gold-position random
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


# These templates match evaluate_helmet_rag.py exactly
PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)


def make_title(context: str) -> str:
    """Generate a synthetic document title from the context."""
    for sep in [" - Wikipedia", " -- Wikipedia", "\n", ". "]:
        if sep in context[:200]:
            candidate = context[:context.index(sep)].strip()
            if 3 < len(candidate) < 100:
                return candidate
    snippet = context[:80]
    if " " in snippet[40:]:
        snippet = snippet[:snippet.rindex(" ", 0, 80)]
    return snippet.strip().rstrip(".,;:-")


def build_example(
    sample: dict,
    distractor_pool: list[dict],
    num_docs: int,
    gold_position: str,
    rng: random.Random,
) -> dict:
    """Build a single training example in alpaca format with HELMET-style prompt.

    Returns:
        Dict with 'instruction', 'input', 'output' fields for Axolotl alpaca format.
    """
    # Build gold document
    gold_title = make_title(sample["context"])
    gold_doc = PASSAGE_TEMPLATE.format(title=gold_title, text=sample["context"])

    # Sample distractors
    num_distractors = num_docs - 1
    distractors = rng.sample(distractor_pool, min(num_distractors, len(distractor_pool)))
    distractor_docs = []
    for d in distractors:
        title = make_title(d["context"])
        distractor_docs.append(PASSAGE_TEMPLATE.format(title=title, text=d["context"]))

    # Pad if needed
    while len(distractor_docs) < num_distractors:
        d = rng.choice(distractor_pool)
        title = make_title(d["context"])
        distractor_docs.append(PASSAGE_TEMPLATE.format(title=title, text=d["context"]))

    # Place gold document
    if gold_position == "first":
        pos = 0
    elif gold_position == "last":
        pos = len(distractor_docs)
    elif gold_position == "middle":
        pos = len(distractor_docs) // 2
    else:  # random
        pos = rng.randint(0, len(distractor_docs))

    all_docs = distractor_docs[:pos] + [gold_doc] + distractor_docs[pos:]

    # Build the input to match HELMET eval prompt format:
    #   {context}\n\nQuestion: {question}
    context = "\n\n".join(all_docs)
    input_text = f"{context}\n\nQuestion: {sample['question']}"

    # The output is just the answer (what comes after "Answer:" in HELMET)
    answer = sample["answer"]

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": answer,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate NQ training data in Axolotl alpaca format")
    parser.add_argument("--num-examples", type=int, default=1000,
                        help="Number of training examples to generate")
    parser.add_argument("--num-docs", type=int, default=20,
                        help="Total documents per example (1 gold + N-1 distractors)")
    parser.add_argument("--gold-position", type=str, default="random",
                        choices=["random", "first", "last", "middle"],
                        help="Where to place the gold document in the context")
    parser.add_argument("--output-dir", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation"],
                        help="Which split of nq_cqa to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading tilyupo/nq_cqa ({args.split} split)...")
    ds = load_dataset("tilyupo/nq_cqa", split=args.split)
    print(f"  Loaded {len(ds)} examples")

    # Shuffle and select examples
    indices = list(range(len(ds)))
    rng.shuffle(indices)

    num_to_generate = min(args.num_examples, len(ds))
    selected_indices = indices[:num_to_generate]

    # Build distractor pool from non-selected examples
    if len(ds) > num_to_generate * 2:
        distractor_indices = indices[num_to_generate:]
    else:
        distractor_indices = indices

    distractor_pool = [ds[i] for i in distractor_indices[:num_to_generate * 5]]
    print(f"  Distractor pool: {len(distractor_pool)} examples")

    # Generate examples
    print(f"Generating {num_to_generate} examples with {args.num_docs} docs each "
          f"(gold_position={args.gold_position})...")
    examples = []
    for idx in selected_indices:
        sample = ds[idx]
        pool = [d for d in distractor_pool if d["question"] != sample["question"]]
        example = build_example(sample, pool, args.num_docs, args.gold_position, rng)
        examples.append(example)

    # Write output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"nq_train_k{args.num_docs}_{args.gold_position}.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Print stats
    input_lens = [len(ex["input"]) for ex in examples]
    avg_input_chars = sum(input_lens) / len(input_lens)
    output_lens = [len(ex["output"]) for ex in examples]
    avg_output_chars = sum(output_lens) / len(output_lens)

    print(f"\nWrote {len(examples)} examples to {out_path}")
    print(f"  Avg input length:  {avg_input_chars:,.0f} chars")
    print(f"  Avg output length: {avg_output_chars:,.0f} chars")
    print(f"  File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Show a sample
    print(f"\n=== Sample example ===")
    ex = examples[0]
    print(f"  instruction: {ex['instruction'][:100]}...")
    print(f"  input (first 300 chars): {ex['input'][:300]}...")
    print(f"  input (last 200 chars): ...{ex['input'][-200:]}")
    print(f"  output: {ex['output']}")


if __name__ == "__main__":
    main()
