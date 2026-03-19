"""Generate NQ training data for Axolotl in HELMET-compatible prompt format.

Loads tilyupo/nq_cqa, places 1 gold document among N-1 distractors using HELMET's
passage template. Output is alpaca format for Axolotl training.

Usage:
    python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20
    python scripts/generate_nq_training_data.py --num-examples 5000 --num-docs 100 --gold-position random
"""

import argparse
import random
from datasets import load_dataset
from lib.io import save_jsonl, print_dataset_stats

PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)


def make_title(context: str) -> str:
    for sep in [" - Wikipedia", " -- Wikipedia", "\n", ". "]:
        if sep in context[:200]:
            candidate = context[:context.index(sep)].strip()
            if 3 < len(candidate) < 100:
                return candidate
    snippet = context[:80]
    if " " in snippet[40:]:
        snippet = snippet[:snippet.rindex(" ", 0, 80)]
    return snippet.strip().rstrip(".,;:-")


def build_example(sample, distractors, num_docs, gold_position, rng):
    gold_doc = PASSAGE_TEMPLATE.format(title=make_title(sample["context"]), text=sample["context"])

    dist_docs = []
    pool = [d for d in distractors if d["question"] != sample["question"]]
    for d in rng.sample(pool, min(num_docs - 1, len(pool))):
        dist_docs.append(PASSAGE_TEMPLATE.format(title=make_title(d["context"]), text=d["context"]))
    while len(dist_docs) < num_docs - 1:
        d = rng.choice(pool)
        dist_docs.append(PASSAGE_TEMPLATE.format(title=make_title(d["context"]), text=d["context"]))

    pos = {"first": 0, "last": len(dist_docs), "middle": len(dist_docs) // 2}.get(
        gold_position, rng.randint(0, len(dist_docs))
    )
    all_docs = dist_docs[:pos] + [gold_doc] + dist_docs[pos:]
    context = "\n\n".join(all_docs)

    return {
        "instruction": INSTRUCTION,
        "input": f"{context}\n\nQuestion: {sample['question']}",
        "output": sample["answer"],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate NQ training data")
    parser.add_argument("--num-examples", type=int, default=1000)
    parser.add_argument("--num-docs", type=int, default=20)
    parser.add_argument("--gold-position", type=str, default="random",
                        choices=["random", "first", "last", "middle"])
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"Loading tilyupo/nq_cqa ({args.split})...")
    ds = load_dataset("tilyupo/nq_cqa", split=args.split)
    print(f"  Loaded {len(ds)} examples")

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n = min(args.num_examples, len(ds))
    selected = indices[:n]
    distractor_pool = [ds[i] for i in indices[n:n + n * 5] if n < len(ds)] or [ds[i] for i in indices[:n * 5]]

    print(f"Generating {n} examples ({args.num_docs} docs, gold={args.gold_position})...")
    examples = [build_example(ds[i], distractor_pool, args.num_docs, args.gold_position, rng) for i in selected]

    path = f"{args.output_dir}/nq_train_k{args.num_docs}_{args.gold_position}.jsonl"
    save_jsonl(path, examples)
    print_dataset_stats(examples, "Train", path)


if __name__ == "__main__":
    main()
