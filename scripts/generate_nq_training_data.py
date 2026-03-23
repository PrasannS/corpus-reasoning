"""Generate NQ training data for Axolotl in HELMET-compatible prompt format.

Loads tilyupo/nq_cqa, places 1 gold document among N-1 distractors using HELMET's
passage template. Output is alpaca format for Axolotl training.

Usage:
    python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20
    python scripts/generate_nq_training_data.py --num-examples 5000 --num-docs 100 --gold-position random
    python scripts/generate_nq_training_data.py --num-examples 2500 --num-docs 20 --no-titles --split-docs
"""

import argparse
import random
import re
from datasets import load_dataset
from lib.io import save_jsonl, print_dataset_stats

PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"
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


def split_into_chunks(text, target_len=450):
    """Split text into sentence-level chunks of approximately target_len chars.

    Splits on sentence boundaries ('. ', '? ', '! ') to produce chunks
    roughly matching eval document lengths (~430 chars avg).
    """
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > target_len:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def format_doc(text, title=None, use_titles=True):
    """Format a document with or without title."""
    if use_titles and title:
        return PASSAGE_TEMPLATE.format(title=title, text=text)
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=text)


def precompute_distractor_chunks(distractors, use_titles=True):
    """Pre-split all distractor contexts into chunks (call once, not per-example)."""
    dist_chunks = []
    for d in distractors:
        chunks = split_into_chunks(d["context"])
        d_title = make_title(d["context"]) if use_titles else None
        for chunk in chunks:
            dist_chunks.append((chunk, d_title))
    return dist_chunks


def build_example(sample, distractors, num_docs, gold_position, rng,
                  use_titles=True, split_docs=False, precomputed_chunks=None):
    gold_text = sample["context"]
    gold_title = make_title(gold_text) if use_titles else None

    if split_docs:
        # Split gold context into chunks; pick the chunk containing the answer if possible,
        # otherwise use the first chunk as the "gold" doc
        gold_chunks = split_into_chunks(gold_text)
        answer = sample["answer"].lower()
        gold_chunk = gold_chunks[0]
        for chunk in gold_chunks:
            if answer in chunk.lower():
                gold_chunk = chunk
                break
        gold_doc = format_doc(gold_chunk, gold_title, use_titles)

        # Use pre-computed distractor chunks
        dist_chunks = precomputed_chunks
        selected = rng.sample(dist_chunks, min(num_docs - 1, len(dist_chunks)))
        dist_docs = [format_doc(text, title, use_titles) for text, title in selected]
        # Pad if needed
        while len(dist_docs) < num_docs - 1:
            text, title = rng.choice(dist_chunks)
            dist_docs.append(format_doc(text, title, use_titles))
    else:
        gold_doc = format_doc(gold_text, gold_title, use_titles)
        dist_docs = []
        pool = [d for d in distractors if d["question"] != sample["question"]]
        for d in rng.sample(pool, min(num_docs - 1, len(pool))):
            d_title = make_title(d["context"]) if use_titles else None
            dist_docs.append(format_doc(d["context"], d_title, use_titles))
        while len(dist_docs) < num_docs - 1:
            d = rng.choice(pool)
            d_title = make_title(d["context"]) if use_titles else None
            dist_docs.append(format_doc(d["context"], d_title, use_titles))

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
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles to avoid train/eval title mismatch")
    parser.add_argument("--split-docs", action="store_true",
                        help="Split documents into ~450-char chunks to match eval doc lengths")
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

    flags = []
    if args.no_titles:
        flags.append("no-titles")
    if args.split_docs:
        flags.append("split")
    flags_str = f"_{'_'.join(flags)}" if flags else ""

    # Pre-compute distractor chunks once if splitting docs
    precomputed_chunks = None
    if args.split_docs:
        print("Pre-computing distractor chunks...")
        precomputed_chunks = precompute_distractor_chunks(
            distractor_pool, use_titles=not args.no_titles
        )
        print(f"  {len(precomputed_chunks)} distractor chunks from {len(distractor_pool)} contexts")

    print(f"Generating {n} examples ({args.num_docs} docs, gold={args.gold_position}, "
          f"titles={'no' if args.no_titles else 'yes'}, split_docs={args.split_docs})...")
    examples = []
    for idx, i in enumerate(selected):
        examples.append(
            build_example(ds[i], distractor_pool, args.num_docs, args.gold_position, rng,
                          use_titles=not args.no_titles, split_docs=args.split_docs,
                          precomputed_chunks=precomputed_chunks)
        )
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{n} examples generated")

    path = f"{args.output_dir}/nq_train_k{args.num_docs}_{args.gold_position}_{n}{flags_str}.jsonl"
    save_jsonl(path, examples)
    print_dataset_stats(examples, "Train", path)


if __name__ == "__main__":
    main()
