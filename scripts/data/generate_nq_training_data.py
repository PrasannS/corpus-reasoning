"""Generate NQ training data in unified structured format.

Loads tilyupo/nq_cqa, places 1 gold document among N-1 distractors.
Output is unified JSONL (documents stored as list, formatting at train/eval time).

Usage:
    python scripts/data/generate_nq_training_data.py --num-examples 1000 --num-docs 20
    python scripts/data/generate_nq_training_data.py --num-examples 5000 --num-docs 100 --gold-position random
"""

import argparse
import random
import re
from datasets import load_dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*

from lib.io import save_jsonl, print_dataset_stats


def make_title(context: str) -> str:
    """Extract a title from NQ context (raw Wikipedia text, no structured title field).

    Heuristically looks for separators like " - Wikipedia" near the start.
    Falls back to the first ~80 chars at a word boundary.
    """
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
    """Split text into sentence-level chunks of approximately target_len chars."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sent in sentences:
        if current and len(current) + len(sent) + 1 > target_len:
            chunks.append(current.strip())
            current = sent
        else:
            current = current + " " + sent if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def precompute_distractor_chunks(distractors, use_titles=True):
    """Pre-split all distractor contexts into ~450-char chunks.

    Called once (not per-example) when --split-docs is used to match eval doc lengths.
    Returns list of {"title": ..., "text": ...} dicts.
    """
    chunks = []
    for d in distractors:
        title = make_title(d["context"]) if use_titles else None
        for chunk_text in split_into_chunks(d["context"]):
            chunks.append({"title": title, "text": chunk_text})
    return chunks


def build_example(sample, distractors, num_docs, gold_position, rng,
                  use_titles=True, split_docs=False, precomputed_chunks=None):
    """Build one unified-format example from an NQ sample.

    Returns dict with: documents, queries, answers, gold_doc_indices, source.
    """
    if num_docs == 0:
        return {
            "documents": [],
            "queries": [sample["question"]],
            "answers": [sample["answer"]],
            "gold_doc_indices": [],
            "source": "nq",
        }

    gold_text = sample["context"]
    gold_title = make_title(gold_text) if use_titles else None

    # --- Step 1: Prepare gold doc and distractors ---
    if split_docs:
        # Split gold context into chunks, pick chunk containing the answer
        gold_chunks = split_into_chunks(gold_text)
        answer_lower = sample["answer"].lower()
        gold_chunk = gold_chunks[0]
        for chunk in gold_chunks:
            if answer_lower in chunk.lower():
                gold_chunk = chunk
                break
        gold_doc = {"title": gold_title, "text": gold_chunk}

        selected = rng.sample(precomputed_chunks, min(num_docs - 1, len(precomputed_chunks)))
        dist_docs = list(selected)
        while len(dist_docs) < num_docs - 1:
            dist_docs.append(rng.choice(precomputed_chunks))
    else:
        gold_doc = {"title": gold_title, "text": gold_text}
        # Exclude same-question distractors to avoid leakage
        pool = [d for d in distractors if d["question"] != sample["question"]]
        sampled = rng.sample(pool, min(num_docs - 1, len(pool)))
        dist_docs = [{"title": make_title(d["context"]) if use_titles else None,
                       "text": d["context"]} for d in sampled]
        while len(dist_docs) < num_docs - 1:
            d = rng.choice(pool)
            dist_docs.append({"title": make_title(d["context"]) if use_titles else None,
                               "text": d["context"]})

    # --- Step 2: Place gold document at specified position ---
    pos = {"first": 0, "last": len(dist_docs), "middle": len(dist_docs) // 2}.get(
        gold_position, rng.randint(0, len(dist_docs))
    )
    documents = dist_docs[:pos] + [gold_doc] + dist_docs[pos:]

    return {
        "documents": documents,
        "queries": [sample["question"]],
        "answers": [sample["answer"]],
        "gold_doc_indices": [pos],
        "source": "nq",
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
                        help="Omit document titles")
    parser.add_argument("--split-docs", action="store_true",
                        help="Split documents into ~450-char chunks to match eval doc lengths")
    parser.add_argument("--no-context", action="store_true",
                        help="Generate closed-book examples (no documents)")
    args = parser.parse_args()

    if args.no_context:
        args.num_docs = 0

    rng = random.Random(args.seed)
    print(f"Loading tilyupo/nq_cqa ({args.split})...")
    ds = load_dataset("tilyupo/nq_cqa", split=args.split)
    print(f"  Loaded {len(ds)} examples")

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n = min(args.num_examples, len(ds))
    selected = indices[:n]
    distractor_pool = ([ds[i] for i in indices[n:n + n * 5] if n < len(ds)]
                       or [ds[i] for i in indices[:n * 5]])

    # Pre-compute distractor chunks once if splitting docs
    precomputed_chunks = None
    if args.split_docs:
        print("Pre-computing distractor chunks...")
        precomputed_chunks = precompute_distractor_chunks(
            distractor_pool, use_titles=not args.no_titles
        )
        print(f"  {len(precomputed_chunks)} chunks from {len(distractor_pool)} contexts")

    print(f"Generating {n} examples ({args.num_docs} docs, gold={args.gold_position})...")
    examples = []
    for idx, i in enumerate(selected):
        examples.append(build_example(
            ds[i], distractor_pool, args.num_docs, args.gold_position, rng,
            use_titles=not args.no_titles, split_docs=args.split_docs,
            precomputed_chunks=precomputed_chunks,
        ))
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{n} examples generated")

    # Build filename
    flags = []
    if args.no_titles:
        flags.append("notitle")
    if args.split_docs:
        flags.append("split")
    tag = f"_{'_'.join(flags)}" if flags else ""
    if args.no_context:
        path = f"{args.output_dir}/nq_{args.split}_nocontext_{n}{tag}.jsonl"
    else:
        path = f"{args.output_dir}/nq_{args.split}_k{args.num_docs}_{args.gold_position}_{n}{tag}.jsonl"

    save_jsonl(path, examples)
    print_dataset_stats(examples, args.split.capitalize(), path)


if __name__ == "__main__":
    main()
