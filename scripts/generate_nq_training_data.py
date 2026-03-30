"""Generate NQ training data for Axolotl in HELMET-compatible prompt format.

Loads tilyupo/nq_cqa, places 1 gold document among N-1 distractors using HELMET's
passage template. Output is alpaca format for Axolotl training.

By default generates retrieval task (output relevant document ID). Use --no-retrieval
for the original QA task (output answer text).

Usage:
    python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20
    python scripts/generate_nq_training_data.py --num-examples 1000 --num-docs 20 --no-retrieval
    python scripts/generate_nq_training_data.py --num-examples 5000 --num-docs 100 --gold-position random
"""

import argparse
import random
import re
from datasets import load_dataset
from lib.io import save_jsonl, print_dataset_stats
from lib.prompts import (
    QA_INSTRUCTION as INSTRUCTION,
    RETRIEVAL_INSTRUCTION_SINGLE as RETRIEVAL_INSTRUCTION,
    format_doc,
)


def make_title(context: str) -> str:
    """Extract a title from the NQ context string.

    NQ contexts are raw Wikipedia text (no structured title field), so we
    heuristically extract a title by looking for common separators like
    " - Wikipedia" at the start of the text. Falls back to the first ~80 chars.
    """
    for sep in [" - Wikipedia", " -- Wikipedia", "\n", ". "]:
        if sep in context[:200]:
            candidate = context[:context.index(sep)].strip()
            if 3 < len(candidate) < 100:
                return candidate
    # Fallback: use first ~80 chars, breaking at a word boundary
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


def precompute_distractor_chunks(distractors, use_titles=True):
    """Pre-split all distractor contexts into chunks (call once, not per-example).

    When --split-docs is used, both gold and distractor texts are chunked to
    ~450 chars to match the eval document lengths (HELMET uses ~430 char passages).
    Pre-computing distractor chunks avoids redundant work per example.
    Returns list of (text, title) tuples.
    """
    dist_chunks = []
    for d in distractors:
        chunks = split_into_chunks(d["context"])
        d_title = make_title(d["context"]) if use_titles else None
        for chunk in chunks:
            dist_chunks.append((chunk, d_title))
    return dist_chunks


def build_example(sample, distractors, num_docs, gold_position, rng,
                  use_titles=True, split_docs=False, precomputed_chunks=None,
                  retrieval=False):
    if num_docs == 0:
        # No-context (closed-book) baseline: just the question, no documents
        instruction = RETRIEVAL_INSTRUCTION if retrieval else INSTRUCTION
        return {
            "instruction": instruction,
            "input": f"Question: {sample['question']}",
            "output": sample["answer"],
        }

    gold_text = sample["context"]
    gold_title = make_title(gold_text) if use_titles else None

    # --- Step 1: Prepare gold doc and distractors as (text, title) tuples ---
    if split_docs:
        # Split mode: chunk the gold context (~450 chars each) and pick the
        # chunk that contains the answer as the "gold" document
        gold_chunks = split_into_chunks(gold_text)
        answer = sample["answer"].lower()
        gold_chunk = gold_chunks[0]  # fallback if answer not found in any chunk
        for chunk in gold_chunks:
            if answer in chunk.lower():
                gold_chunk = chunk
                break

        # Sample from pre-computed distractor chunks (already split to ~450 chars)
        dist_chunks = precomputed_chunks
        selected = rng.sample(dist_chunks, min(num_docs - 1, len(dist_chunks)))
        dist_items = list(selected)
        while len(dist_items) < num_docs - 1:
            dist_items.append(rng.choice(dist_chunks))
        gold_item = (gold_chunk, gold_title)
    else:
        # Default mode: use full NQ contexts as documents (no chunking)
        # Exclude distractors that share the same question to avoid leakage
        pool = [d for d in distractors if d["question"] != sample["question"]]
        sampled = rng.sample(pool, min(num_docs - 1, len(pool)))
        dist_items = [(d["context"], make_title(d["context"]) if use_titles else None)
                      for d in sampled]
        # Pad with duplicates if pool is too small
        while len(dist_items) < num_docs - 1:
            d = rng.choice(pool)
            dist_items.append((d["context"], make_title(d["context"]) if use_titles else None))
        gold_item = (gold_text, gold_title)

    # --- Step 2: Place gold document at the specified position ---
    pos = {"first": 0, "last": len(dist_items), "middle": len(dist_items) // 2}.get(
        gold_position, rng.randint(0, len(dist_items))  # "random" falls through to here
    )
    all_items = dist_items[:pos] + [gold_item] + dist_items[pos:]

    # --- Step 3: Format documents and build output ---
    # Retrieval mode: each doc gets a [N] ID prefix, output is the gold doc's ID
    # QA mode: plain documents, output is the answer text
    if retrieval:
        all_docs = [format_doc(text, title, use_titles, doc_id=i + 1)
                    for i, (text, title) in enumerate(all_items)]
        gold_id = pos + 1  # 1-indexed position of the gold document
    else:
        all_docs = [format_doc(text, title, use_titles)
                    for text, title in all_items]

    context = "\n\n".join(all_docs)
    instruction = RETRIEVAL_INSTRUCTION if retrieval else INSTRUCTION
    output = f"[{gold_id}]" if retrieval else sample["answer"]

    return {
        "instruction": instruction,
        "input": f"{context}\n\nQuestion: {sample['question']}",
        "output": output,
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
    parser.add_argument("--no-context", action="store_true",
                        help="Generate closed-book examples (no documents, just question)")
    parser.add_argument("--retrieval", action="store_true", default=True,
                        help="Generate retrieval task: output relevant document ID instead of answer (default)")
    parser.add_argument("--no-retrieval", action="store_false", dest="retrieval",
                        help="Generate QA task instead of retrieval")
    args = parser.parse_args()

    if args.no_context:
        args.num_docs = 0

    rng = random.Random(args.seed)
    print(f"Loading tilyupo/nq_cqa ({args.split})...")
    ds = load_dataset("tilyupo/nq_cqa", split=args.split)
    print(f"  Loaded {len(ds)} examples")

    # Shuffle indices to randomize which examples become training vs distractors
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n = min(args.num_examples, len(ds))
    selected = indices[:n]
    # Use non-selected examples as distractor pool (5x the training size for variety).
    # If we selected all examples, fall back to reusing them as distractors.
    distractor_pool = [ds[i] for i in indices[n:n + n * 5] if n < len(ds)] or [ds[i] for i in indices[:n * 5]]

    flags = []
    if args.retrieval:
        flags.append("retrieval")
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

    task_type = "retrieval" if args.retrieval else "QA"
    print(f"Generating {n} {task_type} examples ({args.num_docs} docs, gold={args.gold_position}, "
          f"titles={'no' if args.no_titles else 'yes'}, split_docs={args.split_docs})...")
    examples = []
    for idx, i in enumerate(selected):
        examples.append(
            build_example(ds[i], distractor_pool, args.num_docs, args.gold_position, rng,
                          use_titles=not args.no_titles, split_docs=args.split_docs,
                          precomputed_chunks=precomputed_chunks, retrieval=args.retrieval)
        )
        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{n} examples generated")

    if args.no_context:
        path = f"{args.output_dir}/nq_train_nocontext_{n}{flags_str}.jsonl"
    else:
        path = f"{args.output_dir}/nq_train_k{args.num_docs}_{args.gold_position}_{n}{flags_str}.jsonl"
    save_jsonl(path, examples)
    print_dataset_stats(examples, "Train", path)


if __name__ == "__main__":
    main()
