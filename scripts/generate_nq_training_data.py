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

PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"
PASSAGE_TEMPLATE_ID = "Document [{id}] (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE_ID = "Document [{id}]: {text}"
INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)
RETRIEVAL_INSTRUCTION = (
    "Use the given documents to identify which document is most relevant to "
    "answering the question.\n"
    "Write your answer in the following format:\nRelevant Document: [id]"
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


def format_doc(text, title=None, use_titles=True, doc_id=None):
    """Format a document with or without title, optionally with numeric ID."""
    if doc_id is not None:
        if use_titles and title:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=title, text=text)
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=text)
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
                  use_titles=True, split_docs=False, precomputed_chunks=None,
                  retrieval=False):
    if num_docs == 0:
        # No-context (closed-book) mode
        instruction = RETRIEVAL_INSTRUCTION if retrieval else INSTRUCTION
        return {
            "instruction": instruction,
            "input": f"Question: {sample['question']}",
            "output": sample["answer"],
        }

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

        # Use pre-computed distractor chunks
        dist_chunks = precomputed_chunks
        selected = rng.sample(dist_chunks, min(num_docs - 1, len(dist_chunks)))
        dist_items = list(selected)
        while len(dist_items) < num_docs - 1:
            dist_items.append(rng.choice(dist_chunks))
        # gold_item is (text, title)
        gold_item = (gold_chunk, gold_title)
    else:
        pool = [d for d in distractors if d["question"] != sample["question"]]
        sampled = rng.sample(pool, min(num_docs - 1, len(pool)))
        dist_items = [(d["context"], make_title(d["context"]) if use_titles else None)
                      for d in sampled]
        while len(dist_items) < num_docs - 1:
            d = rng.choice(pool)
            dist_items.append((d["context"], make_title(d["context"]) if use_titles else None))
        gold_item = (gold_text, gold_title)

    pos = {"first": 0, "last": len(dist_items), "middle": len(dist_items) // 2}.get(
        gold_position, rng.randint(0, len(dist_items))
    )
    # Assemble all items: list of (text, title) tuples
    all_items = dist_items[:pos] + [gold_item] + dist_items[pos:]

    # Format documents (with IDs in retrieval mode)
    if retrieval:
        all_docs = [format_doc(text, title, use_titles, doc_id=i + 1)
                    for i, (text, title) in enumerate(all_items)]
        gold_id = pos + 1  # 1-indexed
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

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n = min(args.num_examples, len(ds))
    selected = indices[:n]
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
