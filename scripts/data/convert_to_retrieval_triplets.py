"""Convert existing long-context retrieval JSONL into triplets for dense retrieval.

Takes the same training data used by the long-context models (e.g.,
hotpotqa_train_k20_shuffled_retrieval_bridge_5000.jsonl) and extracts
(query, positive_doc, negative_doc) triplets, ensuring the retrieval model
sees the exact same query-document pairs.

For each example with Q gold docs and D-Q distractors, generates Q*(D-Q) triplets
(one per gold × distractor combination), so the model sees every positive-negative
pair from the original context.

Use --max-triplets or adjust epochs to control total training tokens.

Usage:
    # Full exhaustive triplets
    python scripts/convert_to_retrieval_triplets.py \
        data/hotpotqa_train_k20_shuffled_retrieval_bridge_5000.jsonl \
        data/hotpotqa_train_k20_retrieval_triplets.jsonl

    # Subsample to match long-context token budget
    python scripts/convert_to_retrieval_triplets.py \
        data/hotpotqa_train_k20_shuffled_retrieval_bridge_5000.jsonl \
        data/hotpotqa_train_k20_retrieval_triplets_50k.jsonl \
        --max-triplets 50000
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports

from lib.io import load_jsonl, save_jsonl


def parse_documents_from_input(input_text):
    """Extract individual documents and question from the formatted input field."""
    # Split off the question at the end
    question_match = re.search(r'\n\nQuestion:\s*(.+)$', input_text, re.DOTALL)
    if question_match:
        doc_text = input_text[:question_match.start()]
        question = question_match.group(1).strip()
    else:
        doc_text = input_text
        question = ""

    # Locate each document start
    doc_starts = [m.start() for m in re.finditer(r'Document\s+\[', doc_text)]
    doc_chunks = []
    for i, start in enumerate(doc_starts):
        end = doc_starts[i + 1] if i + 1 < len(doc_starts) else len(doc_text)
        doc_chunks.append(doc_text[start:end].strip())

    header_pattern = re.compile(
        r'Document\s+\[(\d+)\]\s*'
        r'(?:\(Title:\s*(.*?)\)\s*:\s*|\:\s*)'
        r'(.*)',
        re.DOTALL
    )

    docs = []
    for chunk in doc_chunks:
        match = header_pattern.match(chunk)
        if not match:
            continue
        doc_id = int(match.group(1))
        title = match.group(2).strip() if match.group(2) else None
        text = match.group(3).strip()
        # Format as "Title: text" to match retrieval model input
        if title:
            full_text = f"{title}: {text}"
        else:
            full_text = text
        docs.append({"id": doc_id, "full_text": full_text})

    return docs, question


def convert_to_triplets(examples, rng, max_triplets=None):
    """Convert long-context retrieval examples to exhaustive triplets.

    Each example has a query, gold doc IDs, and a set of documents.
    Generates one triplet per (gold_doc, distractor) pair.
    """
    triplets = []

    for ex in examples:
        docs, question = parse_documents_from_input(ex["input"])
        gold_ids = set(int(m) for m in re.findall(r'\[(\d+)\]', ex["output"]))

        if not docs or not question or not gold_ids:
            continue

        doc_by_id = {d["id"]: d["full_text"] for d in docs}
        gold_docs = [(did, doc_by_id[did]) for did in sorted(gold_ids) if did in doc_by_id]
        distractor_docs = [(did, doc_by_id[did]) for did in sorted(doc_by_id.keys())
                           if did not in gold_ids]

        for _gid, gold_text in gold_docs:
            for _did, dist_text in distractor_docs:
                triplets.append({
                    "query": question,
                    "positive": gold_text,
                    "negative": dist_text,
                })

    rng.shuffle(triplets)
    if max_triplets and len(triplets) > max_triplets:
        triplets = triplets[:max_triplets]

    return triplets


def main():
    parser = argparse.ArgumentParser(
        description="Convert long-context retrieval JSONL to dense retrieval triplets")
    parser.add_argument("input_file", type=str,
                        help="Input JSONL (e.g., hotpotqa_train_k20_shuffled_retrieval_bridge_5000.jsonl)")
    parser.add_argument("output_file", type=str,
                        help="Output triplet JSONL")
    parser.add_argument("--max-triplets", type=int, default=None,
                        help="Maximum number of triplets to output (subsample if more)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading {args.input_file}...")
    examples = load_jsonl(args.input_file)
    print(f"  {len(examples)} examples")

    triplets = convert_to_triplets(examples, rng, args.max_triplets)

    # Stats
    avg_query_chars = sum(len(t["query"]) for t in triplets) / len(triplets)
    avg_pos_chars = sum(len(t["positive"]) for t in triplets) / len(triplets)
    avg_neg_chars = sum(len(t["negative"]) for t in triplets) / len(triplets)
    avg_total_chars = avg_query_chars + avg_pos_chars + avg_neg_chars
    approx_tokens = sum(len(t["query"]) + len(t["positive"]) + len(t["negative"])
                        for t in triplets) / 4

    print(f"\n  Generated {len(triplets)} triplets")
    print(f"  Avg chars: query={avg_query_chars:.0f}, pos={avg_pos_chars:.0f}, neg={avg_neg_chars:.0f}, total={avg_total_chars:.0f}")
    print(f"  Approx total tokens: {approx_tokens/1e6:.1f}M")

    save_jsonl(args.output_file, triplets)
    print(f"\n  Saved to {args.output_file}")

    # Show samples
    print("\n=== Samples ===")
    for t in triplets[:3]:
        print(f"  Q: {t['query'][:80]}")
        print(f"  +: {t['positive'][:80]}...")
        print(f"  -: {t['negative'][:80]}...")
        print()


if __name__ == "__main__":
    main()
