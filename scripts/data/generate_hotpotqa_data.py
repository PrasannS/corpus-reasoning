"""Generate HotpotQA training data in unified structured format.

Supports both single-query and multi-query modes:
  - Single-query (default): Each example has 1 question with 2 supporting docs
    among distractors. Model must reason across both to answer.
  - Multi-query (--num-queries > 1): Each example bundles N independent questions
    with all their supporting docs shuffled together, optionally padded with
    distractors.

Output is unified JSONL (documents stored as list, formatting at train/eval time).

Usage:
    # Single-query
    python scripts/data/generate_hotpotqa_data.py --num-examples 1000 --num-docs 20
    python scripts/data/generate_hotpotqa_data.py --num-examples 500 --num-docs 30 --question-type bridge

    # Multi-query
    python scripts/data/generate_hotpotqa_data.py --num-queries 10 --num-examples 1000
    python scripts/data/generate_hotpotqa_data.py --num-queries 5 --total-docs 40 --num-examples 500
"""

import argparse
import random
from datasets import load_dataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*

from lib.io import save_jsonl, print_dataset_stats


def paragraphs_from_context(context):
    """Convert HotpotQA context dict to list of {title, text} dicts."""
    titles = context["title"]
    sentences_list = context["sentences"]
    docs = []
    for title, sentences in zip(titles, sentences_list):
        text = " ".join(s.strip() for s in sentences)
        docs.append({"title": title, "text": text})
    return docs


def get_supporting_titles(example):
    """Return set of supporting document titles."""
    return set(example["supporting_facts"]["title"])


def build_distractor_pool(ds, indices, rng):
    """Build external distractor pool from non-supporting docs of given examples."""
    pool = []
    for idx in indices:
        ex = ds[idx]
        sup_titles = get_supporting_titles(ex)
        for doc in paragraphs_from_context(ex["context"]):
            if doc["title"] not in sup_titles:
                pool.append(doc)
    rng.shuffle(pool)
    return pool


def _gather_distractors(local_distractors, distractor_pool, num_needed, rng):
    """Select distractors: local first, then external pool, with repetition if needed."""
    distractors = list(local_distractors)
    rng.shuffle(distractors)
    if len(distractors) < num_needed:
        extra = rng.sample(distractor_pool,
                           min(num_needed - len(distractors), len(distractor_pool)))
        distractors.extend(extra)
    while len(distractors) < num_needed:
        distractors.append(rng.choice(distractor_pool))
    return distractors[:num_needed]


def build_single_example(example, distractor_pool, num_docs, doc_order,
                         gold_position, rng, use_titles=True):
    """Build one unified-format example from a single HotpotQA question.

    Returns dict with: documents, queries, answers, gold_doc_indices, source.
    """
    if num_docs == 0:
        return {
            "documents": [],
            "queries": [example["question"]],
            "answers": [example["answer"]],
            "gold_doc_indices": [],
            "source": "hotpotqa",
        }

    all_docs = paragraphs_from_context(example["context"])
    supporting_titles = get_supporting_titles(example)

    # Separate supporting docs (in reasoning-chain order) from local distractors
    sf_titles_ordered = list(dict.fromkeys(example["supporting_facts"]["title"]))
    supporting = []
    for t in sf_titles_ordered:
        for d in all_docs:
            if d["title"] == t and d not in supporting:
                supporting.append(d)
                break
    local_distractors = [d for d in all_docs if d["title"] not in supporting_titles]

    num_distractors_needed = max(0, num_docs - len(supporting))
    distractors = _gather_distractors(local_distractors, distractor_pool,
                                      num_distractors_needed, rng)

    # Arrange document order
    if doc_order == "shuffled":
        all_paragraphs = supporting + distractors
        rng.shuffle(all_paragraphs)
    else:
        # "reasoning" mode: supporting docs as contiguous block at specified position
        pos_map = {
            "first": 0,
            "last": len(distractors),
            "middle": len(distractors) // 2,
        }
        pos = pos_map.get(gold_position, rng.randint(0, len(distractors)))
        all_paragraphs = distractors[:pos] + supporting + distractors[pos:]

    # Find gold doc indices (where supporting docs ended up)
    gold_indices = [i for i, d in enumerate(all_paragraphs)
                    if d["title"] in supporting_titles]

    return {
        "documents": all_paragraphs,
        "queries": [example["question"]],
        "answers": [example["answer"]],
        "gold_doc_indices": gold_indices,
        "source": "hotpotqa",
    }


def build_multi_example(examples_group, distractor_pool, total_docs, rng,
                        use_titles=True):
    """Build one unified-format multi-query example from N HotpotQA questions.

    Collects supporting docs from all N queries, deduplicates by title,
    shuffles everything together, and optionally pads with distractors.

    Returns dict with: documents, queries, answers, gold_doc_indices (per-query), source.
    """
    all_supporting = []
    all_local_distractors = []
    queries = []
    answers = []
    seen_titles = set()
    per_query_supporting_titles = []

    for ex in examples_group:
        all_docs = paragraphs_from_context(ex["context"])
        supporting_titles = get_supporting_titles(ex)
        per_query_supporting_titles.append(supporting_titles)

        for d in all_docs:
            if d["title"] in supporting_titles and d["title"] not in seen_titles:
                all_supporting.append(d)
                seen_titles.add(d["title"])

        for d in all_docs:
            if d["title"] not in supporting_titles and d["title"] not in seen_titles:
                all_local_distractors.append(d)
                seen_titles.add(d["title"])

        queries.append(ex["question"])
        answers.append(ex["answer"])

    # Pad with distractors to reach total_docs (0 = supporting only)
    num_distractors_needed = max(0, total_docs - len(all_supporting)) if total_docs > 0 else 0
    pool_available = [d for d in distractor_pool if d["title"] not in seen_titles]
    distractors = _gather_distractors(all_local_distractors, pool_available,
                                      num_distractors_needed, rng)

    # Shuffle all documents together
    all_paragraphs = all_supporting + distractors
    rng.shuffle(all_paragraphs)

    # Build per-query gold_doc_indices
    gold_doc_indices = []
    for sup_titles in per_query_supporting_titles:
        indices = [i for i, d in enumerate(all_paragraphs) if d["title"] in sup_titles]
        gold_doc_indices.append(indices)

    return {
        "documents": all_paragraphs,
        "queries": queries,
        "answers": answers,
        "gold_doc_indices": gold_doc_indices,
        "source": "hotpotqa",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HotpotQA training data")
    parser.add_argument("--num-examples", type=int, default=1000,
                        help="Number of examples to generate per split")
    parser.add_argument("--num-queries", type=int, default=1,
                        help="Queries per example (1=single-query, >1=multi-query)")
    parser.add_argument("--num-docs", type=int, default=20,
                        help="Total documents in context (single-query mode, must be >= 2)")
    parser.add_argument("--total-docs", type=int, default=0,
                        help="Total documents in context (multi-query mode, 0=supporting only)")
    parser.add_argument("--question-type", type=str, default="bridge",
                        choices=["bridge", "comparison", "all"],
                        help="Filter by question type")
    parser.add_argument("--level", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Filter by difficulty level")
    parser.add_argument("--doc-order", type=str, default="shuffled",
                        choices=["shuffled", "reasoning"],
                        help="'shuffled' = random order; 'reasoning' = supporting docs "
                             "in supporting_facts order as a block (single-query only)")
    parser.add_argument("--gold-position", type=str, default="random",
                        choices=["random", "first", "last", "middle"],
                        help="Where to place supporting doc block (only for doc-order=reasoning)")
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "both"],
                        help="Which dataset split(s) to use")
    parser.add_argument("--num-eval", type=int, default=500,
                        help="Number of eval examples (only used with --split both)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-context", action="store_true",
                        help="Generate closed-book examples (no documents)")
    args = parser.parse_args()

    multi_query = args.num_queries > 1

    if args.no_context:
        args.num_docs = 0
    elif not multi_query and args.num_docs < 2:
        parser.error("Need at least 2 docs for multi-hop QA (or use --no-context)")

    rng = random.Random(args.seed)

    if args.split == "both":
        splits_to_process = [("train", args.num_examples), ("validation", args.num_eval)]
    else:
        splits_to_process = [(args.split, args.num_examples)]

    for split_name, num_wanted in splits_to_process:
        print(f"\nLoading hotpotqa/hotpot_qa distractor ({split_name})...")
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)
        print(f"  Loaded {len(ds)} examples")

        if args.question_type != "all":
            ds = ds.filter(lambda ex: ex["type"] == args.question_type)
            print(f"  After filtering to type={args.question_type}: {len(ds)} examples")

        if args.level != "all":
            ds = ds.filter(lambda ex: ex["level"] == args.level)
            print(f"  After filtering to level={args.level}: {len(ds)} examples")

        if len(ds) == 0:
            print(f"  No examples remaining for {split_name}, skipping.")
            continue

        indices = list(range(len(ds)))
        rng.shuffle(indices)

        if multi_query:
            # Multi-query: each example needs num_queries source examples
            source_needed = num_wanted * args.num_queries
            if len(ds) < source_needed:
                num_wanted = len(ds) // args.num_queries
                print(f"  Warning: only enough for {num_wanted} multi-query examples")
            if num_wanted == 0:
                continue

            selected_count = num_wanted * args.num_queries
            selected = indices[:selected_count]
            pool_indices = indices[selected_count:selected_count + selected_count] or indices
            distractor_pool = build_distractor_pool(ds, pool_indices, rng)
            print(f"  Distractor pool: {len(distractor_pool)} paragraphs")

            # Group into bundles of num_queries
            groups = []
            for i in range(0, selected_count, args.num_queries):
                groups.append([ds[idx] for idx in selected[i:i + args.num_queries]])

            docs_label = f"total_docs={args.total_docs}" if args.total_docs > 0 else "supporting_only"
            print(f"  Generating {len(groups)} multi-query examples "
                  f"(N={args.num_queries}, {docs_label})...")

            examples = []
            for group in groups:
                examples.append(build_multi_example(
                    group, distractor_pool, args.total_docs, rng,
                    use_titles=not args.no_titles,
                ))
        else:
            # Single-query mode
            n = min(num_wanted, len(ds))
            selected = indices[:n]
            pool_indices = indices[n:n + n * 3] if n < len(ds) else indices
            distractor_pool = build_distractor_pool(ds, pool_indices, rng) if args.num_docs > 0 else []
            if distractor_pool:
                print(f"  Distractor pool: {len(distractor_pool)} paragraphs")

            print(f"  Generating {n} examples (docs={args.num_docs}, "
                  f"order={args.doc_order}, gold_pos={args.gold_position})...")
            examples = []
            for i in selected:
                examples.append(build_single_example(
                    ds[i], distractor_pool, args.num_docs,
                    args.doc_order, args.gold_position, rng,
                    use_titles=not args.no_titles,
                ))

        # Build filename
        if multi_query:
            flags = [f"n{args.num_queries}"]
            flags.append(f"k{args.total_docs}" if args.total_docs > 0 else "suponly")
        elif args.num_docs == 0:
            flags = ["nocontext"]
        else:
            flags = [f"k{args.num_docs}", args.doc_order]

        if args.question_type != "all":
            flags.append(args.question_type)
        if args.level != "all":
            flags.append(args.level)
        if args.no_titles:
            flags.append("notitle")
        tag = "_".join(flags)

        label = "train" if split_name == "train" else "eval"
        prefix = "multi_hotpotqa" if multi_query else "hotpotqa"
        path = f"{args.output_dir}/{prefix}_{label}_{tag}_{len(examples)}.jsonl"

        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)


if __name__ == "__main__":
    main()
