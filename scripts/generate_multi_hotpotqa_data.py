"""Generate multi-query HotpotQA training data for Axolotl.

Each example bundles N independent HotpotQA queries into a single context.
The corpus contains the supporting documents for all N queries (shuffled together),
optionally padded with distractors to reach a target total document count.
The expected answer is a comma-separated list of answers in query order.

Key flags:
  --num-queries       Number of independent queries per example (default: 10)
  --total-docs        Total documents in context (0 = just supporting docs, no distractors)
  --num-examples      Number of multi-query examples to generate

Usage:
    python scripts/generate_multi_hotpotqa_data.py --num-examples 1000
    python scripts/generate_multi_hotpotqa_data.py --num-examples 500 --num-queries 5 --total-docs 40
    python scripts/generate_multi_hotpotqa_data.py --num-examples 2000 --split both
"""

import argparse
import random
from datasets import load_dataset
from lib.io import save_jsonl, print_dataset_stats

PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"
PASSAGE_TEMPLATE_ID = "Document [{id}] (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE_ID = "Document [{id}]: {text}"
INSTRUCTION = (
    "Use the given documents to answer each of the following questions. "
    "Write a concise and short answer for each question, in order, as a comma-separated list.\n"
    "Write your answer in the following format:\nAnswers: [answer1], [answer2], ..."
)
RETRIEVAL_INSTRUCTION = (
    "Use the given documents to identify which documents are relevant to "
    "answering each of the following questions. For each question, list the "
    "relevant document IDs.\n"
    "Write your answer in the following format:\n"
    "Relevant Documents: Q1: [id1], [id2]; Q2: [id3], [id4]; ..."
)


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


def format_doc(doc, use_titles=True, doc_id=None):
    if doc_id is not None:
        if use_titles:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=doc["title"], text=doc["text"])
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=doc["text"])
    if use_titles:
        return PASSAGE_TEMPLATE.format(title=doc["title"], text=doc["text"])
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=doc["text"])


def build_multi_example(examples_group, distractor_pool, total_docs, rng,
                        use_titles=True, retrieval=False):
    """Build one multi-query training example from N HotpotQA questions.

    Args:
        examples_group: List of N HotpotQA dataset rows.
        distractor_pool: List of extra distractor doc dicts from other examples.
        total_docs: Total number of documents in context (0 = supporting only).
        rng: Random instance.
        use_titles: Whether to include document titles.
        retrieval: If True, output relevant document IDs per query instead of answers.

    Returns:
        dict with instruction, input, output fields, or None if dedup fails.
    """
    all_supporting = []
    all_local_distractors = []
    questions = []
    answers = []
    seen_titles = set()
    # Track per-query supporting titles for retrieval mode
    per_query_supporting_titles = []

    for ex in examples_group:
        all_docs = paragraphs_from_context(ex["context"])
        supporting_titles = get_supporting_titles(ex)
        per_query_supporting_titles.append(supporting_titles)

        # Collect supporting docs (deduplicate by title across queries)
        for d in all_docs:
            if d["title"] in supporting_titles and d["title"] not in seen_titles:
                all_supporting.append(d)
                seen_titles.add(d["title"])

        # Collect local distractors
        for d in all_docs:
            if d["title"] not in supporting_titles and d["title"] not in seen_titles:
                all_local_distractors.append(d)
                seen_titles.add(d["title"])

        questions.append(ex["question"])
        answers.append(ex["answer"])

    num_supporting = len(all_supporting)

    # Determine how many distractors we need
    if total_docs > 0:
        num_distractors_needed = max(0, total_docs - num_supporting)
    else:
        num_distractors_needed = 0

    # Build distractor list: prefer local distractors, then pool
    distractors = list(all_local_distractors)
    rng.shuffle(distractors)
    if len(distractors) < num_distractors_needed:
        pool_available = [d for d in distractor_pool if d["title"] not in seen_titles]
        extra = rng.sample(pool_available,
                           min(num_distractors_needed - len(distractors), len(pool_available)))
        distractors.extend(extra)
    # If still not enough, allow duplicates from pool
    while len(distractors) < num_distractors_needed:
        distractors.append(rng.choice(distractor_pool))
    distractors = distractors[:num_distractors_needed]

    # Shuffle all documents together
    all_paragraphs = all_supporting + distractors
    rng.shuffle(all_paragraphs)

    # Format documents (with IDs in retrieval mode)
    if retrieval:
        formatted_docs = [format_doc(d, use_titles, doc_id=i + 1)
                          for i, d in enumerate(all_paragraphs)]
        # Build per-query doc ID output
        query_parts = []
        for qi, sup_titles in enumerate(per_query_supporting_titles):
            gold_ids = sorted(
                i + 1 for i, d in enumerate(all_paragraphs)
                if d["title"] in sup_titles
            )
            ids_str = ", ".join(f"[{gid}]" for gid in gold_ids)
            query_parts.append(f"Q{qi + 1}: {ids_str}")
        output = "; ".join(query_parts)
    else:
        formatted_docs = [format_doc(d, use_titles) for d in all_paragraphs]
        output = ", ".join(answers)

    context = "\n\n".join(formatted_docs)

    # Build questions block
    questions_block = "\n".join(
        f"Question {i+1}: {q}" for i, q in enumerate(questions)
    )

    instruction = RETRIEVAL_INSTRUCTION if retrieval else INSTRUCTION

    return {
        "instruction": instruction,
        "input": f"{context}\n\n{questions_block}",
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate multi-query HotpotQA training data")
    parser.add_argument("--num-examples", type=int, default=1000,
                        help="Number of multi-query examples to generate per split")
    parser.add_argument("--num-queries", type=int, default=10,
                        help="Number of independent queries per example")
    parser.add_argument("--total-docs", type=int, default=0,
                        help="Total documents in context (0 = supporting docs only, no distractors)")
    parser.add_argument("--question-type", type=str, default="bridge",
                        choices=["bridge", "comparison", "all"],
                        help="Filter by question type")
    parser.add_argument("--level", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Filter by difficulty level")
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles")
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "validation", "both"],
                        help="Which dataset split(s) to use")
    parser.add_argument("--num-eval", type=int, default=500,
                        help="Number of eval examples (only used with --split both)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrieval", action="store_true", default=True,
                        help="Generate retrieval task: output relevant document IDs instead of answers (default)")
    parser.add_argument("--no-retrieval", action="store_false", dest="retrieval",
                        help="Generate QA task instead of retrieval")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Each multi-query example needs num_queries source examples
    total_source_needed_per_split = {
        "train": args.num_examples * args.num_queries,
        "validation": args.num_eval * args.num_queries,
    }

    if args.split == "both":
        splits_to_process = [("train", args.num_examples), ("validation", args.num_eval)]
    else:
        splits_to_process = [(args.split, args.num_examples)]

    for split_name, num_wanted in splits_to_process:
        print(f"\nLoading hotpotqa/hotpot_qa distractor ({split_name})...")
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split_name)
        print(f"  Loaded {len(ds)} examples")

        # Filter by question type
        if args.question_type != "all":
            ds = ds.filter(lambda ex: ex["type"] == args.question_type)
            print(f"  After filtering to type={args.question_type}: {len(ds)} examples")

        # Filter by level
        if args.level != "all":
            ds = ds.filter(lambda ex: ex["level"] == args.level)
            print(f"  After filtering to level={args.level}: {len(ds)} examples")

        source_needed = num_wanted * args.num_queries
        if len(ds) < source_needed:
            actual_examples = len(ds) // args.num_queries
            print(f"  Warning: only enough for {actual_examples} multi-query examples "
                  f"(need {source_needed} source examples, have {len(ds)})")
            num_wanted = actual_examples

        if num_wanted == 0:
            print(f"  No examples remaining for {split_name}, skipping.")
            continue

        # Shuffle and partition into groups of num_queries
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        selected_count = num_wanted * args.num_queries
        selected = indices[:selected_count]

        # Build external distractor pool from non-selected examples
        distractor_pool = []
        pool_indices = indices[selected_count:selected_count + selected_count]
        if not pool_indices:
            pool_indices = indices  # fallback
        for idx in pool_indices:
            ex = ds[idx]
            sup_titles = get_supporting_titles(ex)
            for doc in paragraphs_from_context(ex["context"]):
                if doc["title"] not in sup_titles:
                    distractor_pool.append(doc)
        rng.shuffle(distractor_pool)
        print(f"  Distractor pool: {len(distractor_pool)} paragraphs")

        # Group selected examples into multi-query bundles
        groups = []
        for i in range(0, selected_count, args.num_queries):
            group = [ds[idx] for idx in selected[i:i + args.num_queries]]
            groups.append(group)

        docs_label = f"total_docs={args.total_docs}" if args.total_docs > 0 else "supporting_only"
        print(f"  Generating {len(groups)} multi-query examples "
              f"(N={args.num_queries}, {docs_label}, type={args.question_type})...")

        examples = []
        for group in groups:
            ex = build_multi_example(group, distractor_pool, args.total_docs, rng,
                                     use_titles=not args.no_titles, retrieval=args.retrieval)
            examples.append(ex)

        # Build filename
        flags = []
        if args.retrieval:
            flags.append("retrieval")
        flags.append(f"n{args.num_queries}")
        if args.total_docs > 0:
            flags.append(f"k{args.total_docs}")
        else:
            flags.append("suponly")
        if args.question_type != "all":
            flags.append(args.question_type)
        if args.level != "all":
            flags.append(args.level)
        if args.no_titles:
            flags.append("notitle")
        tag = "_".join(flags)
        label = "train" if split_name == "train" else "eval"
        path = f"{args.output_dir}/multi_hotpotqa_{label}_{tag}_{len(examples)}.jsonl"

        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)

        # Show a sample
        if examples:
            ex = examples[0]
            parts = ex["input"].split("\n\n")
            num_doc_parts = len(parts) - 1  # last part is the questions block
            print(f"\n=== Sample ({num_doc_parts} docs, {args.num_queries} queries) ===")
            for part in parts[:3]:
                print(f"  {part[:120]}...")
            if num_doc_parts > 3:
                print(f"  ... ({num_doc_parts - 3} more docs)")
            # Show questions
            print(f"  {parts[-1][:200]}")
            print(f"  Output: {ex['output']}")


if __name__ == "__main__":
    main()
