"""Generate multi-hop QA training data from HotpotQA for Axolotl.

Uses HELMET-compatible prompt format (same passage template & instruction as NQ eval)
so that training and evaluation prompts are aligned.

Each example places 2 supporting documents among distractors. The model must
reason across both documents to answer the multi-hop question.

Key flags:
  --num-docs        Total documents in context (must be >= 2)
  --question-type   Filter by type: "bridge", "comparison", or "all"
  --doc-order       "reasoning" = supporting docs in supporting_facts order;
                    "shuffled"  = all docs randomly shuffled (default)
  --gold-position   Where to place supporting docs when doc-order=reasoning

Usage:
    python scripts/generate_hotpotqa_data.py --num-examples 1000 --num-docs 20
    python scripts/generate_hotpotqa_data.py --num-examples 500 --num-docs 30 --question-type bridge --doc-order reasoning
    python scripts/generate_hotpotqa_data.py --num-examples 2000 --num-docs 20 --split both
"""

import argparse
import random
from datasets import load_dataset
from lib.io import save_jsonl, print_dataset_stats

# Match HELMET eval format exactly (evaluate_helmet_rag.py lines 21-28)
PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"
PASSAGE_TEMPLATE_ID = "Document [{id}] (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE_ID = "Document [{id}]: {text}"
INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)
RETRIEVAL_INSTRUCTION = (
    "Use the given documents to identify which documents are relevant to "
    "answering the question. List all relevant document IDs.\n"
    "Write your answer in the following format:\nRelevant Documents: [id1], [id2]"
)


def paragraphs_from_context(context):
    """Convert HotpotQA context dict to list of {title, text, is_supporting} dicts."""
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
    """Format a document dict as a passage string, optionally with numeric ID."""
    if doc_id is not None:
        if use_titles:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=doc["title"], text=doc["text"])
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=doc["text"])
    if use_titles:
        return PASSAGE_TEMPLATE.format(title=doc["title"], text=doc["text"])
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=doc["text"])


def build_example(example, distractor_pool, num_docs, doc_order, gold_position,
                  rng, use_titles=True, retrieval=False):
    """Build one training example from a HotpotQA question.

    Args:
        example: A HotpotQA dataset row.
        distractor_pool: List of extra distractor doc dicts from other examples.
        num_docs: Total number of documents to include.
        doc_order: "reasoning" or "shuffled".
        gold_position: Where to place supporting docs ("random", "first", "last", "middle").
        rng: Random instance.
        use_titles: Whether to include document titles.
        retrieval: If True, output relevant document IDs instead of answer.
    """
    if num_docs == 0:
        instruction = RETRIEVAL_INSTRUCTION if retrieval else INSTRUCTION
        return {
            "instruction": instruction,
            "input": f"Question: {example['question']}",
            "output": example["answer"],
        }

    all_docs = paragraphs_from_context(example["context"])
    supporting_titles = get_supporting_titles(example)

    # Split into supporting and in-example distractors
    # Preserve supporting_facts order for reasoning mode
    sf_titles_ordered = list(dict.fromkeys(example["supporting_facts"]["title"]))
    supporting = []
    for t in sf_titles_ordered:
        for d in all_docs:
            if d["title"] == t and d not in supporting:
                supporting.append(d)
                break
    local_distractors = [d for d in all_docs if d["title"] not in supporting_titles]

    num_distractors_needed = max(0, num_docs - len(supporting))

    # Build distractor list: prefer in-example distractors, then external pool
    distractors = list(local_distractors)
    rng.shuffle(distractors)
    if len(distractors) < num_distractors_needed:
        extra = rng.sample(distractor_pool,
                           min(num_distractors_needed - len(distractors),
                               len(distractor_pool)))
        distractors.extend(extra)
    while len(distractors) < num_distractors_needed:
        distractors.append(rng.choice(distractor_pool))
    distractors = distractors[:num_distractors_needed]

    if doc_order == "shuffled":
        all_paragraphs = supporting + distractors
        rng.shuffle(all_paragraphs)
    else:
        # reasoning: supporting docs in supporting_facts order, inserted as block
        pos_map = {
            "first": 0,
            "last": len(distractors),
            "middle": len(distractors) // 2,
        }
        pos = pos_map.get(gold_position, rng.randint(0, len(distractors)))
        all_paragraphs = distractors[:pos] + supporting + distractors[pos:]

    # Format documents (with IDs in retrieval mode)
    if retrieval:
        formatted_docs = [format_doc(d, use_titles, doc_id=i + 1)
                          for i, d in enumerate(all_paragraphs)]
        # Find supporting doc positions (1-indexed)
        gold_ids = sorted(
            i + 1 for i, d in enumerate(all_paragraphs)
            if d["title"] in supporting_titles
        )
        output = ", ".join(f"[{gid}]" for gid in gold_ids)
    else:
        formatted_docs = [format_doc(d, use_titles) for d in all_paragraphs]
        output = example["answer"]

    context = "\n\n".join(formatted_docs)
    instruction = RETRIEVAL_INSTRUCTION if retrieval else INSTRUCTION

    return {
        "instruction": instruction,
        "input": f"{context}\n\nQuestion: {example['question']}",
        "output": output,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate HotpotQA multi-hop training data")
    parser.add_argument("--num-examples", type=int, default=1000,
                        help="Number of examples to generate per split")
    parser.add_argument("--num-docs", type=int, default=20,
                        help="Total documents in context (must be >= 2)")
    parser.add_argument("--question-type", type=str, default="bridge",
                        choices=["bridge", "comparison", "all"],
                        help="Filter by question type")
    parser.add_argument("--level", type=str, default="all",
                        choices=["easy", "medium", "hard", "all"],
                        help="Filter by difficulty level")
    parser.add_argument("--doc-order", type=str, default="shuffled",
                        choices=["shuffled", "reasoning"],
                        help="'shuffled' = random order; 'reasoning' = supporting docs "
                             "in supporting_facts order as a block")
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
                        help="Generate closed-book examples (no documents, just question)")
    parser.add_argument("--retrieval", action="store_true", default=True,
                        help="Generate retrieval task: output relevant document IDs instead of answer (default)")
    parser.add_argument("--no-retrieval", action="store_false", dest="retrieval",
                        help="Generate QA task instead of retrieval")
    args = parser.parse_args()

    if args.no_context:
        args.num_docs = 0
    elif args.num_docs < 2:
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

        # Filter by question type
        if args.question_type != "all":
            ds = ds.filter(lambda ex: ex["type"] == args.question_type)
            print(f"  After filtering to type={args.question_type}: {len(ds)} examples")

        # Filter by level
        if args.level != "all":
            ds = ds.filter(lambda ex: ex["level"] == args.level)
            print(f"  After filtering to level={args.level}: {len(ds)} examples")

        if len(ds) == 0:
            print(f"  No examples remaining for {split_name}, skipping.")
            continue

        # Select examples
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        n = min(num_wanted, len(ds))
        selected = indices[:n]

        # Build external distractor pool from non-selected examples
        distractor_pool = []
        if args.num_docs > 0:
            pool_indices = indices[n:n + n * 3] if n < len(ds) else indices
            for idx in pool_indices:
                ex = ds[idx]
                sup_titles = get_supporting_titles(ex)
                for doc in paragraphs_from_context(ex["context"]):
                    if doc["title"] not in sup_titles:
                        distractor_pool.append(doc)
            rng.shuffle(distractor_pool)
            print(f"  Distractor pool: {len(distractor_pool)} paragraphs")

        ctx_label = "no-context" if args.num_docs == 0 else f"docs={args.num_docs}"
        task_type = "retrieval" if args.retrieval else "QA"
        print(f"  Generating {n} {task_type} examples ({ctx_label}, type={args.question_type}, "
              f"order={args.doc_order}, gold_pos={args.gold_position})...")
        examples = []
        for i in selected:
            ex = build_example(ds[i], distractor_pool, args.num_docs,
                               args.doc_order, args.gold_position, rng,
                               use_titles=not args.no_titles, retrieval=args.retrieval)
            examples.append(ex)

        # Build filename
        flags = ["nocontext"] if args.num_docs == 0 else [f"k{args.num_docs}", args.doc_order]
        if args.retrieval:
            flags.append("retrieval")
        if args.question_type != "all":
            flags.append(args.question_type)
        if args.level != "all":
            flags.append(args.level)
        if args.no_titles:
            flags.append("notitle")
        tag = "_".join(flags)
        label = "train" if split_name == "train" else "eval"
        path = f"{args.output_dir}/hotpotqa_{label}_{tag}_{n}.jsonl"

        save_jsonl(path, examples)
        print_dataset_stats(examples, split_name.capitalize(), path)

        # Show a sample
        if examples:
            ex = examples[0]
            parts = ex["input"].split("\n\n")
            print(f"\n=== Sample ({len(parts) - 1} docs) ===")
            for part in parts[:3]:
                print(f"  {part[:120]}...")
            if len(parts) > 4:
                print(f"  ... ({len(parts) - 4} more docs)")
            print(f"  {parts[-1]}")
            print(f"  Output: {ex['output']}")


if __name__ == "__main__":
    main()
