"""Generate human-readable example files showing exactly what the model sees.

For each task variant, shows 3 training examples and 3 eval examples with
the full alpaca-wrapped prompt. Eval examples use the same code path as the
actual eval scripts (HELMET data for QA, generated JSONL for retrieval).

Usage:
    python scripts/generate_examples.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
from lib.io import ALPACA_TEMPLATE, load_jsonl, format_alpaca_prompt, insert_dummy_tokens
from lib.data_format import build_prompt
from lib.prompts import (
    PASSAGE_TEMPLATE,
    QA_INSTRUCTION, MULTI_QA_INSTRUCTION,
    RETRIEVAL_INSTRUCTION_SINGLE, RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY,
)

EXAMPLES_DIR = Path("examples")
EXAMPLES_DIR.mkdir(exist_ok=True)

NUM_EXAMPLES = 3


def count_docs(input_text):
    """Count documents in input text."""
    n = input_text.count("Document (Title:") + input_text.count("Document [")
    if n == 0:
        n = input_text.count("Document:")
    return n


def format_train_example(ex):
    """Format a training example using alpaca template (matches Axolotl training).

    Supports both unified format (documents/queries) and legacy (instruction/input/output).
    """
    if "documents" in ex:
        return build_prompt(ex, task="qa", use_alpaca=True)
    prompt = format_alpaca_prompt(ex["instruction"], ex["input"])
    return prompt, ex["output"]


def fmt_unified(task="retrieval", query_position="after", before_dummy=0, after_dummy=0):
    """Return a formatter function for unified-format examples."""
    def _fmt(ex):
        if "documents" in ex:
            return build_prompt(ex, task=task, query_position=query_position,
                                before_dummy=before_dummy, after_dummy=after_dummy,
                                use_alpaca=True)
        # Fallback for legacy format
        prompt = format_alpaca_prompt(ex["instruction"], ex["input"])
        return prompt, ex["output"]
    return _fmt


def format_helmet_eval_example(sample, instruction, query_position="after",
                               before_dummy=0, after_dummy=0):
    """Format a HELMET eval example the same way evaluate_helmet_rag.py does for trained models.

    This replicates the alpaca-format code path from evaluate_helmet_rag.py:load_dataset_for_eval
    with use_alpaca=True, shots=0 (trained model eval).
    """
    context = "\n\n".join(
        PASSAGE_TEMPLATE.format(**ctx) for ctx in sample.get("ctxs", [])
    )
    question = sample["question"]

    if query_position == "both":
        input_text = f"Question: {question}\n\n{context}\n\nQuestion: {question}"
    elif query_position == "before":
        input_text = f"Question: {question}\n\n{context}"
    else:
        input_text = f"{context}\n\nQuestion: {question}"

    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)

    prompt = format_alpaca_prompt(instruction, input_text)
    answers = sample["answers"]
    return prompt, answers


def format_multi_hotpotqa_eval_example(ex, instruction, query_position="after"):
    """Format a multi-hotpotqa eval example from JSONL the way evaluate_multi_hotpotqa.py does."""
    input_text = ex["input"]

    if query_position == "before":
        parts = input_text.rsplit("\n\n", 1)
        if len(parts) == 2:
            input_text = f"{parts[1]}\n\n{parts[0]}"
    elif query_position == "both":
        parts = input_text.rsplit("\n\n", 1)
        if len(parts) == 2:
            input_text = f"{parts[1]}\n\n{parts[0]}\n\n{parts[1]}"

    prompt = format_alpaca_prompt(instruction, input_text)
    return prompt, ex["output"]


def format_retrieval_eval_example(ex, instruction, task, query_position="after"):
    """Format a retrieval eval example the way evaluate_retrieval.py does."""
    input_text = ex["input"]

    if task == "multi-hotpotqa":
        sep_point = input_text.rfind("\nQuestion 1:")
        if sep_point >= 0:
            context_part = input_text[:sep_point]
            questions_part = input_text[sep_point:]
        else:
            context_part = input_text
            questions_part = ""

        if query_position == "before":
            input_text = f"{questions_part.strip()}\n\n{context_part}"
        elif query_position == "both":
            input_text = f"{questions_part.strip()}\n\n{context_part}\n\n{questions_part.strip()}"
    else:
        parts = input_text.rsplit("\n\nQuestion:", 1)
        if len(parts) == 2:
            context_part = parts[0]
            question_part = f"Question:{parts[1]}"
            if query_position == "before":
                input_text = f"{question_part}\n\n{context_part}"
            elif query_position == "both":
                input_text = f"{question_part}\n\n{context_part}\n\n{question_part}"

    prompt = format_alpaca_prompt(instruction, input_text)
    return prompt, ex["output"]


def write_section(f, label, prompt, output, idx):
    """Write one example section to the file."""
    doc_count = count_docs(prompt)
    f.write(f"{'=' * 80}\n")
    f.write(f"  {label} Example {idx}\n")
    f.write(f"  Documents: {doc_count} | Prompt chars: {len(prompt):,} | Output chars: {len(output):,}\n")
    f.write(f"{'=' * 80}\n\n")
    f.write("--- MODEL INPUT (prompt) ---\n\n")
    f.write(prompt)
    f.write("\n--- EXPECTED OUTPUT (completion) ---\n\n")
    if isinstance(output, list):
        f.write(f"{output[0]}  (gold answers: {output})\n")
    else:
        f.write(output + "\n")
    f.write("\n\n")


# ── Example definitions ──
# Each entry: (name, description, train_config, eval_config)
# train_config: (file, formatter_fn)
# eval_config: (file, formatter_fn) or None

TASK_EXAMPLES = []


def add_task(name, description, train_file, eval_file, train_fmt, eval_fmt):
    TASK_EXAMPLES.append((name, description, train_file, eval_file, train_fmt, eval_fmt))


# ── QA tasks ──

def fmt_train(ex):
    return format_train_example(ex)

def fmt_helmet_qa_eval(sample):
    return format_helmet_eval_example(sample, QA_INSTRUCTION)

def fmt_helmet_qa_eval_qboth(sample):
    return format_helmet_eval_example(sample, QA_INSTRUCTION, query_position="both")

def fmt_multi_qa_eval(ex):
    return format_multi_hotpotqa_eval_example(ex, MULTI_QA_INSTRUCTION)

# ── Retrieval tasks ──

def fmt_retrieval_nq_eval(ex):
    return format_retrieval_eval_example(ex, RETRIEVAL_INSTRUCTION_SINGLE, "nq")

def fmt_retrieval_hotpotqa_eval(ex):
    return format_retrieval_eval_example(ex, RETRIEVAL_INSTRUCTION_MULTI_DOC, "hotpotqa")

def fmt_retrieval_multi_eval(ex):
    return format_retrieval_eval_example(ex, RETRIEVAL_INSTRUCTION_MULTI_QUERY, "multi-hotpotqa")


# QA tasks
add_task("nq_qa",
         "NQ question-answering: 20 documents, output is the answer text",
         "data/nq_train_k20_random_2500.jsonl",
         "data/data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl",
         fmt_train, fmt_helmet_qa_eval)

add_task("nq_qa_qboth",
         "NQ question-answering with query-both: question before and after documents",
         "data/nq_train_k20_random_2500_qboth.jsonl",
         "data/data/kilt/nq-dev-multikilt_1000_k20_dep6.jsonl",
         fmt_train, fmt_helmet_qa_eval_qboth)

add_task("hotpotqa_qa",
         "HotpotQA multi-hop QA: 2 supporting docs among 20 total, output is the answer",
         "data/hotpotqa_train_k20_shuffled_bridge_2500.jsonl",
         "data/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl",
         fmt_train, fmt_helmet_qa_eval)

add_task("hotpotqa_qa_qboth",
         "HotpotQA multi-hop QA with query-both",
         "data/hotpotqa_train_k20_shuffled_bridge_2500_qboth.jsonl",
         "data/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl",
         fmt_train, fmt_helmet_qa_eval_qboth)

add_task("multi_hotpotqa_qa",
         "Multi-query HotpotQA QA: 10 queries, comma-separated answers",
         "data/multi_hotpotqa_train_n10_k50_bridge_5000.jsonl",
         # Eval is generated on-the-fly by evaluate_multi_hotpotqa.py; use training file
         # with offset to show different examples for the eval section
         "data/multi_hotpotqa_train_n10_k50_bridge_5000.jsonl",
         fmt_train, fmt_multi_qa_eval)

# Retrieval tasks
add_task("nq_retrieval",
         "NQ retrieval: 20 docs with IDs, output is the relevant document ID",
         "data/nq_train_k20_random_1000_retrieval.jsonl",
         "data/nq_train_k20_random_500_retrieval.jsonl",
         fmt_train, fmt_retrieval_nq_eval)

add_task("hotpotqa_retrieval",
         "HotpotQA retrieval: 20 docs with IDs, output is the 2 relevant document IDs",
         "data/hotpotqa_train_k20_shuffled_retrieval_bridge_2500.jsonl",
         "data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl",
         fmt_train, fmt_retrieval_hotpotqa_eval)

# ── Dummy token tasks ──

def fmt_train_dummy_before(ex):
    """Format training example with before_dummy=10 (matches convert_to_dummy.py --before-dummy 10)."""
    modified = dict(ex)
    modified["input"] = insert_dummy_tokens(ex["input"], before_dummy=10)
    return format_train_example(modified)

def fmt_helmet_qa_eval_qboth_dummy_before(sample):
    """Format eval example with qboth + before_dummy=10."""
    return format_helmet_eval_example(sample, QA_INSTRUCTION, query_position="both",
                                       before_dummy=10)

add_task("hotpotqa_qa_qboth_dummy_before",
         "HotpotQA QA with query-both and 10 dummy tokens before documents",
         "data/hotpotqa_train_k10_shuffled_retrieval_bridge_5000_qboth_bd10.jsonl",
         "data/data/kilt/hotpotqa-dev-multikilt_1000_k20_dep3.jsonl",
         fmt_train_dummy_before, fmt_helmet_qa_eval_qboth_dummy_before)

add_task("multi_hotpotqa_retrieval",
         "Multi-query HotpotQA retrieval: per-query relevant document IDs",
         "data/multi_hotpotqa_train_retrieval_n10_suponly_bridge_500.jsonl",
         "data/multi_hotpotqa_eval_retrieval_n10_suponly_bridge_100.jsonl",
         fmt_train, fmt_retrieval_multi_eval)


def load_examples(path, n, offset=0):
    """Load n examples from a JSONL file, starting at offset."""
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            if i >= offset + n:
                break
            if line.strip():
                examples.append(json.loads(line))
    return examples


def generate_example_file(name, description, train_file, eval_file, train_fmt, eval_fmt):
    """Generate one example file with train + eval examples."""
    out_path = EXAMPLES_DIR / f"{name}.txt"

    with open(out_path, "w") as f:
        f.write(f"# {description}\n")
        f.write(f"#\n")
        f.write(f"# Train data: {train_file}\n")
        f.write(f"# Eval data:  {eval_file}\n")
        f.write(f"# Examples per section: {NUM_EXAMPLES}\n")
        f.write(f"#\n")
        f.write(f"# This file shows the exact prompt the model sees during training and evaluation.\n")
        f.write(f"# Training uses Axolotl with alpaca format. Eval uses the corresponding eval script\n")
        f.write(f"# with alpaca format + 0 few-shot demos (for trained/finetuned models).\n")
        f.write(f"\n\n")

        # ── Training examples ──
        if Path(train_file).exists():
            train_data = load_examples(train_file, NUM_EXAMPLES)
            f.write(f"{'#' * 80}\n")
            f.write(f"#  TRAINING EXAMPLES (from {train_file})\n")
            f.write(f"#  Formatted by: Axolotl alpaca format (instruction + input -> output)\n")
            f.write(f"{'#' * 80}\n\n")
            for i, ex in enumerate(train_data, 1):
                prompt, output = train_fmt(ex)
                write_section(f, "TRAIN", prompt, output, i)
        else:
            f.write(f"# TRAIN: {train_file} not found, skipping.\n\n")

        # ── Eval examples ──
        if Path(eval_file).exists():
            # If eval and train are the same file, offset to show different examples
            eval_offset = NUM_EXAMPLES if eval_file == train_file else 0
            eval_data = load_examples(eval_file, NUM_EXAMPLES, offset=eval_offset)
            f.write(f"{'#' * 80}\n")
            f.write(f"#  EVAL EXAMPLES (from {eval_file})\n")
            f.write(f"#  Formatted by: eval script with use_alpaca=True, shots=0\n")
            f.write(f"{'#' * 80}\n\n")
            for i, ex in enumerate(eval_data, 1):
                prompt, output = eval_fmt(ex)
                write_section(f, "EVAL", prompt, output, i)
        else:
            f.write(f"# EVAL: {eval_file} not found, skipping.\n\n")

    print(f"  {name} -> {out_path}")


def main():
    print(f"Generating example files in examples/ ({NUM_EXAMPLES} examples each, train + eval)...")
    for name, description, train_file, eval_file, train_fmt, eval_fmt in TASK_EXAMPLES:
        generate_example_file(name, description, train_file, eval_file, train_fmt, eval_fmt)
    print(f"\nDone. See examples/ directory.")


if __name__ == "__main__":
    main()
