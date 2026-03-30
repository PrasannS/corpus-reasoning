"""Evaluate retrieval tasks using vLLM.

Loads pre-generated retrieval JSONL data (produced by generate_*_data.py --retrieval
with --split validation). This ensures eval data uses the same process as training data.

Supports NQ (single gold doc), HotpotQA (2 gold docs), and multi-HotpotQA
(per-query gold docs).

Usage:
    # NQ retrieval eval
    python scripts/evaluate_retrieval.py --eval-data data/nq_train_k20_random_500_retrieval.jsonl \
        --lora-path outputs/nq-retrieval-lora

    # HotpotQA retrieval eval
    python scripts/evaluate_retrieval.py --eval-data data/hotpotqa_eval_k20_shuffled_retrieval_bridge_500.jsonl \
        --lora-path outputs/hotpotqa-retrieval-lora

    # Multi-HotpotQA retrieval eval
    python scripts/evaluate_retrieval.py --eval-data data/multi_hotpotqa_eval_retrieval_n10_suponly_bridge_50.jsonl \
        --lora-path outputs/multi-hotpotqa-retrieval-lora --task multi-hotpotqa
"""

import argparse
import random
import re

from lib.io import load_jsonl, save_results, format_alpaca_prompt, insert_dummy_tokens
from lib.metrics import (
    parse_doc_ids, retrieval_exact_match, retrieval_recall,
    retrieval_precision, retrieval_f1, aggregate,
)

try:
    from vllm import SamplingParams
    from lib.vllm_utils import add_vllm_args, load_model, run_inference
except ImportError:
    SamplingParams = None

# Must match training data instructions (from generate_*_data.py --retrieval)
INSTRUCTION_SINGLE = (
    "Use the given documents to identify which document is most relevant to "
    "answering the question.\n"
    "Write your answer in the following format:\nRelevant Document: [id]"
)
INSTRUCTION_MULTI_DOC = (
    "Use the given documents to identify which documents are relevant to "
    "answering the question. List all relevant document IDs.\n"
    "Write your answer in the following format:\nRelevant Documents: [id1], [id2]"
)
INSTRUCTION_MULTI_QUERY = (
    "Use the given documents to identify which documents are relevant to "
    "answering each of the following questions. For each question, list the "
    "relevant document IDs.\n"
    "Write your answer in the following format:\n"
    "Relevant Documents: Q1: [id1], [id2]; Q2: [id3], [id4]; ..."
)

# Map task names to instructions
TASK_INSTRUCTIONS = {
    "nq": INSTRUCTION_SINGLE,
    "hotpotqa": INSTRUCTION_MULTI_DOC,
    "multi-hotpotqa": INSTRUCTION_MULTI_QUERY,
}


def detect_task(examples):
    """Auto-detect task type from data format."""
    ex = examples[0]
    output = ex["output"]
    if "Q1:" in output or "Q2:" in output:
        return "multi-hotpotqa"
    # Count gold IDs: 1 = NQ, 2+ = HotpotQA
    ids = parse_doc_ids(output)
    return "nq" if len(ids) <= 1 else "hotpotqa"


def extract_after_thinking(text):
    """Extract answer text after </think> tag, if present."""
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_retrieval_output(output):
    """Parse document IDs from model output, trying multiple strategies."""
    text = output.strip()

    # Try extracting after </think> tag
    after_think = extract_after_thinking(text)
    if after_think:
        text = after_think

    # Try prefix removal
    for prefix in ["Relevant Documents:", "Relevant Document:", "relevant documents:",
                   "relevant document:"]:
        idx = text.find(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):].strip()
            break

    # Take first line only
    text = text.split("\n")[0].strip()

    return text, parse_doc_ids(text)


def parse_multi_query_output(output, num_queries):
    """Parse per-query document IDs from multi-query model output.

    Expected format: "Q1: [3], [7]; Q2: [1], [5]; ..."
    Returns list of sets of ints.
    """
    text = output.strip()

    after_think = extract_after_thinking(text)
    if after_think:
        text = after_think

    for prefix in ["Relevant Documents:", "Relevant Document:"]:
        idx = text.find(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):].strip()
            break

    text = text.split("\n")[0].strip()

    # Split on semicolons for per-query parts
    parts = [p.strip() for p in text.split(";")]

    per_query_ids = []
    for part in parts:
        # Remove Q1:, Q2: prefix if present
        part = re.sub(r'^Q\d+:\s*', '', part)
        ids = parse_doc_ids(part)
        per_query_ids.append(ids)

    # Pad or truncate
    while len(per_query_ids) < num_queries:
        per_query_ids.append(set())
    per_query_ids = per_query_ids[:num_queries]

    return per_query_ids


def compute_retrieval_metrics(predicted_ids, gold_ids):
    """Compute retrieval metrics for a single example."""
    return {
        "exact_match": float(retrieval_exact_match(predicted_ids, gold_ids)),
        "recall": retrieval_recall(predicted_ids, gold_ids),
        "precision": retrieval_precision(predicted_ids, gold_ids),
        "f1": retrieval_f1(predicted_ids, gold_ids),
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieval task evaluation")
    add_vllm_args(parser)
    parser.add_argument("--eval-data", type=str, required=True,
                        help="Pre-generated retrieval JSONL file (from generate_*_data.py --retrieval)")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "nq", "hotpotqa", "multi-hotpotqa"],
                        help="Task type (auto-detected from data if not specified)")
    parser.add_argument("--max-test-samples", type=int, default=None,
                        help="Limit number of eval examples")
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"])
    parser.add_argument("--use-alpaca", action="store_true",
                        help="Force alpaca prompt format (for full FT models without --lora-path)")
    parser.add_argument("--before-dummy", type=int, default=0,
                        help="Number of dummy token repetitions to insert before documents")
    parser.add_argument("--after-dummy", type=int, default=0,
                        help="Number of dummy token repetitions to insert after documents")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable thinking mode: append <think> to prompt")
    parser.set_defaults(max_tokens=50, output_file="outputs/retrieval_results.json")
    args = parser.parse_args()

    # Load eval data
    print(f"Loading eval data from {args.eval_data}")
    raw = load_jsonl(args.eval_data)
    if args.max_test_samples and len(raw) > args.max_test_samples:
        random.seed(42)
        raw = random.sample(raw, args.max_test_samples)
    print(f"  {len(raw)} examples")

    # Detect task
    task = args.task if args.task != "auto" else detect_task(raw)
    print(f"  Task: {task}")

    instruction = TASK_INSTRUCTIONS[task]
    use_alpaca = bool(args.lora_path) or args.use_alpaca
    fmt_label = "alpaca" if use_alpaca else "plain"
    print(f"  Prompt format: {fmt_label}")

    # Build prompts
    prompts = []
    gold_data = []
    for ex in raw:
        input_text = ex["input"]

        # Apply query position transformation
        if task == "multi-hotpotqa":
            # Last block is questions, rest is context
            sep_point = input_text.rfind("\nQuestion 1:")
            if sep_point >= 0:
                context_part = input_text[:sep_point]
                questions_part = input_text[sep_point:]
            else:
                context_part = input_text
                questions_part = ""

            if args.query_position == "before":
                input_text = f"{questions_part.strip()}\n\n{context_part}"
            elif args.query_position == "both":
                input_text = f"{questions_part.strip()}\n\n{context_part}\n\n{questions_part.strip()}"
        else:
            # Single question: last part after \n\nQuestion:
            parts = input_text.rsplit("\n\nQuestion:", 1)
            if len(parts) == 2:
                context_part = parts[0]
                question_part = f"Question:{parts[1]}"
                if args.query_position == "before":
                    input_text = f"{question_part}\n\n{context_part}"
                elif args.query_position == "both":
                    input_text = f"{question_part}\n\n{context_part}\n\n{question_part}"

        if args.before_dummy > 0 or args.after_dummy > 0:
            input_text = insert_dummy_tokens(input_text, args.before_dummy, args.after_dummy)

        if use_alpaca:
            prompt = format_alpaca_prompt(instruction, input_text)
        else:
            prompt = f"{instruction}\n\n{input_text}\n"

        prompts.append(prompt)

        # Parse gold IDs from output
        gold_output = ex["output"]
        if task == "multi-hotpotqa":
            # Parse per-query gold IDs
            num_queries = len(re.findall(r"Q\d+:", gold_output))
            per_query_gold = []
            for part in gold_output.split(";"):
                part = re.sub(r'^Q\d+:\s*', '', part.strip())
                per_query_gold.append(parse_doc_ids(part))
            gold_data.append({"per_query_gold": per_query_gold, "num_queries": num_queries})
        else:
            gold_data.append({"gold_ids": parse_doc_ids(gold_output)})

    # Load model and run inference
    llm, lora_request = load_model(args)

    if args.enable_thinking:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max(args.max_tokens, 512))
        prompts = [p + "<think>\n" for p in prompts]
    else:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["\n"])

    print(f"Running inference on {len(prompts)} examples...")
    responses = run_inference(llm, prompts, sampling_params, lora_request)

    # Evaluate
    if task == "multi-hotpotqa":
        all_per_query = []
        all_agg = []
        details = []

        for ex, resp, gold in zip(raw, responses, gold_data):
            num_q = gold["num_queries"]
            predicted_per_query = parse_multi_query_output(resp, num_q)
            per_query_gold = gold["per_query_gold"]

            # Pad gold if needed
            while len(per_query_gold) < num_q:
                per_query_gold.append(set())

            per_query_metrics = []
            for pred_ids, gold_ids in zip(predicted_per_query, per_query_gold):
                m = compute_retrieval_metrics(pred_ids, gold_ids)
                per_query_metrics.append(m)

            all_per_query.append(per_query_metrics)
            n = len(per_query_metrics)
            agg = {
                "exact_match": sum(m["exact_match"] for m in per_query_metrics) / n,
                "recall": sum(m["recall"] for m in per_query_metrics) / n,
                "precision": sum(m["precision"] for m in per_query_metrics) / n,
                "f1": sum(m["f1"] for m in per_query_metrics) / n,
                "all_correct": float(all(m["exact_match"] == 1.0 for m in per_query_metrics)),
            }
            all_agg.append(agg)
            details.append({
                "gold_output": ex["output"],
                "raw_output": resp.strip()[:500],
                **agg,
            })

        # Overall metrics
        n_total = len(all_agg)
        overall = {k: sum(a[k] for a in all_agg) / n_total
                   for k in ["exact_match", "recall", "precision", "f1", "all_correct"]}

        # Per-position metrics
        num_q = gold_data[0]["num_queries"]
        per_position = []
        for pos in range(num_q):
            pos_metrics = {}
            for k in ["exact_match", "recall", "precision", "f1"]:
                pos_metrics[k] = sum(pq[pos][k] for pq in all_per_query) / n_total
            per_position.append(pos_metrics)

        results = {
            "args": vars(args),
            "task": task,
            "overall": overall,
            "per_position": per_position,
            "details": details,
        }

        print(f"\n{'='*60}")
        print(f"RESULTS ({len(raw)} examples, {num_q} queries each)")
        print(f"{'='*60}")
        print(f"  Overall EM:        {overall['exact_match']:.1%}")
        print(f"  Overall Recall:    {overall['recall']:.1%}")
        print(f"  Overall Precision: {overall['precision']:.1%}")
        print(f"  Overall F1:        {overall['f1']:.1%}")
        print(f"  All-correct:       {overall['all_correct']:.1%}")

        print(f"\nPer-position:")
        for pos, m in enumerate(per_position):
            print(f"  Q{pos+1}: EM={m['exact_match']:.1%}  R={m['recall']:.1%}  "
                  f"P={m['precision']:.1%}  F1={m['f1']:.1%}")

    else:
        # NQ or HotpotQA (single query)
        metric_keys = ["exact_match", "recall", "precision", "f1"]
        results_list = []
        details = []

        for ex, resp, gold in zip(raw, responses, gold_data):
            parsed_text, predicted_ids = parse_retrieval_output(resp)
            gold_ids = gold["gold_ids"]
            m = compute_retrieval_metrics(predicted_ids, gold_ids)
            results_list.append(m)
            details.append({
                "gold_ids": sorted(gold_ids),
                "predicted_ids": sorted(predicted_ids),
                "raw_output": resp.strip()[:200],
                **m,
            })

        overall = aggregate(results_list, metric_keys)
        results = {
            "args": vars(args),
            "task": task,
            "overall": overall,
            "details": details,
        }

        print(f"\n{'='*60}")
        print(f"RESULTS ({len(raw)} examples, task={task})")
        print(f"{'='*60}")
        print(f"  EM:        {overall['exact_match']:.1%}")
        print(f"  Recall:    {overall['recall']:.1%}")
        print(f"  Precision: {overall['precision']:.1%}")
        print(f"  F1:        {overall['f1']:.1%}")

    # Show samples
    print(f"\n--- Samples ---")
    for d in details[:5]:
        print(f"  {d}")
        print()

    save_results(args.output_file, results)
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
