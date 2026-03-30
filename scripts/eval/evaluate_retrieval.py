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

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports

from lib.io import load_jsonl, save_results
from lib.data_format import build_prompt, is_multi_query
from lib.metrics import (
    parse_doc_ids, retrieval_exact_match, retrieval_recall,
    retrieval_precision, retrieval_f1, aggregate,
)
from lib.prompts import (
    RETRIEVAL_INSTRUCTION_SINGLE,
    RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY,
)

try:
    from vllm import SamplingParams
    from lib.vllm_utils import add_vllm_args, load_model, run_inference
except ImportError:
    SamplingParams = None

# Map task names to their retrieval instructions (from lib/prompts.py)
TASK_INSTRUCTIONS = {
    "nq": RETRIEVAL_INSTRUCTION_SINGLE,
    "hotpotqa": RETRIEVAL_INSTRUCTION_MULTI_DOC,
    "multi-hotpotqa": RETRIEVAL_INSTRUCTION_MULTI_QUERY,
}


def detect_task(examples):
    """Auto-detect task type from data format.

    Inspects unified-format fields to determine the task:
    - Multiple queries → multi-hotpotqa
    - Multiple gold docs (flat list) → hotpotqa
    - Single gold doc → nq
    """
    ex = examples[0]
    if len(ex["queries"]) > 1:
        return "multi-hotpotqa"
    gold = ex.get("gold_doc_indices", [])
    if isinstance(gold, list) and gold and isinstance(gold[0], list):
        return "multi-hotpotqa"
    if isinstance(gold, list) and len(gold) > 1:
        return "hotpotqa"
    return "nq"


def extract_after_thinking(text):
    """Extract answer text after </think> tag, if present."""
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_retrieval_output(output):
    """Parse document IDs from model output, trying multiple strategies.

    The model may output IDs in different formats depending on whether
    thinking mode is enabled and how it phrases the response:
      1. Plain: "[3], [7]"
      2. With prefix: "Relevant Documents: [3], [7]"
      3. After thinking: "<think>reasoning...</think>[3], [7]"
    We try stripping each layer and parse whatever remains.
    """
    text = output.strip()

    # Strategy 1: strip thinking block if present
    after_think = extract_after_thinking(text)
    if after_think:
        text = after_think

    # Strategy 2: strip "Relevant Document(s):" prefix
    for prefix in ["Relevant Documents:", "Relevant Document:", "relevant documents:",
                   "relevant document:"]:
        idx = text.find(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):].strip()
            break

    # Only use the first line (model sometimes generates extra text)
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

    # --- Build prompts and parse gold labels from unified-format JSONL ---
    prompts = []
    gold_data = []
    for ex in raw:
        prompt, _ = build_prompt(
            ex, task="retrieval", query_position=args.query_position,
            before_dummy=args.before_dummy, after_dummy=args.after_dummy,
            use_alpaca=use_alpaca,
        )
        prompts.append(prompt)

        # Extract gold IDs directly from structured data (convert to 1-indexed)
        gold = ex["gold_doc_indices"]
        if task == "multi-hotpotqa":
            per_query_gold = [set(g + 1 for g in gids) for gids in gold]
            gold_data.append({"per_query_gold": per_query_gold,
                              "num_queries": len(ex["queries"])})
        else:
            gids = gold[0] if gold and isinstance(gold[0], list) else gold
            gold_data.append({"gold_ids": set(g + 1 for g in gids)})

    # Load model and run inference
    llm, lora_request = load_model(args)

    if args.enable_thinking:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max(args.max_tokens, 512))
        prompts = [p + "<think>\n" for p in prompts]
    else:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["\n"])

    print(f"Running inference on {len(prompts)} examples...")
    responses = run_inference(llm, prompts, sampling_params, lora_request)

    # --- Evaluate predictions against gold labels ---
    if task == "multi-hotpotqa":
        # Multi-query eval: compute metrics per-query within each example,
        # then average across queries (per-example) and across examples (overall)
        all_per_query = []  # per_query_metrics for each example
        all_agg = []        # per-example aggregated metrics
        details = []

        for ex, resp, gold in zip(raw, responses, gold_data):
            num_q = gold["num_queries"]
            predicted_per_query = parse_multi_query_output(resp, num_q)
            per_query_gold = gold["per_query_gold"]

            while len(per_query_gold) < num_q:
                per_query_gold.append(set())

            # Score each query independently
            per_query_metrics = []
            for pred_ids, gold_ids in zip(predicted_per_query, per_query_gold):
                m = compute_retrieval_metrics(pred_ids, gold_ids)
                per_query_metrics.append(m)

            all_per_query.append(per_query_metrics)
            # Average metrics across queries within this example
            n = len(per_query_metrics)
            agg = {
                "exact_match": sum(m["exact_match"] for m in per_query_metrics) / n,
                "recall": sum(m["recall"] for m in per_query_metrics) / n,
                "precision": sum(m["precision"] for m in per_query_metrics) / n,
                "f1": sum(m["f1"] for m in per_query_metrics) / n,
                # all_correct: 1.0 only if ALL queries in this example have perfect EM
                "all_correct": float(all(m["exact_match"] == 1.0 for m in per_query_metrics)),
            }
            all_agg.append(agg)
            details.append({
                "gold_output": ex["output"],
                "raw_output": resp.strip()[:500],
                **agg,
            })

        # Average across all examples
        n_total = len(all_agg)
        overall = {k: sum(a[k] for a in all_agg) / n_total
                   for k in ["exact_match", "recall", "precision", "f1", "all_correct"]}

        # Per-position metrics: how well does the model do on Q1 vs Q2 vs Q3 etc.
        # This reveals whether later queries in the sequence are harder.
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
