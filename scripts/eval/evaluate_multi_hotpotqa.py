"""Evaluate multi-query HotpotQA using vLLM.

Generates eval data on-the-fly from HotpotQA validation set, or loads from
a pre-generated JSONL file. Computes per-query and aggregate metrics.

Usage:
    # Generate eval data on-the-fly (base model)
    python scripts/evaluate_multi_hotpotqa.py --num-queries 10 --max-test-samples 100

    # With LoRA finetuned model
    python scripts/evaluate_multi_hotpotqa.py --num-queries 10 --max-test-samples 100 \
        --lora-path outputs/multi-hotpotqa-std-lora

    # From pre-generated eval JSONL
    python scripts/evaluate_multi_hotpotqa.py --eval-data data/multi_hotpotqa_eval_n10_suponly_bridge_50.jsonl
"""

import argparse
import random
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports

from lib.io import load_jsonl, save_results, format_alpaca_prompt
from lib.data_format import build_prompt
from lib.metrics import (
    normalize_answer, exact_match, substring_match, token_f1,
    max_over_answers, aggregate,
)

try:
    from vllm import SamplingParams
    from lib.vllm_utils import add_vllm_args, load_model, run_inference
except ImportError:
    SamplingParams = None

# Must match training data format (generate_hotpotqa_data.py --num-queries N)
INSTRUCTION = (
    "Use the given documents to answer each of the following questions. "
    "Write a concise and short answer for each question, in order, as a comma-separated list.\n"
    "Write your answer in the following format:\nAnswers: [answer1], [answer2], ..."
)

PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"


def parse_multi_answers(output, num_queries):
    """Parse comma-separated answers from model output.

    Tries multiple strategies:
    1. Parse after "Answers:" prefix
    2. Parse after "Answer:" prefix
    3. Parse the raw output directly
    """
    text = output.strip()

    # Try extracting after </think> tag first
    think_match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if think_match:
        text = think_match.group(1).strip()

    # Try "Answers:" prefix
    for prefix in ["Answers:", "Answer:", "answers:", "answer:"]:
        idx = text.find(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):].strip()
            break

    # Take first line only
    text = text.split("\n")[0].strip()

    # Split on commas
    parts = [p.strip() for p in text.split(",")]

    # Pad or truncate to num_queries
    while len(parts) < num_queries:
        parts.append("")
    parts = parts[:num_queries]

    return parts


def compute_per_query_metrics(predicted_answers, gold_answers_list):
    """Compute metrics for each query position.

    Args:
        predicted_answers: List of N predicted answer strings.
        gold_answers_list: List of N gold answer strings.

    Returns:
        List of per-query metric dicts and an aggregate dict.
    """
    per_query = []
    for pred, gold in zip(predicted_answers, gold_answers_list):
        em = exact_match(pred, gold)
        sub_em = substring_match(pred, gold)
        f1 = token_f1(pred, gold)
        per_query.append({
            "exact_match": float(em),
            "substring_exact_match": float(sub_em),
            "f1": f1,
        })

    # Aggregate across queries in this example
    n = len(per_query)
    agg = {
        "exact_match": sum(q["exact_match"] for q in per_query) / n,
        "substring_exact_match": sum(q["substring_exact_match"] for q in per_query) / n,
        "f1": sum(q["f1"] for q in per_query) / n,
        # All-correct: 1 only if every query is exactly correct
        "all_correct": float(all(q["exact_match"] == 1.0 for q in per_query)),
    }
    return per_query, agg


def generate_eval_data(args, rng):
    """Generate eval examples on-the-fly from HotpotQA validation set."""
    from datasets import load_dataset as hf_load_dataset

    print(f"\nLoading hotpotqa/hotpot_qa distractor (validation)...")
    ds = hf_load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    print(f"  Loaded {len(ds)} examples")

    if args.question_type != "all":
        ds = ds.filter(lambda ex: ex["type"] == args.question_type)
        print(f"  After filtering to type={args.question_type}: {len(ds)} examples")

    indices = list(range(len(ds)))
    rng.shuffle(indices)

    source_needed = args.max_test_samples * args.num_queries
    if len(ds) < source_needed:
        args.max_test_samples = len(ds) // args.num_queries
        print(f"  Adjusted to {args.max_test_samples} multi-query examples")

    selected_count = args.max_test_samples * args.num_queries
    selected = indices[:selected_count]

    # Build distractor pool
    distractor_pool = []
    pool_indices = indices[selected_count:selected_count + selected_count]
    if not pool_indices:
        pool_indices = indices
    for idx in pool_indices:
        ex = ds[idx]
        sup_titles = set(ex["supporting_facts"]["title"])
        titles = ex["context"]["title"]
        sentences_list = ex["context"]["sentences"]
        for title, sentences in zip(titles, sentences_list):
            if title not in sup_titles:
                text = " ".join(s.strip() for s in sentences)
                distractor_pool.append({"title": title, "text": text})
    rng.shuffle(distractor_pool)

    examples = []
    for i in range(0, selected_count, args.num_queries):
        group_indices = selected[i:i + args.num_queries]
        group = [ds[idx] for idx in group_indices]

        all_supporting = []
        all_local_distractors = []
        questions = []
        answers = []
        seen_titles = set()

        for ex in group:
            titles = ex["context"]["title"]
            sentences_list = ex["context"]["sentences"]
            sup_titles = set(ex["supporting_facts"]["title"])

            for title, sentences in zip(titles, sentences_list):
                text = " ".join(s.strip() for s in sentences)
                doc = {"title": title, "text": text}
                if title in sup_titles and title not in seen_titles:
                    all_supporting.append(doc)
                    seen_titles.add(title)
                elif title not in sup_titles and title not in seen_titles:
                    all_local_distractors.append(doc)
                    seen_titles.add(title)

            questions.append(ex["question"])
            answers.append(ex["answer"])

        # Add distractors if total_docs specified
        num_supporting = len(all_supporting)
        if args.total_docs > 0:
            num_distractors_needed = max(0, args.total_docs - num_supporting)
        else:
            num_distractors_needed = 0

        distractors = list(all_local_distractors)
        rng.shuffle(distractors)
        if len(distractors) < num_distractors_needed:
            pool_avail = [d for d in distractor_pool if d["title"] not in seen_titles]
            extra = rng.sample(pool_avail,
                               min(num_distractors_needed - len(distractors), len(pool_avail)))
            distractors.extend(extra)
        while len(distractors) < num_distractors_needed:
            distractors.append(rng.choice(distractor_pool))
        distractors = distractors[:num_distractors_needed]

        all_paragraphs = all_supporting + distractors
        rng.shuffle(all_paragraphs)

        no_titles = getattr(args, "no_titles", False)
        if no_titles:
            formatted_docs = [PASSAGE_TEMPLATE_NO_TITLE.format(text=d["text"]) for d in all_paragraphs]
        else:
            formatted_docs = [PASSAGE_TEMPLATE.format(**d) for d in all_paragraphs]
        context = "\n\n".join(formatted_docs)

        questions_block = "\n".join(
            f"Question {j+1}: {q}" for j, q in enumerate(questions)
        )

        examples.append({
            "input": f"{context}\n\n{questions_block}",
            "questions": questions,
            "answers": answers,
            "num_docs": len(all_paragraphs),
        })

    return examples


def load_eval_from_jsonl(path, max_samples):
    """Load pre-generated eval data from unified-format JSONL."""
    raw = load_jsonl(path)
    if max_samples and len(raw) > max_samples:
        random.seed(42)
        raw = random.sample(raw, max_samples)

    examples = []
    for ex in raw:
        examples.append({
            "unified": ex,
            "questions": ex["queries"],
            "answers": ex["answers"],
            "num_docs": len(ex["documents"]),
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Multi-query HotpotQA evaluation")
    add_vllm_args(parser)
    parser.add_argument("--num-queries", type=int, default=10,
                        help="Number of queries per example")
    parser.add_argument("--total-docs", type=int, default=0,
                        help="Total documents (0 = supporting only)")
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--question-type", type=str, default="bridge",
                        choices=["bridge", "comparison", "all"])
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"])
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--use-alpaca", action="store_true",
                        help="Force alpaca prompt format (for full FT models without --lora-path)")
    parser.add_argument("--eval-data", type=str, default="",
                        help="Pre-generated eval JSONL (skip on-the-fly generation)")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.set_defaults(max_tokens=100, output_file="outputs/multi_hotpotqa_results.json")
    args = parser.parse_args()

    rng = random.Random(42)

    # Load or generate eval data
    if args.eval_data:
        print(f"Loading eval data from {args.eval_data}")
        eval_examples = load_eval_from_jsonl(args.eval_data, args.max_test_samples)
        # Infer num_queries from data
        if eval_examples:
            args.num_queries = len(eval_examples[0]["questions"])
    else:
        eval_examples = generate_eval_data(args, rng)

    print(f"  {len(eval_examples)} examples, {args.num_queries} queries each")

    # Build prompts
    use_alpaca = bool(args.lora_path) or args.use_alpaca
    fmt_label = "alpaca" if use_alpaca else "plain"
    print(f"Prompt format: {fmt_label}")

    prompts = []
    for ex in eval_examples:
        if "unified" in ex:
            # Unified format: use build_prompt
            prompt, _ = build_prompt(
                ex["unified"], task="qa", query_position=args.query_position,
                use_alpaca=use_alpaca,
            )
        else:
            # On-the-fly generated: already has "input" field
            input_text = ex["input"]
            if args.query_position == "before":
                parts = input_text.rsplit("\n\n", 1)
                if len(parts) == 2:
                    input_text = f"{parts[1]}\n\n{parts[0]}"
            elif args.query_position == "both":
                parts = input_text.rsplit("\n\n", 1)
                if len(parts) == 2:
                    input_text = f"{parts[1]}\n\n{parts[0]}\n\n{parts[1]}"
            if use_alpaca:
                prompt = format_alpaca_prompt(INSTRUCTION, input_text)
            else:
                prompt = f"{INSTRUCTION}\n\n{input_text}\nAnswers:"

        prompts.append(prompt)

    # Load model and run inference
    llm, lora_request = load_model(args)

    if args.enable_thinking:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=max(args.max_tokens, 512))
        prompts = [p + "<think>\n" for p in prompts]
    else:
        sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["\n\n"])

    print(f"Running inference on {len(prompts)} examples...")
    responses = run_inference(llm, prompts, sampling_params, lora_request)

    # Evaluate
    all_per_query = []
    all_agg = []
    details = []
    for ex, resp in zip(eval_examples, responses):
        predicted = parse_multi_answers(resp, len(ex["answers"]))
        per_query, agg_metrics = compute_per_query_metrics(predicted, ex["answers"])
        all_per_query.append(per_query)
        all_agg.append(agg_metrics)
        details.append({
            "questions": ex["questions"],
            "gold_answers": ex["answers"],
            "predicted_answers": predicted,
            "raw_output": resp.strip()[:500],
            "num_docs": ex["num_docs"],
            **agg_metrics,
        })

    # Compute overall metrics
    n = len(all_agg)
    overall = {
        "exact_match": sum(a["exact_match"] for a in all_agg) / n,
        "substring_exact_match": sum(a["substring_exact_match"] for a in all_agg) / n,
        "f1": sum(a["f1"] for a in all_agg) / n,
        "all_correct": sum(a["all_correct"] for a in all_agg) / n,
    }

    # Per-position metrics (how well does the model do on query 1, 2, ..., N?)
    per_position = []
    for pos in range(args.num_queries):
        pos_metrics = {
            "exact_match": sum(pq[pos]["exact_match"] for pq in all_per_query) / n,
            "substring_exact_match": sum(pq[pos]["substring_exact_match"] for pq in all_per_query) / n,
            "f1": sum(pq[pos]["f1"] for pq in all_per_query) / n,
        }
        per_position.append(pos_metrics)

    # Save results
    results = {
        "args": vars(args),
        "overall": overall,
        "per_position": per_position,
        "details": details,
    }
    save_results(args.output_file, results)

    # Print summary
    print(f"\n{'='*60}\nRESULTS ({len(eval_examples)} examples, {args.num_queries} queries each)\n{'='*60}")
    print(f"  Overall EM:      {overall['exact_match']:.1%}")
    print(f"  Overall SubEM:   {overall['substring_exact_match']:.1%}")
    print(f"  Overall F1:      {overall['f1']:.1%}")
    print(f"  All-correct:     {overall['all_correct']:.1%}")

    print(f"\nPer-position EM:")
    for pos, m in enumerate(per_position):
        print(f"  Q{pos+1}: EM={m['exact_match']:.1%}  SubEM={m['substring_exact_match']:.1%}  F1={m['f1']:.1%}")

    # Show a few samples
    print(f"\n--- Samples ---")
    for d in details[:3]:
        print(f"  Questions: {d['questions'][:3]}...")
        print(f"  Gold:      {d['gold_answers'][:3]}...")
        print(f"  Predicted: {d['predicted_answers'][:3]}...")
        print(f"  EM={d['exact_match']:.1%}  Raw: {d['raw_output'][:100]}")
        print()

    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
