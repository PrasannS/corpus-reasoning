"""
Evaluate models on HELMET RAG tasks using vLLM.

Supports the four KILT-based RAG benchmarks from HELMET:
  - Natural Questions (NQ)
  - TriviaQA
  - HotpotQA
  - PopQA

Replicates HELMET's prompt format and metrics (exact match, substring EM, F1)
without depending on the HELMET codebase.

Usage:
    python scripts/evaluate_helmet_rag.py --base-model NousResearch/Llama-3.2-1B
    python scripts/evaluate_helmet_rag.py --base-model NousResearch/Llama-3.2-1B --lora-path outputs/niah-lora
    python scripts/evaluate_helmet_rag.py --datasets nq,triviaqa --max-test-samples 50
"""

import argparse
import hashlib
import json
import os
import re
import string
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


# ---------------------------------------------------------------------------
# HELMET-compatible metrics (from HELMET utils.py)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        return "".join(ch for ch in text if ch not in string.punctuation)
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def substring_exact_match_score(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]):
    """Compute the max metric score over all ground truth answers."""
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    elif ground_truths and isinstance(ground_truths[0], list):
        ground_truths = [a for sublist in ground_truths for a in sublist]
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def parse_output(output: str, prefix: str = "Answer:") -> str | None:
    """Parse the answer from model output, matching HELMET's parse_output."""
    def lstrip_string(s, sub):
        return re.sub(f"^{re.escape(sub)}", "", s, flags=re.IGNORECASE)
    patterns = [
        re.compile(f"(?:{prefix})(.*?)(?:\\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*?)(?:\n|$)"),
    ]
    for pat in patterns:
        match = pat.search(output)
        if match is not None:
            return lstrip_string(match[1].strip(), prefix).strip()
    return None


def compute_metrics(prediction: str, answers: list[str]) -> dict:
    """Compute HELMET metrics for a single prediction."""
    em = max_over_ground_truths(exact_match_score, prediction, answers)
    sub_em = max_over_ground_truths(substring_exact_match_score, prediction, answers)
    f1 = max_over_ground_truths(f1_score, prediction, answers)

    # Also try parsed output and take max
    parsed = parse_output(prediction)
    if parsed is not None:
        em = max(em, max_over_ground_truths(exact_match_score, parsed, answers))
        sub_em = max(sub_em, max_over_ground_truths(substring_exact_match_score, parsed, answers))
        f1 = max(f1, max_over_ground_truths(f1_score, parsed, answers))

    return {"exact_match": float(em), "substring_exact_match": float(sub_em), "f1": f1}


# ---------------------------------------------------------------------------
# Data loading & prompt formatting (replicating HELMET data.py)
# ---------------------------------------------------------------------------

DATASET_CONFIG = {
    "nq": {
        "test_file": "data/data/kilt/nq-dev-multikilt_1000_k{num_docs}_dep6.jsonl",
        "demo_file": "data/data/kilt/nq-train-multikilt_1000_k3_dep6.jsonl",
    },
    "triviaqa": {
        "test_file": "data/data/kilt/triviaqa-dev-multikilt_1000_k{num_docs}_dep6.jsonl",
        "demo_file": "data/data/kilt/triviaqa-train-multikilt_1000_k3_dep6.jsonl",
    },
    "hotpotqa": {
        "test_file": "data/data/kilt/hotpotqa-dev-multikilt_1000_k{num_docs}_dep3.jsonl",
        "demo_file": "data/data/kilt/hotpotqa-train-multikilt_1000_k3_dep3.jsonl",
    },
    "popqa": {
        "test_file": "data/data/kilt/popqa_test_1000_k{num_docs}_dep6.jsonl",
        "demo_file": "data/data/kilt/popqa_test_1000_k3_dep6.jsonl",
    },
}

PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
DEMO_TEMPLATE = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"
USER_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}{context}\n\nQuestion: {question}"
)
SYSTEM_TEMPLATE = "Answer:"


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_demos(demo_data: list[dict], sample: dict, shots: int, key: str = "question") -> str:
    """Build few-shot demo text, replicating HELMET's deterministic seeding."""
    if shots == 0:
        return ""
    # Deterministic shuffle based on sample key
    h = int(hashlib.sha256(str(sample[key]).encode("utf-8")).hexdigest(), 16) % 2**31
    import random
    rng = random.Random(h)
    # Filter out the current sample and deduplicate
    demos = [d for d in demo_data if d.get(key) != sample.get(key)]
    rng.shuffle(demos)
    seen = set()
    unique_demos = []
    for d in demos:
        k = d.get(key, "")
        if k not in seen:
            seen.add(k)
            unique_demos.append(d)
        if len(unique_demos) >= shots:
            break
    # Format demos
    demo_texts = []
    for d in unique_demos:
        docs = "\n\n".join(PASSAGE_TEMPLATE.format(**c) for c in d.get("ctxs", []))
        answer = d["answers"][0] if isinstance(d["answers"], list) else d["answers"]
        demo_texts.append(DEMO_TEMPLATE.format(documents=docs, question=d["question"], answer=answer))
    return "\n\n".join(demo_texts) + "\n\n" if demo_texts else ""


def load_dataset_for_eval(
    dataset_name: str,
    max_test_samples: int | None = None,
    shots: int = 2,
    num_docs: int = 1000,
) -> list[dict]:
    """Load and format a HELMET RAG dataset.

    Returns list of dicts with 'prompt', 'answers', 'question' keys.
    """
    config = DATASET_CONFIG[dataset_name]
    test_file = config["test_file"].format(num_docs=num_docs)
    demo_file = config["demo_file"]

    # Try fallback num_docs values if file doesn't exist
    if not Path(test_file).exists():
        for fallback in [500, 105, 100, 50, 10, 3]:
            alt = config["test_file"].format(num_docs=fallback)
            if Path(alt).exists():
                print(f"  File {test_file} not found, falling back to {alt}")
                test_file = alt
                break
        else:
            raise FileNotFoundError(f"No test file found for {dataset_name}. Tried num_docs={num_docs} and fallbacks.")

    print(f"  Loading test data from: {test_file}")
    test_data = load_jsonl(test_file)

    demo_data = []
    if shots > 0 and Path(demo_file).exists():
        print(f"  Loading demo data from: {demo_file}")
        demo_data = load_jsonl(demo_file)

    # Subsample
    if max_test_samples and len(test_data) > max_test_samples:
        # Deduplicate by question, then sample
        key = "id" if "id" in test_data[0] else "question"
        seen = set()
        unique = []
        for d in test_data:
            k = d.get(key, d["question"])
            if k not in seen:
                seen.add(k)
                unique.append(d)
        import random
        random.seed(42)
        test_data = random.sample(unique, min(max_test_samples, len(unique)))

    # Build prompts
    examples = []
    for sample in test_data:
        demos = build_demos(demo_data, sample, shots)
        context = "\n\n".join(
            PASSAGE_TEMPLATE.format(**c) for c in sample.get("ctxs", [])
        )
        prompt = USER_TEMPLATE.format(demos=demos, context=context, question=sample["question"])
        prompt = prompt + "\n" + SYSTEM_TEMPLATE

        examples.append({
            "prompt": prompt,
            "answers": sample["answers"],
            "question": sample["question"],
        })

    return examples


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_eval(
    llm: LLM,
    examples: list[dict],
    sampling_params: SamplingParams,
    lora_request: LoRARequest | None = None,
) -> list[dict]:
    """Run inference and compute metrics."""
    prompts = [ex["prompt"] for ex in examples]
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    responses = [o.outputs[0].text for o in outputs]

    results = []
    for ex, resp in zip(examples, responses):
        mets = compute_metrics(resp, ex["answers"])
        results.append({
            "question": ex["question"],
            "answers": ex["answers"],
            "prediction": resp.strip()[:300],
            **mets,
        })
    return results


def aggregate_metrics(results: list[dict]) -> dict:
    """Average metrics across all results."""
    if not results:
        return {}
    keys = ["exact_match", "substring_exact_match", "f1"]
    return {k: sum(r[k] for r in results) / len(results) for k in keys}


def main():
    parser = argparse.ArgumentParser(description="Evaluate on HELMET RAG tasks with vLLM")
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="", help="Path to LoRA adapter (empty to skip)")
    parser.add_argument("--datasets", type=str, default="nq,triviaqa,hotpotqa,popqa",
                        help="Comma-separated dataset names")
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--shots", type=int, default=2, help="Number of few-shot demos")
    parser.add_argument("--num-docs", type=int, default=1000,
                        help="Number of retrieved documents per question")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max model context length")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max new tokens")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--output-file", type=str, default="outputs/helmet_rag_results.json")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",")]

    # Load model
    enable_lora = bool(args.lora_path)
    print(f"Loading model: {args.base_model} (enable_lora={enable_lora})")
    llm = LLM(
        model=args.base_model,
        enable_lora=enable_lora,
        max_lora_rank=64 if enable_lora else None,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.5,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        stop=["\n"],  # HELMET uses stop_new_line: true
    )

    lora_request = None
    if args.lora_path:
        lora_request = LoRARequest("lora", 1, str(Path(args.lora_path).resolve()))

    all_results = {}
    summary = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*60}")

        if dataset_name not in DATASET_CONFIG:
            print(f"  Unknown dataset: {dataset_name}, skipping")
            continue

        try:
            examples = load_dataset_for_eval(
                dataset_name,
                max_test_samples=args.max_test_samples,
                shots=args.shots,
                num_docs=args.num_docs,
            )
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        print(f"  Loaded {len(examples)} examples")
        # Show prompt length stats
        prompt_lens = [len(ex["prompt"]) for ex in examples]
        print(f"  Prompt length (chars): min={min(prompt_lens)}, max={max(prompt_lens)}, "
              f"avg={sum(prompt_lens)/len(prompt_lens):.0f}")

        model_tag = "lora" if lora_request else "base"
        results = run_eval(llm, examples, sampling_params, lora_request)
        metrics = aggregate_metrics(results)

        all_results[dataset_name] = {"metrics": metrics, "details": results}
        summary[dataset_name] = metrics

        print(f"\n  Results ({model_tag}):")
        print(f"    Exact Match:           {metrics['exact_match']:.1%}")
        print(f"    Substring Exact Match: {metrics['substring_exact_match']:.1%}")
        print(f"    F1:                    {metrics['f1']:.1%}")

        # Show a few examples
        print(f"\n  Sample predictions:")
        for r in results[:3]:
            print(f"    Q: {r['question'][:80]}")
            print(f"    A: {r['answers'][:2]}")
            print(f"    P: {r['prediction'][:80]}")
            print(f"    EM={r['exact_match']:.0f} SubEM={r['substring_exact_match']:.0f} F1={r['f1']:.2f}")
            print()

    # Save results
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    output = {"args": vars(args), "summary": summary, "results": all_results}
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Dataset':<15} {'EM':>8} {'SubEM':>8} {'F1':>8}")
    print("-" * 41)
    for ds, mets in summary.items():
        print(f"{ds:<15} {mets['exact_match']:>7.1%} {mets['substring_exact_match']:>7.1%} {mets['f1']:>7.1%}")


if __name__ == "__main__":
    main()
