"""Evaluate on HELMET RAG tasks (NQ, TriviaQA, HotpotQA, PopQA) using vLLM.

Replicates HELMET's prompt format and metrics without depending on their codebase.

Usage:
    python scripts/evaluate_helmet_rag.py --datasets nq,triviaqa --num-docs 20
    python scripts/evaluate_helmet_rag.py --lora-path outputs/nq-rag-lora --datasets nq
"""

import argparse
import hashlib
import random
import re
from pathlib import Path
from vllm import SamplingParams
from lib.io import load_jsonl, save_results, format_alpaca_prompt
from lib.vllm_utils import add_vllm_args, load_model, run_inference
from lib.metrics import exact_match, substring_match, token_f1, max_over_answers, aggregate

# HELMET-compatible templates
PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"
DEMO_TEMPLATE = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"

# Instruction matches training data (generate_nq_training_data.py)
INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)

# Original HELMET-style templates (for base model eval without alpaca wrapper)
HELMET_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}{context}\n\nQuestion: {question}"
)
HELMET_TEMPLATE_QUERY_BEFORE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}Question: {question}\n\n{context}"
)
HELMET_TEMPLATE_QUERY_BOTH = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}Question: {question}\n\n{context}\n\nQuestion: {question}"
)

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


def parse_output(output: str, prefix: str = "Answer:") -> str | None:
    patterns = [
        re.compile(f"(?:{prefix})(.*?)(?:\\n|$)", flags=re.IGNORECASE),
        re.compile(r"(?:^)(.*?)(?:\n|$)"),
    ]
    for pat in patterns:
        match = pat.search(output)
        if match:
            result = re.sub(f"^{re.escape(prefix)}", "", match[1].strip(), flags=re.IGNORECASE).strip()
            if result:
                return result
    return None


def _format_passage(ctx, no_titles=False):
    if no_titles:
        return PASSAGE_TEMPLATE_NO_TITLE.format(text=ctx["text"])
    return PASSAGE_TEMPLATE.format(**ctx)


def build_demos(demo_data, sample, shots, no_titles=False):
    if shots == 0:
        return ""
    h = int(hashlib.sha256(str(sample["question"]).encode()).hexdigest(), 16) % 2**31
    rng = random.Random(h)
    demos = [d for d in demo_data if d.get("question") != sample.get("question")]
    rng.shuffle(demos)
    seen, unique = set(), []
    for d in demos:
        k = d.get("question", "")
        if k not in seen:
            seen.add(k)
            unique.append(d)
        if len(unique) >= shots:
            break
    texts = []
    for d in unique:
        docs = "\n\n".join(_format_passage(c, no_titles) for c in d.get("ctxs", []))
        ans = d["answers"][0] if isinstance(d["answers"], list) else d["answers"]
        texts.append(DEMO_TEMPLATE.format(documents=docs, question=d["question"], answer=ans))
    return "\n\n".join(texts) + "\n\n" if texts else ""


def compute_metrics(prediction, answers):
    em = max_over_answers(exact_match, prediction, answers)
    sub_em = max_over_answers(substring_match, prediction, answers)
    f1 = max_over_answers(token_f1, prediction, answers)
    parsed = parse_output(prediction)
    if parsed:
        em = max(em, max_over_answers(exact_match, parsed, answers))
        sub_em = max(sub_em, max_over_answers(substring_match, parsed, answers))
        f1 = max(f1, max_over_answers(token_f1, parsed, answers))
    return {"exact_match": float(em), "substring_exact_match": float(sub_em), "f1": f1}


def load_dataset_for_eval(dataset_name, max_samples=None, shots=2, num_docs=1000,
                          query_position="after", use_alpaca=True, no_titles=False):
    config = DATASET_CONFIG[dataset_name]
    # Trained models (alpaca format) use 0 shots to match training data
    if use_alpaca and shots > 0:
        print(f"  Note: overriding shots={shots} -> 0 for alpaca format (matches training data)")
        shots = 0
    # For num_docs=0 (no-context baseline), load any available test file for the questions
    search_docs = [num_docs] if num_docs > 0 else []
    search_docs += [500, 105, 100, 50, 20, 10, 3]
    test_file = None
    for nd in search_docs:
        candidate = config["test_file"].format(num_docs=nd)
        if Path(candidate).exists():
            if nd != num_docs:
                print(f"  Fallback: {candidate}")
            test_file = candidate
            break
    if test_file is None:
        raise FileNotFoundError(f"No test file for {dataset_name}")

    fmt_label = "alpaca" if use_alpaca else "helmet"
    print(f"  Loading: {test_file} (format={fmt_label}, titles={'no' if no_titles else 'yes'})")
    test_data = load_jsonl(test_file)
    demo_data = load_jsonl(config["demo_file"]) if shots > 0 and num_docs > 0 and Path(config["demo_file"]).exists() else []

    if max_samples and len(test_data) > max_samples:
        key = "id" if "id" in test_data[0] else "question"
        seen, unique = set(), []
        for d in test_data:
            k = d.get(key, d["question"])
            if k not in seen:
                seen.add(k)
                unique.append(d)
        random.seed(42)
        test_data = random.sample(unique, min(max_samples, len(unique)))

    examples = []
    for s in test_data:
        demos = ""
        context = ""
        if num_docs > 0:
            demos = build_demos(demo_data, s, shots, no_titles=no_titles)
            context = "\n\n".join(_format_passage(c, no_titles) for c in s.get("ctxs", []))

        if use_alpaca:
            # Alpaca format: matches training data (instruction + input wrapped in alpaca template)
            if num_docs == 0:
                input_text = f"Question: {s['question']}"
            else:
                if demos:
                    context = demos + context
                if query_position == "before":
                    input_text = f"Question: {s['question']}\n\n{context}"
                elif query_position == "both":
                    input_text = f"Question: {s['question']}\n\n{context}\n\nQuestion: {s['question']}"
                else:
                    input_text = f"{context}\n\nQuestion: {s['question']}"
            prompt = format_alpaca_prompt(INSTRUCTION, input_text)
        else:
            # Original HELMET format (no alpaca wrapper)
            if num_docs == 0:
                prompt = HELMET_TEMPLATE.format(demos="", context="", question=s["question"]) + "\nAnswer:"
            else:
                if query_position == "before":
                    template = HELMET_TEMPLATE_QUERY_BEFORE
                elif query_position == "both":
                    template = HELMET_TEMPLATE_QUERY_BOTH
                else:
                    template = HELMET_TEMPLATE
                prompt = template.format(demos=demos, context=context, question=s["question"]) + "\nAnswer:"

        examples.append({"prompt": prompt, "answers": s["answers"], "question": s["question"]})
    return examples


def main():
    parser = argparse.ArgumentParser(description="HELMET RAG evaluation")
    add_vllm_args(parser)
    parser.add_argument("--datasets", type=str, default="nq,triviaqa,hotpotqa,popqa")
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--num-docs", type=int, default=1000)
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"],
                        help="Place question before or after documents")
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles from prompts")
    parser.set_defaults(max_tokens=20, output_file="outputs/helmet_rag_results.json")
    args = parser.parse_args()

    llm, lora_request = load_model(args)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["\n"])

    # Use alpaca prompt format for trained models (LoRA/full FT), original HELMET format for base
    use_alpaca = bool(args.lora_path) or args.base_model != "NousResearch/Llama-3.2-1B"
    # Trained models use 0 shots to match training data format (no demos in training)
    if use_alpaca and args.shots != 0:
        print(f"  Auto-setting shots=0 for trained model (training data has no demos)")
        args.shots = 0
    print(f"Prompt format: {'alpaca' if use_alpaca else 'helmet (base model)'}, shots={args.shots}")

    all_results, summary = {}, {}
    for ds_name in args.datasets.split(","):
        ds_name = ds_name.strip()
        print(f"\n{'='*60}\nEvaluating: {ds_name}\n{'='*60}")
        if ds_name not in DATASET_CONFIG:
            print(f"  Unknown dataset: {ds_name}")
            continue
        try:
            examples = load_dataset_for_eval(ds_name, args.max_test_samples, args.shots, args.num_docs,
                                                query_position=args.query_position, use_alpaca=use_alpaca,
                                                no_titles=args.no_titles)
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        print(f"  {len(examples)} examples")
        prompts = [ex["prompt"] for ex in examples]
        responses = run_inference(llm, prompts, sampling_params, lora_request)

        results = []
        for ex, resp in zip(examples, responses):
            m = compute_metrics(resp, ex["answers"])
            results.append({"question": ex["question"], "prediction": resp.strip()[:300], **m})

        metrics = aggregate(results, ["exact_match", "substring_exact_match", "f1"])
        all_results[ds_name] = {"metrics": metrics, "details": results}
        summary[ds_name] = metrics
        print(f"  EM: {metrics['exact_match']:.1%}  SubEM: {metrics['substring_exact_match']:.1%}  F1: {metrics['f1']:.1%}")

    save_results(args.output_file, {"args": vars(args), "summary": summary, "results": all_results})
    print(f"\nResults saved to {args.output_file}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Dataset':<15} {'EM':>8} {'SubEM':>8} {'F1':>8}")
    print("-" * 41)
    for ds, m in summary.items():
        print(f"{ds:<15} {m['exact_match']:>7.1%} {m['substring_exact_match']:>7.1%} {m['f1']:>7.1%}")


if __name__ == "__main__":
    main()
