"""Evaluate with chunked document attention using HuggingFace (not vLLM).

Prefills with chunked attention mask (documents isolated), then generates with
standard causal attention. Supports HELMET RAG format and alpaca JSONL.

Usage:
    python scripts/evaluate_chunked.py --datasets nq --num-docs 20
    python scripts/evaluate_chunked.py --eval-data data/nq_eval_k20.jsonl --lora-path outputs/nq-rag-chunked
    python scripts/evaluate_chunked.py --datasets nq --lora-path outputs/nq-rag-chunked --num-docs 20
"""

import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from lib.io import load_jsonl, save_results
from lib.io import format_alpaca_prompt
from lib.metrics import exact_match, substring_match, token_f1, max_over_answers, aggregate
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    build_chunked_causal_mask,
)
from evaluate_helmet_rag import DATASET_CONFIG, PASSAGE_TEMPLATE, USER_TEMPLATE, build_demos, parse_output


def load_model(args):
    """Load model with SDPA attention and optional LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model.resize_token_embeddings(len(tokenizer))

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print(f"Loaded LoRA from {args.lora_path}")

    model = model.cuda().eval()
    return model, tokenizer, doc_start_id, doc_end_id


@torch.no_grad()
def generate_chunked(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                     max_new_tokens=20, stop_token_ids=None):
    """Generate with chunked attention during prefill, standard causal during decode."""
    device = input_ids.device

    # Phase 1: Prefill with chunked attention mask
    mask = build_chunked_causal_mask(
        input_ids.squeeze(0), doc_start_id, doc_end_id,
    ).to(device)

    outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Phase 2: Autoregressive decode with standard causal attention
    generated = [next_token]
    for _ in range(max_new_tokens - 1):
        outputs = model(input_ids=next_token, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if stop_token_ids and next_token.item() in stop_token_ids:
            break

    gen_ids = torch.cat(generated, dim=-1)
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def load_helmet_examples(dataset_name, num_docs, max_samples, shots):
    """Load HELMET eval data and build prompts with doc boundary tokens."""
    from evaluate_helmet_rag import load_dataset_for_eval
    raw_examples = load_dataset_for_eval(dataset_name, max_samples, shots, num_docs)

    # The prompts from load_dataset_for_eval use HELMET format (not alpaca).
    # We need to wrap documents with boundary tokens.
    wrapped = []
    for ex in raw_examples:
        prompt = wrap_documents(ex["prompt"])
        wrapped.append({**ex, "prompt": prompt})
    return wrapped


def load_alpaca_examples(path, max_samples):
    """Load alpaca-format JSONL and build prompts with doc boundary tokens."""
    examples = load_jsonl(path)
    if max_samples and len(examples) > max_samples:
        import random
        random.seed(42)
        examples = random.sample(examples, max_samples)

    result = []
    for ex in examples:
        wrapped_input = wrap_documents(ex["input"])
        prompt = format_alpaca_prompt(ex["instruction"], wrapped_input)
        answers = ex["output"] if isinstance(ex["output"], list) else [ex["output"]]
        result.append({"prompt": prompt, "answers": answers, "question": ex.get("question", "")})
    return result


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate with chunked document attention")
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--output-file", type=str, default="outputs/chunked_eval_results.json")

    # HELMET format
    parser.add_argument("--datasets", type=str, default="", help="HELMET datasets (e.g. nq,triviaqa)")
    parser.add_argument("--num-docs", type=int, default=20)
    parser.add_argument("--shots", type=int, default=2)

    # Alpaca format
    parser.add_argument("--eval-data", type=str, default="", help="Alpaca-format JSONL")
    args = parser.parse_args()

    if not args.datasets and not args.eval_data:
        parser.error("Specify --datasets (HELMET) or --eval-data (alpaca JSONL)")

    model, tokenizer, doc_start_id, doc_end_id = load_model(args)
    device = next(model.parameters()).device

    newline_id = tokenizer.encode("\n", add_special_tokens=False)
    stop_ids = {tokenizer.eos_token_id} | set(newline_id)

    all_results = {}

    # Determine eval sources
    eval_sources = []
    if args.datasets:
        for ds in args.datasets.split(","):
            eval_sources.append(("helmet", ds.strip()))
    if args.eval_data:
        eval_sources.append(("alpaca", args.eval_data))

    for source_type, source in eval_sources:
        label = source if source_type == "helmet" else Path(source).stem
        print(f"\n{'='*60}\nEvaluating: {label} (chunked attention)\n{'='*60}")

        if source_type == "helmet":
            examples = load_helmet_examples(source, args.num_docs, args.max_test_samples, args.shots)
        else:
            examples = load_alpaca_examples(source, args.max_test_samples)

        print(f"  {len(examples)} examples")
        results = []
        for ex in tqdm(examples, desc=f"  {label}"):
            input_ids = tokenizer(
                ex["prompt"], return_tensors="pt", truncation=True,
            ).input_ids.to(device)

            response = generate_chunked(
                model, tokenizer, input_ids, doc_start_id, doc_end_id,
                max_new_tokens=args.max_tokens, stop_token_ids=stop_ids,
            )

            m = compute_metrics(response, ex["answers"])
            results.append({"prediction": response.strip()[:300], **m})

        metrics = aggregate(results, ["exact_match", "substring_exact_match", "f1"])
        all_results[label] = {"metrics": metrics, "details": results}
        print(f"  EM: {metrics['exact_match']:.1%}  SubEM: {metrics['substring_exact_match']:.1%}  F1: {metrics['f1']:.1%}")

        # Show samples
        for r in results[:3]:
            print(f"    Pred: {r['prediction'][:80]}  EM={r['exact_match']:.0f}")

    save_results(args.output_file, {"args": vars(args), "results": all_results})
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
