"""Fast evaluation with chunked document attention using FlexAttention.

Uses FlexAttention for compiled block-sparse prefill attention, then standard
causal attention for autoregressive decoding. Much faster than SDPA-based
evaluate_chunked.py due to no N×N mask materialization.

Usage:
    python scripts/evaluate_chunked_fast.py --datasets nq --num-docs 20
    python scripts/evaluate_chunked_fast.py --lora-path outputs/nq-rag-chunked --datasets nq --num-docs 20
"""

import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from lib.io import load_jsonl, save_results, format_alpaca_prompt
from lib.metrics import exact_match, substring_match, token_f1, max_over_answers, aggregate
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    reorder_query, find_chunk_spans,
)
from evaluate_helmet_rag import DATASET_CONFIG, PASSAGE_TEMPLATE, build_demos, parse_output

from torch.nn.attention.flex_attention import create_block_mask


def load_model(args):
    """Load model with FlexAttention and optional LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation="flex_attention",
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


def build_chunk_ids(input_ids, doc_start_id, doc_end_id):
    """Build chunk ID tensor for FlexAttention."""
    seq_len = input_ids.size(1) if input_ids.dim() > 1 else len(input_ids)
    ids_1d = input_ids.squeeze(0) if input_ids.dim() > 1 else input_ids
    spans = find_chunk_spans(ids_1d, doc_start_id, doc_end_id)
    chunk_id = torch.full((seq_len,), -1, dtype=torch.int32, device=input_ids.device)
    for idx, (s, e) in enumerate(spans):
        chunk_id[s:e] = idx
    return chunk_id.unsqueeze(0)  # (1, seq_len)


@torch.no_grad()
def generate_chunked_flex(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                          max_new_tokens=20, stop_token_ids=None, standard_attention=False):
    """Generate with FlexAttention chunked prefill, then standard decode."""
    device = input_ids.device
    B, S = input_ids.shape

    if standard_attention:
        # Standard causal — no block mask needed, model uses default causal
        outputs = model(input_ids=input_ids, use_cache=True)
    else:
        # Build BlockMask for chunked attention
        chunk_ids = build_chunk_ids(input_ids, doc_start_id, doc_end_id)

        def mask_mod(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            q_chunk = chunk_ids[b, q_idx]
            kv_chunk = chunk_ids[b, kv_idx]
            same_chunk = (q_chunk == kv_chunk) & (q_chunk >= 0)
            q_free = q_chunk == -1
            kv_free = kv_chunk == -1
            return causal & (same_chunk | q_free | kv_free)

        block_mask = create_block_mask(mask_mod, B=B, H=None, Q_LEN=S, KV_LEN=S, device=device)
        outputs = model(input_ids=input_ids, attention_mask=block_mask, use_cache=True)

    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Autoregressive decode (standard causal)
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


def load_helmet_examples(dataset_name, num_docs, max_samples, shots,
                         query_position="after", use_alpaca=True, no_titles=False):
    from evaluate_helmet_rag import load_dataset_for_eval
    raw_examples = load_dataset_for_eval(dataset_name, max_samples, shots, num_docs,
                                          query_position=query_position, use_alpaca=use_alpaca,
                                          no_titles=no_titles)
    wrapped = []
    for ex in raw_examples:
        prompt = wrap_documents(ex["prompt"])
        wrapped.append({**ex, "prompt": prompt})
    return wrapped


def load_alpaca_examples(path, max_samples, query_position="after"):
    examples = load_jsonl(path)
    if max_samples and len(examples) > max_samples:
        import random
        random.seed(42)
        examples = random.sample(examples, max_samples)

    result = []
    for ex in examples:
        input_text = reorder_query(ex["input"], query_position)
        wrapped_input = wrap_documents(input_text)
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
    parser = argparse.ArgumentParser(description="Fast chunked attention evaluation with FlexAttention")
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--output-file", type=str, default="outputs/chunked_fast_eval_results.json")
    parser.add_argument("--datasets", type=str, default="", help="HELMET datasets")
    parser.add_argument("--num-docs", type=int, default=20)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--eval-data", type=str, default="", help="Alpaca-format JSONL")
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"])
    parser.add_argument("--standard-attention", action="store_true")
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles from prompts")
    args = parser.parse_args()

    if not args.datasets and not args.eval_data:
        parser.error("Specify --datasets (HELMET) or --eval-data (alpaca JSONL)")

    model, tokenizer, doc_start_id, doc_end_id = load_model(args)
    device = next(model.parameters()).device

    newline_id = tokenizer.encode("\n", add_special_tokens=False)
    stop_ids = {tokenizer.eos_token_id} | set(newline_id)

    all_results = {}
    eval_sources = []
    if args.datasets:
        for ds in args.datasets.split(","):
            eval_sources.append(("helmet", ds.strip()))
    if args.eval_data:
        eval_sources.append(("alpaca", args.eval_data))

    use_alpaca = bool(args.lora_path)
    if use_alpaca and args.shots != 0:
        print(f"  Auto-setting shots=0 for trained model (training data has no demos)")
        args.shots = 0
    print(f"Prompt format: {'alpaca' if use_alpaca else 'helmet (base model)'}, shots={args.shots}")

    for source_type, source in eval_sources:
        label = source if source_type == "helmet" else Path(source).stem
        attn_label = "standard" if args.standard_attention else "chunked+flex"
        print(f"\n{'='*60}\nEvaluating: {label} ({attn_label} attention, query {args.query_position})\n{'='*60}")

        if source_type == "helmet":
            examples = load_helmet_examples(source, args.num_docs, args.max_test_samples, args.shots,
                                            query_position=args.query_position, use_alpaca=use_alpaca,
                                            no_titles=args.no_titles)
        else:
            examples = load_alpaca_examples(source, args.max_test_samples,
                                            query_position=args.query_position)

        print(f"  {len(examples)} examples")
        results = []
        for ex in tqdm(examples, desc=f"  {label}"):
            input_ids = tokenizer(
                ex["prompt"], return_tensors="pt", truncation=True,
            ).input_ids.to(device)

            response = generate_chunked_flex(
                model, tokenizer, input_ids, doc_start_id, doc_end_id,
                max_new_tokens=args.max_tokens, stop_token_ids=stop_ids,
                standard_attention=args.standard_attention,
            )

            m = compute_metrics(response, ex["answers"])
            results.append({"prediction": response.strip()[:300], **m})

        metrics = aggregate(results, ["exact_match", "substring_exact_match", "f1"])
        all_results[label] = {"metrics": metrics, "details": results}
        print(f"  EM: {metrics['exact_match']:.1%}  SubEM: {metrics['substring_exact_match']:.1%}  F1: {metrics['f1']:.1%}")

    save_results(args.output_file, {"args": vars(args), "results": all_results})
    print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
