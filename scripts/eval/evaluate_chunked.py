"""Evaluate with chunked document attention using HuggingFace (not vLLM).

Prefills with chunked attention mask (documents isolated), then generates with
standard causal attention. Supports two attention backends:
  - sdpa (default): Materializes full N*N float mask. Works everywhere but uses
    more memory. Good for shorter sequences.
  - flex: Uses PyTorch FlexAttention with block-sparse masks. Faster and more
    memory-efficient for long sequences, but requires PyTorch >= 2.5.

Usage:
    python scripts/evaluate_chunked.py --datasets nq --num-docs 20
    python scripts/evaluate_chunked.py --backend flex --datasets nq --num-docs 20
    python scripts/evaluate_chunked.py --eval-data data/nq_eval_k20.jsonl --lora-path outputs/nq-rag-chunked
"""

import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports
from lib.io import load_jsonl, save_results
from lib.data_format import build_prompt
from lib.metrics import exact_match, substring_match, token_f1, max_over_answers, aggregate
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    build_chunked_causal_mask, find_chunk_spans,
)
from evaluate_helmet_rag import DATASET_CONFIG, PASSAGE_TEMPLATE, build_demos, parse_output


def load_model(args):
    """Load model with SDPA or FlexAttention backend and optional LoRA.

    Unlike vLLM eval, this uses HuggingFace directly because vLLM doesn't
    support custom 4D attention masks needed for chunked attention.
    """
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Chunked attention requires special <doc_start>/<doc_end> tokens to mark
    # document boundaries. Skip adding them for standard attention baseline.
    if args.standard_attention:
        doc_start_id, doc_end_id = None, None
    else:
        doc_start_id, doc_end_id = setup_tokenizer(tokenizer)

    # Choose attention implementation based on backend
    attn_impl = "flex_attention" if args.backend == "flex" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    if not args.standard_attention:
        # Resize embeddings to include the new boundary tokens
        model.resize_token_embeddings(len(tokenizer))

    if args.lora_path:
        # Merge LoRA weights into base model for faster inference
        # (no adapter overhead during generation)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print(f"Loaded LoRA from {args.lora_path}")

    model = model.cuda().eval()
    return model, tokenizer, doc_start_id, doc_end_id


@torch.no_grad()
def generate_chunked(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                     max_new_tokens=20, stop_token_ids=None, standard_attention=False):
    """Generate with chunked (or standard) attention during prefill, standard causal during decode.

    Two-phase generation:
      Phase 1 (prefill): Process the entire prompt with the chunked 4D mask so
        documents can't attend to each other. KV cache is populated here.
      Phase 2 (decode): Generate tokens autoregressively with standard causal
        attention (new tokens can attend to everything in the KV cache).
    """
    device = input_ids.device

    # Phase 1: Prefill — build the custom attention mask for the full prompt
    if standard_attention:
        # Standard causal mask (lower-triangular) as a baseline comparison
        seq_len = input_ids.size(1)
        dtype = torch.bfloat16
        mask = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype), diagonal=1
        ).unsqueeze(0).unsqueeze(0).to(device)
    else:
        # Chunked mask: documents isolated from each other, query/instruction "free"
        mask = build_chunked_causal_mask(
            input_ids.squeeze(0), doc_start_id, doc_end_id,
        ).to(device)

    outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Phase 2: Greedy autoregressive decode — no custom mask needed since
    # generated tokens attend to the full KV cache via standard causal attention
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


def build_chunk_ids(input_ids, doc_start_id, doc_end_id):
    """Build chunk ID tensor for FlexAttention: chunk_id[i] = chunk index if in doc, -1 if free."""
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
    """Generate with FlexAttention chunked prefill, then standard causal decode.

    FlexAttention uses compiled block-sparse masks instead of materializing the
    full N*N mask, making it faster and more memory-efficient for long sequences.
    """
    from torch.nn.attention.flex_attention import create_block_mask

    device = input_ids.device
    B, S = input_ids.shape

    if standard_attention:
        # Standard causal -- no block mask needed, model uses default causal
        outputs = model(input_ids=input_ids, use_cache=True)
    else:
        # Build BlockMask: same chunked attention logic but in FlexAttention's
        # mask_mod function format (evaluated per-block, not per-element)
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

    # Greedy autoregressive decode (standard causal, same as SDPA path)
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
                         query_position="after", use_alpaca=True, no_titles=False,
                         before_dummy=0, after_dummy=0):
    """Load HELMET eval data and build prompts with doc boundary tokens."""
    from evaluate_helmet_rag import load_dataset_for_eval
    raw_examples = load_dataset_for_eval(dataset_name, max_samples, shots, num_docs,
                                          query_position=query_position, use_alpaca=use_alpaca,
                                          no_titles=no_titles,
                                          before_dummy=before_dummy,
                                          after_dummy=after_dummy)

    # Wrap documents with boundary tokens for chunked attention.
    wrapped = []
    for ex in raw_examples:
        prompt = wrap_documents(ex["prompt"])
        wrapped.append({**ex, "prompt": prompt})
    return wrapped


def load_alpaca_examples(path, max_samples, query_position="after",
                         before_dummy=0, after_dummy=0, task="qa",
                         use_titles=True):
    """Load unified-format JSONL and build prompts with doc boundary tokens."""
    examples = load_jsonl(path)
    if max_samples and len(examples) > max_samples:
        import random
        random.seed(42)
        examples = random.sample(examples, max_samples)

    result = []
    for ex in examples:
        prompt, output = build_prompt(
            ex, task=task, query_position=query_position,
            use_titles=use_titles, before_dummy=before_dummy,
            after_dummy=after_dummy, use_alpaca=True,
        )
        prompt = wrap_documents(prompt)
        result.append({
            "prompt": prompt,
            "answers": ex["answers"],
            "question": ex["queries"][0],
        })
    return result


def extract_after_thinking(text):
    """Extract answer text after </think> tag, if present."""
    import re
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        # Take first line of the answer
        first_line = answer.split('\n')[0].strip()
        return first_line if first_line else answer
    return None


def compute_metrics(prediction, answers):
    em = max_over_answers(exact_match, prediction, answers)
    sub_em = max_over_answers(substring_match, prediction, answers)
    f1 = max_over_answers(token_f1, prediction, answers)

    parsed = parse_output(prediction)
    if parsed:
        em = max(em, max_over_answers(exact_match, parsed, answers))
        sub_em = max(sub_em, max_over_answers(substring_match, parsed, answers))
        f1 = max(f1, max_over_answers(token_f1, parsed, answers))

    # Also try extracting answer after </think> tag
    after_think = extract_after_thinking(prediction)
    if after_think:
        em = max(em, max_over_answers(exact_match, after_think, answers))
        sub_em = max(sub_em, max_over_answers(substring_match, after_think, answers))
        f1 = max(f1, max_over_answers(token_f1, after_think, answers))
        # Also try parse_output on the post-thinking text
        parsed_think = parse_output(after_think)
        if parsed_think:
            em = max(em, max_over_answers(exact_match, parsed_think, answers))
            sub_em = max(sub_em, max_over_answers(substring_match, parsed_think, answers))
            f1 = max(f1, max_over_answers(token_f1, parsed_think, answers))

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

    # Query position and attention type
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"],
                        help="Place question before or after documents")
    parser.add_argument("--backend", type=str, default="sdpa", choices=["sdpa", "flex"],
                        help="Attention backend: 'sdpa' materializes N*N mask, "
                             "'flex' uses block-sparse FlexAttention (faster, less memory)")
    parser.add_argument("--standard-attention", action="store_true",
                        help="Use standard causal attention instead of chunked")
    parser.add_argument("--no-titles", action="store_true",
                        help="Omit document titles from prompts")
    parser.add_argument("--use-alpaca", action="store_true",
                        help="Force alpaca prompt format (for full FT models without --lora-path)")
    parser.add_argument("--before-dummy", type=int, default=0,
                        help="Number of dummy token repetitions to insert before documents")
    parser.add_argument("--after-dummy", type=int, default=0,
                        help="Number of dummy token repetitions to insert after documents")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen thinking mode: generate <think>...</think> then answer")
    args = parser.parse_args()

    if not args.datasets and not args.eval_data:
        parser.error("Specify --datasets (HELMET) or --eval-data (alpaca JSONL)")

    model, tokenizer, doc_start_id, doc_end_id = load_model(args)
    device = next(model.parameters()).device

    newline_id = tokenizer.encode("\n", add_special_tokens=False)
    if args.enable_thinking:
        # Don't stop on newlines when thinking — reasoning contains newlines
        stop_ids = {tokenizer.eos_token_id}
        # Increase max tokens to allow room for reasoning
        if args.max_tokens <= 50:
            args.max_tokens = 512
            print(f"  Thinking mode: increased max_tokens to {args.max_tokens}")
    else:
        stop_ids = {tokenizer.eos_token_id} | set(newline_id)

    all_results = {}

    # Determine eval sources
    eval_sources = []
    if args.datasets:
        for ds in args.datasets.split(","):
            eval_sources.append(("helmet", ds.strip()))
    if args.eval_data:
        eval_sources.append(("alpaca", args.eval_data))

    # Use alpaca prompt format for trained models, original HELMET format for base
    use_alpaca = bool(args.lora_path) or args.use_alpaca
    if use_alpaca and args.shots != 0:
        print(f"  Auto-setting shots=0 for trained model (training data has no demos)")
        args.shots = 0
    print(f"Prompt format: {'alpaca' if use_alpaca else 'helmet (base model)'}, shots={args.shots}")

    for source_type, source in eval_sources:
        label = source if source_type == "helmet" else Path(source).stem
        attn_label = "standard" if args.standard_attention else f"chunked+{args.backend}"
        print(f"\n{'='*60}\nEvaluating: {label} ({attn_label} attention, query {args.query_position})\n{'='*60}")

        if source_type == "helmet":
            examples = load_helmet_examples(source, args.num_docs, args.max_test_samples, args.shots,
                                            query_position=args.query_position, use_alpaca=use_alpaca,
                                            no_titles=args.no_titles,
                                            before_dummy=args.before_dummy,
                                            after_dummy=args.after_dummy)
        else:
            examples = load_alpaca_examples(source, args.max_test_samples,
                                            query_position=args.query_position,
                                            before_dummy=args.before_dummy,
                                            after_dummy=args.after_dummy)

        print(f"  {len(examples)} examples")
        results = []
        for ex in tqdm(examples, desc=f"  {label}"):
            prompt = ex["prompt"]
            if args.enable_thinking:
                prompt = prompt + "<think>\n"
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True,
            ).input_ids.to(device)

            # Dispatch to SDPA or FlexAttention backend
            gen_fn = generate_chunked_flex if args.backend == "flex" else generate_chunked
            response = gen_fn(
                model, tokenizer, input_ids, doc_start_id, doc_end_id,
                max_new_tokens=args.max_tokens, stop_token_ids=stop_ids,
                standard_attention=args.standard_attention,
            )

            m = compute_metrics(response, ex["answers"])
            results.append({"prediction": response.strip()[:500], **m})

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
