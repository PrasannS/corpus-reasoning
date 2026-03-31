"""Unified evaluation script for all tasks and inference backends.

Supports any combination of:
  Task:    --task {retrieval, qa}
  Backend: --backend {vllm, chunked-sdpa, chunked-flex, standard}
  Data:    --eval-data (unified JSONL) or --datasets (HELMET KILT format)

The backend controls how inference is run:
  - vllm: Fast batched inference via vLLM (no custom attention masks)
  - chunked-sdpa: HuggingFace with chunked 4D SDPA masks (docs isolated)
  - chunked-flex: HuggingFace with FlexAttention block-sparse masks (faster)
  - standard: HuggingFace with standard causal attention (baseline)

The task controls prompt formatting and metrics:
  - retrieval: Prompts ask for document IDs, metrics are retrieval EM/recall/precision/F1
  - qa: Prompts ask for answer text, metrics are QA EM/SubEM/F1

Usage:
    # Retrieval eval with chunked attention
    python scripts/eval/evaluate.py --backend chunked-sdpa --task retrieval \\
        --eval-data data/hotpotqa_eval_k20_shuffled_bridge_500.jsonl \\
        --lora-path outputs/model

    # QA eval with vLLM
    python scripts/eval/evaluate.py --backend vllm --task qa \\
        --datasets nq,hotpotqa --lora-path outputs/model

    # Retrieval eval with vLLM
    python scripts/eval/evaluate.py --backend vllm --task retrieval \\
        --eval-data data/nq_eval_k20_random_500.jsonl --lora-path outputs/model

    # HELMET base model eval (no fine-tuning)
    python scripts/eval/evaluate.py --backend vllm --task qa \\
        --datasets nq --shots 2
"""

import argparse
import re
import random
import sys
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports

import hashlib

from lib.io import load_jsonl, save_results, format_alpaca_prompt, insert_dummy_tokens
from lib.data_format import build_prompt
from lib.prompts import (
    PASSAGE_TEMPLATE, PASSAGE_TEMPLATE_NO_TITLE,
    QA_INSTRUCTION, DEMO_TEMPLATE,
    HELMET_TEMPLATE, HELMET_TEMPLATE_QUERY_BEFORE, HELMET_TEMPLATE_QUERY_BOTH,
)
from lib.metrics import (
    exact_match, substring_match, token_f1, max_over_answers, aggregate,
    parse_doc_ids, retrieval_exact_match, retrieval_recall,
    retrieval_precision, retrieval_f1,
)
from lib.chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    build_chunked_causal_mask, find_chunk_spans,
)

# Lazy imports for backends
SamplingParams = None


def _import_vllm():
    global SamplingParams
    from vllm import SamplingParams as _SP
    SamplingParams = _SP
    from lib.vllm_utils import add_vllm_args, load_model, run_inference
    return add_vllm_args, load_model, run_inference


# ---------------------------------------------------------------------------
# HuggingFace model loading (chunked / standard backends)
# ---------------------------------------------------------------------------

def load_hf_model(args):
    """Load HuggingFace model for chunked or standard attention eval."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_chunked = args.backend.startswith("chunked")
    if is_chunked:
        doc_start_id, doc_end_id = setup_tokenizer(tokenizer)
    else:
        doc_start_id, doc_end_id = None, None

    attn_impl = "flex_attention" if args.backend == "chunked-flex" else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
    )
    if is_chunked:
        model.resize_token_embeddings(len(tokenizer))

    if args.lora_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        print(f"Loaded LoRA from {args.lora_path}")

    model = model.cuda().eval()
    return model, tokenizer, doc_start_id, doc_end_id


# ---------------------------------------------------------------------------
# HuggingFace generation (chunked / standard / flex)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_hf(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                max_new_tokens=20, stop_token_ids=None, backend="chunked-sdpa"):
    """Generate with HuggingFace: chunked or standard attention prefill, then greedy decode."""
    device = input_ids.device

    if backend == "chunked-flex":
        return _generate_flex(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                              max_new_tokens, stop_token_ids)

    # Build prefill mask
    if backend == "standard":
        seq_len = input_ids.size(1)
        dtype = torch.bfloat16
        mask = torch.triu(
            torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype), diagonal=1
        ).unsqueeze(0).unsqueeze(0).to(device)
    else:
        # chunked-sdpa
        mask = build_chunked_causal_mask(
            input_ids.squeeze(0), doc_start_id, doc_end_id,
        ).to(device)

    outputs = model(input_ids=input_ids, attention_mask=mask, use_cache=True)
    past_kv = outputs.past_key_values
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

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


@torch.no_grad()
def _generate_flex(model, tokenizer, input_ids, doc_start_id, doc_end_id,
                   max_new_tokens=20, stop_token_ids=None):
    """Generate with FlexAttention block-sparse chunked prefill."""
    from torch.nn.attention.flex_attention import create_block_mask

    device = input_ids.device
    B, S = input_ids.shape

    # Build chunk IDs for FlexAttention mask
    ids_1d = input_ids.squeeze(0)
    spans = find_chunk_spans(ids_1d, doc_start_id, doc_end_id)
    chunk_ids = torch.full((1, S), -1, dtype=torch.int32, device=device)
    for idx, (s, e) in enumerate(spans):
        chunk_ids[0, s:e] = idx

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_unified_examples(path, max_samples, task, query_position="after",
                          use_titles=True, before_dummy=0, after_dummy=0,
                          use_alpaca=True, wrap_docs=False):
    """Load unified-format JSONL and build prompts."""
    examples = load_jsonl(path)
    if max_samples and len(examples) > max_samples:
        random.seed(42)
        examples = random.sample(examples, max_samples)

    result = []
    for ex in examples:
        prompt, output = build_prompt(
            ex, task=task, query_position=query_position,
            use_titles=use_titles, before_dummy=before_dummy,
            after_dummy=after_dummy, use_alpaca=use_alpaca,
        )
        if wrap_docs:
            prompt = wrap_documents(prompt)

        entry = {
            "prompt": prompt,
            "expected_output": output,
            "answers": ex["answers"],
            "queries": ex["queries"],
            "gold_doc_indices": ex.get("gold_doc_indices", []),
        }
        result.append(entry)
    return result


HELMET_DATASET_CONFIG = {
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


def _format_passage(ctx, no_titles=False):
    if no_titles:
        return PASSAGE_TEMPLATE_NO_TITLE.format(text=ctx["text"])
    return PASSAGE_TEMPLATE.format(**ctx)


def _build_demos(demo_data, sample, shots, no_titles=False):
    """Build few-shot demos for base model evaluation."""
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


def load_helmet_examples(dataset_name, num_docs, max_samples, shots,
                         query_position="after", use_alpaca=True, no_titles=False,
                         before_dummy=0, after_dummy=0, wrap_docs=False):
    """Load HELMET/KILT eval data and build prompts."""
    config = HELMET_DATASET_CONFIG[dataset_name]

    if use_alpaca and shots > 0:
        print(f"  Note: overriding shots={shots} -> 0 for alpaca format")
        shots = 0

    # Find test file (fall back to other num_docs if exact match missing)
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
    demo_data = (load_jsonl(config["demo_file"])
                 if shots > 0 and num_docs > 0 and Path(config["demo_file"]).exists()
                 else [])

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

    result = []
    for s in test_data:
        demos = ""
        context = ""
        if num_docs > 0:
            demos = _build_demos(demo_data, s, shots, no_titles=no_titles)
            context = "\n\n".join(_format_passage(c, no_titles) for c in s.get("ctxs", []))

        if use_alpaca:
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
            if before_dummy > 0 or after_dummy > 0:
                input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)
            prompt = format_alpaca_prompt(QA_INSTRUCTION, input_text)
        else:
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

        if wrap_docs:
            prompt = wrap_documents(prompt)
        result.append({
            "prompt": prompt,
            "expected_output": None,
            "answers": s["answers"],
            "queries": [s.get("question", "")],
            "gold_doc_indices": [],
        })
    return result


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def extract_after_thinking(text):
    """Extract answer text after </think> tag, if present."""
    match = re.search(r'</think>\s*(.*)', text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
        first_line = answer.split('\n')[0].strip()
        return first_line if first_line else answer
    return None


def parse_output(output, prefix="Answer:"):
    """Extract answer after a prefix pattern."""
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


def parse_retrieval_output(output):
    """Parse document IDs from model output, stripping thinking/prefixes.

    Uses rfind (last occurrence) so that CoT reasoning preceding the final
    'Relevant Document: [X]' line doesn't interfere with parsing.
    """
    text = output.strip()
    after_think = extract_after_thinking(text)
    if after_think:
        text = after_think
    for prefix in ["Relevant Documents:", "Relevant Document:", "relevant documents:",
                   "relevant document:"]:
        idx = text.rfind(prefix)
        if idx >= 0:
            text = text[idx + len(prefix):].strip()
            break
    text = text.split("\n")[0].strip()
    return parse_doc_ids(text)


def parse_multi_query_output(output, num_queries):
    """Parse per-query document IDs: 'Q1: [3], [7]; Q2: [1]; ...'"""
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
    parts = [p.strip() for p in text.split(";")]
    per_query_ids = []
    for part in parts:
        part = re.sub(r'^Q\d+:\s*', '', part)
        per_query_ids.append(parse_doc_ids(part))
    while len(per_query_ids) < num_queries:
        per_query_ids.append(set())
    return per_query_ids[:num_queries]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_qa_metrics(prediction, answers):
    """Compute QA metrics with multiple extraction strategies."""
    em = max_over_answers(exact_match, prediction, answers)
    sub_em = max_over_answers(substring_match, prediction, answers)
    f1 = max_over_answers(token_f1, prediction, answers)

    parsed = parse_output(prediction)
    if parsed:
        em = max(em, max_over_answers(exact_match, parsed, answers))
        sub_em = max(sub_em, max_over_answers(substring_match, parsed, answers))
        f1 = max(f1, max_over_answers(token_f1, parsed, answers))

    after_think = extract_after_thinking(prediction)
    if after_think:
        em = max(em, max_over_answers(exact_match, after_think, answers))
        sub_em = max(sub_em, max_over_answers(substring_match, after_think, answers))
        f1 = max(f1, max_over_answers(token_f1, after_think, answers))
        parsed_think = parse_output(after_think)
        if parsed_think:
            em = max(em, max_over_answers(exact_match, parsed_think, answers))
            sub_em = max(sub_em, max_over_answers(substring_match, parsed_think, answers))
            f1 = max(f1, max_over_answers(token_f1, parsed_think, answers))

    return {"exact_match": float(em), "substring_exact_match": float(sub_em), "f1": f1}


def compute_retrieval_metrics_single(prediction, gold_doc_indices):
    """Compute retrieval metrics for single-query examples."""
    predicted_ids = parse_retrieval_output(prediction)
    # Convert 0-indexed gold to 1-indexed (prompt uses 1-indexed doc IDs)
    gold_ids = set(g + 1 for g in gold_doc_indices)
    return {
        "exact_match": float(retrieval_exact_match(predicted_ids, gold_ids)),
        "recall": retrieval_recall(predicted_ids, gold_ids),
        "precision": retrieval_precision(predicted_ids, gold_ids),
        "f1": retrieval_f1(predicted_ids, gold_ids),
        "predicted_ids": sorted(predicted_ids),
        "gold_ids": sorted(gold_ids),
    }


def compute_retrieval_metrics_multi(prediction, gold_doc_indices, num_queries):
    """Compute retrieval metrics for multi-query examples."""
    per_query_predicted = parse_multi_query_output(prediction, num_queries)
    per_query_metrics = []
    for qi, (pred_ids, gold_indices) in enumerate(zip(per_query_predicted, gold_doc_indices)):
        gold_ids = set(g + 1 for g in gold_indices)
        per_query_metrics.append({
            "exact_match": float(retrieval_exact_match(pred_ids, gold_ids)),
            "recall": retrieval_recall(pred_ids, gold_ids),
            "precision": retrieval_precision(pred_ids, gold_ids),
            "f1": retrieval_f1(pred_ids, gold_ids),
        })
    n = len(per_query_metrics)
    agg = {k: sum(m[k] for m in per_query_metrics) / n
           for k in ["exact_match", "recall", "precision", "f1"]}
    agg["all_correct"] = float(all(m["exact_match"] == 1.0 for m in per_query_metrics))
    return agg, per_query_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation (any task × any backend)")

    # Model
    parser.add_argument("--base-model", type=str, default="NousResearch/Llama-3.2-1B")
    parser.add_argument("--lora-path", type=str, default="")
    parser.add_argument("--max-tokens", type=int, default=50)

    # Task and backend
    parser.add_argument("--task", type=str, default="retrieval",
                        choices=["qa", "retrieval", "cot_retrieval"])
    parser.add_argument("--backend", type=str, default="vllm",
                        choices=["vllm", "chunked-sdpa", "chunked-flex", "standard"],
                        help="Inference backend")

    # Data sources
    parser.add_argument("--eval-data", type=str, default="",
                        help="Unified-format JSONL file")
    parser.add_argument("--datasets", type=str, default="",
                        help="HELMET datasets (e.g. nq,hotpotqa) — QA task only")
    parser.add_argument("--num-docs", type=int, default=20)
    parser.add_argument("--shots", type=int, default=2,
                        help="Few-shot demos for HELMET base model eval")

    # Formatting
    parser.add_argument("--query-position", type=str, default="after",
                        choices=["before", "after", "both"])
    parser.add_argument("--no-titles", action="store_true")
    parser.add_argument("--use-alpaca", action="store_true",
                        help="Force alpaca format (for full FT models without --lora-path)")
    parser.add_argument("--before-dummy", type=int, default=0)
    parser.add_argument("--after-dummy", type=int, default=0)

    # Generation
    parser.add_argument("--max-test-samples", type=int, default=100)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output-file", type=str, default="outputs/eval_results.json")

    # vLLM-specific (only used when backend=vllm)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--tokenizer", type=str, default="")

    args = parser.parse_args()

    if not args.datasets and not args.eval_data:
        parser.error("Specify --eval-data (unified JSONL) or --datasets (HELMET)")

    use_alpaca = bool(args.lora_path) or args.use_alpaca
    is_hf = args.backend != "vllm"
    wrap_docs = is_hf and args.backend.startswith("chunked")

    if use_alpaca and args.shots != 0:
        print(f"  Auto-setting shots=0 for trained model (training data has no demos)")
        args.shots = 0

    if args.task == "cot_retrieval" and args.max_tokens <= 50:
        args.max_tokens = 512
        print(f"  CoT retrieval: increased max_tokens to {args.max_tokens}")

    if args.enable_thinking and args.max_tokens <= 50:
        args.max_tokens = 512
        print(f"  Thinking mode: increased max_tokens to {args.max_tokens}")

    print(f"Task: {args.task} | Backend: {args.backend} | "
          f"Format: {'alpaca' if use_alpaca else 'helmet'} | Shots: {args.shots}")

    # --- Load model ---
    if is_hf:
        model, tokenizer, doc_start_id, doc_end_id = load_hf_model(args)
        device = next(model.parameters()).device
        newline_id = tokenizer.encode("\n", add_special_tokens=False)
        multiline_output = args.enable_thinking or args.task == "cot_retrieval"
        if multiline_output:
            stop_ids = {tokenizer.eos_token_id}
        else:
            stop_ids = {tokenizer.eos_token_id} | set(newline_id)
    else:
        _import_vllm()
        from lib.vllm_utils import load_model as vllm_load_model, run_inference
        llm, lora_request = vllm_load_model(args)
        multiline_output = args.enable_thinking or args.task == "cot_retrieval"
        if multiline_output:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        else:
            sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, stop=["\n"])

    # --- Load data ---
    eval_sources = []
    if args.eval_data:
        eval_sources.append(("unified", args.eval_data))
    if args.datasets:
        for ds in args.datasets.split(","):
            eval_sources.append(("helmet", ds.strip()))

    all_results = {}
    for source_type, source in eval_sources:
        label = source if source_type == "helmet" else Path(source).stem
        print(f"\n{'='*60}\nEvaluating: {label} ({args.backend}, {args.task})\n{'='*60}")

        if source_type == "helmet":
            examples = load_helmet_examples(
                source, args.num_docs, args.max_test_samples, args.shots,
                query_position=args.query_position, use_alpaca=use_alpaca,
                no_titles=args.no_titles,
                before_dummy=args.before_dummy, after_dummy=args.after_dummy,
                wrap_docs=wrap_docs,
            )
        else:
            examples = load_unified_examples(
                source, args.max_test_samples, task=args.task,
                query_position=args.query_position,
                use_titles=not args.no_titles,
                before_dummy=args.before_dummy, after_dummy=args.after_dummy,
                use_alpaca=use_alpaca, wrap_docs=wrap_docs,
            )

        print(f"  {len(examples)} examples")

        # --- Run inference ---
        if is_hf:
            responses = []
            for ex in tqdm(examples, desc=f"  {label}"):
                prompt = ex["prompt"]
                if args.enable_thinking:
                    prompt = prompt + "<think>\n"
                input_ids = tokenizer(
                    prompt, return_tensors="pt", truncation=True,
                ).input_ids.to(device)
                response = generate_hf(
                    model, tokenizer, input_ids, doc_start_id, doc_end_id,
                    max_new_tokens=args.max_tokens, stop_token_ids=stop_ids,
                    backend=args.backend,
                )
                responses.append(response)
        else:
            prompts = [ex["prompt"] for ex in examples]
            if args.enable_thinking:
                prompts = [p + "<think>\n" for p in prompts]
            print(f"  Running vLLM inference...")
            responses = run_inference(llm, prompts, sampling_params, lora_request)

        # --- Compute metrics ---
        if args.task in ("retrieval", "cot_retrieval"):
            results, details = _eval_retrieval(examples, responses)
        else:
            results, details = _eval_qa(examples, responses)

        all_results[label] = {"metrics": results, "details": details}

        # Print summary
        print(f"\n  Results:")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.1%}")

        # Show samples
        for d in details[:3]:
            print(f"    Pred: {d.get('prediction', '')[:80]}")

    save_results(args.output_file, {"args": vars(args), "results": all_results})
    print(f"\nResults saved to {args.output_file}")


def _eval_qa(examples, responses):
    """Evaluate QA task: compute EM, SubEM, F1."""
    results_list = []
    details = []
    for ex, resp in zip(examples, responses):
        m = compute_qa_metrics(resp, ex["answers"])
        results_list.append(m)
        details.append({"prediction": resp.strip()[:500], **m})

    metrics = aggregate(results_list, ["exact_match", "substring_exact_match", "f1"])
    return metrics, details


def _eval_retrieval(examples, responses):
    """Evaluate retrieval task: compute retrieval EM, recall, precision, F1."""
    results_list = []
    details = []

    for ex, resp in zip(examples, responses):
        gold = ex["gold_doc_indices"]
        is_multi = len(ex["queries"]) > 1

        if is_multi:
            agg, per_query = compute_retrieval_metrics_multi(resp, gold, len(ex["queries"]))
            results_list.append(agg)
            details.append({"prediction": resp.strip()[:500], **agg})
        else:
            # Flatten gold indices for single-query
            flat_gold = gold[0] if gold and isinstance(gold[0], list) else gold
            m = compute_retrieval_metrics_single(resp, flat_gold)
            results_list.append({k: m[k] for k in ["exact_match", "recall", "precision", "f1"]})
            details.append({"prediction": resp.strip()[:500], **m})

    metric_keys = ["exact_match", "recall", "precision", "f1"]
    # Add all_correct if multi-query
    if any("all_correct" in r for r in results_list):
        metric_keys.append("all_correct")
    metrics = aggregate(results_list, metric_keys)
    return metrics, details


if __name__ == "__main__":
    main()
