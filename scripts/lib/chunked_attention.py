"""Chunked document attention: documents attend only within themselves,
query/answer tokens attend to everything.

Special tokens <|doc_start|> and <|doc_end|> mark document boundaries.
"""

import re
import torch

DOC_START = "<|doc_start|>"
DOC_END = "<|doc_end|>"


def setup_tokenizer(tokenizer):
    """Add doc boundary tokens to tokenizer. Returns (doc_start_id, doc_end_id)."""
    tokenizer.add_special_tokens({"additional_special_tokens": [DOC_START, DOC_END]})
    return tokenizer.convert_tokens_to_ids(DOC_START), tokenizer.convert_tokens_to_ids(DOC_END)


def wrap_documents(text: str) -> str:
    """Wrap each 'Document (Title: ...): ...' block with boundary tokens.

    Handles both NQ RAG format (documents separated by \\n\\n) and
    HELMET format. The question portion after documents is left unwrapped.
    """
    # Find where the question starts (everything after is non-document)
    question_idx = text.rfind("\n\nQuestion:")
    if question_idx == -1:
        doc_section, query_section = text, ""
    else:
        doc_section, query_section = text[:question_idx], text[question_idx:]

    # Split at document boundaries and wrap each
    parts = re.split(r'\n\n(?=Document \(Title:)', doc_section)
    wrapped = []
    for p in parts:
        p = p.strip()
        if p.startswith("Document ("):
            wrapped.append(f"{DOC_START}{p}{DOC_END}")
        elif p:
            wrapped.append(p)

    return "\n\n".join(wrapped) + query_section


def find_chunk_spans(input_ids, doc_start_id, doc_end_id):
    """Find (start, end_exclusive) index spans for each document chunk."""
    spans = []
    start = None
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    for i, tid in enumerate(ids):
        if tid == doc_start_id:
            start = i
        elif tid == doc_end_id and start is not None:
            spans.append((start, i + 1))
            start = None
    return spans


def build_chunked_causal_mask(input_ids, doc_start_id, doc_end_id, dtype=torch.bfloat16):
    """Build 4D attention mask with block-diagonal document attention.

    - Tokens inside a document chunk attend only within that chunk (causal).
    - Tokens outside any chunk attend to all preceding tokens (standard causal).

    Args:
        input_ids: (seq_len,) tensor of token IDs.
        doc_start_id, doc_end_id: special token IDs.
        dtype: mask dtype.

    Returns:
        (1, 1, seq_len, seq_len) float mask. 0 = attend, -inf = masked.
    """
    seq_len = len(input_ids)
    spans = find_chunk_spans(input_ids, doc_start_id, doc_end_id)

    if not spans:
        # No documents found — return standard causal mask
        mask = torch.triu(torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    # Assign chunk IDs (-1 = not in any chunk)
    chunk_id = torch.full((seq_len,), -1, dtype=torch.long)
    for idx, (s, e) in enumerate(spans):
        chunk_id[s:e] = idx

    # Vectorized mask: causal AND (same_chunk OR not_in_chunk)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    same_chunk = (chunk_id.unsqueeze(0) == chunk_id.unsqueeze(1)) & (chunk_id.unsqueeze(0) >= 0)
    not_in_chunk = (chunk_id < 0).unsqueeze(1).expand(-1, seq_len)
    bool_mask = causal & (same_chunk | not_in_chunk)

    min_val = torch.finfo(dtype).min
    float_mask = torch.where(bool_mask, torch.zeros(1, dtype=dtype), torch.full((1,), min_val, dtype=dtype))
    return float_mask.unsqueeze(0).unsqueeze(0)
