"""Chunked document attention: documents attend within themselves and to
query tokens; query/answer tokens attend to everything.

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
    HELMET format. Supports query appearing before or after documents.
    Non-document text (question, instruction) is left unwrapped.
    """
    # Find question — may be before, after, or both
    question_before_match = re.match(r'^(Question:.*?\n\n)', text)
    question_after_idx = text.rfind("\n\nQuestion:")

    query_before, query_after = "", ""
    if question_before_match:
        query_before = question_before_match.group(1)
        doc_section = text[len(query_before):]
        # Also check for question at end (both case)
        q_after_idx = doc_section.rfind("\n\nQuestion:")
        if q_after_idx != -1:
            query_after = doc_section[q_after_idx:]
            doc_section = doc_section[:q_after_idx]
    elif question_after_idx != -1:
        doc_section = text[:question_after_idx]
        query_after = text[question_after_idx:]
    else:
        doc_section = text

    # Split at all \n\n boundaries and wrap document blocks.
    # Non-document text (e.g. dummy tokens) stays unwrapped so chunked
    # attention treats it like query/instruction tokens.
    parts = doc_section.split("\n\n")
    wrapped = []
    for p in parts:
        p = p.strip()
        if p.startswith(("Document (", "Document [", "Document:")):
            wrapped.append(f"{DOC_START}{p}{DOC_END}")
        elif p:
            wrapped.append(p)

    return query_before + "\n\n".join(wrapped) + query_after


def reorder_query(text: str, position: str = "after") -> str:
    """Move the 'Question: ...' line before or after documents in the input.

    Args:
        text: The input field (documents + question).
        position: "before" or "after".

    Returns:
        Reordered text.
    """
    if position == "after":
        # Check if question is at the start and move to end
        m = re.match(r'^(Question:.*?)(\n\n)([\s\S]*)', text)
        if m:
            return m.group(3) + "\n\n" + m.group(1)
        return text  # already after or no question found
    elif position == "before":
        # Check if question is at the end and move to start
        m = re.search(r'\n\n(Question:.*)$', text)
        if m:
            return m.group(1) + "\n\n" + text[:m.start()]
        return text  # already before or no question found
    elif position == "both":
        # Put question both before and after documents
        # First, find and extract the question
        m = re.search(r'\n\n(Question:.*)$', text)
        if m:
            question = m.group(1)
            docs = text[:m.start()]
            return question + "\n\n" + docs + "\n\n" + question
        # Maybe question is already at the start
        m = re.match(r'^(Question:.*?)(\n\n)([\s\S]*)', text)
        if m:
            question = m.group(1)
            docs = m.group(3)
            return question + "\n\n" + docs + "\n\n" + question
        return text
    else:
        raise ValueError(f"Invalid query position: {position!r}, expected 'before', 'after', or 'both'")


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

    - Tokens inside a document chunk attend within that chunk AND to
      non-chunk (query/instruction) tokens that precede them (causal).
    - Tokens outside any chunk attend to all preceding tokens (standard causal).

    This allows each document to "see" the query, while remaining isolated
    from other documents.

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

    # Vectorized mask: causal AND (same_chunk OR row_not_in_chunk OR col_not_in_chunk)
    # - same_chunk: tokens in the same document chunk can attend to each other
    # - row_not_in_chunk: query/answer tokens attend to all preceding tokens
    # - col_not_in_chunk: doc tokens can attend to query tokens (key change)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    same_chunk = (chunk_id.unsqueeze(0) == chunk_id.unsqueeze(1)) & (chunk_id.unsqueeze(0) >= 0)
    row_not_in_chunk = (chunk_id < 0).unsqueeze(1).expand(-1, seq_len)
    col_not_in_chunk = (chunk_id < 0).unsqueeze(0).expand(seq_len, -1)
    bool_mask = causal & (same_chunk | row_not_in_chunk | col_not_in_chunk)

    min_val = torch.finfo(dtype).min
    float_mask = torch.where(bool_mask, torch.zeros(1, dtype=dtype), torch.full((1,), min_val, dtype=dtype))
    return float_mask.unsqueeze(0).unsqueeze(0)
