"""Chunked document attention with block-diagonal masking.

Standard causal attention lets every token attend to all preceding tokens,
meaning each document can "see" other documents in the context. Chunked
attention restricts this: document tokens can only attend within their own
document AND to "free" tokens (query, instruction, padding) that precede them.

This isolates documents from each other while still allowing each document
to attend to the query/instruction. The hypothesis is that this prevents
shortcut learning where the model relies on cross-document attention patterns.

Implementation:
  - Special tokens <|doc_start|> and <|doc_end|> mark document boundaries.
  - wrap_documents() inserts these tokens around each "Document ..." block.
  - build_chunked_causal_mask() constructs a 4D attention mask where:
      * "Free" tokens (outside any doc) attend causally to all preceding tokens
      * Document tokens attend to: same-doc tokens + free tokens (causal)
      * Document tokens do NOT attend to tokens in other documents

The mask logic: causal AND (same_chunk OR row_free OR col_free)
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
    # Step 1: Separate the question(s) from the document section.
    # The question can appear before docs (query-before), after docs (query-after),
    # or both (query-both). We extract it so we don't accidentally wrap it.
    question_before_match = re.match(r'^(Question:.*?\n\n)', text)
    question_after_idx = text.rfind("\n\nQuestion:")

    query_before, query_after = "", ""
    if question_before_match:
        # Question appears at the start — extract it
        query_before = question_before_match.group(1)
        doc_section = text[len(query_before):]
        # Also check for a trailing question (query-both case)
        q_after_idx = doc_section.rfind("\n\nQuestion:")
        if q_after_idx != -1:
            query_after = doc_section[q_after_idx:]
            doc_section = doc_section[:q_after_idx]
    elif question_after_idx != -1:
        # Question appears only at the end
        doc_section = text[:question_after_idx]
        query_after = text[question_after_idx:]
    else:
        # No question found (shouldn't happen in normal usage)
        doc_section = text

    # Step 2: Split the document section on paragraph boundaries (\n\n) and
    # wrap each document block with boundary tokens. Non-document text (e.g.
    # dummy tokens from ablation studies) is left unwrapped — the attention
    # mask will treat unwrapped tokens as "free" (visible to all).
    parts = doc_section.split("\n\n")
    wrapped = []
    for p in parts:
        p = p.strip()
        if p.startswith(("Document (", "Document [", "Document:")):
            # This is a document — wrap it so the mask can isolate it
            wrapped.append(f"{DOC_START}{p}{DOC_END}")
        elif p:
            # Non-document text (dummy tokens, etc.) — leave as free tokens
            wrapped.append(p)

    # Step 3: Reassemble with the question(s) in their original positions
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
    """Find (start, end_exclusive) index spans for each document chunk.

    Scans token IDs for matching <|doc_start|>...<|doc_end|> pairs.
    The boundary tokens themselves are included in the span, so the mask
    treats them as part of the document (they only attend within their chunk).
    """
    spans = []
    start = None
    ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    for i, tid in enumerate(ids):
        if tid == doc_start_id:
            start = i
        elif tid == doc_end_id and start is not None:
            spans.append((start, i + 1))  # +1 to include the end token
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
        # No documents found — fall back to standard causal mask.
        # This happens when standard_attention=True or the input has no docs.
        mask = torch.triu(torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    # Step 1: Assign each token a chunk ID. Tokens outside any document get -1
    # ("free" tokens: instruction, question, padding, dummy tokens).
    chunk_id = torch.full((seq_len,), -1, dtype=torch.long)
    for idx, (s, e) in enumerate(spans):
        chunk_id[s:e] = idx

    # Step 2: Build the attention mask as a boolean matrix.
    # The mask is: causal AND (same_chunk OR row_free OR col_free)
    #
    # Intuition for each condition (all subject to causal — can't look ahead):
    #   same_chunk:  tokens in document 3 can attend to other tokens in document 3
    #   row_free:    free tokens (question, instruction) can attend to everything
    #   col_free:    all tokens can attend to free tokens (so docs can see the query)
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    same_chunk = (chunk_id.unsqueeze(0) == chunk_id.unsqueeze(1)) & (chunk_id.unsqueeze(0) >= 0)
    row_free = (chunk_id < 0).unsqueeze(1).expand(-1, seq_len)
    col_free = (chunk_id < 0).unsqueeze(0).expand(seq_len, -1)
    bool_mask = causal & (same_chunk | row_free | col_free)

    # Step 3: Convert bool mask to float mask for SDPA.
    # SDPA expects 0.0 = attend, -inf = masked (added to attention logits).
    min_val = torch.finfo(dtype).min
    float_mask = torch.where(bool_mask, torch.zeros(1, dtype=dtype), torch.full((1,), min_val, dtype=dtype))
    # Shape: (1, 1, seq_len, seq_len) — batch=1, heads=1 (broadcast across heads)
    return float_mask.unsqueeze(0).unsqueeze(0)
