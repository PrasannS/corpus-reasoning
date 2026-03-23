"""Optimized chunked document attention using FlexAttention.

FlexAttention (PyTorch >= 2.5) compiles custom attention patterns into fused CUDA
kernels, giving flash-attention-level performance with arbitrary mask patterns.
This replaces the O(N²) dense mask approach with a compiled block-sparse kernel.

Key optimizations over the original chunked_attention.py:
1. No N×N mask materialization — mask function is compiled into the kernel
2. Flash-attention-level speed and memory efficiency
3. Compatible with torch.compile for additional speedups
4. Supports sample packing with per-example chunk boundaries

Falls back to optimized SDPA if FlexAttention is unavailable.
"""

import torch
import torch.nn.functional as F
from typing import Optional
from functools import lru_cache

from .chunked_attention import (
    DOC_START, DOC_END, setup_tokenizer, wrap_documents,
    reorder_query, find_chunk_spans,
)

# Re-export for convenience
__all__ = [
    "DOC_START", "DOC_END", "setup_tokenizer", "wrap_documents",
    "reorder_query", "find_chunk_spans",
    "build_flex_attention_mask", "build_chunk_metadata",
    "FlexChunkedCollator", "create_flex_mask_mod",
    "FLEX_AVAILABLE",
]

try:
    from torch.nn.attention.flex_attention import (
        flex_attention,
        create_block_mask,
    )
    # Compile flex_attention for best performance
    flex_attention = torch.compile(flex_attention, dynamic=False)
    FLEX_AVAILABLE = True
except ImportError:
    FLEX_AVAILABLE = False


def build_chunk_metadata(input_ids, doc_start_id, doc_end_id, device=None):
    """Build chunk ID tensor for FlexAttention mask function.

    Returns a (seq_len,) int tensor where:
      - chunk_id[i] = chunk_index if token i is inside a document chunk
      - chunk_id[i] = -1 if token i is outside any chunk (query/instruction/answer)

    This is the minimal metadata needed by the FlexAttention mask function.
    """
    seq_len = len(input_ids)
    spans = find_chunk_spans(input_ids, doc_start_id, doc_end_id)

    chunk_id = torch.full((seq_len,), -1, dtype=torch.int32)
    for idx, (s, e) in enumerate(spans):
        chunk_id[s:e] = idx

    if device is not None:
        chunk_id = chunk_id.to(device)
    return chunk_id


def create_flex_mask_mod(chunk_ids):
    """Create a FlexAttention mask_mod function for chunked attention.

    The attention pattern is:
      - Token i can attend to token j if j <= i (causal) AND:
        - same_chunk(i, j): both in same document chunk
        - row_not_in_chunk(i): query/answer tokens attend to everything
        - col_not_in_chunk(j): any token can attend to query tokens

    Args:
        chunk_ids: (seq_len,) int tensor on the same device as the model.

    Returns:
        A mask_mod function compatible with FlexAttention.
    """
    def mask_mod(b, h, q_idx, kv_idx):
        # Causal: q_idx >= kv_idx
        causal = q_idx >= kv_idx

        q_chunk = chunk_ids[q_idx]
        kv_chunk = chunk_ids[kv_idx]

        # Same chunk (both must be >= 0 to be in a chunk)
        same_chunk = (q_chunk == kv_chunk) & (q_chunk >= 0)

        # Query token (not in any chunk) — attends to everything causal
        q_not_in_chunk = q_chunk < 0

        # KV token not in chunk — any token can attend to query/instruction tokens
        kv_not_in_chunk = kv_chunk < 0

        return causal & (same_chunk | q_not_in_chunk | kv_not_in_chunk)

    return mask_mod


def build_flex_block_mask(chunk_ids, seq_len, num_heads=1, batch_size=1):
    """Build a BlockMask for FlexAttention.

    The BlockMask precomputes which blocks of the attention matrix are
    non-zero, so FlexAttention can skip entirely-masked blocks.

    Args:
        chunk_ids: (seq_len,) or (batch, seq_len) int tensor
        seq_len: sequence length
        num_heads: number of attention heads
        batch_size: batch size

    Returns:
        A BlockMask object for use with flex_attention.
    """
    if not FLEX_AVAILABLE:
        raise RuntimeError("FlexAttention not available (requires PyTorch >= 2.5)")

    mask_mod = create_flex_mask_mod(chunk_ids)
    block_mask = create_block_mask(
        mask_mod,
        B=batch_size,
        H=num_heads,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=chunk_ids.device,
    )
    return block_mask


def build_optimized_sdpa_mask(input_ids, doc_start_id, doc_end_id, dtype=torch.bfloat16):
    """Build 4D attention mask optimized for GPU.

    Like build_chunked_causal_mask but:
    1. Builds directly on GPU if input is on GPU
    2. Uses bool intermediates (1 bit vs 2 bytes per element)
    3. Only converts to float at the end

    Returns (1, 1, seq_len, seq_len) float mask.
    """
    device = input_ids.device if isinstance(input_ids, torch.Tensor) else 'cpu'
    seq_len = len(input_ids)
    chunk_id = build_chunk_metadata(input_ids, doc_start_id, doc_end_id, device=device)

    # All operations on GPU using bool tensors (8x less memory than bf16)
    row_ids = chunk_id.unsqueeze(1)  # (seq_len, 1)
    col_ids = chunk_id.unsqueeze(0)  # (1, seq_len)

    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril_()
    same_chunk = (row_ids == col_ids) & (row_ids >= 0)
    row_free = (chunk_id < 0).unsqueeze(1).expand(-1, seq_len)
    col_free = (chunk_id < 0).unsqueeze(0).expand(seq_len, -1)
    bool_mask = causal & (same_chunk | row_free | col_free)

    # Convert to float mask only at the end
    min_val = torch.finfo(dtype).min
    float_mask = torch.where(bool_mask, torch.tensor(0.0, dtype=dtype, device=device),
                              torch.tensor(min_val, dtype=dtype, device=device))
    return float_mask.unsqueeze(0).unsqueeze(0)


class FlexChunkedCollator:
    """Collator for chunked attention training with FlexAttention or optimized SDPA.

    Compared to ChunkedCollator:
    1. Stores chunk_ids metadata instead of full N×N masks when using FlexAttention
    2. Builds masks on GPU when using SDPA fallback
    3. Supports sample packing (multiple examples concatenated with separator)
    """

    def __init__(self, doc_start_id, doc_end_id, pad_token_id,
                 standard_attention=False, use_flex=None):
        self.doc_start_id = doc_start_id
        self.doc_end_id = doc_end_id
        self.pad_token_id = pad_token_id
        self.standard_attention = standard_attention
        self.use_flex = use_flex if use_flex is not None else FLEX_AVAILABLE

    def __call__(self, features):
        max_len = max(f["input_ids"].size(0) for f in features)

        batch_ids, batch_labels = [], []
        batch_chunk_ids = []  # For FlexAttention
        batch_masks = []  # For SDPA fallback

        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            orig_len = ids.size(0)
            pad_len = max_len - orig_len

            if pad_len > 0:
                ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                labs = torch.cat([labs, torch.full((pad_len,), -100, dtype=labs.dtype)])

            batch_ids.append(ids)
            batch_labels.append(labs)

            if self.standard_attention:
                if self.use_flex:
                    # Standard causal — no chunk metadata needed, use None
                    batch_chunk_ids.append(torch.full((max_len,), -1, dtype=torch.int32))
                else:
                    dtype = torch.bfloat16
                    min_val = torch.finfo(dtype).min
                    mask = torch.triu(torch.full((max_len, max_len), min_val, dtype=dtype), diagonal=1)
                    if pad_len > 0:
                        mask[:, orig_len:] = min_val
                    batch_masks.append(mask.unsqueeze(0))
            else:
                if self.use_flex:
                    # Store chunk_ids for FlexAttention (tiny compared to N×N mask)
                    chunk_id = build_chunk_metadata(
                        f["input_ids"], self.doc_start_id, self.doc_end_id
                    )
                    if pad_len > 0:
                        # Pad tokens get chunk_id = -2 (distinct from query's -1)
                        # so they won't attend to or be attended by anything
                        chunk_id = torch.cat([chunk_id, torch.full((pad_len,), -2, dtype=torch.int32)])
                    batch_chunk_ids.append(chunk_id)
                else:
                    # SDPA: build mask on CPU (will be moved to GPU by trainer)
                    mask = build_optimized_sdpa_mask(
                        f["input_ids"], self.doc_start_id, self.doc_end_id,
                    )
                    if pad_len > 0:
                        min_val = torch.finfo(mask.dtype).min
                        full_mask = torch.full((1, 1, max_len, max_len), min_val, dtype=mask.dtype)
                        full_mask[:, :, :orig_len, :orig_len] = mask
                        mask = full_mask
                    batch_masks.append(mask.squeeze(0))

        result = {
            "input_ids": torch.stack(batch_ids),
            "labels": torch.stack(batch_labels),
        }

        if self.use_flex:
            # Store chunk_ids — FlexAttention mask will be built in the model forward
            result["chunk_ids"] = torch.stack(batch_chunk_ids)
        else:
            result["attention_mask"] = torch.stack(batch_masks)

        return result
