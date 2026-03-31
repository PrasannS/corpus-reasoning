"""Unified data format and prompt building.

All training data is stored in a structured JSONL format with documents kept
separate from formatting. Prompt construction (query position, dummy tokens,
document IDs, alpaca wrapping) happens at train/eval time via build_prompt().

Unified JSONL format:
    {
        "documents": [{"title": "...", "text": "..."}, ...],
        "queries": ["question text"],           # list, even for single-query
        "answers": ["answer text"],             # list, even for single-answer
        "gold_doc_indices": [3],                # 0-indexed positions in documents list
        "source": "nq|hotpotqa",               # dataset origin
    }

For multi-query tasks (multiple independent queries over shared documents):
    {
        "documents": [...],
        "queries": ["q1", "q2", ...],
        "answers": ["a1", "a2", ...],
        "gold_doc_indices": [[0, 2], [1, 4]],  # per-query gold doc indices
        "source": "hotpotqa",
    }

At train/eval time, build_prompt() converts this into a formatted prompt string
with the appropriate instruction, document formatting, query position, etc.
"""

import re

from lib.io import format_alpaca_prompt, insert_dummy_tokens
from lib.prompts import (
    PASSAGE_TEMPLATE, PASSAGE_TEMPLATE_NO_TITLE,
    PASSAGE_TEMPLATE_ID, PASSAGE_TEMPLATE_NO_TITLE_ID,
    QA_INSTRUCTION, MULTI_QA_INSTRUCTION,
    RETRIEVAL_INSTRUCTION_SINGLE, RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY,
    COT_RETRIEVAL_INSTRUCTION_SINGLE, COT_RETRIEVAL_INSTRUCTION_MULTI_DOC,
)


def is_multi_query(example):
    """Check if example has multiple independent queries."""
    return len(example["queries"]) > 1


def _has_multi_gold(example):
    """Check if example has multiple gold documents for any query."""
    gold = example["gold_doc_indices"]
    if not gold:
        return False
    if isinstance(gold[0], list):
        return any(len(g) > 1 for g in gold)
    return len(gold) > 1


def _get_instruction(example, task):
    """Select the appropriate instruction based on task type and query count."""
    multi = is_multi_query(example)
    if task == "cot_retrieval":
        # CoT retrieval doesn't support multi-query
        return (COT_RETRIEVAL_INSTRUCTION_MULTI_DOC if _has_multi_gold(example)
                else COT_RETRIEVAL_INSTRUCTION_SINGLE)
    elif task == "retrieval":
        if multi:
            return RETRIEVAL_INSTRUCTION_MULTI_QUERY
        return (RETRIEVAL_INSTRUCTION_MULTI_DOC if _has_multi_gold(example)
                else RETRIEVAL_INSTRUCTION_SINGLE)
    else:
        return MULTI_QA_INSTRUCTION if multi else QA_INSTRUCTION


def _format_doc(doc, use_titles=True, doc_id=None):
    """Format a single document dict using the passage template."""
    title = doc.get("title")
    text = doc["text"]
    if doc_id is not None:
        if use_titles and title:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=title, text=text)
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=text)
    if use_titles and title:
        return PASSAGE_TEMPLATE.format(title=title, text=text)
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=text)


def _format_documents(documents, task, use_titles=True):
    """Format all documents, adding [N] IDs for retrieval tasks."""
    use_ids = task in ("retrieval", "cot_retrieval")
    formatted = []
    for i, doc in enumerate(documents):
        doc_id = i + 1 if use_ids else None  # 1-indexed for retrieval
        formatted.append(_format_doc(doc, use_titles=use_titles, doc_id=doc_id))
    return "\n\n".join(formatted)


def remap_cot_doc_ids(cot_text, id_mapping):
    """Remap document IDs in CoT text when document positions change.

    Used when scaling to more documents or reshuffling: the CoT was generated
    with documents at certain positions, but at training time positions differ.

    Args:
        cot_text: The chain-of-thought string containing references like
            "Document [3]", "[7]", etc.
        id_mapping: Dict mapping old 1-indexed IDs to new 1-indexed IDs.
            E.g., {3: 45, 7: 72} means old Document [3] is now Document [45].

    Returns:
        CoT text with all [N] references remapped.
    """
    if not id_mapping or not cot_text:
        return cot_text

    def _replace_id(match):
        old_id = int(match.group(1))
        new_id = id_mapping.get(old_id, old_id)
        return f"[{new_id}]"

    return re.sub(r'\[(\d+)\]', _replace_id, cot_text)


def _build_retrieval_ids(gold):
    """Format gold doc indices as 1-indexed ID string: '[3]' or '[3], [7]'."""
    if isinstance(gold[0], list):
        gids = gold[0]
    else:
        gids = gold
    return ", ".join(f"[{g + 1}]" for g in sorted(gids))


def _build_output(example, task):
    """Build the expected output string from the structured example."""
    if task == "cot_retrieval":
        gold = example["gold_doc_indices"]
        cot = example.get("chain_of_thought", "")
        ids_str = _build_retrieval_ids(gold)
        has_multi = _has_multi_gold(example)
        prefix = "Relevant Documents" if has_multi else "Relevant Document"
        # Remap doc IDs in CoT if positions have changed (e.g., scaled to more docs)
        id_mapping = example.get("cot_id_mapping")
        if cot and id_mapping:
            cot = remap_cot_doc_ids(cot, id_mapping)
        if cot:
            return f"{cot}\n{prefix}: {ids_str}"
        else:
            # Fallback: no CoT available, just output IDs
            return f"{prefix}: {ids_str}"
    elif task == "retrieval":
        gold = example["gold_doc_indices"]
        if is_multi_query(example):
            # Multi-query: "Q1: [3], [7]; Q2: [1], [5]; ..."
            parts = []
            for qi, gids in enumerate(gold):
                ids_str = ", ".join(f"[{g + 1}]" for g in sorted(gids))  # 0→1 indexed
                parts.append(f"Q{qi + 1}: {ids_str}")
            return "; ".join(parts)
        else:
            return _build_retrieval_ids(gold)
    else:
        # QA task
        if is_multi_query(example):
            return ", ".join(example["answers"])
        else:
            return example["answers"][0]


def _build_questions_block(queries):
    """Format the question(s) section of the prompt."""
    if len(queries) == 1:
        return f"Question: {queries[0]}"
    return "\n".join(f"Question {i+1}: {q}" for i, q in enumerate(queries))


def build_prompt(example, task="retrieval", query_position="after",
                 use_titles=True, before_dummy=0, after_dummy=0,
                 use_alpaca=True):
    """Build a formatted prompt + output from a unified example.

    This is the single entry point for converting structured data into
    the text format consumed by training and evaluation.

    Args:
        example: Dict with unified format (documents, queries, answers, gold_doc_indices).
        task: "retrieval" (output doc IDs) or "qa" (output answer text).
        query_position: "after" (default), "before", or "both".
        use_titles: Whether to include document titles.
        before_dummy: Number of dummy token repetitions before documents.
        after_dummy: Number of dummy token repetitions after documents.
        use_alpaca: Whether to wrap in alpaca template (True for trained models).

    Returns:
        (prompt, output) tuple of strings.
    """
    docs = example["documents"]
    queries = example["queries"]

    # Handle no-document (closed-book) case
    if not docs:
        instruction = _get_instruction(example, task)
        questions = _build_questions_block(queries)
        output = _build_output(example, task)
        if use_alpaca:
            prompt = format_alpaca_prompt(instruction, questions)
        else:
            prompt = f"{instruction}\n\n{questions}\n"
        return prompt, output

    # Format documents and questions
    context = _format_documents(docs, task, use_titles=use_titles)
    questions = _build_questions_block(queries)

    # Arrange query position relative to documents
    if query_position == "before":
        input_text = f"{questions}\n\n{context}"
    elif query_position == "both":
        input_text = f"{questions}\n\n{context}\n\n{questions}"
    else:  # "after" (default)
        input_text = f"{context}\n\n{questions}"

    # Insert dummy tokens if requested (positional ablation)
    if before_dummy > 0 or after_dummy > 0:
        input_text = insert_dummy_tokens(input_text, before_dummy, after_dummy)

    # Build instruction and output
    instruction = _get_instruction(example, task)
    output = _build_output(example, task)

    # Wrap in alpaca template or plain format
    if use_alpaca:
        prompt = format_alpaca_prompt(instruction, input_text)
    else:
        prompt = f"{instruction}\n\n{input_text}\n"

    return prompt, output
