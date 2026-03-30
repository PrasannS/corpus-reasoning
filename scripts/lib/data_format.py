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

from lib.io import format_alpaca_prompt, insert_dummy_tokens
from lib.prompts import (
    PASSAGE_TEMPLATE, PASSAGE_TEMPLATE_NO_TITLE,
    PASSAGE_TEMPLATE_ID, PASSAGE_TEMPLATE_NO_TITLE_ID,
    QA_INSTRUCTION, MULTI_QA_INSTRUCTION,
    RETRIEVAL_INSTRUCTION_SINGLE, RETRIEVAL_INSTRUCTION_MULTI_DOC,
    RETRIEVAL_INSTRUCTION_MULTI_QUERY,
)


def is_multi_query(example):
    """Check if example has multiple independent queries."""
    return len(example["queries"]) > 1


def _get_instruction(example, task):
    """Select the appropriate instruction based on task type and query count."""
    multi = is_multi_query(example)
    if task == "retrieval":
        if multi:
            return RETRIEVAL_INSTRUCTION_MULTI_QUERY
        # Single vs multi-doc: check if any query has >1 gold doc
        gold = example["gold_doc_indices"]
        has_multi_gold = isinstance(gold[0], list) and any(len(g) > 1 for g in gold)
        if not isinstance(gold[0], list):
            has_multi_gold = len(gold) > 1
        return RETRIEVAL_INSTRUCTION_MULTI_DOC if has_multi_gold else RETRIEVAL_INSTRUCTION_SINGLE
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
    use_ids = (task == "retrieval")
    formatted = []
    for i, doc in enumerate(documents):
        doc_id = i + 1 if use_ids else None  # 1-indexed for retrieval
        formatted.append(_format_doc(doc, use_titles=use_titles, doc_id=doc_id))
    return "\n\n".join(formatted)


def _build_output(example, task):
    """Build the expected output string from the structured example."""
    if task == "retrieval":
        gold = example["gold_doc_indices"]
        if is_multi_query(example):
            # Multi-query: "Q1: [3], [7]; Q2: [1], [5]; ..."
            parts = []
            for qi, gids in enumerate(gold):
                ids_str = ", ".join(f"[{g + 1}]" for g in sorted(gids))  # 0→1 indexed
                parts.append(f"Q{qi + 1}: {ids_str}")
            return "; ".join(parts)
        else:
            # Single/multi-doc: "[3]" or "[3], [7]"
            if isinstance(gold[0], list):
                # Shouldn't happen for single-query, but handle it
                gids = gold[0]
            else:
                gids = gold
            return ", ".join(f"[{g + 1}]" for g in sorted(gids))
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


def build_alpaca_example(example, task="retrieval", query_position="after",
                         use_titles=True):
    """Build an alpaca-format dict (instruction, input, output) from unified example.

    This is a convenience for generating training data in the old alpaca format,
    useful for Axolotl standard attention training which expects this format.
    """
    docs = example["documents"]
    queries = example["queries"]
    instruction = _get_instruction(example, task)
    output = _build_output(example, task)

    if not docs:
        input_text = _build_questions_block(queries)
    else:
        context = _format_documents(docs, task, use_titles=use_titles)
        questions = _build_questions_block(queries)
        if query_position == "before":
            input_text = f"{questions}\n\n{context}"
        elif query_position == "both":
            input_text = f"{questions}\n\n{context}\n\n{questions}"
        else:
            input_text = f"{context}\n\n{questions}"

    return {"instruction": instruction, "input": input_text, "output": output}
