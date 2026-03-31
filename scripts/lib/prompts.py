"""Shared prompt templates and task instructions.

All data generation scripts, evaluation scripts, and example generators import
from here to ensure training and eval prompts stay in sync.

There are two axes of variation:
  1. Task type: QA (answer the question) vs Retrieval (identify relevant doc IDs)
  2. Task scope: single-doc (NQ), multi-doc (HotpotQA), multi-query (multi-HotpotQA)

Passage templates control how individual documents are formatted within the prompt.
The _ID variants add numeric document identifiers for the retrieval task.
"""

# ── Passage templates ──
# Used by all data generation and eval scripts to format individual documents.
# The base template matches HELMET eval format exactly.
PASSAGE_TEMPLATE = "Document (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE = "Document: {text}"
PASSAGE_TEMPLATE_ID = "Document [{id}] (Title: {title}): {text}"
PASSAGE_TEMPLATE_NO_TITLE_ID = "Document [{id}]: {text}"

# ── QA task instructions ──
# Single-query QA (used by NQ and HotpotQA): model outputs a short answer.
QA_INSTRUCTION = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]"
)

# Multi-query QA (used by multi-HotpotQA): model outputs comma-separated answers.
MULTI_QA_INSTRUCTION = (
    "Use the given documents to answer each of the following questions. "
    "Write a concise and short answer for each question, in order, as a comma-separated list.\n"
    "Write your answer in the following format:\nAnswers: [answer1], [answer2], ..."
)

# ── Retrieval task instructions ──
# Single-doc retrieval (NQ): model identifies one relevant document.
RETRIEVAL_INSTRUCTION_SINGLE = (
    "Use the given documents to identify which document is most relevant to "
    "answering the question.\n"
    "Write your answer in the following format:\nRelevant Document: [id]"
)

# Multi-doc retrieval (HotpotQA): model identifies multiple relevant documents.
RETRIEVAL_INSTRUCTION_MULTI_DOC = (
    "Use the given documents to identify which documents are relevant to "
    "answering the question. List all relevant document IDs.\n"
    "Write your answer in the following format:\nRelevant Documents: [id1], [id2]"
)

# CoT retrieval: single-doc (NQ) — reason about relevance, then output ID.
COT_RETRIEVAL_INSTRUCTION_SINGLE = (
    "Use the given documents to identify which document is most relevant to "
    "answering the question. Think step by step about why the document is "
    "relevant, then give your answer.\n"
    "Write your answer in the following format:\n"
    "[chain of thought reasoning]\n"
    "Relevant Document: [id]"
)

# CoT retrieval: multi-doc (HotpotQA) — reason about relevance, then output IDs.
COT_RETRIEVAL_INSTRUCTION_MULTI_DOC = (
    "Use the given documents to identify which documents are relevant to "
    "answering the question. Think step by step about why the documents are "
    "relevant, then list all relevant document IDs.\n"
    "Write your answer in the following format:\n"
    "[chain of thought reasoning]\n"
    "Relevant Documents: [id1], [id2]"
)

# Multi-query retrieval (multi-HotpotQA): per-query relevant document IDs.
RETRIEVAL_INSTRUCTION_MULTI_QUERY = (
    "Use the given documents to identify which documents are relevant to "
    "answering each of the following questions. For each question, list the "
    "relevant document IDs.\n"
    "Write your answer in the following format:\n"
    "Relevant Documents: Q1: [id1], [id2]; Q2: [id3], [id4]; ..."
)

# ── HELMET base-model eval templates (non-alpaca) ──
# These are used when evaluating base models (no fine-tuning) with few-shot demos.
# Trained models use the alpaca template from lib/io.py instead.
DEMO_TEMPLATE = "{documents}\n\nQuestion: {question}\nAnswer: {answer}"

HELMET_TEMPLATE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}{context}\n\nQuestion: {question}"
)
HELMET_TEMPLATE_QUERY_BEFORE = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}Question: {question}\n\n{context}"
)
HELMET_TEMPLATE_QUERY_BOTH = (
    "Use the given documents to write a concise and short answer to the question. "
    "Write your answer in the following format:\nAnswer: [answer]\n\n"
    "{demos}Question: {question}\n\n{context}\n\nQuestion: {question}"
)


def format_doc(text, title=None, use_titles=True, doc_id=None):
    """Format a document string using the appropriate passage template.

    Args:
        text: Document text content.
        title: Document title (optional).
        use_titles: Whether to include the title in the formatted output.
        doc_id: Numeric document ID for retrieval tasks. When provided,
                uses the _ID template variants that show [id] before the doc.

    Returns:
        Formatted document string matching the HELMET passage template format.
    """
    if doc_id is not None:
        if use_titles and title:
            return PASSAGE_TEMPLATE_ID.format(id=doc_id, title=title, text=text)
        return PASSAGE_TEMPLATE_NO_TITLE_ID.format(id=doc_id, text=text)
    if use_titles and title:
        return PASSAGE_TEMPLATE.format(title=title, text=text)
    return PASSAGE_TEMPLATE_NO_TITLE.format(text=text)


def format_doc_dict(doc, use_titles=True, doc_id=None):
    """Format a document dict (with 'title' and 'text' keys) using the passage template.

    Convenience wrapper around format_doc() for HotpotQA-style document dicts.
    """
    title = doc.get("title") if use_titles else None
    return format_doc(doc["text"], title=title, use_titles=use_titles, doc_id=doc_id)
