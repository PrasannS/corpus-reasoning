"""Shared I/O utilities and prompt templates."""

import json
import re
from pathlib import Path


# Alpaca prompt template — used by training (Axolotl) and evaluation scripts.
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input "
    "that provides further context. Write a response that appropriately "
    "completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "### Input:\n{input}\n\n"
    "### Response:\n"
)


def format_alpaca_prompt(instruction: str, input_text: str) -> str:
    return ALPACA_TEMPLATE.format(instruction=instruction, input=input_text)


def load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: str, examples: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def save_results(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def insert_dummy_tokens(input_text: str, before_dummy: int = 0, after_dummy: int = 0,
                        dummy_token: str = "* ") -> str:
    """Insert dummy tokens before and/or after the document block in the input text.

    Args:
        input_text: The 'input' field of an alpaca example (docs + question).
        before_dummy: Number of dummy token repetitions to insert before documents.
        after_dummy: Number of dummy token repetitions to insert after documents.
        dummy_token: The token string to repeat (default "* ").

    Returns:
        Modified input text with dummy tokens inserted.
    """
    if before_dummy == 0 and after_dummy == 0:
        return input_text

    # Find first document (handles all formats: "Document (Title:", "Document:", "Document [")
    doc_match = re.search(r'Document[\s\[\(:]', input_text)
    if not doc_match:
        return input_text
    doc_start_idx = doc_match.start()

    # Find end of document block (trailing question)
    doc_end_idx = len(input_text)
    # Single-query trailing question
    trailing = input_text.rfind("\n\nQuestion:")
    if trailing > doc_start_idx:
        doc_end_idx = trailing
    else:
        # Multi-query trailing questions
        trailing = input_text.rfind("\nQuestion 1:")
        if trailing > doc_start_idx:
            doc_end_idx = trailing

    before_text = input_text[:doc_start_idx]
    doc_text = input_text[doc_start_idx:doc_end_idx]
    after_text = input_text[doc_end_idx:]

    result = before_text
    if before_dummy > 0:
        result += dummy_token * before_dummy + "\n\n"
    result += doc_text
    if after_dummy > 0:
        result += "\n\n" + dummy_token * after_dummy
    result += after_text

    return result


def print_dataset_stats(examples: list[dict], label: str, path: str) -> None:
    if not examples:
        print(f"\n{label}: 0 examples (skipped)")
        return
    input_lens = [len(ex.get("input", "")) for ex in examples]
    output_lens = [len(ex.get("output", "")) for ex in examples]
    size_mb = Path(path).stat().st_size / 1024 / 1024
    print(f"\n{label}: {len(examples)} examples -> {path}")
    print(f"  Avg input:  {sum(input_lens)/len(input_lens):,.0f} chars")
    print(f"  Avg output: {sum(output_lens)/len(output_lens):,.0f} chars")
    print(f"  File size:  {size_mb:.1f} MB")
