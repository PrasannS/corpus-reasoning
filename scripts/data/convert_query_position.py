"""Convert query position in training data between after/before/both formats.

Training data is generated with query-after format by default:
    {context}\n\nQuestion: {question}

This script converts to other query positions:
  - "before":  Question: {question}\n\n{context}
  - "both":    Question: {question}\n\n{context}\n\nQuestion: {question}

Usage:
    python scripts/convert_query_position.py --mode before data/input.jsonl data/output.jsonl
    python scripts/convert_query_position.py --mode both   data/input.jsonl data/output.jsonl
"""

import argparse
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # scripts/ — for lib.*
sys.path.insert(0, str(Path(__file__).resolve().parent))  # same subdir — for sibling imports

from lib.io import load_jsonl, save_jsonl


def convert_to_qbefore(input_text: str) -> str:
    """Move question from after documents to before."""
    idx = input_text.rfind("\n\nQuestion: ")
    if idx == -1:
        return input_text  # No question found (e.g. closed-book), leave as-is
    context = input_text[:idx]
    question_part = input_text[idx + 2:]  # "Question: ..."
    return f"{question_part}\n\n{context}"


def convert_to_qboth(input_text: str) -> str:
    """Duplicate question to appear both before and after documents."""
    m = re.search(r'\n\n(Question:.*)$', input_text)
    if m:
        question = m.group(1)
        docs = input_text[:m.start()]
        return f"{question}\n\n{docs}\n\n{question}"
    return input_text


CONVERTERS = {
    "before": convert_to_qbefore,
    "both": convert_to_qboth,
}


def main():
    parser = argparse.ArgumentParser(description="Convert query position in training data")
    parser.add_argument("input", help="Input JSONL file (query-after format)")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument("--mode", required=True, choices=["before", "both"],
                        help="Target query position: 'before' or 'both'")
    args = parser.parse_args()

    convert_fn = CONVERTERS[args.mode]
    data = load_jsonl(args.input)
    converted = []
    for i, ex in enumerate(data):
        ex["input"] = convert_fn(ex["input"])
        converted.append(ex)
        if (i + 1) % 10000 == 0:
            print(f"  Converted {i + 1} examples...")

    save_jsonl(args.output, converted)
    print(f"Done: {len(converted)} examples -> {args.output}")
    size_mb = __import__('os').path.getsize(args.output) / 1e6
    print(f"  File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
