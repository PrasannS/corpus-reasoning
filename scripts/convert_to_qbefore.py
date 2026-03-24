"""Convert query-after training data to query-before format.

Transforms:
    {context}\n\nQuestion: {question}
Into:
    Question: {question}\n\n{context}

Usage:
    python scripts/convert_to_qbefore.py data/input.jsonl data/output.jsonl
"""

import sys
from lib.io import load_jsonl, save_jsonl


def convert_to_qbefore(example):
    text = example["input"]
    # Find the last "Question: " which is the actual question (after all docs)
    idx = text.rfind("\n\nQuestion: ")
    if idx == -1:
        return example  # Already no-context format, leave as-is
    context = text[:idx]
    question_part = text[idx + 2:]  # "Question: ..."
    return {
        "instruction": example["instruction"],
        "input": f"{question_part}\n\n{context}",
        "output": example["output"],
    }


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.jsonl> <output.jsonl>")
        sys.exit(1)

    data = load_jsonl(sys.argv[1])
    converted = []
    for i, ex in enumerate(data):
        converted.append(convert_to_qbefore(ex))
        if (i + 1) % 10000 == 0:
            print(f"  Converted {i + 1} examples...")

    save_jsonl(sys.argv[2], converted)
    print(f"Done: {len(converted)} examples -> {sys.argv[2]}")
    print(f"  File size: {__import__('os').path.getsize(sys.argv[2]) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
