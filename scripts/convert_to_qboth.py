"""Convert NQ training data from query-after to query-both format.

Reads JSONL where each example has input = "docs...\n\nQuestion: ..."
and outputs with input = "Question: ...\n\ndocs...\n\nQuestion: ..."

Usage:
    python scripts/convert_to_qboth.py data/nq_train_k20_random.jsonl data/nq_train_k20_random_50000_qboth.jsonl
"""

import json
import re
import sys
from pathlib import Path

def convert_to_qboth(input_text: str) -> str:
    """Move question to both before and after documents."""
    m = re.search(r'\n\n(Question:.*)$', input_text)
    if m:
        question = m.group(1)
        docs = input_text[:m.start()]
        return question + "\n\n" + docs + "\n\n" + question
    return input_text

def main():
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    count = 0
    with open(in_path) as fin, open(out_path, 'w') as fout:
        for line in fin:
            ex = json.loads(line)
            ex["input"] = convert_to_qboth(ex["input"])
            fout.write(json.dumps(ex) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  Converted {count} examples...")

    print(f"Done: {count} examples -> {out_path}")
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
