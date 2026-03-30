"""Insert dummy tokens before and/or after the document block in training data.

Adds "* " repeated N times before documents (after query/instruction) and/or
after documents (before final answer). Works with any query position format
(qafter, qbefore, qboth).

Optionally verifies token counts with a specified tokenizer to ensure each
"* " repetition adds exactly the expected number of tokens.

Usage:
    python scripts/convert_to_dummy.py data/input.jsonl data/output.jsonl --before-dummy 10
    python scripts/convert_to_dummy.py data/input.jsonl data/output.jsonl --after-dummy 10
    python scripts/convert_to_dummy.py data/input.jsonl data/output.jsonl --before-dummy 10 --after-dummy 10

    # Verify token counts with a specific tokenizer
    python scripts/convert_to_dummy.py data/input.jsonl data/output.jsonl --before-dummy 10 \
        --tokenizer Qwen/Qwen3.5-0.8B-Base
"""

import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from lib.io import insert_dummy_tokens, load_jsonl, save_jsonl, print_dataset_stats


def verify_token_count(tokenizer_name, before_dummy, after_dummy, dummy_token="* "):
    """Verify that dummy tokens produce the expected number of extra tokens."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Test with a realistic context snippet
    base_text = "Document (Title: Example): Some text here.\n\nQuestion: What is this?"
    modified = insert_dummy_tokens(base_text, before_dummy, after_dummy, dummy_token)

    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    modified_tokens = tokenizer.encode(modified, add_special_tokens=False)
    extra_tokens = len(modified_tokens) - len(base_tokens)

    # Expected: each dummy insertion adds some tokens for the dummy string + separator \n\n
    # Count tokens in just the dummy string itself
    for label, count in [("before", before_dummy), ("after", after_dummy)]:
        if count == 0:
            continue
        dummy_str = dummy_token * count
        dummy_tokens = tokenizer.encode(dummy_str, add_special_tokens=False)
        sep_tokens = tokenizer.encode("\n\n", add_special_tokens=False)
        print(f"  {label}_dummy={count}: '{dummy_token}' x {count} = {len(dummy_tokens)} tokens")
        print(f"    Dummy string: {repr(dummy_str)}")
        print(f"    Token IDs: {dummy_tokens}")
        print(f"    Separator '\\n\\n': {len(sep_tokens)} tokens")
        if len(dummy_tokens) != count:
            print(f"    WARNING: Expected {count} tokens, got {len(dummy_tokens)}. "
                  f"Each '{dummy_token}' is not exactly 1 token for this tokenizer.")

    print(f"  Total extra tokens in context: {extra_tokens}")
    print(f"  Base tokens: {len(base_tokens)}, Modified tokens: {len(modified_tokens)}")


def main():
    parser = argparse.ArgumentParser(description="Insert dummy tokens into training data")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("output", help="Output JSONL file")
    parser.add_argument("--before-dummy", type=int, default=0,
                        help="Number of dummy token repetitions before documents")
    parser.add_argument("--after-dummy", type=int, default=0,
                        help="Number of dummy token repetitions after documents")
    parser.add_argument("--dummy-token", type=str, default="* ",
                        help="Dummy token string to repeat (default: '* ')")
    parser.add_argument("--tokenizer", type=str, default="",
                        help="Tokenizer name to verify token counts (e.g. Qwen/Qwen3.5-0.8B-Base)")
    args = parser.parse_args()

    if args.before_dummy == 0 and args.after_dummy == 0:
        parser.error("Specify at least one of --before-dummy or --after-dummy")

    # Verify token counts if tokenizer specified
    if args.tokenizer:
        print(f"Verifying token counts with {args.tokenizer}...")
        verify_token_count(args.tokenizer, args.before_dummy, args.after_dummy, args.dummy_token)
        print()

    # Convert
    print(f"Converting {args.input} -> {args.output}")
    print(f"  before_dummy={args.before_dummy}, after_dummy={args.after_dummy}, "
          f"dummy_token={repr(args.dummy_token)}")

    count = 0
    with open(args.input) as fin, open(args.output, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            ex = json.loads(line)
            ex["input"] = insert_dummy_tokens(
                ex["input"], args.before_dummy, args.after_dummy, args.dummy_token
            )
            fout.write(json.dumps(ex) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  Converted {count} examples...")

    print(f"Done: {count} examples -> {args.output}")
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"  File size: {size_mb:.1f} MB")

    # Show a sample
    examples = load_jsonl(args.output)
    if examples:
        ex = examples[0]
        inp = ex["input"]
        # Show first 300 chars and last 200 chars
        print(f"\n=== Sample (first example) ===")
        print(f"  Input start: {inp[:300]}...")
        print(f"  Input end:   ...{inp[-200:]}")


if __name__ == "__main__":
    main()
