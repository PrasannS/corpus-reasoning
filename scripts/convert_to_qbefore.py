"""Convert query-after training data to query-before format.

This is a compatibility wrapper. Prefer using convert_query_position.py directly:
    python scripts/convert_query_position.py --mode before data/input.jsonl data/output.jsonl

Usage:
    python scripts/convert_to_qbefore.py data/input.jsonl data/output.jsonl
"""
import sys
from convert_query_position import convert_to_qbefore, main as _main

if __name__ == "__main__":
    # Rewrite argv to match convert_query_position.py's interface
    if len(sys.argv) == 3:
        sys.argv = [sys.argv[0], "--mode", "before", sys.argv[1], sys.argv[2]]
    _main()
