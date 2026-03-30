"""Chunked attention evaluation with FlexAttention -- backward-compat wrapper.

This script has been merged into evaluate_chunked.py, which now supports both
SDPA and FlexAttention backends via --backend {sdpa,flex}.

This file remains as an alias so existing scripts that reference
evaluate_chunked_fast.py continue to work. It defaults to --backend flex.
"""

import sys

# Inject --backend flex as default if not already specified
if "--backend" not in sys.argv:
    sys.argv.insert(1, "--backend")
    sys.argv.insert(2, "flex")

from evaluate_chunked import main

if __name__ == "__main__":
    main()
