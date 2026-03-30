"""Training with chunked document attention -- backward-compat wrapper.

This script has been merged into train_chunked_fast.py, which is the canonical
version with all features (optimized SDPA masks, LoRA + full FT, tf32, etc.).

This file remains as an alias so existing configs and sbatch scripts that
reference train_chunked.py continue to work.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_chunked_fast import main

if __name__ == "__main__":
    main()
