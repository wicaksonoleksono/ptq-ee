"""
00_download_data.py
-------------------
Pre-downloads all datasets used in the benchmark to HuggingFace cache.
Run once before starting experiments. No GPU needed.

Usage:
    python scripts/00_download_data.py
"""

import json
import os
import sys
from pathlib import Path

# Load config
CONFIG_PATH = Path(__file__).parent / "experiment_config.json"
with open(CONFIG_PATH) as f:
    CFG = json.load(f)

CACHE_DIR = (Path(__file__).parent / CFG["paths"]["model_cache_dir"]).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Map task names to (hf_dataset_id, subset, split) tuples
DATASETS_TO_DOWNLOAD = [
    ("cnn_dailymail",                          "3.0.0",                   "test"),
    ("xsum",                                   None,                      "test"),
    ("openai_humaneval",                       None,                      "test"),
    ("wikitext",                               "wikitext-2-raw-v1",       "validation"),
    ("Rowan/hellaswag",                        None,                      "validation"),
    ("allenai/ai2_arc",                        "ARC-Challenge",           "test"),
    ("cais/mmlu",                              "all",                     "test"),
    ("gsm8k",                                  "main",                    "test"),
]


def download_dataset(name, subset, split):
    from datasets import load_dataset
    label = f"{name}" + (f"/{subset}" if subset else "") + f" [{split}]"
    try:
        ds = load_dataset(name, subset, split=split, cache_dir=str(CACHE_DIR))
        print(f"  OK  {label}  ({len(ds)} examples)")
    except Exception as e:
        print(f"  FAIL {label}: {e}")


def main():
    try:
        from datasets import load_dataset  # noqa: F401
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print("=" * 60)
    print("Downloading benchmark datasets")
    print(f"Cache dir: {CACHE_DIR}")
    print("=" * 60)

    for name, subset, split in DATASETS_TO_DOWNLOAD:
        download_dataset(name, subset, split)

    print("\nDone. All datasets cached.")


if __name__ == "__main__":
    main()
