#!/usr/bin/env python
"""Sanity-check locomo10.json. Prints counts and halts if they don't match expectations."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.config import DATASET_PATH, EVAL_CATEGORIES
from src.data import dataset_stats, load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DATASET_PATH)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: dataset not found at {args.path}")
        print("Download from: https://github.com/snap-research/locomo → data/locomo10.json")
        print(f"Place at: {DATASET_PATH}")
        sys.exit(1)

    print(f"Loading {args.path} ...")
    conversations = load_dataset(args.path)
    stats = dataset_stats(conversations, EVAL_CATEGORIES)

    print(f"\nConversations : {stats['conversations']}")
    print(f"Total QA      : {stats['total_qa']}")
    print(f"Filtered QA   : {stats['filtered_qa']}  (categories {sorted(EVAL_CATEGORIES)})")

    ok = True
    if stats["conversations"] != 10:
        print(f"WARNING: expected 10 conversations, got {stats['conversations']}")
        ok = False
    if stats["total_qa"] != 1986:
        print(f"WARNING: expected 1986 total QA, got {stats['total_qa']}")
        ok = False
    if stats["filtered_qa"] != 1540:
        print(f"WARNING: expected ~1540 filtered QA, got {stats['filtered_qa']}")
        ok = False

    if ok:
        print("\nAll counts match. Dataset looks good.")
    else:
        print("\nCounts don't match expectations — inspect before running eval.")
        sys.exit(1)


if __name__ == "__main__":
    main()
