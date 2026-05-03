#!/usr/bin/env python
"""Download locomo10.json from the snap-research GitHub release."""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.config import DATASET_PATH

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"


def main() -> None:
    if DATASET_PATH.exists():
        print(f"Already exists: {DATASET_PATH}")
        return

    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    try:
        urllib.request.urlretrieve(URL, DATASET_PATH)
        print(f"Saved to {DATASET_PATH}")
    except Exception as e:
        print(f"Download failed: {e}")
        print(f"Download manually and place at: {DATASET_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
