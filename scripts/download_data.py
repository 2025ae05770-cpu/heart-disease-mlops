"""Standalone dataset download script (deliverable: "download script").

    python scripts/download_data.py           # downloads + cleans
    python scripts/download_data.py --force   # re-download
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.clean_data import run_cleaning  # noqa: E402
from src.data.load_data import download_dataset  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    raw_path = download_dataset(force=args.force)
    clean, _ = run_cleaning()
    print(f"raw:   {raw_path}")
    print(f"clean: {clean.shape[0]} rows x {clean.shape[1]} columns -> data/processed/heart_disease_clean.csv")
