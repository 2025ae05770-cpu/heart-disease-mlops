"""Download and load the UCI Heart Disease (Cleveland) dataset.

Usage:
    python -m src.data.load_data            # download if missing
    python -m src.data.load_data --force    # always re-download
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import DATA_URL, RAW_COLUMNS, RAW_DATA_PATH

logger = logging.getLogger(__name__)


def download_dataset(url=DATA_URL, output_path=RAW_DATA_PATH, force=False, timeout=30):
    output_path = Path(output_path)
    if output_path.exists() and not force:
        logger.info("Dataset already present at %s", output_path)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading dataset from %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    if not response.content.strip():
        raise ValueError("Downloaded dataset is empty")

    output_path.write_bytes(response.content)
    logger.info("Saved raw dataset to %s (%d bytes)", output_path, len(response.content))
    return output_path


def load_raw_data(file_path=RAW_DATA_PATH, download_if_missing=True):
    """Load the headerless UCI file; '?' marks missing values in the source."""
    file_path = Path(file_path)
    if not file_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Raw dataset not found: {file_path}")
        download_dataset(output_path=file_path)

    data = pd.read_csv(file_path, header=None, names=RAW_COLUMNS, na_values="?")
    logger.info("Loaded raw dataset with shape %s", data.shape)
    return data


def main():
    parser = argparse.ArgumentParser(description="Download the UCI Heart Disease dataset")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    parser.add_argument("--url", default=DATA_URL)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    path = download_dataset(url=args.url, force=args.force)
    print(f"Raw dataset available at: {path}")


if __name__ == "__main__":
    main()
