#!/usr/bin/env python3
"""Download and extract the public Kaggle dataset without credentials."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "kirtandelwadia/corporate-credit-rating-with-financial-ratios"
)
EXPECTED_FILE = "corporateCreditRatingWithFinancialRatios.csv"


def download(destination: Path) -> Path:
    """Download the source archive and copy the expected CSV."""
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / EXPECTED_FILE

    with tempfile.TemporaryDirectory() as temporary_directory:
        archive_path = Path(temporary_directory) / "dataset.zip"
        request = urllib.request.Request(
            DATASET_URL,
            headers={"User-Agent": "corporate-credit-rating-benchmark/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            archive_path.write_bytes(response.read())

        with zipfile.ZipFile(archive_path) as archive:
            candidates = [
                name
                for name in archive.namelist()
                if Path(name).name == EXPECTED_FILE
            ]
            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one {EXPECTED_FILE}; found {candidates or 'none'}."
                )
            with archive.open(candidates[0]) as source, output.open("wb") as target:
                shutil.copyfileobj(source, target)

    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Destination directory (default: data/raw).",
    )
    args = parser.parse_args()
    output = download(args.output_dir)
    print(f"Downloaded dataset to {output}")


if __name__ == "__main__":
    main()
