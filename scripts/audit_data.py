#!/usr/bin/env python3
"""Generate a reproducible source-data audit and target-distribution chart."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib")
)

import matplotlib.pyplot as plt

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_rating.audit import build_data_audit  # noqa: E402
from credit_rating.data import (  # noqa: E402
    SEVEN_BAND_ORDER,
    load_credit_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/corporateCreditRatingWithFinancialRatios.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
    )
    args = parser.parse_args()

    frame = load_credit_data(args.input)
    audit = build_data_audit(frame)

    metrics_dir = args.output_dir / "metrics"
    charts_dir = args.output_dir / "charts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "data_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )

    counts = (
        frame["target_7band"]
        .value_counts()
        .reindex(SEVEN_BAND_ORDER)
        .fillna(0)
    )
    _, axis = plt.subplots(figsize=(9, 5))
    counts.plot.bar(ax=axis, color="#2f6f9f")
    axis.set_title("Historical rating observations by seven-band target")
    axis.set_xlabel("Rating band")
    axis.set_ylabel("Observations")
    axis.tick_params(axis="x", rotation=0)
    axis.set_xticklabels(
        ["AAA", "AA", "A", "BBB", "BB", "B", "CCC or lower"]
    )
    for position, value in enumerate(counts):
        axis.text(position, value + 25, f"{int(value):,}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(charts_dir / "seven_band_distribution.png", dpi=180)
    plt.close()

    print(
        f"Audited {audit['rows']:,} rows; "
        f"{audit['label_ambiguity']['predictor_vectors_with_multiple_ratings']:,} "
        "predictor vectors map to multiple ratings."
    )


if __name__ == "__main__":
    main()
