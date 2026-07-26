"""Data validation, target construction, and leakage-aware splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

NUMERIC_FEATURES = [
    "Current Ratio",
    "Long-term Debt / Capital",
    "Debt/Equity Ratio",
    "Gross Margin",
    "Operating Margin",
    "EBIT Margin",
    "EBITDA Margin",
    "Pre-Tax Profit Margin",
    "Net Profit Margin",
    "Asset Turnover",
    "ROE - Return On Equity",
    "Return On Tangible Equity",
    "ROA - Return On Assets",
    "ROI - Return On Investment",
    "Operating Cash Flow Per Share",
    "Free Cash Flow Per Share",
]

CATEGORICAL_FEATURES = ["Rating Agency", "Sector", "SIC Code"]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

INVESTMENT_GRADE = {
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
}

RATING_TO_BAND = {
    "AAA": "AAA",
    "AA+": "AA",
    "AA": "AA",
    "AA-": "AA",
    "A+": "A",
    "A": "A",
    "A-": "A",
    "BBB+": "BBB",
    "BBB": "BBB",
    "BBB-": "BBB",
    "BB+": "BB",
    "BB": "BB",
    "BB-": "BB",
    "B+": "B",
    "B": "B",
    "B-": "B",
    "CCC+": "CCC_or_lower",
    "CCC": "CCC_or_lower",
    "CCC-": "CCC_or_lower",
    "CC+": "CCC_or_lower",
    "CC": "CCC_or_lower",
    "C": "CCC_or_lower",
    "D": "CCC_or_lower",
}

SEVEN_BAND_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC_or_lower"]

IDENTIFIER_COLUMNS = ["Corporation", "CIK", "Ticker"]
FORBIDDEN_MODEL_FEATURES = {
    "Rating",
    "Binary Rating",
    "target_binary",
    "target_7band",
    "Corporation",
    "CIK",
    "Ticker",
    "Rating Date",
    "rating_year",
}

REQUIRED_COLUMNS = set(
    MODEL_FEATURES
    + IDENTIFIER_COLUMNS
    + ["Rating", "Rating Date", "Binary Rating"]
)


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological development, validation, and holdout partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def validate_schema(frame: pd.DataFrame) -> None:
    """Raise a useful error when the source schema is incomplete."""
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def load_credit_data(path: str | Path) -> pd.DataFrame:
    """Load the public dataset and construct leakage-free targets."""
    frame = pd.read_csv(path)
    validate_schema(frame)

    frame = frame.copy()
    frame["Rating"] = frame["Rating"].astype(str).str.strip()
    unknown = sorted(set(frame["Rating"]).difference(RATING_TO_BAND))
    if unknown:
        raise ValueError(f"Unmapped rating labels: {unknown}")

    frame["Rating Date"] = pd.to_datetime(
        frame["Rating Date"], errors="raise", format="mixed"
    )
    frame["rating_year"] = frame["Rating Date"].dt.year.astype(int)
    frame["SIC Code"] = frame["SIC Code"].round().astype("Int64").astype(str)
    frame["target_binary"] = frame["Rating"].map(
        lambda rating: (
            "investment_grade"
            if rating in INVESTMENT_GRADE
            else "speculative_grade"
        )
    )
    frame["target_7band"] = frame["Rating"].map(RATING_TO_BAND)

    if frame[MODEL_FEATURES].isna().any().any():
        # The pipeline can impute missing values, but keeping this branch explicit
        # makes unexpected source changes visible in the quality report.
        frame.attrs["missing_model_values"] = int(
            frame[MODEL_FEATURES].isna().sum().sum()
        )
    else:
        frame.attrs["missing_model_values"] = 0

    return frame.sort_values(["Rating Date", "CIK", "Rating Agency"]).reset_index(
        drop=True
    )


def split_temporally(
    frame: pd.DataFrame,
    train_end_year: int = 2014,
    validation_year: int = 2015,
    test_year: int = 2016,
) -> TemporalSplit:
    """Create a forward-looking split without random date mixing."""
    if not train_end_year < validation_year < test_year:
        raise ValueError("Expected train_end_year < validation_year < test_year.")

    train = frame.loc[frame["rating_year"] <= train_end_year].copy()
    validation = frame.loc[frame["rating_year"] == validation_year].copy()
    test = frame.loc[frame["rating_year"] == test_year].copy()

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty partition; check available years."
        )

    return TemporalSplit(train=train, validation=validation, test=test)


def get_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only fields available to the model at prediction time."""
    features = frame.loc[:, MODEL_FEATURES].copy()
    overlap = FORBIDDEN_MODEL_FEATURES.intersection(features.columns)
    if overlap:
        raise AssertionError(f"Forbidden model features detected: {sorted(overlap)}")
    return features


def get_target(frame: pd.DataFrame, task: str) -> pd.Series:
    """Return the approved binary or seven-band target."""
    target_column = {
        "binary": "target_binary",
        "seven_band": "target_7band",
    }.get(task)
    if target_column is None:
        raise ValueError("task must be 'binary' or 'seven_band'")
    return frame[target_column].copy()
