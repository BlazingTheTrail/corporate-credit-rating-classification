from pathlib import Path

import pandas as pd
import pytest

from credit_rating.audit import build_data_audit
from credit_rating.data import (
    FORBIDDEN_MODEL_FEATURES,
    get_features,
    get_target,
    load_credit_data,
    split_temporally,
)


def sample_csv(tmp_path: Path) -> Path:
    rows = []
    ratings = [("BBB", 2014), ("BB+", 2015), ("AAA", 2016)]
    for index, (rating, year) in enumerate(ratings):
        row = {
            "Rating Agency": "Agency",
            "Corporation": f"Issuer {index}",
            "Rating": rating,
            "Rating Date": f"1/1/{year}",
            "CIK": index + 1,
            "Binary Rating": int(rating in {"BBB", "AAA"}),
            "SIC Code": 1000 + index,
            "Sector": "Industrials",
            "Ticker": f"T{index}",
        }
        numeric_names = [
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
        row.update({name: float(index + 1) for name in numeric_names})
        rows.append(row)

    path = tmp_path / "credit.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_targets_are_constructed_from_rating(tmp_path: Path) -> None:
    frame = load_credit_data(sample_csv(tmp_path))
    assert get_target(frame, "binary").tolist() == [
        "investment_grade",
        "speculative_grade",
        "investment_grade",
    ]
    assert get_target(frame, "seven_band").tolist() == ["BBB", "BB", "AAA"]


def test_binary_rating_and_identifiers_are_not_features(tmp_path: Path) -> None:
    frame = load_credit_data(sample_csv(tmp_path))
    features = get_features(frame)
    assert not FORBIDDEN_MODEL_FEATURES.intersection(features.columns)


def test_temporal_split_preserves_order(tmp_path: Path) -> None:
    frame = load_credit_data(sample_csv(tmp_path))
    split = split_temporally(frame)
    assert split.train["rating_year"].max() == 2014
    assert split.validation["rating_year"].unique().tolist() == [2015]
    assert split.test["rating_year"].unique().tolist() == [2016]


def test_unknown_task_is_rejected(tmp_path: Path) -> None:
    frame = load_credit_data(sample_csv(tmp_path))
    with pytest.raises(ValueError):
        get_target(frame, "twenty_three_classes")


def test_audit_confirms_binary_target_equivalence(tmp_path: Path) -> None:
    frame = load_credit_data(sample_csv(tmp_path))
    audit = build_data_audit(frame)
    assert (
        audit["leakage_controls"]["source_binary_matches_derived_target_rows"]
        == len(frame)
    )
    assert audit["leakage_controls"]["source_binary_excluded_from_features"]
