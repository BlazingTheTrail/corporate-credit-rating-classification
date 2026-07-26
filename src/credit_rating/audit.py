"""Source-data diagnostics for transparent portfolio reporting."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .data import MODEL_FEATURES, split_temporally


def build_data_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize coverage, target consistency, and label ambiguity."""
    split = split_temporally(frame)
    development = pd.concat([split.train, split.validation], ignore_index=True)

    source_binary = frame["Binary Rating"].astype(int)
    derived_binary = (frame["target_binary"] == "investment_grade").astype(int)

    predictor_rating_counts = (
        frame.groupby(MODEL_FEATURES, dropna=False)["Rating"].nunique()
    )
    ambiguous_groups = predictor_rating_counts[predictor_rating_counts > 1]
    ambiguous_keys = ambiguous_groups.index
    if len(ambiguous_keys):
        grouped = frame.groupby(MODEL_FEATURES, dropna=False).size()
        ambiguous_rows = int(grouped.loc[ambiguous_keys].sum())
    else:
        ambiguous_rows = 0

    development_issuers = set(development["CIK"])
    test_issuers = set(split.test["CIK"])

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "date_range": {
            "start": frame["Rating Date"].min().date().isoformat(),
            "end": frame["Rating Date"].max().date().isoformat(),
        },
        "issuers": {
            "unique_cik": int(frame["CIK"].nunique()),
            "unique_ticker": int(frame["Ticker"].nunique()),
            "repeated_cik": int((frame["CIK"].value_counts() > 1).sum()),
        },
        "ratings": {
            "original_classes": int(frame["Rating"].nunique()),
            "seven_band_classes": int(frame["target_7band"].nunique()),
            "binary_classes": int(frame["target_binary"].nunique()),
            "original_distribution": {
                str(key): int(value)
                for key, value in frame["Rating"].value_counts().items()
            },
            "seven_band_distribution": {
                str(key): int(value)
                for key, value in frame["target_7band"].value_counts().items()
            },
            "binary_distribution": {
                str(key): int(value)
                for key, value in frame["target_binary"].value_counts().items()
            },
        },
        "leakage_controls": {
            "source_binary_matches_derived_target_rows": int(
                (source_binary == derived_binary).sum()
            ),
            "source_binary_total_rows": int(len(frame)),
            "source_binary_excluded_from_features": True,
            "issuer_identifiers_excluded_from_features": True,
        },
        "label_ambiguity": {
            "predictor_vectors_with_multiple_ratings": int(len(ambiguous_groups)),
            "rows_in_ambiguous_predictor_groups": ambiguous_rows,
        },
        "temporal_holdout": {
            "development_rows_2010_2015": int(len(development)),
            "test_rows_2016": int(len(split.test)),
            "test_issuers": int(len(test_issuers)),
            "test_issuers_seen_in_development": int(
                len(test_issuers.intersection(development_issuers))
            ),
            "test_issuers_unseen_in_development": int(
                len(test_issuers.difference(development_issuers))
            ),
        },
    }
