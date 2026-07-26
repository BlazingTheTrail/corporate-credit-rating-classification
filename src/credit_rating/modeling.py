"""Preprocessing and model definitions."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing that is fitted inside each validation fold."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_model(model_name: str, random_state: int = 42) -> Pipeline:
    """Build a complete leakage-resistant training pipeline."""
    if model_name == "dummy":
        classifier = DummyClassifier(strategy="prior")
    elif model_name == "logistic":
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=3_000,
            random_state=random_state,
        )
    elif model_name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        raise ValueError(
            "model_name must be 'dummy', 'logistic', or 'random_forest'"
        )

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
