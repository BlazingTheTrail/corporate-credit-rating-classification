import numpy as np
import pandas as pd

from credit_rating.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from credit_rating.evaluation import evaluate_predictions
from credit_rating.modeling import build_model


def training_frame() -> tuple[pd.DataFrame, pd.Series]:
    records = []
    labels = []
    for index in range(20):
        record = {column: float(index % 5) for column in NUMERIC_FEATURES}
        record.update(
            {
                CATEGORICAL_FEATURES[0]: "Agency A" if index % 2 else "Agency B",
                CATEGORICAL_FEATURES[1]: "Industrials",
                CATEGORICAL_FEATURES[2]: str(1000 + index % 3),
            }
        )
        records.append(record)
        labels.append(
            "investment_grade" if index % 2 else "speculative_grade"
        )
    return pd.DataFrame(records), pd.Series(labels)


def test_pipeline_fits_and_predicts() -> None:
    features, target = training_frame()
    model = build_model("logistic")
    model.fit(features, target)
    prediction = model.predict(features)
    assert len(prediction) == len(target)


def test_binary_metrics_include_risk_recall() -> None:
    truth = pd.Series(
        ["investment_grade", "speculative_grade", "speculative_grade"]
    )
    prediction = np.array(
        ["investment_grade", "investment_grade", "speculative_grade"]
    )
    metrics = evaluate_predictions(
        truth,
        prediction,
        task="binary",
        positive_probability=np.array([0.1, 0.4, 0.8]),
    )
    assert metrics["speculative_recall"] == 0.5
    assert "roc_auc" in metrics


def test_seven_band_metrics_are_ordinal() -> None:
    truth = pd.Series(["AAA", "AA", "BBB", "BB", "CCC_or_lower"])
    prediction = np.array(["AA", "AA", "BB", "BB", "CCC_or_lower"])
    metrics = evaluate_predictions(truth, prediction, task="seven_band")
    assert metrics["ordinal_mae"] == 0.4
    assert metrics["within_one_band_accuracy"] == 1.0
