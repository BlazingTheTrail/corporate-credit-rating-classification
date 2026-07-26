"""Evaluation helpers for classification and ordered rating bands."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import SEVEN_BAND_ORDER


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    task: str,
    positive_probability: np.ndarray | None = None,
) -> dict[str, Any]:
    """Return metrics appropriate to the selected business task."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
    }

    if task == "binary":
        positive = "speculative_grade"
        true_binary = (y_true == positive).astype(int)
        pred_binary = (pd.Series(y_pred, index=y_true.index) == positive).astype(int)
        metrics.update(
            {
                "speculative_precision": float(
                    precision_score(true_binary, pred_binary, zero_division=0)
                ),
                "speculative_recall": float(
                    recall_score(true_binary, pred_binary, zero_division=0)
                ),
                "speculative_f1": float(
                    f1_score(true_binary, pred_binary, zero_division=0)
                ),
            }
        )
        if positive_probability is not None and true_binary.nunique() == 2:
            metrics["roc_auc"] = float(
                roc_auc_score(true_binary, positive_probability)
            )
            metrics["average_precision"] = float(
                average_precision_score(true_binary, positive_probability)
            )
        labels = ["investment_grade", "speculative_grade"]
    elif task == "seven_band":
        labels = SEVEN_BAND_ORDER
        order = {label: index for index, label in enumerate(labels)}
        true_ordinal = y_true.map(order).to_numpy()
        pred_ordinal = pd.Series(y_pred).map(order).to_numpy()
        metrics["ordinal_mae"] = float(np.mean(np.abs(true_ordinal - pred_ordinal)))
        metrics["within_one_band_accuracy"] = float(
            np.mean(np.abs(true_ordinal - pred_ordinal) <= 1)
        )
    else:
        raise ValueError("task must be 'binary' or 'seven_band'")

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(
        y_true, y_pred, labels=labels
    ).tolist()
    metrics["labels"] = labels
    return metrics
