#!/usr/bin/env python3
"""Train and evaluate binary and seven-band rating classifiers."""

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
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from credit_rating.data import (  # noqa: E402
    get_features,
    get_target,
    load_credit_data,
    split_temporally,
)
from credit_rating.evaluation import evaluate_predictions  # noqa: E402
from credit_rating.modeling import build_model  # noqa: E402


def human_label(label: str) -> str:
    """Format internal target labels for portfolio charts."""
    replacements = {
        "investment_grade": "Investment Grade",
        "speculative_grade": "Speculative Grade",
        "CCC_or_lower": "CCC or lower",
    }
    return replacements.get(label, label)


def positive_probability(model, features: pd.DataFrame) -> pd.Series | None:
    """Return speculative-grade probabilities for a binary classifier."""
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(model.classes_)
    if "speculative_grade" not in classes:
        return None
    index = classes.index("speculative_grade")
    return model.predict_proba(features)[:, index]


def grouped_cv_summary(
    frame: pd.DataFrame,
    task: str,
    model_name: str,
    random_state: int,
) -> dict[str, float]:
    """Estimate performance on issuers not seen in a fold's training set."""
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=random_state
    )
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }
    scores = cross_validate(
        build_model(model_name, random_state),
        get_features(frame),
        get_target(frame, task),
        groups=frame["CIK"],
        cv=splitter,
        scoring=scoring,
        n_jobs=1,
        error_score="raise",
    )
    summary: dict[str, float] = {}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(values.mean())
        summary[f"{metric}_std"] = float(values.std(ddof=1))
    return summary


def select_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    task: str,
    random_state: int,
) -> tuple[str, dict[str, dict[str, float]]]:
    """Select a candidate on 2015 data without consulting the 2016 holdout."""
    selection_metric = "balanced_accuracy" if task == "binary" else "macro_f1"
    results: dict[str, dict[str, float]] = {}

    for model_name in ["dummy", "logistic", "random_forest"]:
        model = build_model(model_name, random_state)
        model.fit(get_features(train), get_target(train, task))
        prediction = model.predict(get_features(validation))
        metrics = evaluate_predictions(
            get_target(validation, task), prediction, task
        )
        results[model_name] = {
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
        }
        if task == "seven_band":
            results[model_name]["ordinal_mae"] = metrics["ordinal_mae"]

    candidates = ["logistic", "random_forest"]
    selected = max(
        candidates,
        key=lambda name: results[name][selection_metric],
    )
    results["selection"] = {
        "metric": selection_metric,
        "selected_model": selected,
    }
    return selected, results


def save_confusion_matrix(
    metrics: dict,
    output_path: Path,
    title: str,
) -> None:
    """Render a version-controlled confusion matrix."""
    matrix = np.asarray(metrics["confusion_matrix"])
    labels = metrics["labels"]
    display_labels = [human_label(label) for label in labels]
    display = ConfusionMatrixDisplay(matrix, display_labels=display_labels)
    figure_size = (8, 5.5) if len(labels) == 2 else (8.5, 7.5)
    rotation = 0 if len(labels) == 2 else 35
    _, axis = plt.subplots(figsize=figure_size)
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=rotation,
    )
    axis.set_title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_task(
    frame: pd.DataFrame,
    task: str,
    model_name: str,
    output_dir: Path,
    random_state: int,
) -> dict:
    """Fit on historical development data and evaluate the 2016 holdout."""
    split = split_temporally(frame)
    development = pd.concat([split.train, split.validation], ignore_index=True)

    selection_results = None
    selected_model = model_name
    if model_name == "auto":
        selected_model, selection_results = select_model(
            split.train, split.validation, task, random_state
        )

    model = build_model(selected_model, random_state)
    model.fit(get_features(development), get_target(development, task))

    test_features = get_features(split.test)
    y_test = get_target(split.test, task)
    y_pred = model.predict(test_features)
    probability = (
        positive_probability(model, test_features) if task == "binary" else None
    )
    metrics = evaluate_predictions(y_test, y_pred, task, probability)
    metrics["task"] = task
    metrics["model"] = selected_model
    if selection_results is not None:
        metrics["model_selection"] = selection_results
    metrics["split"] = {
        "development_years": [
            int(development["rating_year"].min()),
            int(development["rating_year"].max()),
        ],
        "test_year": int(split.test["rating_year"].unique().item()),
        "development_rows": int(len(development)),
        "test_rows": int(len(split.test)),
        "development_issuers": int(development["CIK"].nunique()),
        "test_issuers": int(split.test["CIK"].nunique()),
    }
    metrics["unseen_issuer_group_cv"] = grouped_cv_summary(
        development, task, selected_model, random_state
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_path = output_dir / "metrics" / f"{task}_{selected_model}.json"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(
        metrics,
        output_dir
        / "charts"
        / f"{task}_{selected_model}_confusion_matrix.png",
        (
            f"{task.replace('_', ' ').title()} — "
            f"{selected_model.replace('_', ' ').title()} — 2016 holdout"
        ),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/corporateCreditRatingWithFinancialRatios.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs")
    )
    parser.add_argument(
        "--model",
        choices=["auto", "dummy", "logistic", "random_forest"],
        default="auto",
    )
    parser.add_argument(
        "--task",
        choices=["binary", "seven_band", "all"],
        default="all",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    frame = load_credit_data(args.input)
    tasks = ["binary", "seven_band"] if args.task == "all" else [args.task]
    for task in tasks:
        metrics = run_task(
            frame=frame,
            task=task,
            model_name=args.model,
            output_dir=args.output_dir,
            random_state=args.random_state,
        )
        print(
            f"{task} ({metrics['model']}): "
            f"balanced_accuracy={metrics['balanced_accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
