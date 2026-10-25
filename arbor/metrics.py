"""Evaluation metrics, thin wrappers over ``sklearn.metrics``.

Averaged precision, recall and F1 are computed over the classes present in
either the true or the predicted labels, and a class that is never predicted
scores 0 instead of raising a warning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

METRICS = ("accuracy", "precision", "recall", "f1")
METRIC_LABELS = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1-score"}
AVERAGES = {"macro": "Macro", "weighted": "Weighted"}


def score(y_true, y_pred, average: str = "macro") -> dict[str, float]:
    """Accuracy plus averaged precision, recall and F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def per_class(y_true, y_pred, labels) -> pd.DataFrame:
    """Precision, recall, F1 and support for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return pd.DataFrame(
        {"class": labels, "precision": precision, "recall": recall, "f1": f1, "support": support.astype(int)}
    )


def confusion(y_true, y_pred, labels) -> np.ndarray:
    """Rows are actual classes, columns predicted classes, both in ``labels`` order."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def scorers(average: str = "macro") -> dict:
    """Scorer dictionary for ``sklearn.model_selection.cross_validate``."""
    return {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, average=average, zero_division=0),
        "recall": make_scorer(recall_score, average=average, zero_division=0),
        "f1": make_scorer(f1_score, average=average, zero_division=0),
    }
