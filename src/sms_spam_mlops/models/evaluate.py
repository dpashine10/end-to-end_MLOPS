"""Evaluation and model acceptance policy."""

from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(actual: list[str], predicted: list[str]) -> dict[str, Any]:
    """Calculate labeled test-set metrics for the spam class."""

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_spam": float(precision_score(actual, predicted, pos_label="spam", zero_division=0)),
        "recall_spam": float(recall_score(actual, predicted, pos_label="spam", zero_division=0)),
        "f1_spam": float(f1_score(actual, predicted, pos_label="spam", zero_division=0)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=["not_spam", "spam"]).tolist(),
        "classification_report": classification_report(actual, predicted, output_dict=True, zero_division=0),
    }


def candidate_is_accepted(metrics: dict[str, Any], minimum_f1: float) -> bool:
    """Return whether the candidate meets the explicit spam F1 threshold."""

    return float(metrics["f1_spam"]) >= minimum_f1