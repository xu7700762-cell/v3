from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def subject_sort_key(value: str) -> tuple[str, int | str]:
    text = str(value)
    suffix = "".join(character for character in text if character.isdigit())
    return text.rstrip("0123456789"), int(suffix) if suffix else text


def select_balanced_accuracy_threshold(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Threshold selection requires non-empty rows")
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if set(np.unique(labels).tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires finite scores and both classes")
    candidates = np.unique(
        np.concatenate([scores, np.asarray([np.nextafter(scores.max(), np.inf)])])
    )
    values = np.asarray(
        [balanced_accuracy_score(labels, scores >= threshold) for threshold in candidates],
        dtype=np.float64,
    )
    best_value = float(values.max())
    tied = candidates[np.isclose(values, best_value, rtol=0.0, atol=1e-12)]
    threshold = float(min(tied, key=lambda value: (abs(float(value) - 0.5), float(value))))
    return {
        "method": "max_balanced_accuracy",
        "threshold": threshold,
        "balanced_accuracy": best_value,
        "tie_break": "closest_to_0p5_then_lower",
        "n_samples": int(labels.size),
        "class_counts": {
            "0": int(np.sum(labels == 0)),
            "1": int(np.sum(labels == 1)),
        },
    }


def binary_metrics(rows: list[dict], *, subject_key: str = "subject_id") -> dict:
    if not rows:
        raise ValueError("Metrics require non-empty rows")
    labels = np.asarray([int(row["y_true"]) for row in rows], dtype=np.int64)
    predictions = np.asarray([int(row["y_pred"]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
    if set(np.unique(labels).tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Binary metrics require finite scores and both classes")
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    subjects = np.asarray([str(row[subject_key]) for row in rows], dtype=object)
    subject_values = []
    for subject in sorted(set(subjects.tolist()), key=subject_sort_key):
        mask = subjects == subject
        metric = balanced_accuracy_score if len(np.unique(labels[mask])) == 2 else accuracy_score
        subject_values.append(float(metric(labels[mask], predictions[mask])))
    return {
        "n_samples": int(len(rows)),
        "n_subjects": int(len(set(subjects.tolist()))),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "subject_macro_balanced_accuracy": float(np.mean(subject_values)),
        "F1": float(f1_score(labels, predictions)),
        "AUROC": float(roc_auc_score(labels, scores)),
        "AUPRC": float(average_precision_score(labels, scores)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix": matrix.tolist(),
    }
