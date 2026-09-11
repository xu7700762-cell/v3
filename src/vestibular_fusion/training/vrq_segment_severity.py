"""Segment and subject-audit scoring for VRQ window-level severity."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..evaluation.anchor_pilot import metrics_from_rows
from .token_pilot import score_head


def _subject_metrics(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["subject_id"]].append(row)
    subject_rows = []
    accuracies = []
    for subject, values in sorted(grouped.items()):
        labels = {int(value["y_true"]) for value in values}
        if len(labels) != 1:
            raise RuntimeError("A VRQ subject has inconsistent severity labels")
        probability = float(np.mean([float(value["score"]) for value in values]))
        label = labels.pop()
        prediction = int(probability >= 0.5)
        accuracies.append(np.mean([int(value["correct"]) for value in values]))
        subject_rows.append({"sample_id": subject, "subject_id": subject, "session": "aggregate",
                             "window_indices": "", "logit": 0.0, "score": probability,
                             "threshold": 0.5, "y_pred": prediction, "y_true": label,
                             "correct": int(prediction == label)})
    result = metrics_from_rows(subject_rows)
    return {"subject_macro_segment_accuracy": float(np.mean(accuracies)),
            "subject_aggregated_accuracy": result["metrics"]["accuracy"],
            "subject_aggregated_balanced_accuracy": result["metrics"]["balanced_accuracy"],
            "subject_aggregated_AUROC": result["metrics"]["AUROC"],
            "subject_count": len(subject_rows)}


def score_segment_severity(head, tokens, examples, device, micro):
    rows, result = score_head(head, tokens, examples, device, micro)
    result["subject_audit"] = _subject_metrics(rows)
    return rows, result
