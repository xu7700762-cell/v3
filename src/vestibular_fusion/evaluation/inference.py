from __future__ import annotations

import numpy as np
import torch
from sklearn.cluster import KMeans

from ..data.anchors import anchor_indices, context_features, context_windows, task_indices
from ..data.types import FeatureBank
from ..model.main import FEMBAKANMultiTaskModel
from ..protocol import DEFAULT_STATE_THRESHOLD
from ..training.data import SeverityExample
from .metrics import subject_sort_key
from .smoothing import smooth_current_future


def apply_anchor_oriented_embedding_kmeans(
    rows: list[dict], embeddings: dict[str, np.ndarray]
) -> list[dict]:
    output = []
    for subject in sorted({str(row["subject_id"]) for row in rows}, key=subject_sort_key):
        current = [row for row in rows if str(row["subject_id"]) == subject]
        anchors = [row for row in current if int(row["calibration_anchor"])]
        if len(anchors) != 4:
            raise ValueError(f"{subject} requires exactly four calibration anchors")
        values = np.asarray(embeddings[subject], dtype=np.float64)
        if values.ndim != 2 or values.shape != (len(current), 160):
            raise ValueError(
                f"{subject} requires aligned KAN embeddings [windows,160], got {values.shape}"
            )
        local_indices = np.asarray(
            [int(row["local_index"]) for row in current], dtype=np.int64
        )
        if not np.array_equal(local_indices, np.arange(len(current), dtype=np.int64)):
            raise ValueError(f"{subject} state rows are not in local window order")
        model = KMeans(n_clusters=2, n_init=50, random_state=1001).fit(values)
        centers = model.cluster_centers_
        anchor_indices = np.asarray(
            [int(row["local_index"]) for row in anchors], dtype=np.int64
        )
        anchor_mean = values[anchor_indices].mean(axis=0)
        rest_cluster = int(
            np.argmin(np.linalg.norm(centers - anchor_mean[None, :], axis=1))
        )
        state_cluster = 1 - rest_cluster
        assignments = model.predict(values)
        distances = model.transform(values)
        margins = distances[:, rest_cluster] - distances[:, state_cluster]
        margin_scale = float(np.std(margins))
        if margin_scale < 1e-8:
            raise RuntimeError(f"{subject} KAN embeddings cannot form two state clusters")
        logits = margins / margin_scale
        scores = 1.0 / (1.0 + np.exp(-logits))
        for row, assignment, logit, score in zip(
            current, assignments, logits, scores
        ):
            prediction = int(int(assignment) == state_cluster)
            if int(row["calibration_anchor"]):
                prediction = 0
            output.append(
                {
                    **row,
                    "network_logit": float(row["logit"]),
                    "network_score": float(row["score"]),
                    "logit": float(logit),
                    "score": float(score),
                    "threshold": 0.5,
                    "positive_above_threshold": 1,
                    "anchor_embedding_norm": float(np.linalg.norm(anchor_mean)),
                    "rest_cluster_center_norm": float(
                        np.linalg.norm(centers[rest_cluster])
                    ),
                    "state_cluster_center_norm": float(
                        np.linalg.norm(centers[state_cluster])
                    ),
                    "y_pred": prediction,
                }
            )
    output.sort(key=lambda row: int(row["sample_index"]))
    return output


@torch.no_grad()
def score_state_subjects(
    model: FEMBAKANMultiTaskModel,
    bank: FeatureBank,
    subjects: tuple[str, ...] | list[str],
    reference_sessions: dict[str, str],
    device: torch.device,
    *,
    batch_size: int = 128,
    threshold: float = DEFAULT_STATE_THRESHOLD,
    pooled_features: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    model.eval()
    sample_lookup = {
        (sample.subject_id, sample.local_index): sample for sample in bank.samples
    }
    rows = []
    subject_embeddings = {}
    for subject in sorted(subjects, key=subject_sort_key):
        record = bank.records[subject]
        anchors = anchor_indices(record, reference_sessions[subject])
        subject_pooled = None if pooled_features is None else pooled_features[subject]
        anchor_values = (
            context_windows(record, anchors)
            if subject_pooled is None
            else context_features(record, subject_pooled, anchors)
        )
        anchor_tensor = torch.as_tensor(
            anchor_values[None].astype(np.float32), device=device
        )
        center, anchor_scale = (
            model.anchor_statistics(anchor_tensor)
            if subject_pooled is None
            else model.anchor_statistics_from_pooled(anchor_tensor)
        )
        center, anchor_scale = center[0], anchor_scale[0]
        anchor_set = set(int(index) for index in anchors)
        embedding_batches = []
        for start in range(0, len(record.windows), int(batch_size)):
            stop = min(start + int(batch_size), len(record.windows))
            values = (
                context_windows(record, np.arange(start, stop))
                if subject_pooled is None
                else context_features(record, subject_pooled, np.arange(start, stop))
            )
            windows = torch.as_tensor(values.astype(np.float32), device=device)
            embeddings = (
                model.encode_contexts(windows)
                if subject_pooled is None
                else model.encode_pooled_contexts(windows)
            ) - center.unsqueeze(0)
            embedding_batches.append(embeddings.float().cpu().numpy())
            logits = model.state_logits_from_calibrated(embeddings, anchor_scale)
            scores = torch.sigmoid(logits).float().cpu().numpy()
            logit_values = logits.float().cpu().numpy()
            for offset, (logit, score) in enumerate(zip(logit_values, scores)):
                local_index = start + offset
                sample = sample_lookup[(subject, local_index)]
                rows.append(
                    {
                        "sample_index": int(sample.sample_index),
                        "subject_id": subject,
                        "session": sample.session,
                        "window_index": int(sample.window_index),
                        "local_index": int(local_index),
                        "calibration_anchor": int(local_index in anchor_set),
                        "logit": float(logit),
                        "score": float(score),
                        "threshold": float(threshold),
                        "y_pred": int(score >= threshold),
                    }
                )
        subject_embeddings[subject] = np.concatenate(embedding_batches, axis=0)
    return apply_anchor_oriented_embedding_kmeans(rows, subject_embeddings)


def attach_state_labels(rows: list[dict], bank: FeatureBank) -> list[dict]:
    output = []
    for row in rows:
        label = int(bank.records[str(row["subject_id"])].labels[int(row["local_index"])])
        current = {**row, "y_true": label}
        current["correct"] = int(current["y_pred"] == label)
        output.append(current)
    return output


def apply_state_threshold(rows: list[dict], threshold: float) -> list[dict]:
    output = []
    for row in rows:
        prediction = int(float(row["score"]) >= float(threshold))
        current = {**row, "threshold": float(threshold), "y_pred": prediction}
        if "y_true" in current:
            current["correct"] = int(prediction == int(current["y_true"]))
        output.append(current)
    return output


def add_smoothed_state(
    rows: list[dict], windows: int = 3, threshold: float = DEFAULT_STATE_THRESHOLD
) -> list[dict]:
    scored = [row for row in rows if not int(row.get("calibration_anchor", 0))]
    scores = smooth_current_future(
        scored, np.asarray([float(row["score"]) for row in scored], dtype=np.float64), windows
    )
    by_sample = {}
    for row, score in zip(scored, scores):
        current_threshold = float(row.get("threshold", threshold))
        positive_above = bool(int(row.get("positive_above_threshold", 1)))
        prediction = int(
            float(score) >= current_threshold
            if positive_above
            else float(score) < current_threshold
        )
        by_sample[int(row["sample_index"])] = (float(score), prediction)
    output = []
    for row in rows:
        if int(row.get("calibration_anchor", 0)):
            output.append(
                {
                    **row,
                    "smoothed_score": None,
                    "smoothed_y_pred": None,
                    "smoothed_correct": None,
                }
            )
            continue
        score, prediction = by_sample[int(row["sample_index"])]
        current = {
            **row,
            "smoothed_score": score,
            "smoothed_y_pred": prediction,
        }
        if "y_true" in current:
            current["smoothed_correct"] = int(prediction == int(current["y_true"]))
        output.append(current)
    return output


@torch.no_grad()
def score_severity_examples(
    model: FEMBAKANMultiTaskModel,
    bank: FeatureBank,
    examples: tuple[SeverityExample, ...] | list[SeverityExample],
    device: torch.device,
    *,
    batch_size: int = 32,
    threshold: float = 0.5,
    pooled_features: dict[str, np.ndarray] | None = None,
    token_dynamics: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    if (pooled_features is None) != (token_dynamics is None):
        raise ValueError(
            "Cached severity inference requires pooled features and token dynamics together"
        )
    model.eval()
    ordered = sorted(
        examples,
        key=lambda example: (subject_sort_key(example.subject_id), example.task_session),
    )
    rows = []
    for start in range(0, len(ordered), int(batch_size)):
        current = ordered[start : start + int(batch_size)]
        anchors, tasks, dynamics = [], [], []
        for example in current:
            record = bank.records[example.subject_id]
            subject_pooled = (
                None
                if pooled_features is None
                else pooled_features[example.subject_id]
            )
            subject_dynamics = (
                None
                if token_dynamics is None
                else token_dynamics[example.subject_id]
            )
            anchor_rows = anchor_indices(record, example.reference_session)
            task_rows = task_indices(record, example.task_session, 11)
            if subject_pooled is None:
                anchors.append(context_windows(record, anchor_rows))
                tasks.append(context_windows(record, task_rows))
            else:
                anchors.append(context_features(record, subject_pooled, anchor_rows))
                tasks.append(context_features(record, subject_pooled, task_rows))
                dynamics.append(
                    context_features(record, subject_dynamics, task_rows)
                )
        anchor_tensor = torch.as_tensor(np.stack(anchors).astype(np.float32), device=device)
        task_tensor = torch.as_tensor(np.stack(tasks).astype(np.float32), device=device)
        logits = (
            model.severity_logits(task_tensor, anchor_tensor)
            if pooled_features is None
            else model.severity_logits_from_pooled(
                task_tensor,
                anchor_tensor,
                torch.as_tensor(np.stack(dynamics).astype(np.float32), device=device),
            )
        )
        scores = torch.sigmoid(logits).float().cpu().numpy()
        logit_values = logits.float().cpu().numpy()
        for example, logit, score in zip(current, logit_values, scores):
            rows.append(
                {
                    "subject_id": example.subject_id,
                    "reference_session": example.reference_session,
                    "task_session": example.task_session,
                    "num_anchor_windows": 4,
                    "num_task_windows": 11,
                    "logit": float(logit),
                    "score": float(score),
                    "threshold": float(threshold),
                    "y_pred": int(score >= threshold),
                }
            )
    return rows


def attach_severity_labels(
    rows: list[dict], examples: tuple[SeverityExample, ...] | list[SeverityExample]
) -> list[dict]:
    labels = {
        (example.subject_id, example.task_session): int(example.label) for example in examples
    }
    output = []
    for row in rows:
        label = labels[(str(row["subject_id"]), str(row["task_session"]))]
        current = {**row, "y_true": label}
        current["correct"] = int(current["y_pred"] == label)
        output.append(current)
    return output
