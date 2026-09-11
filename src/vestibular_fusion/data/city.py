from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .features import fit_and_apply_subject_ea
from .types import AuditMetadata, FeatureBank, StateSample, SubjectRecord
from .vrq import subject_sort_key


WINDOW_SIZE = 1280
STRIDE = 1280
CHANNEL_INDICES = tuple(range(22)) + tuple(range(24, 32))
SUBJECTS = tuple(f"acq{index:02d}" for index in range(1, 27) if index != 22)


def load_subject_mat(path: Path, mat_key: str = "data256") -> np.ndarray:
    payload = loadmat(path, variable_names=[mat_key])
    if mat_key not in payload:
        raise KeyError(f"{path} has no {mat_key!r} array")
    raw = np.asarray(payload[mat_key])
    if raw.ndim != 2 or raw.shape[0] != 37 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid city-cruise MAT data: {path} {raw.shape}")
    return raw


def session_alias(segment: dict, anchor_session: str) -> str:
    if segment["session"] == anchor_session:
        return "rest01"
    return f"{segment['state']}_seg_{int(segment['segment_index']):02d}"


def build_raw_bank(data_root: Path, audit: dict, *, mat_key: str = "data256") -> FeatureBank:
    records: dict[str, SubjectRecord] = {}
    samples: list[StateSample] = []
    subject_manifest = {}
    root = Path(data_root)
    for subject_position, subject in enumerate(SUBJECTS, start=1):
        metadata = audit["subjects"][subject]
        path = root / Path(metadata["mat_path"]).name
        raw = load_subject_mat(path, mat_key)
        parts, labels, sessions, window_indices = [], [], [], []
        segment_counts = {}
        for segment in metadata["segments"]:
            alias = session_alias(segment, metadata["anchor_session"])
            selected = raw[
                np.asarray(CHANNEL_INDICES),
                int(segment["start_sample"]) : int(segment["end_sample"]),
            ]
            starts = list(range(0, selected.shape[1] - WINDOW_SIZE + 1, STRIDE))
            if len(starts) < 11 and segment.get("path_score") is not None:
                raise AssertionError(f"{subject}/{alias} cannot provide 11 task windows")
            windows = np.stack(
                [selected[:, start : start + WINDOW_SIZE] for start in starts]
            ).astype(np.float32)
            parts.append(windows)
            labels.extend([int(segment["state"] == "task")] * len(windows))
            sessions.extend([alias] * len(windows))
            window_indices.extend(range(len(windows)))
            segment_counts[alias] = len(windows)
        combined = np.concatenate(parts).astype(np.float32)
        permutation = np.random.RandomState(9000 + subject_position).permutation(len(combined))
        aligned, _, ea_diagnostics = fit_and_apply_subject_ea(combined[permutation])
        aligned = aligned[np.argsort(permutation)]
        for local_index, (session, label, window_index) in enumerate(
            zip(sessions, labels, window_indices)
        ):
            samples.append(
                StateSample(
                    sample_index=len(samples),
                    subject_id=subject,
                    session=session,
                    label=int(label),
                    window_index=int(window_index),
                    local_index=local_index,
                    mat_path=str(path),
                )
            )
        records[subject] = SubjectRecord(
            windows=aligned.astype(np.float16),
            labels=np.asarray(labels, dtype=np.int64),
            sessions=list(sessions),
        )
        subject_manifest[subject] = {
            "segments": segment_counts,
            "num_windows": len(aligned),
            "ea": ea_diagnostics,
        }
    return FeatureBank(
        records=records,
        samples=samples,
        audit=AuditMetadata(
            {
                "num_subjects": len(records),
                "num_windows": len(samples),
                "offline_transductive_subject_EA": True,
                "subjects": subject_manifest,
            }
        ),
    )
