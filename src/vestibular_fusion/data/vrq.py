from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

from .features import fit_and_apply_subject_ea
from .types import AuditMetadata, FeatureBank, StateSample, SubjectRecord


WINDOW_SIZE = 1280
STRIDE = 1280
CHANNEL_INDICES = tuple(range(22)) + tuple(range(24, 32))


@dataclass(frozen=True)
class SubjectProtocol:
    subject_id: str
    final_task: str
    post_rest: Optional[str]
    expected_post_rest: str


def subject_sort_key(subject: str) -> tuple[str, int | str]:
    text = str(subject)
    suffix = "".join(character for character in text if character.isdigit())
    return text.rstrip("0123456789"), int(suffix) if suffix else text


def load_windows(path: Path, mat_key: str) -> np.ndarray:
    payload = loadmat(path, variable_names=[mat_key])
    if mat_key not in payload:
        raise KeyError(f"{path} has no {mat_key!r} array")
    raw = np.asarray(payload[mat_key])
    if raw.ndim != 2 or raw.shape[0] <= max(CHANNEL_INDICES):
        raise ValueError(f"Invalid EEG shape in {path}: {raw.shape}")
    if not np.issubdtype(raw.dtype, np.number) or not np.isfinite(raw).all():
        raise ValueError(f"Non-numeric or non-finite EEG in {path}")
    selected = raw[np.asarray(CHANNEL_INDICES)].astype(np.float32)
    starts = range(0, selected.shape[1] - WINDOW_SIZE + 1, STRIDE)
    windows = [selected[:, start : start + WINDOW_SIZE] for start in starts]
    if not windows:
        raise ValueError(f"{path} produced no complete five-second windows")
    return np.stack(windows)


def session_order(protocol: SubjectProtocol) -> list[str]:
    sessions = ["rest01", "rest02", protocol.final_task]
    if protocol.post_rest:
        sessions.append(protocol.post_rest)
    return sessions


def build_raw_bank(
    data_root: Path,
    mat_key: str,
    audit: dict,
    protocols: list[SubjectProtocol],
) -> FeatureBank:
    records: dict[str, SubjectRecord] = {}
    samples: list[StateSample] = []
    subject_manifest = {}
    root = Path(data_root)
    for subject_number, protocol in enumerate(protocols, start=1):
        raw_parts, labels, sessions, window_indices, paths = [], [], [], [], []
        session_counts = {}
        for session in session_order(protocol):
            path = root / f"{protocol.subject_id}_{session}.mat"
            windows = load_windows(path, mat_key)
            label = 0 if session in {"rest01", "rest02"} else 1
            session_counts[session] = len(windows)
            raw_parts.append(windows)
            labels.extend([label] * len(windows))
            sessions.extend([session] * len(windows))
            window_indices.extend(range(len(windows)))
            paths.extend([str(path)] * len(windows))
        combined = np.concatenate(raw_parts).astype(np.float32)
        permutation = np.random.RandomState(9000 + subject_number).permutation(len(combined))
        aligned, _, ea_diagnostics = fit_and_apply_subject_ea(combined[permutation])
        aligned = aligned[np.argsort(permutation)]
        for local_index, (session, label, window_index, path) in enumerate(
            zip(sessions, labels, window_indices, paths)
        ):
            samples.append(
                StateSample(
                    sample_index=len(samples),
                    subject_id=protocol.subject_id,
                    session=session,
                    label=int(label),
                    window_index=int(window_index),
                    local_index=int(local_index),
                    mat_path=path,
                )
            )
        records[protocol.subject_id] = SubjectRecord(
            windows=aligned.astype(np.float16),
            labels=np.asarray(labels, dtype=np.int64),
            sessions=list(sessions),
        )
        subject_manifest[protocol.subject_id] = {
            "sessions": session_counts,
            "num_windows": int(len(aligned)),
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
                "read_only_qc": audit.get("read_only_qc", True),
            }
        ),
    )
