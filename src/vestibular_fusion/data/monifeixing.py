from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .features import fit_and_apply_subject_ea
from .types import AuditMetadata, FeatureBank, StateSample, SubjectRecord


SUBJECTS = tuple(f"sub{index}" for index in range(1, 19))
SESSIONS = (("rest1", 0), ("rest2", 1))
CHANNEL_INDICES = tuple(range(22)) + tuple(range(24, 32))
WINDOW_SIZE = 1280


def _load_windows(path: Path, mat_key: str) -> np.ndarray:
    payload = loadmat(path, variable_names=[mat_key])
    if mat_key not in payload:
        raise KeyError(f"{path} does not contain {mat_key!r}")
    signal = np.asarray(payload[mat_key], dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] <= max(CHANNEL_INDICES):
        raise ValueError(f"Invalid EEG shape in {path}: {signal.shape}")
    signal = signal[np.asarray(CHANNEL_INDICES)]
    starts = range(0, signal.shape[1] - WINDOW_SIZE + 1, WINDOW_SIZE)
    windows = [signal[:, start : start + WINDOW_SIZE] for start in starts]
    if not windows:
        raise ValueError(f"{path} produced no complete five-second windows")
    return np.stack(windows).astype(np.float32, copy=False)


def build_raw_bank(source_dir: Path, *, mat_key: str = "data256") -> FeatureBank:
    records: dict[str, SubjectRecord] = {}
    samples: list[StateSample] = []
    subject_manifest = {}
    for subject_number, subject in enumerate(SUBJECTS, start=1):
        session_windows = {
            session: _load_windows(Path(source_dir) / f"{subject}_{session}_q.mat", mat_key)
            for session, _ in SESSIONS
        }
        combined = np.concatenate([session_windows[session] for session, _ in SESSIONS])
        permutation = np.random.RandomState(9000 + subject_number).permutation(len(combined))
        aligned, _, ea_diagnostics = fit_and_apply_subject_ea(combined[permutation])
        inverse = np.argsort(permutation)
        aligned = aligned[inverse]
        labels, sessions = [], []
        offset = 0
        for session, label in SESSIONS:
            path = Path(source_dir) / f"{subject}_{session}_q.mat"
            for window_index in range(len(session_windows[session])):
                labels.append(label)
                sessions.append(session)
                samples.append(
                    StateSample(
                        sample_index=len(samples),
                        subject_id=subject,
                        session=session,
                        label=label,
                        window_index=window_index,
                        local_index=offset + window_index,
                        mat_path=str(path),
                    )
                )
            offset += len(session_windows[session])
        records[subject] = SubjectRecord(
            windows=aligned.astype(np.float16),
            labels=np.asarray(labels, dtype=np.int64),
            sessions=sessions,
        )
        subject_manifest[subject] = {
            "sessions": {name: int(len(values)) for name, values in session_windows.items()},
            "num_windows": int(len(aligned)),
            "ea": ea_diagnostics,
        }
    if len(samples) != 2159:
        raise AssertionError(f"Expected 2,159 monifeixing windows, got {len(samples)}")
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
