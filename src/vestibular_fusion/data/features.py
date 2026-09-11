from __future__ import annotations

import numpy as np


WINDOW_SIZE = 1280
EA_SHRINKAGE = 0.0


def fit_and_apply_subject_ea(unlabeled_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    if not isinstance(unlabeled_windows, np.ndarray) or unlabeled_windows.ndim != 3:
        raise TypeError("EA expects [windows,channels,time]")
    if unlabeled_windows.shape[1:] != (30, WINDOW_SIZE):
        raise ValueError(f"Unexpected EA input shape: {unlabeled_windows.shape}")
    values = unlabeled_windows.astype(np.float64)
    centered = values - values.mean(axis=-1, keepdims=True)
    standardized = centered / np.maximum(centered.std(axis=(1, 2), keepdims=True), 1e-8)
    covariance = standardized @ standardized.transpose(0, 2, 1) / float(WINDOW_SIZE - 1)
    reference = covariance.mean(axis=0)
    isotropic = float(np.trace(reference)) / reference.shape[0]
    reference = (1.0 - EA_SHRINKAGE) * reference + EA_SHRINKAGE * isotropic * np.eye(
        reference.shape[0], dtype=np.float64
    )
    ridge = max(isotropic * 1e-6, 1e-8)
    reference += np.eye(reference.shape[0], dtype=np.float64) * ridge
    eigenvalues, eigenvectors = np.linalg.eigh(reference)
    transform = (eigenvectors * np.maximum(eigenvalues, 1e-10) ** -0.5) @ eigenvectors.T
    aligned = np.einsum("cd,ndt->nct", transform, standardized, optimize=True).astype(np.float32)
    if not np.isfinite(aligned).all() or not np.isfinite(transform).all():
        raise FloatingPointError("Subject EA produced non-finite values")
    diagnostics = {
        "mode": "subject_unlabeled",
        "num_windows": int(len(unlabeled_windows)),
        "condition_number": float(np.linalg.cond(reference)),
        "ridge": float(ridge),
    }
    return aligned, transform.astype(np.float32), diagnostics
