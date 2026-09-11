from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SubjectRecord:
    """EA-aligned raw windows and metadata for one subject."""

    windows: np.ndarray
    labels: np.ndarray
    sessions: list[str]


@dataclass(frozen=True)
class StateSample:
    sample_index: int
    subject_id: str
    session: str
    label: int
    window_index: int
    local_index: int
    mat_path: str


@dataclass(frozen=True)
class AuditMetadata:
    manifest: dict[str, Any]


@dataclass
class FeatureBank:
    records: dict[str, SubjectRecord]
    samples: list[StateSample]
    audit: AuditMetadata
