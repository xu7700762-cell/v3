"""Source-only pilot data: unlabeled subject EA, no designated reference segment."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..data.features import fit_and_apply_subject_ea
from ..data.vrq import load_windows, CHANNEL_INDICES
from ..data.city import load_subject_mat
from ..evaluation.io import read_json, sha256_file
from ..ssl_protocol import digest_json


@dataclass(frozen=True)
class TokenExample:
    subject_id: str
    session: str
    indices: tuple
    label: int
    sample_id: str

    def identity(self):
        return {"subject_id": self.subject_id, "session": self.session,
                "indices": list(self.indices), "sample_id": self.sample_id, "label": self.label}


def uniform_windows(indices, count):
    count = int(count)
    if count <= 0 or len(indices) < count:
        raise ValueError(f"Target session cannot supply {count} distinct windows")
    return tuple(int(indices[i]) for i in np.rint(np.linspace(0, len(indices) - 1, count)).astype(int))


def uniform_eleven(indices):
    return uniform_windows(indices, 11)


def sample_hash(examples):
    return digest_json([e.identity() for e in examples])


def checked_file(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"MAT/protocol SHA-256 mismatch: {path}")
    return {"file": Path(path).name, "sha256": actual}


def source_split(manifest, dataset):
    raw = (manifest["folds"] if dataset == "vrq" else manifest["fold_manifest"]["folds"])["fold_1"]
    split = {k: sorted(str(s) for s in raw[k]) for k in ("train_subjects", "val_subjects", "test_subjects")}
    train, val, test = (set(split[k]) for k in split)
    if not train or not val or train & val or (train | val) & test:
        raise ValueError("Pilot identity partition is invalid")
    return split


def load_source_data(config, dataset, *, severity_windows=11):
    if dataset not in ("vrq", "city"):
        raise ValueError("Pilot supports VRQ and city only")
    protocol_root = Path(config["protocol_root"])
    manifest_path = protocol_root / dataset / "audit_manifest.json"
    bundle = read_json(protocol_root / "manifest.json")
    checked_file(manifest_path, bundle["protocol_files"][f"protocols/{dataset}/audit_manifest.json"]["sha256"])
    manifest = read_json(manifest_path)
    split = source_split(manifest, dataset)
    subjects = sorted(set(split["train_subjects"]) | set(split["val_subjects"]))
    data_root = Path(config["paths"][f"{dataset}_data_root"])
    records, examples, assets, ea_audit = [], {"state": [], "severity": []}, [], {}
    offset = 0
    for subject in subjects:
        parts, labels, targets = [], [], []
        if dataset == "vrq":
            p = next(p for p in manifest["subject_protocols"] if p["subject_id"] == subject)
            # These names locate fixed protocol samples and define supervised targets only.
            # They never enter EA, the model, or a calibration operation.
            sessions = ["rest01", "rest02", p["final_task"]] + ([p["post_rest"]] if p["post_rest"] else [])
            for session in sessions:
                path = data_root / f"{subject}_{session}.mat"
                assets.append(checked_file(path, manifest["run_fingerprint_payload"]["inputs"]["mat_sha256"][path.name]))
                parts.append(load_windows(path, manifest["run_fingerprint_payload"]["mat_key"]))
                labels.append(int(session not in ("rest01", "rest02")))
                targets.append(int(manifest["audit"]["subjects"][subject]["ssq_label"])
                               if session == p["final_task"] else None)
        else:
            metadata = manifest["audit"]["subjects"][subject]
            path = data_root / Path(metadata["mat_path"].replace("\\", "/")).name
            assets.append(checked_file(path, metadata["mat_sha256"]))
            raw = load_subject_mat(path)
            path_labels = {int(p["route_order"]): int(p["path_label"])
                           for p in manifest["audit"]["path_labels"] if p["subject_id"] == subject}
            for segment in metadata["segments"]:
                selected = raw[np.asarray(CHANNEL_INDICES), int(segment["start_sample"]):int(segment["end_sample"])]
                starts = list(range(0, selected.shape[1] - 1280 + 1, 1280))
                if not starts:
                    raise ValueError("Protocol segment has no complete five-second window")
                parts.append(np.stack([selected[:, i:i + 1280] for i in starts]).astype(np.float32))
                labels.append(int(segment["state"] == "task"))
                targets.append(path_labels[int(segment["route_order"])] if segment.get("path_score") is not None else None)
        # Fit independently per subject, using every protocol window without labels or stage selection.
        combined = np.concatenate(parts).astype(np.float32)
        aligned, _, diagnostics = fit_and_apply_subject_ea(combined)
        records.append(aligned)
        ea_audit[subject] = diagnostics
        local = 0
        for number, (part, label, target) in enumerate(zip(parts, labels, targets)):
            session = f"segment_{number:02d}"
            indices = list(range(offset + local, offset + local + len(part)))
            for j, index in enumerate(indices):
                examples["state"].append(TokenExample(subject, session, (index,), label, f"{subject}/{session}/{j:05d}"))
            if target is not None:
                selected = (uniform_windows(indices, severity_windows) if severity_windows is not None
                            else uniform_windows(indices, len(indices)))
                examples["severity"].append(TokenExample(
                    subject, session, selected, target, f"{subject}/{session}"
                ))
            local += len(part)
        offset += len(combined)
    windows = np.concatenate(records).astype(np.float32)
    partitions = {}
    for task, values in examples.items():
        values.sort(key=lambda e: e.sample_id)
        partitions[task] = {}
        for role, key in (("train", "train_subjects"), ("val", "val_subjects")):
            selected = [e for e in values if e.subject_id in split[key]]
            if {e.subject_id for e in selected} != set(split[key]) or len({e.sample_id for e in selected}) != len(selected):
                raise ValueError("Missing or duplicate source examples")
            partitions[task][role] = selected
        if {e.label for e in partitions[task]["train"]} != {0, 1}:
            raise ValueError("Source-train must contain both classes")
    identity = {"dataset": dataset, "split": split, "assets": assets,
                "manifest_sha256": sha256_file(manifest_path), "window_shape": list(windows.shape),
                "subject_ea": ea_audit, "offline_transductive_subject_EA": True,
                "reference_calibration": False, "outer_test_loaded": False,
                "severity_windows": int(severity_windows) if severity_windows is not None else "all_available",
                "sample_hashes": {t: {r: sample_hash(e) for r, e in p.items()} for t, p in partitions.items()}}
    return windows, partitions, identity
