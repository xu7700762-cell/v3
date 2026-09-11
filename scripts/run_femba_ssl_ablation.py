"""Run the independent 2x2 FEMBA SSL ablation; see docs/FEMBA_SSL_ABLATION.md."""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "train", "evaluate", "all", "summarize"), default="all")
    p.add_argument("--variants", nargs="+", choices=("A1", "A2", "A3", "A4"), default=["A1", "A2", "A3", "A4"])
    p.add_argument("--tasks", nargs="+", choices=("state", "severity"), default=["state", "severity"])
    p.add_argument("--datasets", nargs="+", choices=("monifeixing", "vrq", "city"), default=["monifeixing", "vrq", "city"])
    p.add_argument("--folds", nargs="+", type=int, choices=range(1, 6), default=list(range(1, 6)))
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "paths.local.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "femba_ssl_ablation")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true", help="Verify and retain completed folds; archive and restart interrupted folds")
    return p


def make_plan(args) -> dict:
    if args.resume and (args.smoke or args.stage != "all"):
        raise ValueError("--resume requires --stage all without --smoke")
    if args.seed <= 0:
        raise ValueError("--seed must be positive")
    for name in ("variants", "tasks", "datasets", "folds"):
        values = getattr(args, name)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate --{name} entries")
    if args.smoke and args.stage in {"evaluate", "summarize"}:
        raise ValueError("Smoke checkpoints cannot be used for outer evaluation or complete summaries")
    if args.stage == "summarize" and (set(args.variants) != {"A1", "A2", "A3", "A4"}
            or set(args.datasets) != {"monifeixing", "vrq", "city"} or set(args.folds) != set(range(1, 6))):
        raise ValueError("Summary requires all four variants, three datasets and five folds")
    root = args.output_root.resolve()
    if args.smoke:
        root = root / "smoke"
    root = root / f"seed_{args.seed}"
    jobs = [{"dataset": d, "task": t, "variant": v, "fold_id": f"fold_{f}",
             "output": str(root / t / v / d / f"fold_{f}")}
            for d, t, v, f in itertools.product(args.datasets, args.tasks, args.variants, args.folds)]
    return {"stage": args.stage, "smoke": args.smoke, "seed": args.seed, "resume": args.resume,
            "job_count": len(jobs), "root": str(root), "jobs": jobs,
            "complete_matrix": len(jobs) == 120 and not args.smoke}


def resume_fold(job_root, root, identity):
    """Return a verified completed training report, or preserve an interrupted attempt."""
    from datetime import datetime, timezone
    from vestibular_fusion.evaluation.io import read_json, read_csv, sha256_file
    from vestibular_fusion.evaluation.metrics import binary_metrics
    from vestibular_fusion.ssl_protocol import validate_metadata
    job_root, root = Path(job_root).resolve(), Path(root).resolve()
    relative = job_root.relative_to(root)
    if len(relative.parts) != 4:
        raise ValueError("Resume must target exactly one task/variant/dataset/fold directory")
    if not job_root.exists() or not any(job_root.iterdir()):
        return None
    training = job_root / "training" / "report.json"
    evaluation = job_root / "evaluation" / "report.json"
    if training.is_file() and evaluation.is_file():
        train_report, eval_report = read_json(training), read_json(evaluation)
        expected = {k: v for k, v in identity.items() if k != "environment"}
        expected["status"] = "complete"
        validate_metadata(train_report, expected)
        validate_metadata(eval_report, expected)
        checkpoint = job_root / "training" / "checkpoint.pt"
        predictions = job_root / "evaluation" / "predictions.csv"
        if sha256_file(checkpoint) != eval_report["checkpoint_sha256"]:
            raise RuntimeError(f"Completed checkpoint hash mismatch: {checkpoint}")
        if sha256_file(predictions) != eval_report["predictions_sha256"]:
            raise RuntimeError(f"Completed prediction hash mismatch: {predictions}")
        if binary_metrics(read_csv(predictions)) != eval_report["metrics"]:
            raise RuntimeError(f"Completed metrics do not reproduce: {evaluation}")
        if train_report["initialization"] != eval_report["initialization"]:
            raise RuntimeError("Completed training/evaluation initialization mismatch")
        if not train_report["refit_reset_verified"] or train_report["reload_max_abs_error"] != 0:
            raise RuntimeError("Completed fold has not passed refit/reload checks")
        print(f"RETAIN {relative.as_posix()} verified complete", flush=True)
        return train_report
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = root.parent / "interrupted_attempts"
    destination = archive_root / stamp / root.name / relative
    destination.resolve().relative_to(archive_root.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    job_root.rename(destination)
    print(f"ARCHIVE {relative.as_posix()} -> {destination}; restart from original initialization", flush=True)
    return None


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    # Set determinism environment before importing torch or starting CUDA.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    import torch
    from vestibular_fusion.config import load_config
    from vestibular_fusion.preflight import _check_environment, _check_protocol
    from vestibular_fusion.evaluation.io import read_json, write_json
    from vestibular_fusion.evaluation.ssl_ablation import evaluate_fold, summarize
    from vestibular_fusion.training.data import load_training_dataset
    from vestibular_fusion.training.probe_data import make_examples
    from vestibular_fusion.training.ssl_ablation import train_fold
    from vestibular_fusion.ssl_protocol import (SCHEMA, VARIANTS, experiment_protocol,
        protocol_binding, split_identity, assert_empty)

    pretrained = any(VARIANTS[v]["pretrained"] for v in args.variants)
    # Evaluation uses full saved encoder state; only construction requires the source weights.
    need_weights = pretrained and args.stage in {"preflight", "train", "all"}
    config = load_config(args.config, require_pretrain=need_weights)
    protocol = experiment_protocol(args.seed)
    environment = _check_environment()
    assets = _check_protocol(config, require_pretrain=need_weights)
    binding = protocol_binding(protocol, assets, config["protocol_root"])
    bundle = read_json(config["protocol_root"] / "manifest.json")
    official_sha = bundle["pretrained_femba"]["sha256"]
    print(json.dumps({"preflight": "passed", "files": assets["file_count"],
                      "binding": binding, "jobs": len(plan["jobs"]), "smoke": args.smoke}, indent=2), flush=True)
    if args.stage == "preflight":
        return 0
    root = Path(plan["root"])
    # Fail before performing any work if any requested output would be overwritten.
    for job in plan["jobs"]:
        if args.resume:
            continue
        if args.stage in {"all", "train"}:
            assert_empty(Path(job["output"]) / "training")
        if args.stage == "evaluate" or (args.stage == "all" and not args.smoke):
            assert_empty(Path(job["output"]) / "evaluation")
    if args.stage == "summarize":
        assert_empty(root / "summary")
    if args.smoke:
        assert_empty(root / "validation")
    device = torch.device("cuda")
    expected_folds = {}
    smoke_reports = []
    paired_initializations = {}
    for dataset in args.datasets:
        data = load_training_dataset(config, dataset)
        expected_folds[dataset] = {k: split_identity(v) for k, v in data.folds.items()}
        if args.stage == "summarize":
            continue
        for job in [j for j in plan["jobs"] if j["dataset"] == dataset]:
            task, variant, fold_id = job["task"], job["variant"], job["fold_id"]
            fold = data.folds[fold_id]
            partition = {name: make_examples(data.bank, fold, task, getattr(fold, name)) for name in (
                "source_train_subjects", "source_val_subjects", "source_subjects", "test_subjects")}
            identity = {"checkpoint_schema": SCHEMA, "dataset": dataset, "task": task,
                        "variant": variant, "fold_id": fold_id, "training_seed": args.seed,
                        "encoder_trainable": VARIANTS[variant]["encoder_trainable"],
                        "smoke": args.smoke, "protocol": protocol, "binding": binding,
                        "split": split_identity(fold), "official_pretrain_sha256": official_sha,
                        "environment": environment}
            job_root = Path(job["output"])
            retained = resume_fold(job_root, root, identity) if args.resume else None
            if args.stage in {"all", "train"}:
                if retained is None:
                    print(f"START {task}/{variant}/{dataset}/{fold_id} smoke={args.smoke}", flush=True)
                    report = train_fold(job_root / "training", identity, data.bank,
                                        partition["source_train_subjects"], partition["source_val_subjects"],
                                        partition["source_subjects"], partition["test_subjects"], device,
                                        checkpoint_path=config["paths"].get("pretrain_checkpoint"))
                else:
                    report = retained
                # Pairing is checked during smoke as well as final result aggregation.
                init = report["initialization"]
                family = "pretrained" if VARIANTS[variant]["pretrained"] else "random"
                for key, value in (("head", init["head_sha256"]),
                                   (family, init["encoder_sha256"]),
                                   ("shapes", init["parameter_shapes"])):
                    if paired_initializations.setdefault(key, value) != value:
                        raise AssertionError(f"Initialization pairing failed: {key}")
                if args.smoke:
                    smoke_reports.append({"task": task, "variant": variant, "dataset": dataset,
                                          "fold_id": fold_id, "report": report})
                print(f"PASS {task}/{variant}/{dataset}/{fold_id} "
                      f"encoder_delta={report['audit']['encoder_delta_l2']:.6g} "
                      f"reload_error={report['reload_max_abs_error']}", flush=True)
            if retained is None and (args.stage == "evaluate" or (args.stage == "all" and not args.smoke)):
                evaluate_fold(job_root / "training" / "checkpoint.pt", job_root / "evaluation",
                              identity, data.bank, partition["test_subjects"], device)
            torch.cuda.empty_cache()
    if args.smoke:
        destination = root / "validation"
        assert_empty(destination)
        result = {"status": "passed", "smoke": True, "full_experiment_completed": False,
                  "checks": len(smoke_reports), "plan": plan, "environment": environment,
                  "binding": binding, "pairing": paired_initializations, "results": smoke_reports}
        write_json(destination / "smoke_summary.json", result)
    elif args.stage == "summarize" or (args.stage == "all" and plan["complete_matrix"]):
        if args.resume and (root / "summary" / "aggregate_report.json").is_file():
            from vestibular_fusion.ssl_protocol import validate_metadata
            validate_metadata(read_json(root / "summary" / "aggregate_report.json"),
                              {"status": "complete", "smoke": False, "protocol": protocol,
                               "binding": binding, "tasks": args.tasks})
        else:
            summarize(root, protocol, binding, official_sha, args.tasks, expected_folds)
    print(json.dumps({"status": "passed" if args.smoke else "complete",
                      "stage": args.stage, "smoke": args.smoke, "output": str(root)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
