"""Run C0's 120 full folds, then C1's 120 full folds; store all outputs together."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "all", "summarize"), default="all")
    p.add_argument("--conditions", nargs="+", choices=("C0", "C1"), default=["C0", "C1"])
    p.add_argument("--variants", nargs="+", choices=("A1", "A2", "A3", "A4"), default=["A1", "A2", "A3", "A4"])
    p.add_argument("--tasks", nargs="+", choices=("state", "severity"), default=["state", "severity"])
    p.add_argument("--datasets", nargs="+", choices=("monifeixing", "vrq", "city"), default=["monifeixing", "vrq", "city"])
    p.add_argument("--folds", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--config", type=Path, default=ROOT / "configs/paths.local.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/femba_ssl_fivefold_v3")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def make_plan(args):
    if args.seed <= 0:
        raise ValueError("Seed must be positive")
    for key in ("conditions", "variants", "tasks", "datasets", "folds"):
        v = getattr(args, key)
        if len(v) != len(set(v)):
            raise ValueError(f"Duplicate --{key}")
    if args.conditions != sorted(args.conditions):
        raise ValueError("C0 must precede C1")
    per_condition = len(args.variants) * len(args.tasks) * len(args.datasets) * len(args.folds)
    if args.stage == "summarize" and (args.smoke or per_condition != 120):
        raise ValueError("Summary requires complete non-smoke fivefold conditions")
    root = args.output_root.resolve() / ("smoke" if args.smoke else "full") / f"seed_{args.seed}"
    jobs = [{"condition": c, "dataset": d, "task": t, "variant": v, "fold_id": f"fold_{f}",
             "output": str(root / c / t / v / d / f"fold_{f}")}
            for c, d, t, f, v in itertools.product(args.conditions, args.datasets, args.tasks, args.folds, args.variants)]
    return {"root": str(root), "jobs": jobs, "job_count": len(jobs), "jobs_per_condition": per_condition,
            "condition_order": args.conditions, "smoke": args.smoke, "refit": True,
            "storage": "shared_lossless_frozen_encoders_independent_trainable_models"}


def main(argv=None):
    args = parser().parse_args(argv)
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    import torch
    from vestibular_fusion.config import load_config
    from vestibular_fusion.preflight import _check_environment, _check_protocol
    from vestibular_fusion.ssl_protocol import VARIANTS, split_identity, assert_empty, validate_metadata
    from vestibular_fusion.evaluation.io import read_json, write_json
    from vestibular_fusion.training.data import load_training_dataset
    from vestibular_fusion.training.ssl_fivefold import (SCHEMA, experiment_protocol, protocol_binding,
        full_partitions, run_fold, summarize_condition, compare_conditions)
    require_weights = args.stage != "summarize" and any(VARIANTS[v]["pretrained"] for v in args.variants)
    config = load_config(args.config, require_pretrain=require_weights)
    environment = _check_environment()
    assets = _check_protocol(config, require_pretrain=require_weights)
    protocol = experiment_protocol(args.seed)
    binding = protocol_binding(protocol, assets, config["protocol_root"])
    official = read_json(config["protocol_root"] / "manifest.json")["pretrained_femba"]["sha256"]
    root, device = Path(plan["root"]), torch.device("cuda")
    print(json.dumps({"preflight": "passed", "files": assets["file_count"], "jobs": plan["job_count"],
                      "condition_order": args.conditions, "root": str(root), "binding": binding}), flush=True)
    if args.stage == "preflight":
        return 0
    if args.stage == "all" and not args.resume:
        for job in plan["jobs"]:
            assert_empty(Path(job["output"]))
    root.mkdir(parents=True, exist_ok=True)
    smoke_results = []
    for condition in args.conditions:
        if condition == "C1" and not args.smoke and args.stage == "all":
            previous = read_json(root / "summary/C0/aggregate_report.json")
            validate_metadata(previous, {"status": "complete", "jobs": 120, "protocol": protocol,
                                         "binding": binding, "condition": "C0"})
        identities = []
        for dataset in args.datasets:
            data = load_training_dataset(config, dataset)
            for task in args.tasks:
                for fold_number in args.folds:
                    fold_id = f"fold_{fold_number}"
                    fold = data.folds[fold_id]
                    train, val, source, test, anchors, outer, samples = full_partitions(data.bank, fold, task)
                    for variant in args.variants:
                        identity = {"checkpoint_schema": SCHEMA, "training_seed": args.seed,
                            "condition": condition, "task": task, "dataset": dataset, "variant": variant,
                            "fold_id": fold_id, "smoke": args.smoke, "protocol": protocol, "binding": binding,
                            "split": split_identity(fold), **samples, "official_pretrain_sha256": official,
                            "encoder_trainable": VARIANTS[variant]["encoder_trainable"], "environment": environment}
                        identities.append(identity)
                        if args.stage == "summarize":
                            continue
                        if shutil.disk_usage(root).free < 2_000_000_000:
                            raise RuntimeError("Less than 2 GB free; stopping safely before the next fold")
                        job = root / condition / task / variant / dataset / fold_id
                        print(f"START {condition}/{task}/{variant}/{dataset}/{fold_id} smoke={args.smoke}", flush=True)
                        result = run_fold(job, root, identity, data.bank, (train, val, source, test, anchors, outer), device,
                            checkpoint_path=config["paths"].get("pretrain_checkpoint"), resume=args.resume)
                        if args.smoke:
                            smoke_results.append({"job": str(job), "refit": result["refit"]})
                        print(f"PASS {condition}/{task}/{variant}/{dataset}/{fold_id}", flush=True)
                        torch.cuda.empty_cache()
            del data
        if not args.smoke and len(identities) == 120:
            summarize_condition(root, condition, identities)
            print(f"CONDITION COMPLETE {condition}: 120/120", flush=True)
    if args.smoke:
        write_json(root / "smoke_summary.json", {"status": "passed", "checks": len(smoke_results),
            "full_fivefold_completed": False, "results": smoke_results, "plan": plan})
    elif all((root / "summary" / c / "aggregate_report.json").is_file() for c in ("C0", "C1")):
        compare_conditions(root)
    print(json.dumps({"status": "completed_requested_jobs", "jobs": plan["job_count"], "output": str(root)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
