"""Run 80 fixed C0 DoG/MLP folds against 40 immutable linear controls."""
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
    p.add_argument("--stage", choices=("preflight", "train", "evaluate", "all", "summarize"), default="all")
    p.add_argument("--heads", nargs="+", choices=("fractional_dog_polykan", "mlp"), default=["fractional_dog_polykan", "mlp"])
    p.add_argument("--variants", nargs="+", choices=("A3", "A4"), default=["A3", "A4"])
    p.add_argument("--tasks", nargs="+", choices=("state", "severity"), default=["state", "severity"])
    p.add_argument("--datasets", nargs="+", choices=("vrq", "city"), default=["vrq", "city"])
    p.add_argument("--folds", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    p.add_argument("--seed", type=int, choices=(2001,), default=2001)
    p.add_argument("--config", type=Path, default=ROOT / "configs/paths.local.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/femba_c0_head_comparison_v1")
    p.add_argument("--linear-baseline-root", type=Path, default=ROOT / "outputs/femba_ssl_fivefold_v3/full/seed_2001")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def make_plan(args):
    for name in ("heads", "variants", "tasks", "datasets", "folds"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate --{name}")
    if args.smoke and args.stage in ("evaluate", "summarize"):
        raise ValueError("Smoke cannot evaluate outer-test or summarize full results")
    if args.smoke:
        args.folds = [1]
    root = args.output_root.resolve() / ("smoke" if args.smoke else "full") / f"seed_{args.seed}"
    # This machine's output contract is explicit: no experiment results on C or WSL ext4.
    normalized = str(root).replace("\\", "/").lower()
    if not (normalized.startswith("d:/") or normalized.startswith("/mnt/d/")):
        raise ValueError("All experiment outputs must be on D:/ or /mnt/d/")
    if root.is_relative_to(args.linear_baseline_root.resolve()) or args.linear_baseline_root.resolve().is_relative_to(root):
        raise ValueError("New output and old baseline directories must be separate")
    jobs = [{"head": h, "task": t, "variant": v, "dataset": d, "fold_id": f"fold_{f}"}
            for d, t, f, v, h in itertools.product(args.datasets, args.tasks, args.folds, args.variants, args.heads)]
    if args.stage == "summarize" and len(jobs) != 80:
        raise ValueError("Summary requires all 80 jobs")
    return {"root": str(root), "jobs": jobs, "job_count": len(jobs), "smoke": args.smoke}


def main(argv=None):
    args = parser().parse_args(argv)
    plan = make_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if os.environ.get("PYTHONHASHSEED") != str(args.seed):
        os.environ["PYTHONHASHSEED"] = str(args.seed)
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)])
    import torch
    from vestibular_fusion.config import load_config
    from vestibular_fusion.preflight import _check_environment, _check_protocol
    from vestibular_fusion.ssl_protocol import split_identity, assert_empty
    from vestibular_fusion.evaluation.io import read_json, write_json
    from vestibular_fusion.training.data import load_training_dataset
    from vestibular_fusion.training.ssl_fivefold import full_partitions
    from vestibular_fusion.training.c0_head_comparison import (SCHEMA, experiment_protocol, protocol_binding,
        verify_linear_baselines, check_baseline_pair, run_job)
    from vestibular_fusion.evaluation.c0_head_comparison import summarize
    config = load_config(args.config)
    environment, assets = _check_environment(), _check_protocol(config)
    print("Verifying all 40 read-only C0 linear baselines...", flush=True)
    baseline = verify_linear_baselines(args.linear_baseline_root)
    protocol = experiment_protocol(args.seed)
    binding = protocol_binding(protocol, assets, config["protocol_root"], baseline)
    official = read_json(config["protocol_root"] / "manifest.json")["pretrained_femba"]["sha256"]
    print(json.dumps({"preflight": "passed", "linear_folds": 40, "jobs": len(plan["jobs"]), "binding": binding}), flush=True)
    root, device = Path(plan["root"]), torch.device("cuda")
    if args.stage == "preflight":
        return 0
    if args.stage in ("all", "train") and not args.resume:
        for j in plan["jobs"]:
            assert_empty(root / j["head"] / j["task"] / j["variant"] / j["dataset"] / j["fold_id"])
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "baseline_manifest.json", {"sha256": baseline["manifest_sha256"], "jobs": baseline["manifest"]})
    identities, smoke_results = [], []
    for dataset in args.datasets:
        data = load_training_dataset(config, dataset)
        for task, number in itertools.product(args.tasks, args.folds):
            fold_id = f"fold_{number}"
            fold = data.folds[fold_id]
            *partitions, samples = full_partitions(data.bank, fold, task)
            for variant, head in itertools.product(args.variants, args.heads):
                identity = {"checkpoint_schema": SCHEMA, "training_seed": args.seed, "condition": "C0",
                    "head": head, "task": task, "dataset": dataset, "variant": variant, "fold_id": fold_id,
                    "smoke": args.smoke, "protocol": protocol, "binding": binding, "split": split_identity(fold),
                    **samples, "official_pretrain_sha256": official, "encoder_trainable": variant == "A4", "environment": environment}
                check_baseline_pair(identity, partitions, baseline)
                identities.append(identity)
                if args.stage == "summarize":
                    continue
                if shutil.disk_usage(root).free < 3_000_000_000:
                    raise RuntimeError("Less than 3 GB free on D; stopped before the next fold")
                job = root / head / task / variant / dataset / fold_id
                print(f"START {head}/{task}/{variant}/{dataset}/{fold_id} smoke={args.smoke}", flush=True)
                initial = baseline["jobs"][task, variant, dataset, fold_id]["report"]["initialization"]
                result = run_job(job, root, identity, data.bank, partitions, device, stage=args.stage,
                    resume=args.resume, checkpoint_path=config["paths"]["pretrain_checkpoint"], baseline_initial=initial)
                if args.smoke:
                    smoke_results.append({"job": str(job), "status": result["status"]})
                write_json(root / "progress.json", {"last_job": str(job), "last_status": result["status"],
                    "requested_jobs": len(plan["jobs"]), "visited_jobs": len(identities), "smoke": args.smoke})
                print(f"PASS {head}/{task}/{variant}/{dataset}/{fold_id}", flush=True)
                torch.cuda.empty_cache()
        del data
    if args.smoke:
        write_json(root / "smoke_summary.json", {"status": "passed", "checks": len(smoke_results),
            "full_smoke_matrix_completed": len(smoke_results) == 16, "outer_test_scored": False,
            "full_fivefold_completed": False, "results": smoke_results, "binding": binding})
    elif args.stage in ("all", "evaluate", "summarize") and len(identities) == 80:
        summarize(root, identities, baseline)
    print(json.dumps({"status": "completed_requested_jobs", "jobs": len(identities), "root": str(root)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
