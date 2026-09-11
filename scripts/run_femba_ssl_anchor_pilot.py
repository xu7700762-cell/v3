"""Validation-only FEMBA pretraining x four-anchor pilot (48 paired fits)."""
import argparse
import itertools
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "train", "evaluate", "all", "summarize"), default="all")
    p.add_argument("--conditions", nargs="+", choices=("C0", "C1"), default=["C0", "C1"])
    p.add_argument("--variants", nargs="+", choices=("A1", "A2", "A3", "A4"), default=["A1", "A2", "A3", "A4"])
    p.add_argument("--tasks", nargs="+", choices=("state", "severity"), default=["state", "severity"])
    p.add_argument("--datasets", nargs="+", choices=("monifeixing", "vrq", "city"), default=["monifeixing", "vrq", "city"])
    p.add_argument("--folds", nargs="+", type=int, choices=(1,), default=[1])
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--config", type=Path, default=ROOT / "configs/paths.local.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/femba_ssl_anchor_v2")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def make_plan(args):
    if args.seed <= 0:
        raise ValueError("Seed must be positive")
    for key in ("conditions", "variants", "tasks", "datasets", "folds"):
        values = getattr(args, key)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate --{key}")
    if args.resume and args.stage not in ("all", "train"):
        raise ValueError("Resume requires all or train")
    if args.smoke and args.stage in ("summarize", "evaluate"):
        raise ValueError("Smoke cannot be used for pilot evaluation/summary")
    root = args.output_root.resolve() / ("smoke" if args.smoke else "pilot") / f"seed_{args.seed}"
    jobs = [{"dataset": d, "task": t, "condition": c, "variant": v, "fold_id": "fold_1",
             "output": str(root / t / c / v / d / "fold_1")}
            for d, t, v, c in itertools.product(args.datasets, args.tasks, args.variants, args.conditions)]
    if args.stage == "summarize" and len(jobs) != 48:
        raise ValueError("Summary requires all 48 conditions")
    return {"root": str(root), "jobs": jobs, "job_count": len(jobs), "smoke": args.smoke,
            "complete_matrix": len(jobs) == 48 and not args.smoke, "outer_test_scoring": False}


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
    from vestibular_fusion.anchor_protocol import SCHEMA, experiment_protocol, protocol_binding
    from vestibular_fusion.ssl_protocol import VARIANTS, split_identity, assert_empty
    from vestibular_fusion.training.data import load_training_dataset
    from vestibular_fusion.training.anchor_data import pilot_partitions
    from vestibular_fusion.training.anchor_pilot import train_pilot
    from vestibular_fusion.evaluation.anchor_pilot import evaluate, summarize, verify_result
    from vestibular_fusion.evaluation.io import read_json, write_json

    require_weights = any(VARIANTS[v]["pretrained"] for v in args.variants) and args.stage in ("preflight", "train", "all")
    config = load_config(args.config, require_pretrain=require_weights)
    environment = _check_environment()
    assets = _check_protocol(config, require_pretrain=require_weights)
    protocol = experiment_protocol(args.seed)
    binding = protocol_binding(protocol, assets, config["protocol_root"])
    official_sha = read_json(config["protocol_root"] / "manifest.json")["pretrained_femba"]["sha256"]
    print(json.dumps({"preflight": "passed", "jobs": plan["job_count"], "files": assets["file_count"],
                      "binding": binding, "environment": environment}), flush=True)
    if args.stage == "preflight":
        return 0
    root, device = Path(plan["root"]), torch.device("cuda")
    for job in plan["jobs"]:
        if not args.resume and args.stage in ("train", "all"):
            assert_empty(Path(job["output"]))
        if args.stage == "evaluate":
            assert_empty(Path(job["output"]) / "evaluation")
    identities, results, pairing = [], [], {}
    for dataset in args.datasets:
        data = load_training_dataset(config, dataset)
        fold = data.folds["fold_1"]
        for task in args.tasks:
            train, val, anchors, samples = pilot_partitions(data.bank, fold, task)
            for job in (j for j in plan["jobs"] if j["dataset"] == dataset and j["task"] == task):
                c, v = job["condition"], job["variant"]
                identity = {"checkpoint_schema": SCHEMA, "training_seed": args.seed,
                            "dataset": dataset, "task": task, "condition": c, "variant": v,
                            "fold_id": "fold_1", "smoke": args.smoke, "protocol": protocol,
                            "binding": binding, "split": split_identity(fold), **samples,
                            "encoder_trainable": VARIANTS[v]["encoder_trainable"],
                            "official_pretrain_sha256": official_sha, "environment": environment}
                identities.append(identity)
                destination = Path(job["output"])
                if args.stage == "summarize":
                    continue
                if args.resume and (destination / "evaluation/report.json").is_file():
                    report, _, _ = verify_result(destination, identity)
                    print(f"RETAIN {task}/{c}/{v}/{dataset}", flush=True)
                else:
                    if args.stage in ("train", "all"):
                        print(f"START {task}/{c}/{v}/{dataset} smoke={args.smoke}", flush=True)
                        report = train_pilot(destination / "training", identity, data.bank, train, val,
                                             anchors, device, checkpoint_path=config["paths"].get("pretrain_checkpoint"),
                                             resume=args.resume)
                    else:
                        report = read_json(destination / "training/report.json")
                    if args.stage in ("evaluate", "all"):
                        partial = destination / "evaluation"
                        if args.resume and partial.exists() and any(partial.iterdir()):
                            import time
                            partial.resolve().relative_to(root.resolve())
                            archived = destination / f"evaluation.interrupted-{time.time_ns()}"
                            partial.rename(archived)
                            print(f"ARCHIVE partial validation -> {archived}", flush=True)
                        evaluate(destination / "training/best.pt", destination / "evaluation", identity,
                                 data.bank, val, anchors, device)
                initial = report["initialization"]
                checks = {("head",): initial["head_sha256"],
                          ("shapes",): initial["parameter_shapes"],
                          ("encoder", VARIANTS[v]["pretrained"]): initial["encoder_sha256"],
                          ("shuffle", task, dataset): report["first_epoch_order_sha256"]}
                for key, value in checks.items():
                    if pairing.setdefault(key, value) != value:
                        raise AssertionError(f"Pilot pairing failure: {key}")
                results.append({"task": task, "condition": c, "variant": v, "dataset": dataset,
                                "report": report})
                print(f"PASS {task}/{c}/{v}/{dataset} best_step={report['best_global_step']} "
                      f"reload_error={report['reload_max_abs_error']}", flush=True)
                torch.cuda.empty_cache()
        del data
    if args.smoke:
        write_json(root / "validation/smoke_summary.json", {"status": "passed", "smoke": True,
            "checks": len(results), "complete_smoke_matrix": len(results) == 48,
            "pilot_completed": False, "plan": plan, "binding": binding, "results": results})
    elif args.stage == "summarize" or (args.stage == "all" and plan["complete_matrix"]):
        summary = root / "summary/aggregate_report.json"
        if args.resume and summary.parent.exists():
            import time
            summary.parent.resolve().relative_to(root.resolve())
            summary.parent.rename(root / f"summary.previous-{time.time_ns()}")
        summarize(root, identities)
    print(json.dumps({"stage": args.stage, "status": "completed_requested_jobs", "output": str(root),
                      "pilot_completed": plan["complete_matrix"] and args.stage in ("all", "summarize"),
                      "full_fivefold_completed": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
