"""Run paired VRQ state/severity shared-base development without anchors or outer-test access."""
import argparse
import itertools
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parser():
    from vestibular_fusion.model.severity_dynamics import CONFIGS
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preflight", "train", "evaluate", "all", "summarize"),
                   default="all")
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    p.add_argument("--tasks", nargs="+", choices=("severity", "state"), default=["severity", "state"])
    p.add_argument("--seeds", nargs="+", type=int, choices=(2001, 3001, 4001),
                   default=[2001, 3001, 4001])
    p.add_argument("--splits", nargs="+", choices=("split_1", "split_2", "split_3"),
                   default=["split_1", "split_2", "split_3"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/paths.local.json")
    p.add_argument("--output-root", type=Path,
                   default=ROOT / "outputs/femba_vrq_boosted_development_v1")
    p.add_argument("--cache-root", type=Path,
                   default=ROOT / "outputs/femba_reference_free_token_pilot_v1")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ("configs", "tasks", "seeds", "splits"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            raise ValueError(f"Duplicate --{key}")
    for path in (args.output_root.resolve(), args.cache_root.resolve()):
        normalized = str(path).replace("\\", "/").lower()
        if not (normalized.startswith("d:/") or normalized.startswith("/mnt/d/")):
            raise ValueError("All experiment artifacts and caches must stay on D")
    jobs = list(itertools.product(args.seeds, args.tasks, args.splits, args.configs))
    if args.dry_run:
        import json
        print(json.dumps({"jobs": jobs, "job_count": len(jobs), "dataset": "vrq",
                          "outer_test_eeg_loaded": False, "outer_test_scored": False}, indent=2))
        return 0
    from vestibular_fusion.evaluation.vrq_boosted import summarize
    mode = "smoke" if args.smoke else "development"
    root = args.output_root.resolve() / mode
    if args.stage == "summarize":
        print(summarize(root, smoke=args.smoke))
        return 0
    import torch
    from vestibular_fusion.config import load_config
    from vestibular_fusion.evaluation.anchor_pilot import load_verified
    from vestibular_fusion.evaluation.io import read_json, sha256_file, write_json
    from vestibular_fusion.model.linear_probe import tensor_state_hash
    from vestibular_fusion.model.vrq_boosted import VRQBoostedHead
    from vestibular_fusion.preflight import _check_environment
    from vestibular_fusion.ssl_protocol import assert_empty, digest_json
    from vestibular_fusion.training.ssl_ablation import seed_runtime
    from vestibular_fusion.training.token_data import sample_hash
    from vestibular_fusion.training.token_pilot import (online_smoke, prepare_encoder, require_space,
        restored_head, score_head, token_cache, train_job, verify_artifacts)
    from vestibular_fusion.training.vrq_boosted import boosted_diagnostics
    from vestibular_fusion.training.vrq_boosted_data import load_vrq_development_data
    if os.environ.get("PYTHONHASHSEED") != "2001":
        os.environ["PYTHONHASHSEED"] = "2001"
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                  *(sys.argv[1:] if argv is None else argv)])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    config = load_config(args.config)
    locked = read_json(ROOT / "reproducibility/protocols/femba_vrq_boosted_development.json")
    if set(args.seeds) - set(locked["seeds"]) or set(args.tasks) - set(locked["tasks"]) or (
        set(args.splits) - set(locked["development_splits"])):
        raise ValueError("Requested jobs are outside the locked protocol")
    environment = _check_environment()
    device = torch.device("cuda")
    code = {str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted((ROOT / "src/vestibular_fusion").rglob("*.py"))}
    code["entrypoint"] = sha256_file(Path(__file__))
    code_sha = digest_json(code)
    require_space(args.output_root, 500 * 1024**2)
    encoder, encoder_identity = prepare_encoder(config, args.cache_root, device)
    windows, partitions, data_identity, cache_identity = load_vrq_development_data(config, locked)
    if args.stage == "preflight":
        write_json(root / "preflight.json", {"status": "passed", "protocol": locked,
                   "environment": environment, "encoder": encoder_identity,
                   "data": data_identity, "code": code})
        print({task: {split: {"train": len(value["train"]), "val": len(value["val"])}
                      for split, value in task_partitions.items()}
               for task, task_partitions in partitions.items()})
        return 0
    tokens, cache = token_cache(args.cache_root, windows, cache_identity,
                                encoder, encoder_identity, device)
    if args.stage in ("train", "all") and not args.resume:
        for seed, task, split, name in jobs:
            assert_empty(root / f"seed_{seed}" / task / split / name)
    visited = 0
    for seed, task, split in itertools.product(args.seeds, args.tasks, args.splits):
        seed_runtime(seed)
        train, val = partitions[task][split]["train"], partitions[task][split]["val"]
        job_protocol = {**locked, "seed": seed}
        base_folder = root / f"seed_{seed}" / task / split / "base"
        base_identity = {
            "schema": locked["schema"], "protocol": job_protocol, "config": "base",
            "task": task, "dataset": "vrq", "split_name": split, "smoke": args.smoke,
            "encoder": encoder_identity, "cache": cache, "data_sha256": digest_json(data_identity),
            "train_sha256": sample_hash(train), "val_sha256": sample_hash(val),
            "code_sha256": code_sha, "environment": environment,
        }
        base_factory = lambda name, value_seed, selected_task=task: VRQBoostedHead(
            "base", selected_task, value_seed)
        if "base" in args.configs:
            if args.smoke:
                base_identity["online_smoke"] = online_smoke(
                    base_factory("base", seed).to(device), encoder, windows, tokens, device)
            if args.stage == "evaluate":
                base_report = verify_artifacts(base_folder, base_identity)
            else:
                base_report = train_job(base_folder, base_identity, tokens, train, val, device,
                                        resume=args.resume, head_factory=base_factory,
                                        diagnostics_fn=boosted_diagnostics)
            visited += 1
            print(f"Completed {visited}/{len(jobs)}: seed={seed}/{task}/{split}/base", flush=True)
        elif not (base_folder / "report.json").exists():
            raise RuntimeError("Residual jobs require the matching completed shared base")
        else:
            base_report = verify_artifacts(base_folder)
        base_payload = load_verified(base_folder / "best.pt")
        base_state = {name: value for name, value in base_payload["state"].items()
                      if name.startswith(VRQBoostedHead._base_prefixes())}
        base_state_sha = tensor_state_hash(base_state)
        for name in [value for value in args.configs if value != "base"]:
            folder = root / f"seed_{seed}" / task / split / name
            identity = {**base_identity, "config": name,
                        "base_checkpoint_sha256": sha256_file(base_folder / "best.pt"),
                        "base_state_sha256": base_state_sha,
                        "base_best_step": base_report["best_step"]}
            factory = lambda config_name, value_seed, state=base_state, selected_task=task: (
                VRQBoostedHead(config_name, selected_task, value_seed, state))
            if args.smoke:
                identity["online_smoke"] = online_smoke(
                    factory(name, seed).to(device), encoder, windows, tokens, device)
            if args.stage == "evaluate":
                report = verify_artifacts(folder, identity)
                head, checkpoint = restored_head(folder / "best.pt", identity, device, factory)
                _, validation = score_head(head, tokens, val, device, locked["microbatch_size"][task])
                if validation != report["best_validation"] or validation != checkpoint["validation"]:
                    raise RuntimeError("Repeated VRQ boosted evaluation differs")
            else:
                report = train_job(folder, identity, tokens, train, val, device,
                                   resume=args.resume, head_factory=factory,
                                   diagnostics_fn=boosted_diagnostics)
            restored, _ = restored_head(folder / "best.pt", identity, device, factory)
            if tensor_state_hash(restored.base_state()) != base_state_sha:
                raise RuntimeError("A VRQ residual job changed the frozen shared base")
            visited += 1
            print(f"Completed {visited}/{len(jobs)}: seed={seed}/{task}/{split}/{name}", flush=True)
    print(summarize(root, smoke=args.smoke), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
