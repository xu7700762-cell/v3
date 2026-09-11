"""Run the locked reference-free FEMBA severity dynamics development experiment."""
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
    p.add_argument("--stage", choices=("preflight", "train", "evaluate", "all", "summarize"), default="all")
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    p.add_argument("--datasets", nargs="+", choices=("vrq", "city"), default=["vrq", "city"])
    p.add_argument("--config", type=Path, default=ROOT / "configs/paths.local.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "outputs/femba_severity_dynamics_pilot_v1")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    from vestibular_fusion.evaluation.io import read_json, sha256_file, write_json
    from vestibular_fusion.ssl_protocol import assert_empty, digest_json
    args = parser().parse_args(argv)
    for key in ("configs", "datasets"):
        if len(getattr(args, key)) != len(set(getattr(args, key))):
            raise ValueError(f"Duplicate --{key}")
    output = args.output_root.resolve()
    normalized = str(output).replace("\\", "/").lower()
    if not (normalized.startswith("d:/") or normalized.startswith("/mnt/d/")):
        raise ValueError("All experiment results and caches must be on D")
    root = output / ("smoke" if args.smoke else "pilot") / "seed_2001"
    jobs = list(itertools.product(args.datasets, args.configs))
    if args.dry_run:
        import json
        print(json.dumps({"jobs": jobs, "job_count": len(jobs), "root": str(root),
                          "seed": 2001, "fold": "fold_1", "task": "severity",
                          "outer_test_loaded": False, "outer_test_scored": False}, indent=2))
        return 0
    from vestibular_fusion.evaluation.severity_dynamics import summarize
    if args.stage == "summarize":
        print(summarize(root, smoke=args.smoke))
        return 0
    import torch
    from vestibular_fusion.config import load_config
    from vestibular_fusion.model.linear_probe import tensor_state_hash
    from vestibular_fusion.model.severity_dynamics import SeverityDynamicsHead
    from vestibular_fusion.preflight import _check_environment
    from vestibular_fusion.training.severity_dynamics import severity_diagnostics
    from vestibular_fusion.training.ssl_ablation import seed_runtime
    from vestibular_fusion.training.token_data import load_source_data, sample_hash
    from vestibular_fusion.training.token_pilot import (online_smoke, prepare_encoder, require_space,
        restored_head, score_head, token_cache, train_job, verify_artifacts)
    if os.environ.get("PYTHONHASHSEED") != "2001":
        os.environ["PYTHONHASHSEED"] = "2001"
        os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                                  *(sys.argv[1:] if argv is None else argv)])
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    seed_runtime(2001)
    config = load_config(args.config)
    environment = _check_environment()
    device = torch.device("cuda")
    protocol = read_json(ROOT / "reproducibility/protocols/femba_severity_dynamics_pilot.json")
    factory = lambda name, seed: SeverityDynamicsHead(name, seed, protocol["minimum_severity_windows"])
    code = {str(p.relative_to(ROOT)): sha256_file(p)
            for p in sorted((ROOT / "src/vestibular_fusion").rglob("*.py"))}
    code["entrypoint"] = sha256_file(Path(__file__))
    code_sha = digest_json(code)
    require_space(output, 800 * 1024**2)
    if args.stage in ("train", "all") and not args.resume:
        for dataset, name in jobs:
            assert_empty(root / name / dataset / "fold_1")
    encoder, encoder_identity = prepare_encoder(config, output, device)
    completed = 0
    for dataset in args.datasets:
        windows, partitions, data_identity = load_source_data(
            config, dataset, severity_windows=None
        )
        train, val = partitions["severity"]["train"], partitions["severity"]["val"]
        if any(len(example.indices) < protocol["minimum_severity_windows"] for example in train + val):
            raise RuntimeError("A severity path is shorter than the locked minimum")
        require_space(output, windows.shape[0] * 80 * 525 * 4 + 800 * 1024**2)
        write_json(root / "data" / f"{dataset}.json", {
            "identity": data_identity,
            "samples": {role: [example.identity() for example in values]
                        for role, values in partitions["severity"].items()},
        })
        if args.stage == "preflight":
            print(f"preflight {dataset}: {len(windows)} source windows, "
                  f"{len(train)}/{len(val)} train/val severity paths, no outer-test EEG loaded", flush=True)
            continue
        tokens, cache = token_cache(output, windows, data_identity, encoder, encoder_identity, device)
        for name in args.configs:
            identity = {
                "schema": protocol["schema"], "protocol": protocol, "config": name,
                "task": "severity", "dataset": dataset, "smoke": args.smoke,
                "encoder": encoder_identity, "cache": cache,
                "data_sha256": digest_json(data_identity), "split": data_identity["split"],
                "train_sha256": sample_hash(train), "val_sha256": sample_hash(val),
                "code_sha256": code_sha, "environment": environment,
            }
            if args.smoke:
                identity["online_smoke"] = online_smoke(factory(name, 2001).to(device), encoder,
                                                        windows, tokens, device)
            folder = root / name / dataset / "fold_1"
            if args.stage == "evaluate":
                report = verify_artifacts(folder, identity)
                head, checkpoint = restored_head(folder / "best.pt", identity, device, factory)
                _, validation = score_head(head, tokens, val, device, protocol["microbatch_size"]["severity"])
                if validation != report["best_validation"] or validation != checkpoint["validation"]:
                    raise RuntimeError("Repeated source-validation evaluation differs")
            else:
                report = train_job(folder, identity, tokens, train, val, device,
                                   resume=args.resume, head_factory=factory,
                                   diagnostics_fn=severity_diagnostics)
            if tensor_state_hash(encoder.state_dict()) != encoder_identity["encoder_sha256"]:
                raise RuntimeError("Shared frozen encoder changed")
            completed += 1
            write_json(root / "progress.json", {
                "visited_jobs": completed, "requested_jobs": len(jobs),
                "last_job": [dataset, name], "last_best_validation": report["best_validation"],
            })
            print(f"Completed {completed}/{len(jobs)}: {dataset}/severity/{name}", flush=True)
        del tokens, windows
    if args.stage == "preflight":
        write_json(root / "preflight.json", {"status": "passed", "environment": environment,
                   "encoder": encoder_identity, "protocol": protocol, "code": code})
    else:
        print(summarize(root, smoke=args.smoke), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
