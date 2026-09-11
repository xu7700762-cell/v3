"""Aggregate paired VRQ state and severity shared-base development runs."""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np

from .anchor_pilot import metrics_from_rows
from .io import read_csv, read_json, write_csv, write_json
from ..model.severity_dynamics import CONFIGS
from ..ssl_protocol import digest_json
from ..training.token_pilot import verify_artifacts


METRICS = ("accuracy", "balanced_accuracy", "AUROC", "BCE")


def summarize(root, *, smoke=False):
    root = Path(root)
    locked = read_json(Path(__file__).resolve().parents[3]
                       / "reproducibility/protocols/femba_vrq_boosted_development.json")
    expected = list(itertools.product(locked["seeds"], locked["tasks"],
                                      locked["development_splits"], CONFIGS))
    reports, metrics, pair_binding = {}, [], {}
    common = None
    for seed, task, split, config in expected:
        folder = root / f"seed_{seed}" / task / split / config
        if not (folder / "report.json").exists():
            continue
        report = verify_artifacts(folder)
        identity = report["identity"]
        job_protocol = dict(identity["protocol"])
        job_seed = job_protocol.pop("seed", None)
        if job_protocol != locked or job_seed != seed or identity["schema"] != locked["schema"]:
            raise RuntimeError("Result does not use the locked VRQ boosted protocol")
        if (identity["config"], identity["split_name"], identity["dataset"], identity["task"]) != (
            config, split, "vrq", task
        ) or identity["smoke"] != smoke:
            raise RuntimeError("Mixed VRQ boosted result identity")
        binding = (identity["code_sha256"], identity["encoder"], identity["data_sha256"])
        if common is None:
            common = binding
        if common != binding:
            raise RuntimeError("Mixed code, encoder or data provenance")
        rows = read_csv(folder / "predictions.csv")
        samples = [{"subject_id": row["subject_id"], "session": row["session"],
                    "indices": [int(index) for index in str(row["window_indices"]).split(",")],
                    "sample_id": row["sample_id"], "label": row["y_true"]} for row in rows]
        if digest_json(samples) != identity["val_sha256"]:
            raise RuntimeError("Saved rows do not match the locked VRQ samples")
        key = (seed, task, split)
        current = (identity["train_sha256"], identity["val_sha256"],
                   report["first_epoch_order_sha256"],
                   [(row["sample_id"], row["window_indices"], row["y_true"]) for row in rows])
        if key in pair_binding and pair_binding[key] != current:
            raise RuntimeError("A paired VRQ run changed samples, labels or shuffle")
        pair_binding[key] = current
        scored = metrics_from_rows(rows)
        validation = report["best_validation"]
        if scored["loss"] != validation["loss"] or any(
            scored["metrics"][name] != validation["metrics"][name]
            for name in ("accuracy", "balanced_accuracy", "AUROC")
        ):
            raise RuntimeError("VRQ predictions do not reproduce metrics")
        if config != "base" and identity["base_state_sha256"] != report["initialization"]["base_sha256"]:
            raise RuntimeError("Residual initialization does not contain the selected VRQ base")
        reports[(seed, task, split, config)] = report
        metrics.append({"seed": seed, "task": task, "split": split, "config": config,
                        "accuracy": validation["metrics"]["accuracy"],
                        "balanced_accuracy": validation["metrics"]["balanced_accuracy"],
                        "AUROC": validation["metrics"]["AUROC"], "BCE": validation["loss"],
                        "n_samples": validation["metrics"]["n_samples"],
                        "n_subjects": validation["metrics"]["n_subjects"],
                        "best_step": report["best_step"],
                        "parameters": report["initialization"]["parameters"],
                        "trainable_parameters": report["initialization"]["trainable_parameters"]})
    complete = len(reports) == len(expected)
    output = root / "summary"
    result = {"status": "complete" if complete and not smoke else
              "smoke_passed" if reports and smoke else "partial",
              "completed_jobs": len(reports), "expected_jobs": len(expected),
              "full_matrix_completed": complete, "outer_test_scored": False,
              "evaluation_partition": "nested_source_validation", "smoke": smoke,
              "limits": locked["limits"]}
    if smoke:
        result["all_two_step_checks"] = bool(reports) and all(
            report["global_step"] >= 2 for report in reports.values())
        result["all_online_checks"] = bool(reports) and all(
            report["identity"].get("online_smoke", {}).get("encoder_unchanged")
            for report in reports.values())
        write_json(output / "aggregate_report.json", result)
        return result
    aggregates, differences = [], []
    for task, config in itertools.product(locked["tasks"], CONFIGS):
        values = [row for row in metrics if row["task"] == task and row["config"] == config]
        if values:
            aggregate = {"task": task, "config": config, "runs": len(values)}
            for metric in METRICS:
                numbers = [row[metric] for row in values if row[metric] is not None]
                aggregate[metric + "_mean"] = float(np.mean(numbers)) if numbers else None
                aggregate[metric + "_std"] = float(np.std(numbers)) if numbers else None
            aggregates.append(aggregate)
    for seed, task, split in itertools.product(locked["seeds"], locked["tasks"],
                                               locked["development_splits"]):
        values = {row["config"]: row for row in metrics
                  if row["seed"] == seed and row["task"] == task and row["split"] == split}
        if "dog" in values:
            for baseline in ("base", "mlp", "poly", "fractional"):
                if baseline in values:
                    differences.append({"seed": seed, "task": task, "split": split,
                                        "contrast": f"dog - {baseline}",
                                        **{metric: values["dog"][metric] - values[baseline][metric]
                                           for metric in METRICS}})
    promotions = {}
    for task in locked["tasks"]:
        promotion = {"evaluated": complete, "passed": False}
        if complete:
            dog_mlp = [row for row in differences
                       if row["task"] == task and row["contrast"] == "dog - mlp"]
            dog_poly = [row for row in differences
                        if row["task"] == task and row["contrast"] == "dog - poly"]
            promotion.update(
                dog_minus_mlp_mean_BACC=float(np.mean(
                    [row["balanced_accuracy"] for row in dog_mlp])),
                dog_minus_poly_mean_BACC=float(np.mean(
                    [row["balanced_accuracy"] for row in dog_poly])),
                dog_positive_runs_vs_mlp=sum(
                    row["balanced_accuracy"] > 0 for row in dog_mlp),
            )
            ablation = []
            for seed, split in itertools.product(locked["seeds"], locked["development_splits"]):
                report = reports[(seed, task, split, "dog")]
                diagnostic = read_json(root / f"seed_{seed}" / task / split / "dog/diagnostics.json")
                ablation.append(report["best_validation"]["metrics"]["balanced_accuracy"]
                                - diagnostic["counterfactual_no_dog"]["metrics"]["balanced_accuracy"])
            promotion["dog_minus_no_dog_mean_BACC"] = float(np.mean(ablation))
            rule = locked["promotion_rule"]
            promotion["passed"] = (
                promotion["dog_minus_mlp_mean_BACC"] > 0
                and promotion["dog_minus_poly_mean_BACC"] > 0
                and promotion["dog_positive_runs_vs_mlp"] >= rule["dog_positive_runs_minimum_of_nine"]
                and promotion["dog_minus_no_dog_mean_BACC"] > 0
            )
        promotions[task] = promotion
    result["promotion"] = promotions
    if metrics:
        write_csv(output / "run_metrics.csv", metrics)
    if aggregates:
        write_csv(output / "aggregate_metrics.csv", aggregates)
    if differences:
        write_csv(output / "paired_differences.csv", differences)
    write_json(output / "aggregate_report.json", result)
    pct = lambda value: "NA" if value is None else f"{100 * value:.2f}"
    text = ["# FEMBA VRQ 共享基线残差开发实验", "",
            f"状态：{result['status']}；{len(reports)}/{len(expected)} 项。outer-test 未评分。", ""]
    for task in locked["tasks"]:
        text += [f"## {task}", "", "| 配置 | ACC % | BACC % | AUROC % | BCE | 运行数 |",
                 "|---|---:|---:|---:|---:|---:|"]
        for row in aggregates:
            if row["task"] == task:
                text.append(f"| {row['config']} | {pct(row['accuracy_mean'])} | "
                            f"{pct(row['balanced_accuracy_mean'])} | {pct(row['AUROC_mean'])} | "
                            f"{row['BCE_mean']:.3f} | {row['runs']} |")
        text += ["", f"DoG 晋级：`passed={promotions[task]['passed']}`", ""]
    text += ["## 解释边界", "", *[f"- {value}" for value in locked["limits"]]]
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result
