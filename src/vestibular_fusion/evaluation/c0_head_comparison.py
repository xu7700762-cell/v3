"""Read-only paired scoring of the locked 80 nonlinear and 40 linear folds."""
import itertools
from pathlib import Path

import numpy as np

from .anchor_pilot import metrics_from_rows
from .io import write_csv, write_json, sha256_file
from ..training.c0_head_comparison import (SCHEMA, HEADS, VARIANTS, TASKS, DATASETS, FOLDS,
                                          verify_artifacts)

METRICS = ("accuracy", "balanced_accuracy", "AUROC", "BCE")


def flat_metrics(rows):
    scored = metrics_from_rows(rows)
    return {**{k: scored["metrics"][k] for k in METRICS[:-1]}, "BCE": scored["loss"]}


def summary_from_folds(folds):
    """Require the entire registered matrix, including the immutable linear control."""
    expected = set(itertools.product(("linear", *HEADS), TASKS, VARIANTS, DATASETS, FOLDS))
    if set(folds) != expected:
        raise ValueError("Summary requires 80 nonlinear and 40 verified linear folds")
    paired, grouped, fold_table = {}, {}, []
    for (head, task, variant, dataset, fold), rows in sorted(folds.items()):
        identity = [(r["sample_id"], r["subject_id"], r["session"], str(r["window_indices"]), int(r["y_true"])) for r in rows]
        pair_key = (task, dataset, fold)
        if paired.setdefault(pair_key, identity) != identity:
            raise RuntimeError("Scored samples or labels differ across heads/variants")
        if len(set(r["sample_id"] for r in rows)) != len(rows):
            raise RuntimeError("Duplicate scored samples")
        metrics = flat_metrics(rows)
        grouped.setdefault((head, task, variant, dataset), []).append((fold, rows, metrics))
        fold_table.append({"head": head, "task": task, "variant": variant, "dataset": dataset,
                           "fold": fold, "n_samples": len(rows), **metrics})
    dataset_table = []
    for (head, task, variant, dataset), groups in sorted(grouped.items()):
        groups.sort(key=lambda x: x[0])
        rows = [r for _, data, _ in groups for r in data]
        subjects = [set(r["subject_id"] for r in data) for _, data, _ in groups]
        if sum(map(len, subjects)) != len(set().union(*subjects)):
            raise RuntimeError("Outer-test subject occurs in multiple folds")
        scored, metrics = metrics_from_rows(rows), flat_metrics(rows)
        entry = {"head": head, "task": task, "variant": variant, "dataset": dataset,
                 "n_samples": len(rows), "n_subjects": len(set().union(*subjects)),
                 "negative_count": scored["class_counts"]["0"], "positive_count": scored["class_counts"]["1"],
                 "positive_fraction": scored["positive_fraction"], "majority_accuracy": scored["majority_accuracy"]}
        for metric in METRICS:
            values = [m[metric] for _, _, m in groups]
            entry[metric] = metrics[metric]
            entry[metric + "_fold_mean"] = None if any(x is None for x in values) else float(np.mean(values))
            entry[metric + "_fold_std"] = None if any(x is None for x in values) else float(np.std(values, ddof=1))
        dataset_table.append(entry)
    macros = []
    for head, task, variant in itertools.product(("linear", *HEADS), TASKS, VARIANTS):
        selected = [r for r in dataset_table if (r["head"], r["task"], r["variant"]) == (head, task, variant)]
        macros.append({"head": head, "task": task, "variant": variant, "dataset": "dataset_macro",
            **{m: None if any(r[m] is None for r in selected) else float(np.mean([r[m] for r in selected])) for m in METRICS}})
    lookup = {(r["head"], r["task"], r["variant"], r["dataset"]): r for r in dataset_table + macros}
    comparisons = []
    pairs = (("fractional_dog_polykan", "linear"), ("mlp", "linear"), ("fractional_dog_polykan", "mlp"))
    for task, dataset, variant, (a, b), metric in itertools.product(TASKS, (*DATASETS, "dataset_macro"), VARIANTS, pairs, METRICS):
        x, y = lookup[a, task, variant, dataset][metric], lookup[b, task, variant, dataset][metric]
        comparisons.append({"task": task, "dataset": dataset, "comparison": f"{a}-{b}", "variant_or_head": variant,
            "metric": metric, "difference": None if x is None or y is None else (x - y) * (1 if metric == "BCE" else 100),
            "unit": "BCE" if metric == "BCE" else "percentage_points"})
    for task, dataset, head, metric in itertools.product(TASKS, (*DATASETS, "dataset_macro"), ("linear", *HEADS), METRICS):
        x, y = lookup[head, task, "A4", dataset][metric], lookup[head, task, "A3", dataset][metric]
        comparisons.append({"task": task, "dataset": dataset, "comparison": "A4-A3", "variant_or_head": head,
            "metric": metric, "difference": None if x is None or y is None else (x - y) * (1 if metric == "BCE" else 100),
            "unit": "BCE" if metric == "BCE" else "percentage_points"})
    return {"folds": fold_table, "datasets": dataset_table, "dataset_macros": macros, "comparisons": comparisons}


def summarize(root, identities, baseline):
    expected = set(itertools.product(HEADS, TASKS, VARIANTS, DATASETS, FOLDS))
    actual = {tuple(i[k] for k in ("head", "task", "variant", "dataset", "fold_id")) for i in identities}
    if actual != expected or len(identities) != 80 or any(i["smoke"] or i["checkpoint_schema"] != SCHEMA for i in identities):
        raise ValueError("Complete summary requires exactly 80 non-smoke jobs")
    folds, pairs, provenance = {}, {}, []
    for key, value in baseline["jobs"].items():
        folds[("linear", *key)] = value["rows"]
    for identity in identities:
        head, task, variant, dataset, fold = (identity[k] for k in ("head", "task", "variant", "dataset", "fold_id"))
        job = Path(root) / head / task / variant / dataset / fold
        report, rows = verify_artifacts(job, root, identity)
        previous = baseline["jobs"][task, variant, dataset, fold]["report"]
        init = report["initialization"]
        for key in ("selection_order_sha256", "refit_order_sha256"):
            if report[key] != previous[key]:
                raise RuntimeError("Baseline shuffle changed")
        if init["encoder_sha256"] != previous["initialization"]["encoder_sha256"]:
            raise RuntimeError("Baseline encoder initialization differs")
        checks = {("head", head): init["head_sha256"], ("output",): init["output_layer_sha256"],
                  ("norm",): init["normalization_sha256"], ("encoder",): init["encoder_sha256"],
                  ("protocol",): identity["protocol"], ("binding",): identity["binding"]}
        for key, value in checks.items():
            if pairs.setdefault(key, value) != value:
                raise RuntimeError(f"Head pairing mismatch: {key}")
        folds[head, task, variant, dataset, fold] = rows
        provenance.append({"job": "/".join((head, task, variant, dataset, fold)),
                           "report_sha256": sha256_file(job / "report.json")})
    result = {"status": "complete", "new_jobs": 80, "linear_baseline_jobs": 40,
              "full_fivefold_completed": True, "schema": SCHEMA, "protocol": identities[0]["protocol"],
              "binding": identities[0]["binding"], **summary_from_folds(folds),
              "baseline_manifest_sha256": baseline["manifest_sha256"], "new_report_manifest": provenance,
              "limits": identities[0]["protocol"]["limits"]}
    destination = Path(root) / "summary"
    for key, filename in (("folds", "fold_metrics.csv"), ("datasets", "dataset_metrics.csv"),
                          ("dataset_macros", "dataset_macros.csv"), ("comparisons", "paired_comparisons.csv")):
        write_csv(destination / filename, result[key])
    write_json(destination / "aggregate_report.json", result)
    lines = ["# FEMBA C0 无锚点下游比较", "", "80 个新实验与 40 个已核验线性基线；seed=2001。", ""]
    for dataset in (*DATASETS, "dataset_macro"):
        for task in TASKS:
            lines += [f"## {dataset} / {task}", "", "|组别|下游|ACC %|BACC %|AUROC %|BCE|",
                      "|---|---|---:|---:|---:|---:|"]
            for r in result["datasets"] + result["dataset_macros"]:
                if r["dataset"] == dataset and r["task"] == task:
                    values = ["NA" if r[m] is None else f"{r[m] * (1 if m == 'BCE' else 100):.3f}" for m in METRICS]
                    lines.append("|" + "|".join([r["variant"], r["head"], *values]) + "|")
            lines.append("")
    lines += ["## 解释边界", "", *[f"- {v}" for v in result["limits"]], ""]
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return result
