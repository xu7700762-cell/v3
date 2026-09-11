"""Paired source-validation summary for the severity dynamics experiment."""
import itertools
from pathlib import Path

from .anchor_pilot import metrics_from_rows
from .io import read_csv, read_json, write_csv, write_json
from ..model.severity_dynamics import CONFIGS
from ..ssl_protocol import digest_json
from ..training.token_pilot import verify_artifacts


METRICS = ("accuracy", "balanced_accuracy", "AUROC", "BCE")


def summarize(root, *, smoke=False):
    root = Path(root)
    protocol_path = Path(__file__).resolve().parents[3] / "reproducibility/protocols/femba_severity_dynamics_pilot.json"
    locked = read_json(protocol_path)
    reports, rows_out = {}, []
    common, dataset_binding = None, {}
    for config, dataset in itertools.product(CONFIGS, ("vrq", "city")):
        folder = root / config / dataset / "fold_1"
        if not (folder / "report.json").exists():
            continue
        report = verify_artifacts(folder)
        identity = report["identity"]
        if identity["protocol"] != locked or identity["schema"] != locked["schema"]:
            raise RuntimeError("Result does not use the locked severity protocol")
        if (identity["config"], identity["dataset"], identity["task"]) != (config, dataset, "severity"):
            raise RuntimeError("Mixed severity result identity")
        if identity["smoke"] != smoke or identity["protocol"]["outer_test_scored"]:
            raise RuntimeError("Wrong experiment partition")
        binding = {key: identity[key] for key in ("protocol", "code_sha256", "encoder")}
        if common is None:
            common = binding
        if common != binding:
            raise RuntimeError("Mixed protocol, code or encoder provenance")
        predictions = read_csv(folder / "predictions.csv")
        samples = [{"sample_id": x["sample_id"], "subject_id": x["subject_id"], "session": x["session"],
                    "indices": [int(i) for i in str(x["window_indices"]).split(",")],
                    "label": x["y_true"]} for x in predictions]
        if digest_json(samples) != identity["val_sha256"]:
            raise RuntimeError("Validation predictions do not match the locked samples")
        current = (identity["data_sha256"], identity["train_sha256"], identity["val_sha256"],
                   report["first_epoch_order_sha256"],
                   [(x["sample_id"], x["subject_id"], x["window_indices"], x["y_true"])
                    for x in predictions])
        if dataset in dataset_binding and dataset_binding[dataset] != current:
            raise RuntimeError("Paired configurations do not share samples, labels and shuffle")
        dataset_binding[dataset] = current
        scored = metrics_from_rows(predictions)
        expected = report["best_validation"]
        if scored["loss"] != expected["loss"] or any(
            scored["metrics"][metric] != expected["metrics"][metric] for metric in METRICS[:-1]
        ):
            raise RuntimeError("Saved predictions do not reproduce metrics")
        reports[(config, dataset)] = report
        rows_out.append({
            "config": config, "dataset": dataset,
            "accuracy": expected["metrics"]["accuracy"],
            "balanced_accuracy": expected["metrics"]["balanced_accuracy"],
            "subject_macro_balanced_accuracy": expected["metrics"]["subject_macro_balanced_accuracy"],
            "AUROC": expected["metrics"]["AUROC"], "BCE": expected["loss"],
            "n_samples": len(predictions), "n_subjects": expected["metrics"]["n_subjects"],
            "positive_fraction": expected["positive_fraction"], "best_step": report["best_step"],
            "parameters": report["initialization"]["parameters"],
        })
    complete = len(reports) == len(CONFIGS) * 2
    output = root / "summary"
    result = {
        "status": ("smoke_passed" if smoke else "complete") if complete else "partial",
        "completed_jobs": len(reports), "expected_jobs": len(CONFIGS) * 2,
        "full_matrix_completed": complete, "evaluation_partition": "source_val",
        "outer_test_scored": False, "smoke": smoke,
        "limits": locked["limits"],
    }
    if smoke:
        result["all_two_step_checks"] = complete and all(r["global_step"] >= 2 for r in reports.values())
        result["all_online_checks"] = complete and all(
            r["identity"].get("online_smoke", {}).get("encoder_unchanged") for r in reports.values()
        )
        write_json(output / "aggregate_report.json", result)
        return result
    differences = []
    for dataset in ("vrq", "city"):
        values = {r["config"]: r for r in rows_out if r["dataset"] == dataset}
        if "dog" not in values:
            continue
        for baseline in ("base", "mlp", "poly", "fractional"):
            if baseline in values:
                differences.append({"contrast": f"dog - {baseline}", "dataset": dataset,
                                    **{metric: values["dog"][metric] - values[baseline][metric]
                                       for metric in METRICS}})
    if rows_out:
        write_csv(output / "dataset_metrics.csv", rows_out)
    if differences:
        write_csv(output / "paired_differences.csv", differences)
    write_json(output / "aggregate_report.json", result)
    pct = lambda value: "NA" if value is None else f"{100 * value:.2f}"
    dec = lambda value: "NA" if value is None else f"{value:.3f}"
    text = ["# FEMBA 无锚点严重度动态残差小试", "",
            f"状态：{result['status']}；{len(reports)}/{len(CONFIGS) * 2} 项。seed=2001，fold_1 source-val。", "",
            "结果用于结构开发，不是独立测试结果。EA 为逐被试无标签离线 transductive 预处理。", ""]
    for dataset in ("vrq", "city"):
        text += [f"## {dataset} / severity", "",
                 "| 配置 | ACC % | BACC % | subject-macro BACC % | AUROC % | BCE | N | 被试 |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows_out:
            if row["dataset"] == dataset:
                text.append(f"| {row['config']} | {pct(row['accuracy'])} | {pct(row['balanced_accuracy'])} | "
                            f"{pct(row['subject_macro_balanced_accuracy'])} | {pct(row['AUROC'])} | "
                            f"{dec(row['BCE'])} | {row['n_samples']} | {row['n_subjects']} |")
        text.append("")
    text += ["## 解释边界", "", *[f"- {value}" for value in locked["limits"]]]
    output.mkdir(parents=True, exist_ok=True)
    (output / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result
