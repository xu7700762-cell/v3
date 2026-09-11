"""Paired validation-only reports; incomplete matrices cannot be called complete."""
import itertools
from pathlib import Path

from .io import read_json, read_csv, write_json, write_csv
from .anchor_pilot import metrics_from_rows
from ..model.token_probe import CONFIGS
from ..training.token_pilot import verify_artifacts
from ..ssl_protocol import digest_json

METRICS = ("accuracy", "balanced_accuracy", "AUROC", "BCE")


def flat_metrics(report):
    v = report["best_validation"]
    return {**{k: v["metrics"][k] for k in METRICS[:-1]}, "BCE": v["loss"]}


def summarize(root, *, smoke=False):
    root = Path(root)
    locked = read_json(Path(__file__).resolve().parents[3] / "reproducibility/protocols/femba_reference_free_token_pilot.json")
    reports, metrics, identities = {}, [], {}
    common = None
    for config, task, dataset in itertools.product(CONFIGS, ("state", "severity"), ("vrq", "city")):
        folder = root / config / task / dataset / "fold_1"
        if not (folder / "report.json").exists():
            continue
        r = verify_artifacts(folder)
        ident = r["identity"]
        if ident["protocol"] != locked or ident["schema"] != locked["schema"]:
            raise RuntimeError("Result does not use the locked pilot protocol")
        if (ident["config"], ident["task"], ident["dataset"]) != (config, task, dataset) or ident["smoke"] != smoke:
            raise RuntimeError("Mixed job results")
        if ident["protocol"]["seed"] != 2001 or ident["protocol"]["fold"] != "fold_1" or ident["protocol"]["outer_test_scored"]:
            raise RuntimeError("Not the locked validation pilot")
        binding = {k: ident[k] for k in ("protocol", "code_sha256", "encoder")}
        if common is None:
            common = binding
        if common != binding:
            raise RuntimeError("Mixed protocol, code or encoder provenance")
        rows = read_csv(folder / "predictions.csv")
        sample_records = [{"sample_id": x["sample_id"], "subject_id": x["subject_id"], "session": x["session"],
                           "indices": [int(i) for i in str(x["window_indices"]).split(",")], "label": x["y_true"]} for x in rows]
        if digest_json(sample_records) != ident["val_sha256"] or {x["subject_id"] for x in rows} != set(ident["split"]["val_subjects"]):
            raise RuntimeError("Scoring rows do not match the locked validation partition")
        scored = metrics_from_rows(rows)
        expected = r["best_validation"]
        if scored["loss"] != expected["loss"] or any(scored["metrics"][k] != expected["metrics"][k] for k in METRICS[:-1]):
            raise RuntimeError("Predictions do not reproduce metrics")
        pair = (task, dataset)
        current = (ident["data_sha256"], ident["train_sha256"], ident["val_sha256"], r["first_epoch_order_sha256"],
                   [(x["sample_id"], x["subject_id"], x["window_indices"], x["y_true"]) for x in rows])
        if pair in identities and identities[pair] != current:
            raise RuntimeError("Comparison samples, labels, shuffle or data differ")
        identities[pair] = current
        reports[(config, task, dataset)] = r
        metrics.append({"config": config, "task": task, "dataset": dataset, **flat_metrics(r),
                        "n_samples": len(rows), "n_subjects": len({x["subject_id"] for x in rows}),
                        "positive_fraction": expected["positive_fraction"], "majority_accuracy": expected["majority_accuracy"],
                        "best_step": r["best_step"], "total_steps": r["global_step"],
                        "last_BCE": r["last_validation"]["loss"], "parameters": r["initialization"]["parameters"],
                        "trainable_parameters": r["initialization"]["trainable_parameters"]})
    complete = len(reports) == 52
    out = root / "summary"
    result = {"status": ("smoke_passed" if smoke else "complete") if complete else "partial",
              "completed_jobs": len(reports), "expected_jobs": 52, "smoke": smoke,
              "evaluation_partition": "source_val", "outer_test_scored": False,
              "full_matrix_completed": complete, "limits": common["protocol"]["limits"] if common else []}
    if smoke:
        result["all_two_step_checks"] = complete and all(r["global_step"] >= 2 for r in reports.values())
        result["all_online_checks"] = complete and all(r["identity"].get("online_smoke", {}).get("encoder_unchanged") for r in reports.values())
        write_json(out / "aggregate_report.json", result)
        return result
    macros = []
    for c, t in itertools.product(CONFIGS, ("state", "severity")):
        values = [r for r in metrics if r["config"] == c and r["task"] == t]
        if len(values) == 2:
            macros.append({"config": c, "task": t, **{m: None if any(v[m] is None for v in values)
                                                       else sum(v[m] for v in values) / 2 for m in METRICS}})
    comparisons = []
    contrasts = [(f"R{b}_{h}", f"R{a}_{h}") for h in ("dog", "mlp", "poly") for a, b in ((0, 1), (1, 2))]
    contrasts += [(f"R{r}_dog", f"R{r}_{h}") for r in range(3) for h in ("mlp", "poly")]
    contrasts += [("R2_dog", c) for c in ("linear", "R2_order1", "R2_no_dog", "R2_order1_no_dog")]
    for a, b in contrasts:
        for t, d in itertools.product(("state", "severity"), ("vrq", "city", "macro")):
            pool = macros if d == "macro" else [m for m in metrics if m["dataset"] == d]
            av = next((v for v in pool if v["config"] == a and v["task"] == t), None)
            bv = next((v for v in pool if v["config"] == b and v["task"] == t), None)
            if av and bv:
                comparisons.append({"contrast": f"{a} - {b}", "task": t, "dataset": d,
                                    **{m: None if av[m] is None or bv[m] is None else av[m] - bv[m] for m in METRICS}})
    if metrics:
        write_csv(out / "dataset_metrics.csv", metrics)
    if macros:
        write_csv(out / "dataset_macros.csv", macros)
    if comparisons:
        write_csv(out / "paired_differences.csv", comparisons)
    write_json(out / "aggregate_report.json", result)
    text = ["# FEMBA 无锚点逐 token 映射小试（保留无标签 EA）", "",
            f"状态：{result['status']}；{len(reports)}/52 项。seed=2001，fold_1 source-val；不是独立测试结果。", "",
            "EA 使用每名被试协议内全部无标签窗口，属于离线 transductive 处理。静息身份不进入预测或校准。", ""]
    def value(x, percent=True):
        return "NA" if x is None else f"{x * (100 if percent else 1):.2f}"
    for t, d in itertools.product(("state", "severity"), ("vrq", "city")):
        text += [f"## {d} / {t}", "", "| 配置 | ACC % | BACC % | AUROC % | BCE | 样本数 | 最佳步 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for v in metrics:
            if (v["task"], v["dataset"]) == (t, d):
                text.append(f"| {v['config']} | {value(v['accuracy'])} | {value(v['balanced_accuracy'])} | {value(v['AUROC'])} | {value(v['BCE'], False)} | {v['n_samples']} | {v['best_step']} |")
        text.append("")
    text += ["## 解释边界", "", "不自动将正差值解释为统计显著；只改善汇总而未超过同结构 MLP，不能归因于 DoG。",
             "关闭分支的推理诊断与重新训练的成分消融分别保存在 diagnostics.json 和配对表。", "",
             *[f"- {s}" for s in result["limits"]]]
    (out / "RESULTS.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return result
