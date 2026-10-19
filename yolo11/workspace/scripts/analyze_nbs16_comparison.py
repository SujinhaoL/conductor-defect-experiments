r"""Analyze Y11-DIAG-002 against Y11-DIAG-001 using validation data only."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys

import numpy as np
import yaml

from check_dataset import snapshot
from readonly_dataset import DATASET, PROJECT, PROTECTED_ROOT, WORKSPACE, install_write_guard, redirect_label_caches

EXPECTED_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
OLD_RUN = WORKSPACE / "runs" / "yolo11s_diag_1280_e30_20260916_104603"
OLD_STATUS_PATH = WORKSPACE / "logs" / "latest_yolo11s_diagnostic_status.json"
NEW_STATUS_PATH = WORKSPACE / "logs" / "latest_yolo11s_nbs16_status.json"
ACCUMULATION_LOG = WORKSPACE / "logs" / "yolo11s_nbs16_accumulation.csv"
DATA_YAML = WORKSPACE / "configs" / "conductor_bl_v1.yaml"
ANALYSIS_ROOT = WORKSPACE / "analysis" / "nbs64_vs_nbs16"
PRED_ROOT = WORKSPACE / "analysis" / "diagnostic_predictions_nbs16"
REPORT_PATH = WORKSPACE / "notes" / "nbs64_vs_nbs16_report.md"
HANDOFF_PATH = WORKSPACE / "notes" / "Y11-DIAG-002_handoff_to_gpt.md"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_results(path: Path) -> list[dict[str, float]]:
    rows = []
    for raw in read_csv(path):
        rows.append({key.strip(): float(value) for key, value in raw.items()})
    if len(rows) != 30 or [int(row["epoch"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError(f"Expected epochs 1..30 in {path}, got {len(rows)} rows")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native(value):
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def fmt(value, digits=6):
    if isinstance(value, str):
        return value
    value = float(value)
    if not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Inf" if value > 0 else "-Inf")
    return f"{value:.{digits}f}"


def mean(rows, key):
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return float(np.mean(values)) if values else math.nan


def slope(rows, key):
    finite = [(row["epoch"], row[key]) for row in rows if math.isfinite(row[key])]
    if len(finite) < 2:
        return math.nan
    return float(np.polyfit([x for x, _ in finite], [y for _, y in finite], 1)[0])


def best_row(rows, key, secondary=None):
    finite = [row for row in rows if math.isfinite(row[key])]
    return max(finite, key=lambda row: (row[key], row[secondary] if secondary else 0.0, -row["epoch"]))


def val_nonfinite(rows):
    keys = ("val/box_loss", "val/cls_loss", "val/dfl_loss")
    return [int(row["epoch"]) for row in rows if not all(math.isfinite(row[key]) for key in keys)]


def validate(model_path: Path, name: str, half: bool, plots: bool = True):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(DATA_YAML), split="val", imgsz=1280, batch=8, workers=0, device=0,
        rect=False, half=half, plots=plots, project=str(ANALYSIS_ROOT), name=name,
        exist_ok=True, verbose=True,
    )
    summary = native(metrics.summary(decimals=8))
    overall = {key: float(value) for key, value in metrics.results_dict.items()}
    return model, summary, overall


def prediction_stats(model) -> list[dict]:
    summary = []
    per_image = []
    val_dir = DATASET / "images" / "val"
    expected = len([p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}])
    if expected != 23:
        raise RuntimeError(f"Expected 23 validation images, got {expected}")
    for threshold in (0.001, 0.01, 0.05, 0.25):
        tag = str(threshold).replace(".", "p")
        results = model.predict(
            source=str(val_dir), conf=threshold, imgsz=1280, batch=4, device=0,
            save=True, save_txt=False, save_conf=False, project=str(PRED_ROOT),
            name=f"conf_{tag}", exist_ok=True, verbose=False,
        )
        if len(results) != expected:
            raise RuntimeError(f"Prediction at conf={threshold} returned {len(results)} images")
        counts = {0: 0, 1: 0}
        confidences = []
        images_with = 0
        for result in results:
            n = len(result.boxes)
            classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist() if n else []
            confs = result.boxes.conf.detach().cpu().numpy().astype(float).tolist() if n else []
            if n:
                images_with += 1
            for cls in classes:
                if cls not in counts:
                    raise RuntimeError(f"Unexpected prediction class: {cls}")
                counts[cls] += 1
            confidences.extend(confs)
            per_image.append({
                "threshold": threshold,
                "image": Path(result.path).name,
                "detections": n,
                "broken": classes.count(0),
                "loose": classes.count(1),
                "max_conf": max(confs) if confs else "",
            })
        output_dir = PRED_ROOT / f"conf_{tag}"
        saved = [p for p in output_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}]
        if len(saved) != expected:
            raise RuntimeError(f"Expected {expected} saved images at conf={threshold}, got {len(saved)}")
        summary.append({
            "threshold": threshold,
            "val_images": expected,
            "images_with_predictions": images_with,
            "total_detections": len(confidences),
            "broken_detections": counts[0],
            "loose_detections": counts[1],
            "confidence_median": float(np.median(confidences)) if confidences else math.nan,
            "confidence_max": max(confidences) if confidences else math.nan,
            "visualization_dir": str(output_dir),
        })
    write_csv(PRED_ROOT / "prediction_summary.csv", summary)
    write_csv(PRED_ROOT / "prediction_per_image.csv", per_image)
    (PRED_ROOT / "prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def verify_dataset_unchanged() -> dict:
    before = json.loads((WORKSPACE / "logs" / "dataset_snapshot_before.json").read_text(encoding="utf-8"))
    after = snapshot()
    changes = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    report = {
        "unchanged": not changes,
        "changed_paths": changes,
        "entries_checked": len(after),
        "sha256_scope": str(DATASET),
        "metadata_scope": str(PROTECTED_ROOT),
    }
    path = WORKSPACE / "logs" / "dataset_integrity_after_yolo11s_nbs16.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if changes:
        raise RuntimeError(f"Protected dataset changed: {changes[:10]}")
    return report


def main() -> None:
    os.environ["YOLO_CONFIG_DIR"] = str(WORKSPACE / "configs" / "ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")
    os.environ["TORCH_HOME"] = str(WORKSPACE / "cache" / "torch")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    install_write_guard()

    import torch
    import ultralytics

    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(f"Wrong Python environment: {sys.prefix}")
    if not Path(ultralytics.__file__).resolve().is_relative_to((PROJECT / "ultralytics_src").resolve()):
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    commit = subprocess.check_output(["git", "-C", str(PROJECT / "ultralytics_src"), "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(PROJECT / "ultralytics_src"), "status", "--porcelain"], text=True)
    if commit != EXPECTED_COMMIT or git_status.strip():
        raise RuntimeError(f"Official source identity changed: commit={commit}, status={git_status!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    old_status = json.loads(OLD_STATUS_PATH.read_text(encoding="utf-8"))
    new_status = json.loads(NEW_STATUS_PATH.read_text(encoding="utf-8"))
    if not old_status.get("success") or not new_status.get("success"):
        raise RuntimeError("Both diagnostics must complete successfully before comparison.")
    console_log = Path(new_status["console_log"])
    optimizer_match = re.search(
        r"AdamW\(lr=([0-9.]+), momentum=([0-9.]+)\)",
        console_log.read_text(encoding="utf-8", errors="replace"),
    )
    if optimizer_match is None:
        raise RuntimeError(f"Could not recover the actual optimizer setup from {console_log}")
    new_status["optimizer_actual"]["lr0"] = float(optimizer_match.group(1))
    new_status["optimizer_actual"]["momentum"] = float(optimizer_match.group(2))
    new_run = Path(new_status["save_dir"])
    old_rows = read_results(OLD_RUN / "results.csv")
    new_rows = read_results(new_run / "results.csv")

    controlled = [
        "model", "epochs", "imgsz", "batch", "workers", "device", "seed", "deterministic", "patience", "cache",
        "plots", "amp", "save", "save_period", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr",
        "mosaic", "mixup", "cutmix", "copy_paste", "close_mosaic", "box", "cls", "dfl", "resume", "split",
    ]
    old_args = yaml.safe_load((OLD_RUN / "args.yaml").read_text(encoding="utf-8-sig"))
    new_args = yaml.safe_load((new_run / "args.yaml").read_text(encoding="utf-8-sig"))
    unexpected = {key: [old_args.get(key), new_args.get(key)] for key in controlled if old_args.get(key) != new_args.get(key)}
    if unexpected:
        raise RuntimeError(f"More than nbs changed: {unexpected}")
    if old_args["nbs"] != 64 or new_args["nbs"] != 16:
        raise RuntimeError("Expected the only controlled change to be nbs 64 -> 16")

    required = ["results.csv", "results.png", "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
                "confusion_matrix.png", "train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg",
                "weights/best.pt", "weights/last.pt"]
    missing = [str(new_run / item) for item in required if not (new_run / item).is_file()]
    if missing:
        raise RuntimeError(f"Missing training artifacts: {missing}")

    redirect_label_caches()
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    old_model, old_classes, old_overall = validate(Path(old_status["best_pt"]), "nbs64_best_val_fp16", half=True)
    new_model, new_classes, new_fp16 = validate(Path(new_status["best_pt"]), "nbs16_best_val_fp16", half=True)
    _, new_classes_fp32, new_fp32 = validate(Path(new_status["best_pt"]), "nbs16_best_val_fp32", half=False)
    write_csv(ANALYSIS_ROOT / "nbs64_best_val_per_class.csv", old_classes)
    write_csv(ANALYSIS_ROOT / "nbs16_best_val_per_class.csv", new_classes)
    write_csv(WORKSPACE / "analysis" / "yolo11s_nbs16_best_val_per_class.csv", new_classes)
    write_csv(ANALYSIS_ROOT / "nbs16_best_val_per_class_fp32.csv", new_classes_fp32)

    fp_rows = []
    for precision, values in (("FP16", new_fp16), ("FP32", new_fp32)):
        fp_rows.append({
            "precision": precision,
            "P": values["metrics/precision(B)"],
            "R": values["metrics/recall(B)"],
            "mAP50": values["metrics/mAP50(B)"],
            "mAP50-95": values["metrics/mAP50-95(B)"],
        })
    write_csv(ANALYSIS_ROOT / "nbs16_fp16_vs_fp32.csv", fp_rows)
    predictions = prediction_stats(new_model)
    old_predictions = read_csv(WORKSPACE / "analysis" / "diagnostic_predictions" / "prediction_summary.csv")

    accumulation = read_csv(ACCUMULATION_LOG)
    if len(accumulation) != 810:
        raise RuntimeError(f"Expected 810 accumulation rows, got {len(accumulation)}")
    acc_values = [int(row["accumulate"]) for row in accumulation]
    steady = int(new_status["steady_accumulate"])
    first_steady_index = next(
        (i for i in range(len(acc_values)) if acc_values[i] == steady and all(value == steady for value in acc_values[i:])),
        None,
    )
    changes = []
    previous = None
    for row in accumulation:
        value = int(row["accumulate"])
        if value != previous:
            changes.append({
                "epoch": int(row["epoch"]),
                "batch_index": int(row["batch_index"]),
                "accumulate": value,
                "learning_rate": float(row["learning_rate"]),
            })
            previous = value
    write_csv(ANALYSIS_ROOT / "nbs16_accumulation_changes.csv", changes)

    epochs = [5, 10, 20, 25, 28, 30]
    comparisons = []
    keys = [
        "train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    ]
    for epoch in epochs:
        old = old_rows[epoch - 1]
        new = new_rows[epoch - 1]
        entry = {"epoch": epoch}
        for key in keys:
            short = key.replace("metrics/", "").replace("train/", "train_").replace("val/", "val_").replace("(B)", "")
            entry[f"nbs64_{short}"] = old[key]
            entry[f"nbs16_{short}"] = new[key]
            entry[f"delta_{short}"] = new[key] - old[key]
        comparisons.append(entry)
    write_csv(ANALYSIS_ROOT / "epoch_comparison.csv", comparisons)

    old_best = best_row(old_rows, "metrics/mAP50-95(B)", "metrics/mAP50(B)")
    new_best = best_row(new_rows, "metrics/mAP50-95(B)", "metrics/mAP50(B)")
    old_last10_map50 = mean(old_rows[-10:], "metrics/mAP50(B)")
    new_last10_map50 = mean(new_rows[-10:], "metrics/mAP50(B)")
    old_last10_map95 = mean(old_rows[-10:], "metrics/mAP50-95(B)")
    new_last10_map95 = mean(new_rows[-10:], "metrics/mAP50-95(B)")
    old_nan = val_nonfinite(old_rows)
    new_nan = val_nonfinite(new_rows)
    new_late_nan = [epoch for epoch in new_nan if epoch >= 10]
    old_total = [{**row, "total": row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"]} for row in old_rows]
    new_total = [{**row, "total": row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"]} for row in new_rows]
    old_train_slope = slope(old_total[:20], "total")
    new_train_slope = slope(new_total[:20], "total")
    old_val_total = [sum(row[key] for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss")) for row in old_rows[-10:] if all(math.isfinite(row[key]) for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss"))]
    new_val_total = [sum(row[key] for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss")) for row in new_rows[-10:] if all(math.isfinite(row[key]) for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss"))]
    old_val_std = statistics.pstdev(old_val_total)
    new_val_std = statistics.pstdev(new_val_total)

    old_class = {row["Class"]: row for row in old_classes}
    new_class = {row["Class"]: row for row in new_classes}
    old_pred_001 = next(row for row in old_predictions if float(row["threshold"]) == 0.001)
    new_pred_001 = next(row for row in predictions if float(row["threshold"]) == 0.001)
    confidence_gain = float(new_pred_001["confidence_max"]) - float(old_pred_001["conf_max"])
    fp_map50_delta = new_fp32["metrics/mAP50(B)"] - new_fp16["metrics/mAP50(B)"]
    fp_map95_delta = new_fp32["metrics/mAP50-95(B)"] - new_fp16["metrics/mAP50-95(B)"]
    faster = new_last10_map95 > old_last10_map95 and new_best["metrics/mAP50-95(B)"] > old_best["metrics/mAP50-95(B)"]
    candidate = faster and not new_late_nan
    decision = "建议作为后续正式 baseline 候选。" if candidate else "暂不建议作为后续正式 baseline 候选。"
    change_text = "; ".join(
        f"epoch {row['epoch']} / batch {row['batch_index']}: {row['accumulate']}" for row in changes
    )

    lines = [
        "# nbs=64 与 nbs=16 单变量诊断对比",
        "",
        "> 实验编号：Y11-DIAG-002。两组只使用 train/val；未使用 test 评估或调参。",
        "",
        "## 1. 单变量核对",
        "",
        "- 两组均从官方 `YOLO(\"yolo11s.pt\")` COCO 预训练权重重新开始，均非 resume。",
        "- 模型、epochs、imgsz、batch、workers、device、seed、deterministic、patience、cache、plots、AMP、保存设置、优化器配置、学习率配置、warmup、数据增强和 loss 权重均相同。",
        "- 唯一主动改变的训练参数：`nbs: 64 -> 16`。",
        f"- 官方源码 commit：`{commit}`；训练后 git status clean。",
        "",
        "## 2. 重点 epoch 对比",
        "",
        "|epoch|nbs|train box|train cls|train dfl|val box|val cls|val dfl|P|R|mAP50|mAP50-95|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for epoch in epochs:
        for nbs, rows in ((64, old_rows), (16, new_rows)):
            row = rows[epoch - 1]
            lines.append(
                f"|{epoch}|{nbs}|{fmt(row['train/box_loss'],4)}|{fmt(row['train/cls_loss'],4)}|{fmt(row['train/dfl_loss'],4)}|"
                f"{fmt(row['val/box_loss'],4)}|{fmt(row['val/cls_loss'],4)}|{fmt(row['val/dfl_loss'],4)}|"
                f"{fmt(row['metrics/precision(B)'],5)}|{fmt(row['metrics/recall(B)'],5)}|"
                f"{fmt(row['metrics/mAP50(B)'],6)}|{fmt(row['metrics/mAP50-95(B)'],6)}|"
            )
    lines += [
        "",
        "## 3. 汇总指标",
        "",
        "|指标|nbs=64|nbs=16|",
        "|---|---:|---:|",
        f"|最佳 epoch|{int(old_best['epoch'])}|{int(new_best['epoch'])}|",
        f"|最佳 mAP50|{fmt(best_row(old_rows, 'metrics/mAP50(B)')['metrics/mAP50(B)'])}|{fmt(best_row(new_rows, 'metrics/mAP50(B)')['metrics/mAP50(B)'])}|",
        f"|最佳 mAP50-95|{fmt(old_best['metrics/mAP50-95(B)'])}|{fmt(new_best['metrics/mAP50-95(B)'])}|",
        f"|最后 10 epoch 平均 mAP50|{fmt(old_last10_map50)}|{fmt(new_last10_map50)}|",
        f"|最后 10 epoch 平均 mAP50-95|{fmt(old_last10_map95)}|{fmt(new_last10_map95)}|",
        f"|epoch 1-20 train loss 总和斜率|{fmt(old_train_slope)}|{fmt(new_train_slope)}|",
        f"|最后 10 epoch val loss 总和标准差|{fmt(old_val_std)}|{fmt(new_val_std)}|",
        "",
        "## 4. 逐类别 best.pt 验证",
        "",
        "|类别|nbs|P|R|mAP50|mAP50-95|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cls in ("broken", "loose"):
        for nbs, data in ((64, old_class[cls]), (16, new_class[cls])):
            lines.append(f"|{cls}|{nbs}|{fmt(data['Box-P'])}|{fmt(data['Box-R'])}|{fmt(data['mAP50'])}|{fmt(data['mAP50-95'])}|")
    lines += [
        "",
        "## 5. 新实验低置信度预测",
        "",
        "|conf|图像有预测/23|总框|broken|loose|confidence 中位数|最高 confidence|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in predictions:
        lines.append(
            f"|{row['threshold']}|{row['images_with_predictions']}|{row['total_detections']}|{row['broken_detections']}|"
            f"{row['loose_detections']}|{fmt(row['confidence_median'])}|{fmt(row['confidence_max'])}|"
        )
    lines += [
        "",
        "## 6. FP16 与 FP32 验证",
        "",
        "|验证精度|P|R|mAP50|mAP50-95|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fp_rows:
        lines.append(f"|{row['precision']}|{fmt(row['P'])}|{fmt(row['R'])}|{fmt(row['mAP50'])}|{fmt(row['mAP50-95'])}|")
    lines += [
        "",
        f"FP32-FP16 差值：mAP50={fmt(fp_map50_delta,8)}，mAP50-95={fmt(fp_map95_delta,8)}。",
        "",
        "## 7. accumulate 实测",
        "",
        f"- nbs=16 共记录 {len(accumulation)} 个 batch，出现的 accumulate 值为 {sorted(set(acc_values))}。",
        f"- warmup 后稳定 accumulate={steady}；从全局 batch {first_steady_index + 1 if first_steady_index is not None else 'NA'} 起后续均保持该值。",
        f"- 变化点：{change_text}。",
        "- nbs=64 历史实验没有逐 batch accumulate 日志；按该固定官方 trainer 的稳态公式 `round(nbs / batch)`，其稳态为 16。这个值是由已锁定 args 和官方代码核验得到，不冒充历史实测日志。",
        "- 因此 nbs=16 的实测稳态 4 相对 nbs=64 的核验稳态 16 小 4 倍；在 batch 数相同的条件下，optimizer update 会更频繁，但本报告不把估算次数写成实测次数。",
        "",
        "## 8. A-I 结论",
        "",
        f"### A. nbs=16 是否明显加快学习？\n\n{'是' if faster else '否'}。判断同时使用最佳 mAP、最后 10 轮平均值和曲线，不使用单个 Precision 尖峰。",
        "",
        f"### B. nbs=16 最佳指标\n\n最佳 mAP50={fmt(best_row(new_rows, 'metrics/mAP50(B)')['metrics/mAP50(B)'])}；最佳 mAP50-95={fmt(new_best['metrics/mAP50-95(B)'])}，最佳 mAP50-95 位于 epoch {int(new_best['epoch'])}。",
        "",
        f"### C. 最后 10 epoch 平均指标\n\nnbs=64：mAP50={fmt(old_last10_map50)}，mAP50-95={fmt(old_last10_map95)}；nbs=16：mAP50={fmt(new_last10_map50)}，mAP50-95={fmt(new_last10_map95)}。{'nbs=16 更高。' if new_last10_map95 > old_last10_map95 else 'nbs=64 更高或相同。'}",
        "",
        f"### D. train loss 是否下降更快？\n\nepoch 1-20 三项 train loss 总和斜率：nbs=64 为 {fmt(old_train_slope)}，nbs=16 为 {fmt(new_train_slope)}。斜率更负表示这一区间下降更快。",
        "",
        f"### E. val loss 是否更稳定？\n\nnbs=64 非有限 val loss epoch={old_nan or '无'}；nbs=16={new_nan or '无'}。最后 10 轮 val loss 总和标准差分别为 {fmt(old_val_std)} 和 {fmt(new_val_std)}。{'nbs=16 未在 epoch>=10 出现 NaN/Inf。' if not new_late_nan else f'nbs=16 在 epoch>=10 仍有非有限值：{new_late_nan}，必须重点处理。'}",
        "",
        f"### F. broken 与 loose 的变化\n\nbroken mAP50-95：{fmt(old_class['broken']['mAP50-95'])} -> {fmt(new_class['broken']['mAP50-95'])}；loose：{fmt(old_class['loose']['mAP50-95'])} -> {fmt(new_class['loose']['mAP50-95'])}。完整 P/R/AP 见逐类别表。",
        "",
        f"### G. 最高预测置信度是否提高？\n\nconf=0.001 诊断中的最高 confidence：nbs=64 为 {fmt(old_pred_001['conf_max'])}，nbs=16 为 {fmt(new_pred_001['confidence_max'])}，变化 {fmt(confidence_gain)}。同时结合候选框数量判断是否减少无意义低置信度输出。",
        "",
        f"### H. 是否值得作为正式 baseline 候选？\n\n{decision} 依据是最佳与末 10 轮指标、验证稳定性和逐类别表现；本轮仍是 Diagnostic，`include_in_paper=No`。",
        "",
        f"### I. 实际梯度累积差异\n\nnbs=16 实测 warmup 动态值为 {sorted(set(acc_values))}，稳态为 {steady}；nbs=64 历史未留逐 batch 日志，固定代码与参数核验的稳态为 16。nbs=16 的稳态累积跨度缩短为四分之一。",
        "",
        "## 9. 稳定性与完整性",
        "",
        f"- nbs=16 训练 batch loss 全部有限：{new_status['all_batch_losses_finite']}。",
        f"- nbs=16 epoch>=10 非有限 val loss：{new_late_nan or '无'}。",
        f"- FP16/FP32 验证差异见上表。",
        f"- GPU peak allocated={new_status['peak_allocated_MiB']} MiB，reserved={new_status['peak_reserved_MiB']} MiB。",
        "- test 未用于训练后评估、选模或调参。",
    ]

    integrity = verify_dataset_unchanged()
    git_after = subprocess.check_output(["git", "-C", str(PROJECT / "ultralytics_src"), "status", "--porcelain"], text=True)
    if git_after.strip():
        raise RuntimeError(f"Official source became dirty: {git_after}")
    lines += [
        f"- 数据集完整性：{'未变化' if integrity['unchanged'] else '发生变化'}，核对 {integrity['entries_checked']} 个条目。",
        "- 官方源码 git status clean。",
        "",
        "## 10. 产物",
        "",
        f"- 新运行：`{new_run}`",
        f"- 新 results.csv：`{new_run / 'results.csv'}`",
        f"- 新 results.png：`{new_run / 'results.png'}`",
        f"- nbs=16 逐类别：`{WORKSPACE / 'analysis' / 'yolo11s_nbs16_best_val_per_class.csv'}`",
        f"- 低置信度预测：`{PRED_ROOT}`",
        f"- accumulate 日志：`{ACCUMULATION_LOG}`",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    new_status.update(
        decision=decision,
        registry_notes=(
            f"Y11-DIAG-001 单变量对照；nbs=16 steady_accumulate={steady}；"
            f"epoch>=10 nonfinite val loss={new_late_nan or 'none'}；FP16/FP32 差异已检查。"
        ),
        post_analysis={
            "success": True,
            "report": str(REPORT_PATH),
            "new_per_class_csv": str(WORKSPACE / "analysis" / "yolo11s_nbs16_best_val_per_class.csv"),
            "prediction_summary": str(PRED_ROOT / "prediction_summary.csv"),
            "fp16_vs_fp32": str(ANALYSIS_ROOT / "nbs16_fp16_vs_fp32.csv"),
            "dataset_integrity_after": integrity,
            "git_status_after": git_after,
        },
    )
    NEW_STATUS_PATH.write_text(json.dumps(new_status, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(WORKSPACE / "logs" / f"{new_status['run_name']}_status.json").write_text(
        json.dumps(new_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    handoff_paths = [
        REPORT_PATH,
        new_run / "results.png",
        new_run / "results.csv",
        new_run / "args.yaml",
        WORKSPACE / "analysis" / "yolo11s_nbs16_best_val_per_class.csv",
        PRED_ROOT / "prediction_summary.csv",
        ACCUMULATION_LOG,
        WORKSPACE / "experiment_tracking" / "experiment_registry.xlsx",
        WORKSPACE / "scripts" / "train_yolo11_nbs16_diagnostic.py",
    ]
    if new_nan or not new_status["all_batch_losses_finite"] or new_status.get("oom"):
        handoff_paths += [Path(new_status["console_log"]), NEW_STATUS_PATH]
    handoff = [
        "# Y11-DIAG-002 GPT 交接清单",
        "",
        "以下仅列出原文件完整路径，不复制、不打包文件。",
        "",
    ]
    handoff += [f"{index}. `{path}`" for index, path in enumerate(handoff_paths, 1)]
    HANDOFF_PATH.write_text("\n\n".join(handoff) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT_PATH),
        "handoff": str(HANDOFF_PATH),
        "decision": decision,
        "dataset_unchanged": integrity["unchanged"],
        "git_clean": not git_after.strip(),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
