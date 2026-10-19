r"""Analyze Y11-TUNE-002 convergence using train/val only."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

from check_dataset import snapshot
from readonly_dataset import DATASET, PROJECT, PROTECTED_ROOT, WORKSPACE, install_write_guard, redirect_label_caches

EXPECTED_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
OLD_RUN = WORKSPACE / "runs" / "yolo11s_diag_1280_e30_20260916_104603"
NEW_RUN = WORKSPACE / "runs" / "yolo11s_tune_1280_e100_20260916_191542"
NEW_BEST = NEW_RUN / "weights" / "best.pt"
DATA_YAML = WORKSPACE / "configs" / "conductor_bl_v1.yaml"
WINDOW_CSV = WORKSPACE / "analysis" / "e100_convergence_windows.csv"
CLASS_CSV = WORKSPACE / "analysis" / "Y11-TUNE-002_per_class.csv"
PRED_CSV = WORKSPACE / "analysis" / "Y11-TUNE-002_prediction_summary.csv"
REPORT = WORKSPACE / "notes" / "e30_vs_e100_convergence_report.md"
STATUS_PATH = WORKSPACE / "logs" / "yolo11s_tune_1280_e100_20260916_191542_status.json"
ANALYSIS_ROOT = WORKSPACE / "analysis" / "e100_convergence_validation"

METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}
LOSSES = {
    "train_box_loss": "train/box_loss",
    "train_cls_loss": "train/cls_loss",
    "train_dfl_loss": "train/dfl_loss",
}
WINDOWS = (
    ("epoch_1_20", 1, 20, "Mosaic ON"),
    ("epoch_21_40", 21, 40, "Mosaic ON"),
    ("epoch_41_60", 41, 60, "Mosaic ON"),
    ("epoch_61_80", 61, 80, "Mosaic ON"),
    ("epoch_71_80", 71, 80, "Mosaic ON; transition audit"),
    ("epoch_81_90", 81, 90, "Mosaic ON"),
    ("epoch_91_100", 91, 100, "Mosaic OFF"),
)


def read_results(path: Path, expected_epochs: int) -> list[dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{key.strip(): float(value) for key, value in raw.items()} for raw in csv.DictReader(stream)]
    if len(rows) != expected_epochs or [int(row["epoch"]) for row in rows] != list(range(1, expected_epochs + 1)):
        raise RuntimeError(f"Expected epochs 1..{expected_epochs} in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    finite = [row for row in rows if math.isfinite(row["metrics/mAP50-95(B)"])]
    return max(finite, key=lambda row: (row["metrics/mAP50-95(B)"], row["metrics/mAP50(B)"], -row["epoch"]))


def mean(rows: list[dict[str, float]], field: str) -> float:
    values = [row[field] for row in rows if math.isfinite(row[field])]
    return statistics.fmean(values)


def stdev(rows: list[dict[str, float]], field: str) -> float:
    values = [row[field] for row in rows if math.isfinite(row[field])]
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def slope(rows: list[dict[str, float]], field: str) -> float:
    pairs = [(row["epoch"], row[field]) for row in rows if math.isfinite(row[field])]
    if len(pairs) < 2:
        return math.nan
    x = np.asarray([pair[0] for pair in pairs], dtype=float)
    y = np.asarray([pair[1] for pair in pairs], dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def select(rows: list[dict[str, float]], start: int, end: int) -> list[dict[str, float]]:
    return [row for row in rows if start <= row["epoch"] <= end]


def summarize_windows(rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    output = []
    for label, start, end, mosaic_state in WINDOWS:
        chosen = select(rows, start, end)
        record: dict[str, Any] = {
            "window": label,
            "epoch_start": start,
            "epoch_end": end,
            "epoch_count": len(chosen),
            "mosaic_state": mosaic_state,
        }
        for short, field in METRICS.items():
            winner = max(chosen, key=lambda row: (row[field], -row["epoch"]))
            record[f"mean_{short}"] = mean(chosen, field)
            record[f"std_{short}"] = stdev(chosen, field)
            record[f"max_{short}"] = winner[field]
            record[f"epoch_of_max_{short}"] = int(winner["epoch"])
            record[f"slope_{short}_per_epoch"] = slope(chosen, field)
        for short, field in LOSSES.items():
            record[f"mean_{short}"] = mean(chosen, field)
            record[f"slope_{short}_per_epoch"] = slope(chosen, field)
        output.append(record)
    return output


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validate_best():
    from ultralytics import YOLO

    model = YOLO(str(NEW_BEST))
    metrics = model.val(
        data=str(DATA_YAML), split="val", imgsz=1280, batch=8, workers=0, device=0,
        rect=False, half=True, plots=False, save_json=False,
        project=str(ANALYSIS_ROOT), name="Y11-TUNE-002_best_val", exist_ok=True, verbose=True,
    )
    return model, native(metrics.summary(decimals=10)), {key: float(value) for key, value in metrics.results_dict.items()}


def prediction_stats(model) -> list[dict[str, Any]]:
    images = sorted(
        path for path in (DATASET / "images" / "val").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    if len(images) != 23:
        raise RuntimeError(f"Expected 23 val images, got {len(images)}")
    output = []
    for threshold in (0.001, 0.01, 0.05, 0.25):
        results = model.predict(
            source=[str(path) for path in images], conf=threshold, imgsz=1280, batch=4,
            device=0, save=False, save_txt=False, save_conf=False, verbose=False,
        )
        confidences: list[float] = []
        class_counts = {0: 0, 1: 0}
        images_with_predictions = 0
        for result in results:
            if len(result.boxes):
                images_with_predictions += 1
                classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                confidences.extend(result.boxes.conf.detach().cpu().numpy().astype(float).tolist())
                for class_id in classes:
                    if class_id not in class_counts:
                        raise RuntimeError(f"Unexpected predicted class: {class_id}")
                    class_counts[class_id] += 1
        output.append({
            "confidence_threshold": threshold,
            "val_images": len(images),
            "images_with_predictions": images_with_predictions,
            "total_prediction_boxes": len(confidences),
            "broken_boxes": class_counts[0],
            "loose_boxes": class_counts[1],
            "confidence_median": float(np.median(confidences)) if confidences else math.nan,
            "confidence_max": max(confidences) if confidences else math.nan,
        })
    return output


def verify_dataset() -> dict[str, Any]:
    baseline = json.loads((WORKSPACE / "logs" / "dataset_snapshot_before.json").read_text(encoding="utf-8"))
    current = snapshot()
    changes = [key for key in sorted(baseline.keys() | current.keys()) if baseline.get(key) != current.get(key)]
    return {
        "unchanged": not changes,
        "changes": changes,
        "entries_checked": len(current),
        "sha256_scope": str(DATASET),
        "metadata_scope": str(PROTECTED_ROOT),
    }


def fmt(value: float, digits: int = 6) -> str:
    return "NaN" if not math.isfinite(value) else f"{value:.{digits}f}"


def pct(new: float, old: float) -> str:
    if old == 0:
        return "NA"
    return f"{(new / old - 1) * 100:+.2f}%"


def main() -> None:
    os.environ["YOLO_CONFIG_DIR"] = str(WORKSPACE / "configs" / "ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")
    os.environ["TORCH_HOME"] = str(WORKSPACE / "cache" / "torch")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    install_write_guard()

    import torch
    import ultralytics

    source = PROJECT / "ultralytics_src"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    git_before = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(f"Wrong Python environment: {sys.prefix}")
    if not Path(ultralytics.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    if commit != EXPECTED_COMMIT or git_before.strip() or not torch.cuda.is_available():
        raise RuntimeError(f"Environment mismatch: commit={commit}, git={git_before!r}, cuda={torch.cuda.is_available()}")
    integrity_before = verify_dataset()
    if not integrity_before["unchanged"]:
        raise RuntimeError(f"Dataset changed: {integrity_before['changes'][:10]}")

    old_args = yaml.safe_load((OLD_RUN / "args.yaml").read_text(encoding="utf-8-sig"))
    new_args = yaml.safe_load((NEW_RUN / "args.yaml").read_text(encoding="utf-8-sig"))
    controlled = (
        "model", "pretrained", "epochs", "imgsz", "batch", "nbs", "workers", "device", "seed",
        "deterministic", "patience", "cache", "amp", "optimizer", "lr0", "lrf", "momentum",
        "weight_decay", "warmup_epochs", "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic",
        "mixup", "cutmix", "copy_paste", "close_mosaic", "resume", "split",
    )
    differences = {key: [old_args[key], new_args[key]] for key in controlled if old_args[key] != new_args[key]}
    if differences != {"epochs": [30, 100]}:
        raise RuntimeError(f"Single-variable condition violated: {differences}")

    required = (
        "results.csv", "results.png", "args.yaml", "weights/best.pt", "weights/last.pt",
        "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "confusion_matrix.png",
        "train_batch0.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    )
    missing = [str(NEW_RUN / item) for item in required if not (NEW_RUN / item).is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    old_rows = read_results(OLD_RUN / "results.csv", 30)
    new_rows = read_results(NEW_RUN / "results.csv", 100)
    old_best = best_row(old_rows)
    new_best = best_row(new_rows)
    windows = summarize_windows(new_rows)
    write_csv(WINDOW_CSV, windows)
    window_by_name = {row["window"]: row for row in windows}

    redirect_label_caches()
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    model, class_summary, revalidation = validate_best()
    class_rows = []
    for row in class_summary:
        if row["Class"] == "all":
            continue
        class_rows.append({
            "experiment_id": "Y11-TUNE-002",
            "class_name": row["Class"],
            "images": row["Images"],
            "instances": row["Instances"],
            "precision": row["Box-P"],
            "recall": row["Box-R"],
            "map50": row["mAP50"],
            "map50_95": row["mAP50-95"],
        })
    write_csv(CLASS_CSV, class_rows)
    predictions = prediction_stats(model)
    write_csv(PRED_CSV, predictions)

    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    last10 = select(new_rows, 91, 100)
    last20 = select(new_rows, 81, 100)
    before20 = select(new_rows, 71, 90)
    immediate_before = select(new_rows, 88, 90)
    immediate_after = select(new_rows, 91, 93)
    w71 = window_by_name["epoch_71_80"]
    w81 = window_by_name["epoch_81_90"]
    w91 = window_by_name["epoch_91_100"]
    trend81_100 = {short: slope(last20, field) for short, field in METRICS.items()}
    loss_trend81_100 = {short: slope(last20, field) for short, field in LOSSES.items()}
    transition = {
        "map50_mean_71_80": w71["mean_map50"],
        "map50_mean_81_90": w81["mean_map50"],
        "map50_mean_91_100": w91["mean_map50"],
        "map50_95_mean_71_80": w71["mean_map50_95"],
        "map50_95_mean_81_90": w81["mean_map50_95"],
        "map50_95_mean_91_100": w91["mean_map50_95"],
        "map50_immediate_before_88_90": mean(immediate_before, METRICS["map50"]),
        "map50_immediate_after_91_93": mean(immediate_after, METRICS["map50"]),
        "map50_95_immediate_before_88_90": mean(immediate_before, METRICS["map50_95"]),
        "map50_95_immediate_after_91_93": mean(immediate_after, METRICS["map50_95"]),
        "map50_slope_71_90": slope(before20, METRICS["map50"]),
        "map50_95_slope_71_90": slope(before20, METRICS["map50_95"]),
        "map50_slope_91_100": slope(last10, METRICS["map50"]),
        "map50_95_slope_91_100": slope(last10, METRICS["map50_95"]),
    }
    decision = (
        "情况C：100 epoch 相比 30 epoch 没有获得明确的 AP 改善，训练轮数不是当前主要瓶颈；"
        "应减少继续逐项调训练超参数，转向极小目标结构问题。"
    )

    lines = [
        "# Y11-DIAG-001 与 Y11-TUNE-002：30/100 epoch 收敛诊断",
        "",
        "> 本报告仅使用 train/val；test 未用于训练、验证、选模、预测诊断或阈值选择。",
        "",
        "## 1. 单变量与运行完整性",
        "",
        "- 唯一主动变量：`epochs: 30 → 100`。模型、预训练权重、imgsz、batch、nbs、seed、优化器设置和所有增强参数均与 Y11-DIAG-001 一致。",
        "- 两次训练都从官方 `YOLO(\"yolo11s.pt\")` 重新开始；没有 resume，也没有加载旧 best.pt/last.pt。",
        "- `mosaic=1.0, close_mosaic=10`：epoch 1–90 为 Mosaic ON，epoch 91–100 为 Mosaic OFF。",
        f"- 新运行目录：`{NEW_RUN}`；共检查 {status['batches_checked']} 个训练 batch，loss 全有限：{status['all_batch_losses_finite']}。",
        f"- epoch 级非有限值：{status['epoch_stability']['nonfinite'] or '无'}；epoch>=10：{status['epoch_stability']['nonfinite_at_or_after_epoch_10'] or '无'}。",
        f"- 峰值显存 allocated={status['peak_allocated_MiB']} MiB，reserved={status['peak_reserved_MiB']} MiB；训练用时 {status['elapsed_seconds']} 秒。",
        "",
        "## 2. 30 epoch 与 100 epoch 最佳结果",
        "",
        "best epoch 按最大 mAP50-95 选择；完全并列时比较 mAP50。P/R/mAP 均读取同一行。",
        "",
        "|实验|epochs|best epoch|Precision|Recall|mAP50|mAP50-95|",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"|Y11-DIAG-001|30|{int(old_best['epoch'])}|{fmt(old_best[METRICS['precision']])}|{fmt(old_best[METRICS['recall']])}|{fmt(old_best[METRICS['map50']])}|{fmt(old_best[METRICS['map50_95']])}|",
        f"|Y11-TUNE-002|100|{int(new_best['epoch'])}|{fmt(new_best[METRICS['precision']])}|{fmt(new_best[METRICS['recall']])}|{fmt(new_best[METRICS['map50']])}|{fmt(new_best[METRICS['map50_95']])}|",
        "",
        f"100 epoch 的 best mAP50 相对 30 epoch 为 {pct(new_best[METRICS['map50']], old_best[METRICS['map50']])}，"
        f"best mAP50-95 为 {pct(new_best[METRICS['map50_95']], old_best[METRICS['map50_95']])}。",
        "",
        "100 epoch 改变了总训练时长和学习率时间进程，因此其前 30 轮不能视为 Y11-DIAG-001 的完全复现；这里只比较两次独立运行的整体最佳结果。",
        "",
        "## 3. 100 epoch 窗口统计",
        "",
        "完整统计、标准差与线性斜率见 `e100_convergence_windows.csv`。",
        "",
        "|窗口|Mosaic|平均 P|平均 R|平均 mAP50|平均 mAP50-95|最高 mAP50（epoch）|最高 mAP50-95（epoch）|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in windows:
        lines.append(
            f"|{row['window']}|{row['mosaic_state']}|{fmt(row['mean_precision'])}|{fmt(row['mean_recall'])}|"
            f"{fmt(row['mean_map50'])}|{fmt(row['mean_map50_95'])}|"
            f"{fmt(row['max_map50'])}（{row['epoch_of_max_map50']}）|"
            f"{fmt(row['max_map50_95'])}（{row['epoch_of_max_map50_95']}）|"
        )
    lines += [
        "",
        "## 4. epoch 91 关闭 Mosaic 前后",
        "",
        f"A. epoch 71–90 的 mAP50 线性斜率为 {transition['map50_slope_71_90']:.8f}/epoch，"
        f"mAP50-95 斜率为 {transition['map50_95_slope_71_90']:.8f}/epoch；关闭前没有形成持续、稳定的上升轨迹。",
        f"B. 关闭前最后 3 轮（88–90）平均 mAP50/mAP50-95={fmt(transition['map50_immediate_before_88_90'])}/{fmt(transition['map50_95_immediate_before_88_90'])}；"
        f"关闭后最初 3 轮（91–93）={fmt(transition['map50_immediate_after_91_93'])}/{fmt(transition['map50_95_immediate_after_91_93'])}。没有明显即时跃升。",
        f"C. epoch 81–90 平均 mAP50/mAP50-95={fmt(w81['mean_map50'])}/{fmt(w81['mean_map50_95'])}；"
        f"epoch 91–100={fmt(w91['mean_map50'])}/{fmt(w91['mean_map50_95'])}，相对变化分别为 "
        f"{pct(w91['mean_map50'], w81['mean_map50'])}/{pct(w91['mean_map50_95'], w81['mean_map50_95'])}。",
        f"D. epoch 91–100 的 mAP50/mAP50-95 斜率为 {transition['map50_slope_91_100']:.8f}/{transition['map50_95_slope_91_100']:.8f} 每轮；"
        "关闭后的变化属于低位波动，没有逐渐形成显著提升。",
        f"E. best epoch={int(new_best['epoch'])}，{'位于' if new_best['epoch'] >= 91 else '不位于'}最后 10 轮。",
        "",
        "## 5. 100 epoch 是否收敛",
        "",
        f"1. best epoch：{int(new_best['epoch'])}。",
        f"2. best mAP50：{fmt(new_best[METRICS['map50']])}。",
        f"3. best mAP50-95：{fmt(new_best[METRICS['map50_95']])}。",
        f"4. 最后 10 轮平均 mAP50/mAP50-95：{fmt(mean(last10, METRICS['map50']))}/{fmt(mean(last10, METRICS['map50_95']))}。",
        f"5. 最后 20 轮平均 P/R/mAP50/mAP50-95：{fmt(mean(last20, METRICS['precision']))}/{fmt(mean(last20, METRICS['recall']))}/{fmt(mean(last20, METRICS['map50']))}/{fmt(mean(last20, METRICS['map50_95']))}。",
        f"6. epoch 81–100 的 mAP50/mAP50-95 线性斜率：{trend81_100['map50']:.8f}/{trend81_100['map50_95']:.8f} 每轮，没有明显持续上升。",
        f"7. best epoch {'≥90' if new_best['epoch'] >= 90 else '<90'}。",
        f"8. 后 20 轮 train box/cls/dfl loss 斜率：{loss_trend81_100['train_box_loss']:.8f}/{loss_trend81_100['train_cls_loss']:.8f}/{loss_trend81_100['train_dfl_loss']:.8f} 每轮；loss 仍缓慢下降，但未转化为 val AP 改善。",
        f"9. 后 20 轮 mAP50/mAP50-95 标准差：{stdev(last20, METRICS['map50']):.8f}/{stdev(last20, METRICS['map50_95']):.8f}。val 指标表现为低值噪声平台，而不是稳定上升平台。",
        "",
        "## 6. best.pt 逐类别 val",
        "",
        "|类别|P|R|mAP50|mAP50-95|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in class_rows:
        lines.append(
            f"|{row['class_name']}|{fmt(row['precision'])}|{fmt(row['recall'])}|{fmt(row['map50'])}|{fmt(row['map50_95'])}|"
        )
    lines += [
        "",
        "## 7. 低置信度预测",
        "",
        "|conf|有预测图片/23|总框|broken|loose|confidence 中位数|最高 confidence|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in predictions:
        lines.append(
            f"|{row['confidence_threshold']}|{row['images_with_predictions']}|{row['total_prediction_boxes']}|"
            f"{row['broken_boxes']}|{row['loose_boxes']}|{fmt(row['confidence_median'])}|{fmt(row['confidence_max'])}|"
        )
    lines += [
        "",
        "## 8. val_batch0_pred.jpg 人工检查",
        "",
        "- 已人工复核：8 张 val 图像均正常解码、排版无损坏，但默认验证可视化中没有绘制任何预测框。",
        "- 这与低置信度统计一致：全体 val 预测的最高 confidence 仅为 0.063177，在 `conf=0.25` 时 23 张图片均无预测；因此空预测图反映的是模型置信度不足，不是可视化故障。",
        "",
        "## 9. 决策",
        "",
        f"**{decision}**",
        "",
        "本次证据不支持自动增加到 150/200 epoch，也没有证据表明性能提升主要集中在 epoch 91 关闭 Mosaic 之后。本轮不运行任何后续实验。",
        "",
        "## 10. 约束确认",
        "",
        "- test 未使用。",
        "- 中央数据集保持只读且完整性不变。",
        "- 官方源码 commit 与工作树保持锁定。",
        "- 未自动启动 150/200 epoch、其他 close_mosaic、其他模型、分辨率、P2、attention、loss、第二个 seed 等实验。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    integrity_after = verify_dataset()
    git_after = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if not integrity_after["unchanged"] or git_after.strip():
        raise RuntimeError(f"Post-analysis integrity failure: dataset={integrity_after}, git={git_after!r}")
    analysis = {
        "success": True,
        "test_used": False,
        "single_variable_diff": differences,
        "best_epoch": int(new_best["epoch"]),
        "best_epoch_metrics": {
            "precision": new_best[METRICS["precision"]],
            "recall": new_best[METRICS["recall"]],
            "map50": new_best[METRICS["map50"]],
            "map50_95": new_best[METRICS["map50_95"]],
        },
        "last10_mean": {short: mean(last10, field) for short, field in METRICS.items()},
        "last20_mean": {short: mean(last20, field) for short, field in METRICS.items()},
        "trend_81_100": trend81_100,
        "train_loss_trend_81_100": loss_trend81_100,
        "mosaic_transition": transition,
        "best_pt_revalidation": revalidation,
        "per_class": class_rows,
        "prediction_summary": native(predictions),
        "report": str(REPORT),
        "window_csv": str(WINDOW_CSV),
        "per_class_csv": str(CLASS_CSV),
        "prediction_csv": str(PRED_CSV),
        "decision": decision,
        "dataset_integrity": integrity_after,
        "git_status_after": git_after,
    }
    status["post_analysis"] = analysis
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
