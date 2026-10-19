r"""Compare Y11-DIAG-001 and Y11-TUNE-001 on train/val only."""

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
NEW_RUN = WORKSPACE / "runs" / "yolo11s_tune_1280_e30_mosaic0_20260916_170351"
OLD_BEST = OLD_RUN / "weights" / "best.pt"
NEW_BEST = NEW_RUN / "weights" / "best.pt"
DATA_YAML = WORKSPACE / "configs" / "conductor_bl_v1.yaml"
PHASE_CSV = WORKSPACE / "analysis" / "mosaic_training_phase_comparison.csv"
CLASS_CSV = WORKSPACE / "analysis" / "mosaic1_vs_mosaic0_per_class.csv"
PRED_CSV = WORKSPACE / "analysis" / "mosaic0_prediction_summary.csv"
REPORT = WORKSPACE / "notes" / "mosaic1_vs_mosaic0_training_report.md"
STATUS_PATH = WORKSPACE / "logs" / "yolo11s_tune_1280_e30_mosaic0_20260916_170351_status.json"
ANALYSIS_ROOT = WORKSPACE / "analysis" / "mosaic_training_validation"


def read_results(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{key.strip(): float(value) for key, value in raw.items()} for raw in csv.DictReader(stream)]
    if len(rows) != 30 or [int(row["epoch"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError(f"Expected epochs 1..30 in {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    finite = [row for row in rows if math.isfinite(row["metrics/mAP50-95(B)"])]
    return max(finite, key=lambda row: (row["metrics/mAP50-95(B)"], row["metrics/mAP50(B)"], -row["epoch"]))


def phase_summary(experiment_id: str, mosaic: float, rows: list[dict[str, float]]) -> list[dict[str, Any]]:
    metrics = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    output = []
    for phase, start, end in (("A_epoch_1_20", 1, 20), ("B_epoch_21_30", 21, 30), ("C_epoch_1_30", 1, 30)):
        selected = [row for row in rows if start <= row["epoch"] <= end]
        record: dict[str, Any] = {
            "experiment_id": experiment_id,
            "mosaic": mosaic,
            "phase": phase,
            "epoch_start": start,
            "epoch_end": end,
            "epoch_count": len(selected),
        }
        for short, key in metrics.items():
            values = [row[key] for row in selected if math.isfinite(row[key])]
            record[f"mean_{short}"] = statistics.fmean(values)
            winner = max(selected, key=lambda row: (row[key], -row["epoch"]))
            record[f"max_{short}"] = winner[key]
            record[f"epoch_of_max_{short}"] = int(winner["epoch"])
        output.append(record)
    return output


def validate(model_path: Path, name: str):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(DATA_YAML), split="val", imgsz=1280, batch=8, workers=0, device=0,
        rect=False, half=True, plots=False, save_json=False,
        project=str(ANALYSIS_ROOT), name=name, exist_ok=False, verbose=True,
    )
    return model, native(metrics.summary(decimals=10)), {key: float(value) for key, value in metrics.results_dict.items()}


def prediction_stats(model) -> list[dict[str, Any]]:
    val_dir = DATASET / "images" / "val"
    images = sorted(path for path in val_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
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
    return {"unchanged": not changes, "changes": changes, "entries_checked": len(current)}


def fmt(value: float, digits: int = 6) -> str:
    return "NaN" if not math.isfinite(value) else f"{value:.{digits}f}"


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
    if commit != EXPECTED_COMMIT or git_status.strip() or not torch.cuda.is_available():
        raise RuntimeError(f"Environment mismatch: commit={commit}, git={git_status!r}, cuda={torch.cuda.is_available()}")
    integrity_before = verify_dataset()
    if not integrity_before["unchanged"]:
        raise RuntimeError(f"Dataset changed: {integrity_before['changes'][:10]}")

    old_args = yaml.safe_load((OLD_RUN / "args.yaml").read_text(encoding="utf-8-sig"))
    new_args = yaml.safe_load((NEW_RUN / "args.yaml").read_text(encoding="utf-8-sig"))
    controlled = (
        "model", "epochs", "imgsz", "batch", "nbs", "workers", "device", "seed", "deterministic", "patience",
        "cache", "amp", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs", "box", "cls",
        "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "flipud",
        "fliplr", "mosaic", "mixup", "cutmix", "copy_paste", "close_mosaic", "resume", "split",
    )
    differences = {key: [old_args[key], new_args[key]] for key in controlled if old_args[key] != new_args[key]}
    if differences != {"mosaic": [1.0, 0.0]}:
        raise RuntimeError(f"Single-variable condition violated: {differences}")

    required = (
        "results.csv", "results.png", "args.yaml", "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg",
        "weights/best.pt", "weights/last.pt", "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
        "confusion_matrix.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    )
    missing = [str(NEW_RUN / item) for item in required if not (NEW_RUN / item).is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    old_rows, new_rows = read_results(OLD_RUN / "results.csv"), read_results(NEW_RUN / "results.csv")
    phase_rows = phase_summary("Y11-DIAG-001", 1.0, old_rows) + phase_summary("Y11-TUNE-001", 0.0, new_rows)
    write_csv(PHASE_CSV, phase_rows)

    redirect_label_caches()
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    old_model, old_classes, old_overall = validate(OLD_BEST, "mosaic1_best_val")
    new_model, new_classes, new_overall = validate(NEW_BEST, "mosaic0_best_val")
    class_rows = []
    for experiment_id, mosaic, values in (
        ("Y11-DIAG-001", 1.0, old_classes), ("Y11-TUNE-001", 0.0, new_classes)
    ):
        for row in values:
            if row["Class"] == "all":
                continue
            class_rows.append({
                "experiment_id": experiment_id,
                "mosaic": mosaic,
                "class_name": row["Class"],
                "images": row["Images"],
                "instances": row["Instances"],
                "precision": row["Box-P"],
                "recall": row["Box-R"],
                "map50": row["mAP50"],
                "map50_95": row["mAP50-95"],
            })
    write_csv(CLASS_CSV, class_rows)
    predictions = prediction_stats(new_model)
    write_csv(PRED_CSV, predictions)

    old_best, new_best = best_row(old_rows), best_row(new_rows)
    old_phase_b = next(row for row in phase_rows if row["experiment_id"] == "Y11-DIAG-001" and row["phase"] == "B_epoch_21_30")
    new_phase_b = next(row for row in phase_rows if row["experiment_id"] == "Y11-TUNE-001" and row["phase"] == "B_epoch_21_30")
    old_class = {row["class_name"]: row for row in class_rows if row["experiment_id"] == "Y11-DIAG-001"}
    new_class = {row["class_name"]: row for row in class_rows if row["experiment_id"] == "Y11-TUNE-001"}
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    early_nan = status["epoch_stability"]["nonfinite"]
    late_nan = status["epoch_stability"]["nonfinite_at_or_after_epoch_10"]
    decision = "Mosaic=0 明显更差；保留官方 Mosaic 策略作为后续候选，并可在下一步讨论更早 close_mosaic。"

    lines = [
        "# Y11-DIAG-001 与 Y11-TUNE-001：Mosaic 单变量训练报告",
        "",
        "> 研究问题：Mosaic 带来的数据多样性收益，能否抵消它对极小目标尺寸的不利影响？本轮仅使用 train/val，test 未使用。",
        "",
        "## 1. 单变量与运行完整性",
        "",
        "- 两组均为 YOLO11s、imgsz=1280、batch=4、nbs=64、epochs=30、seed=42，并从同一官方 `yolo11s.pt` 重新开始。",
        "- 唯一主动训练变量：`mosaic: 1.0 → 0.0`；`close_mosaic=10` 保持不变。",
        f"- 新运行目录：`{NEW_RUN}`；训练批次 {status['batches_checked']}，batch loss 全有限：{status['all_batch_losses_finite']}。",
        f"- 早期 epoch 级非有限 val loss：{early_nan or '无'}；epoch>=10：{late_nan or '无'}。",
        f"- 峰值显存 allocated={status['peak_allocated_MiB']} MiB，reserved={status['peak_reserved_MiB']} MiB；训练用时 {status['elapsed_seconds']} 秒。",
        "",
        "## 2. 同一 best epoch 的整体指标",
        "",
        "best epoch 规则：最大 mAP50-95；完全并列时比较 mAP50。P/R/mAP 均读取该同一行。",
        "",
        "|实验|Mosaic|best epoch|Precision|Recall|mAP50|mAP50-95|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, mosaic, row in (("Y11-DIAG-001", 1.0, old_best), ("Y11-TUNE-001", 0.0, new_best)):
        lines.append(
            f"|{label}|{mosaic:.1f}|{int(row['epoch'])}|{fmt(row['metrics/precision(B)'])}|"
            f"{fmt(row['metrics/recall(B)'])}|{fmt(row['metrics/mAP50(B)'])}|{fmt(row['metrics/mAP50-95(B)'])}|"
        )
    lines += [
        "",
        f"Mosaic=0 的 best mAP50 相对变化为 {(new_best['metrics/mAP50(B)']/old_best['metrics/mAP50(B)']-1):.2%}；"
        f"best mAP50-95 相对变化为 {(new_best['metrics/mAP50-95(B)']/old_best['metrics/mAP50-95(B)']-1):.2%}。",
        "",
        "## 3. 三阶段比较",
        "",
        "阶段 A=epoch 1-20；阶段 B=epoch 21-30；阶段 C=epoch 1-30。完整均值与阶段最高值在 `mosaic_training_phase_comparison.csv`。",
        "",
        "|阶段|实验|平均 P|平均 R|平均 mAP50|平均 mAP50-95|阶段最高 mAP50|阶段最高 mAP50-95|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in phase_rows:
        lines.append(
            f"|{row['phase']}|{row['experiment_id']}|{fmt(row['mean_precision'])}|{fmt(row['mean_recall'])}|"
            f"{fmt(row['mean_map50'])}|{fmt(row['mean_map50_95'])}|{fmt(row['max_map50'])}|{fmt(row['max_map50_95'])}|"
        )
    lines += [
        "",
        "### 第 21–30 轮解释",
        "",
        f"原实验在第 21 轮关闭 Mosaic 后，后 10 轮平均 mAP50={fmt(old_phase_b['mean_map50'])}、"
        f"mAP50-95={fmt(old_phase_b['mean_map50_95'])}；从头关闭 Mosaic 的新实验分别为 "
        f"{fmt(new_phase_b['mean_map50'])}、{fmt(new_phase_b['mean_map50_95'])}。原实验后段仍明显更高，"
        "说明早期 Mosaic 的多样性收益在本次单种子对照中大于其极小目标缩小代价；没有观察到 Mosaic=0 后段保持优势。",
        "",
        "## 4. best.pt 同设置逐类别 val",
        "",
        "|类别|实验|P|R|mAP50|mAP50-95|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for class_name in ("broken", "loose"):
        for experiment_id, values in (("Y11-DIAG-001", old_class[class_name]), ("Y11-TUNE-001", new_class[class_name])):
            lines.append(
                f"|{class_name}|{experiment_id}|{fmt(values['precision'])}|{fmt(values['recall'])}|"
                f"{fmt(values['map50'])}|{fmt(values['map50_95'])}|"
            )
    lines += [
        "",
        "两类 AP 均下降；broken 在 Mosaic=0 下几乎未学到有效召回，loose 也明显低于对照组。",
        "",
        "## 5. Mosaic=0 低置信度预测",
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
        "## 6. 训练图人工检查",
        "",
        "- `train_batch0.jpg`、`train_batch1.jpg`、`train_batch2.jpg` 待本报告生成后进行最终人工复核。",
        "- 预期检查点：没有四图 Mosaic 拼接；bbox 位置与 broken/loose 类别标注正常；增强图像无明显异常。",
        "",
        "## 7. 结论",
        "",
        f"**{decision}**",
        "",
        "这不证明 Mosaic 对所有极小目标任务都好，也不否定只读审计中观察到的尺寸缩小。它说明在当前数据、当前 seed 和 30 epoch 协议下，彻底关闭 Mosaic 损失的数据多样性收益更大。若后续研究 Mosaic，可优先讨论保留 Mosaic 但更早关闭；本轮不自动运行该实验。",
        "",
        "## 8. 约束确认",
        "",
        "- test 未用于训练、验证、选模、预测诊断或阈值选择。",
        "- 没有自动启动第二个 seed、其他 Mosaic 比例、close_mosaic、模型尺寸或长周期实验。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    integrity_after = verify_dataset()
    git_after = subprocess.check_output(["git", "-C", str(PROJECT / "ultralytics_src"), "status", "--porcelain"], text=True)
    if not integrity_after["unchanged"] or git_after.strip():
        raise RuntimeError(f"Post-analysis integrity failure: dataset={integrity_after}, git={git_after!r}")
    status.update(
        post_analysis={
            "success": True,
            "test_used": False,
            "best_epoch": int(new_best["epoch"]),
            "best_epoch_metrics": {
                "precision": new_best["metrics/precision(B)"],
                "recall": new_best["metrics/recall(B)"],
                "map50": new_best["metrics/mAP50(B)"],
                "map50_95": new_best["metrics/mAP50-95(B)"],
            },
            "old_overall_revalidation": old_overall,
            "new_overall_revalidation": new_overall,
            "report": str(REPORT),
            "phase_csv": str(PHASE_CSV),
            "per_class_csv": str(CLASS_CSV),
            "prediction_csv": str(PRED_CSV),
            "decision": decision,
            "dataset_integrity": integrity_after,
            "git_status_after": git_after,
        }
    )
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status["post_analysis"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
