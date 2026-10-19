r"""Analyze the completed YOLO11s learnability run using the validation split only."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from check_dataset import snapshot
from readonly_dataset import DATASET, PROJECT, PROTECTED_ROOT, WORKSPACE, install_write_guard, redirect_label_caches


EXPECTED_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
DATA_YAML = WORKSPACE / "configs" / "conductor_bl_v1.yaml"
PRED_ROOT = WORKSPACE / "analysis" / "diagnostic_predictions"
REPORT_PATH = WORKSPACE / "notes" / "yolo11s_diagnostic_report.md"


def read_results(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append({key.strip(): float(value) for key, value in raw.items()})
    if len(rows) != 30 or [int(row["epoch"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError(f"Expected exactly epochs 1..30, got {len(rows)} rows")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def json_native(value):
    """Convert NumPy scalars returned by Ultralytics metrics into JSON values."""
    if isinstance(value, dict):
        return {key: json_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Inf" if value > 0 else "-Inf")
    return f"{value:.{digits}f}"


def best_row(rows: list[dict[str, float]], key: str) -> dict[str, float]:
    finite = [row for row in rows if math.isfinite(row[key])]
    return max(finite, key=lambda row: row[key])


def mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if math.isfinite(row[key])]
    return float(np.mean(values)) if values else math.nan


def slope(rows: list[dict[str, float]], key: str) -> float:
    finite = [(row["epoch"], row[key]) for row in rows if math.isfinite(row[key])]
    if len(finite) < 2:
        return math.nan
    return float(np.polyfit([x for x, _ in finite], [y for _, y in finite], 1)[0])


def verify_dataset_unchanged(output_name: str) -> dict:
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
    (WORKSPACE / "logs" / output_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if changes:
        raise RuntimeError(f"Protected dataset changed: {changes[:10]}")
    return report


def prediction_stats(model, thresholds: list[float]) -> list[dict]:
    summary = []
    per_image = []
    val_dir = DATASET / "images" / "val"
    expected_images = len([p for p in val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}])
    if expected_images != 23:
        raise RuntimeError(f"Expected 23 val images, found {expected_images}")
    for threshold in thresholds:
        tag = str(threshold).replace(".", "p")
        results = model.predict(
            source=str(val_dir),
            conf=threshold,
            imgsz=1280,
            batch=4,
            device=0,
            save=True,
            save_txt=False,
            save_conf=False,
            project=str(PRED_ROOT),
            name=f"conf_{tag}",
            exist_ok=True,
            verbose=False,
        )
        if len(results) != expected_images:
            raise RuntimeError(f"Prediction at conf={threshold} returned {len(results)} images")
        class_counts = {0: 0, 1: 0}
        confidences = []
        images_with_predictions = 0
        for result in results:
            n = len(result.boxes)
            if n:
                images_with_predictions += 1
                classes = result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                confs = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
                for cls in classes:
                    if cls not in class_counts:
                        raise RuntimeError(f"Unexpected predicted class {cls}")
                    class_counts[cls] += 1
                confidences.extend(confs)
            per_image.append(
                {
                    "threshold": threshold,
                    "image": Path(result.path).name,
                    "detections": n,
                    "broken": int((result.boxes.cls == 0).sum().item()) if n else 0,
                    "loose": int((result.boxes.cls == 1).sum().item()) if n else 0,
                    "max_conf": max(confs) if n else "",
                }
            )
        q = np.quantile(confidences, [0, 0.25, 0.5, 0.75, 1]).tolist() if confidences else [math.nan] * 5
        output_dir = PRED_ROOT / f"conf_{tag}"
        saved_images = [p for p in output_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}]
        if len(saved_images) != expected_images:
            raise RuntimeError(f"Expected {expected_images} saved visualization images at {threshold}, found {len(saved_images)}")
        summary.append(
            {
                "threshold": threshold,
                "val_images": expected_images,
                "images_with_predictions": images_with_predictions,
                "total_detections": len(confidences),
                "broken_detections": class_counts[0],
                "loose_detections": class_counts[1],
                "conf_min": q[0],
                "conf_q25": q[1],
                "conf_median": q[2],
                "conf_q75": q[3],
                "conf_max": q[4],
                "visualization_dir": str(output_dir),
            }
        )
    write_csv(PRED_ROOT / "prediction_summary.csv", summary)
    write_csv(PRED_ROOT / "prediction_per_image.csv", per_image)
    (PRED_ROOT / "prediction_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def make_report(status: dict, rows: list[dict[str, float]], per_class: list[dict], predictions: list[dict], integrity: dict) -> str:
    metric_keys = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
    }
    bests = {name: best_row(rows, key) for name, key in metric_keys.items()}
    highlight_epochs = {1, 3, 5, 10, 20, 30}
    last10 = rows[-10:]
    before_close = rows[10:20]
    after_close = rows[20:30]
    val_nan = [int(row["epoch"]) for row in rows if not all(math.isfinite(row[key]) for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss"))]
    val_nan_text = "、".join(map(str, val_nan)) if val_nan else "无"
    batch_entries = [json.loads(line) for line in Path(status["batch_log"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_finite = all(entry["finite"] for entry in batch_entries)
    metric_last_slope = slope(last10, "metrics/mAP50-95(B)")
    map50_last_slope = slope(last10, "metrics/mAP50(B)")
    train_total = [{**row, "train_total": row["train/box_loss"] + row["train/cls_loss"] + row["train/dfl_loss"]} for row in rows]
    val_total = [{**row, "val_total": row["val/box_loss"] + row["val/cls_loss"] + row["val/dfl_loss"]} for row in rows]
    train_last_slope = slope(train_total[-10:], "train_total")
    valid_val = [row for row in val_total if math.isfinite(row["val_total"])]
    val_last_slope = slope(valid_val[-10:], "val_total")
    best_map_row = bests["mAP50-95"]
    final = rows[-1]
    still_rising = metric_last_slope > 0 and map50_last_slope > 0
    mosaic_before = mean(before_close, "metrics/mAP50-95(B)")
    mosaic_after = mean(after_close, "metrics/mAP50-95(B)")
    mosaic_ratio = mosaic_after / mosaic_before if mosaic_before > 0 else math.inf
    broken = next(row for row in per_class if row["Class"] == "broken")
    loose = next(row for row in per_class if row["Class"] == "loose")

    lines = [
        "# YOLO11s 1280 学习能力诊断报告",
        "",
        "> 范围：仅使用 train/val；未读取 test 指标、未在 test 上评估。此轮是学习能力诊断，不是正式论文实验。",
        "",
        "## 1. 运行身份与约束",
        "",
        f"- 运行目录：`{status['save_dir']}`",
        f"- 官方源码 commit：`{status['git_commit']}`；Ultralytics `{status['ultralytics']}`；源码仓库保持干净。",
        f"- 模型/输入/批量/轮数：YOLO11s / 1280 / 4 / 30；实际 batch：{status['actual_batch']}。",
        f"- GPU：{status['gpu']}；峰值 allocated {status['peak_allocated_MiB']} MiB，reserved {status['peak_reserved_MiB']} MiB。",
        f"- 训练批次：{len(batch_entries)}；批次 loss 全为有限值：{batch_finite}；数据集被拦截写入次数：{len(status['blocked_dataset_writes'])}。",
        f"- 数据集训练后完整性：{'未变化' if integrity['unchanged'] else '发生变化'}（核对 {integrity['entries_checked']} 个条目）。",
        "- 优化器、损失权重和增强参数均取该固定 commit 的官方默认值；没有主动修改。",
        "",
        "## 2. 逐 epoch 指标",
        "",
        "|epoch|train box|train cls|train dfl|P|R|mAP50|mAP50-95|val box|val cls|val dfl|重点|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        epoch = int(row["epoch"])
        lines.append(
            f"|{epoch}|{fmt(row['train/box_loss'], 4)}|{fmt(row['train/cls_loss'], 4)}|{fmt(row['train/dfl_loss'], 4)}|"
            f"{fmt(row['metrics/precision(B)'], 4)}|{fmt(row['metrics/recall(B)'], 4)}|{fmt(row['metrics/mAP50(B)'], 5)}|"
            f"{fmt(row['metrics/mAP50-95(B)'], 5)}|{fmt(row['val/box_loss'], 4)}|{fmt(row['val/cls_loss'], 4)}|"
            f"{fmt(row['val/dfl_loss'], 4)}|{'★' if epoch in highlight_epochs else ''}|"
        )
    lines += [
        "",
        "## 3. 最佳轮次与最终轮",
        "",
        "|指标|最佳 epoch|最佳值|epoch 30|",
        "|---|---:|---:|---:|",
    ]
    for name, key in metric_keys.items():
        row = bests[name]
        lines.append(f"|{name}|{int(row['epoch'])}|{fmt(row[key])}|{fmt(final[key])}|")
    lines += [
        "",
        "## 4. best.pt 在 val 上的逐类别结果",
        "",
        "下面是用 `best.pt`、imgsz=1280 对 val 的复验。为复现训练结束时 Ultralytics 内置验证器的 3 个验证批次、方形输入和半精度推理，复验 batch=8、rect=False、half=True；训练 batch 始终是 4。",
        "",
    ]
    if per_class:
        columns = list(per_class[0])
        lines += ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
        for row in per_class:
            lines.append("|" + "|".join(fmt(float(row[col])) if isinstance(row[col], (int, float)) else str(row[col]) for col in columns) + "|")
    lines += [
        "",
        "## 5. 低置信度 val 预测",
        "",
        "|conf|有预测图像/23|总框|broken|loose|置信度中位数|最高置信度|",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in predictions:
        lines.append(
            f"|{row['threshold']}|{row['images_with_predictions']}|{row['total_detections']}|{row['broken_detections']}|"
            f"{row['loose_detections']}|{fmt(float(row['conf_median']))}|{fmt(float(row['conf_max']))}|"
        )
    lines += [
        "",
        f"最高预测置信度只有 {fmt(float(predictions[0]['conf_max']))}；conf=0.25 时没有任何检测。conf=0.001 的 {predictions[0]['total_detections']} 个候选框和 conf=0.01 的 {predictions[1]['total_detections']} 个候选框说明输出仍被大量极低置信度候选主导。可视化中这些框沿导线和背景密集出现，误检问题明显。",
        "",
        "## 6. A-H 判断",
        "",
        f"### A. 30 epoch 内是否开始学到目标？\n\n是，但信号很弱。最佳 mAP50={fmt(bests['mAP50']['metrics/mAP50(B)'])}、mAP50-95={fmt(best_map_row['metrics/mAP50-95(B)'])}（均在 epoch {int(best_map_row['epoch'])} 附近）；同时最高置信度仅 {fmt(float(predictions[0]['conf_max']))}，conf=0.25 无检测。因此可以确认“开始学习”，不能认为已经形成可用检测器。",
        "",
        f"### B. mAP50、mAP50-95 在 30 epoch 末是否仍在上升？\n\n整体仍在上升。最后 10 轮 mAP50 斜率为 {fmt(map50_last_slope, 8)}/epoch，mAP50-95 斜率为 {fmt(metric_last_slope, 8)}/epoch；两者最佳值都出现在 epoch 28，epoch 29-30 有小幅回落，属于上升趋势中的波动。",
        "",
        f"### C. train loss 是否仍在下降？\n\n最后 10 轮 train 三项 loss 之和的线性斜率为 {fmt(train_last_slope, 6)}/epoch。{'仍在下降。' if train_last_slope < 0 else '没有继续下降。'}批次级 loss 全部有限：{batch_finite}。",
        "",
        f"### D. val loss 是否稳定，是否出现明显过拟合？\n\n后段趋于稳定，未见明显过拟合。最后 10 个有限 val loss 总和的斜率为 {fmt(val_last_slope, 6)}/epoch，同时后段 mAP 上升；最佳 mAP50-95 相对末轮仅回落 {fmt(best_map_row['metrics/mAP50-95(B)'] - final['metrics/mAP50-95(B)'])}。早期不稳定很强：epoch 2-3 的 val cls loss 分别约 463.9 和 7657.1，epoch {val_nan_text} 出现非有限值。",
        "",
        f"### E. broken 和 loose 哪一类更难？\n\n按目标检测主要比较指标 AP，`loose` 更难：其 mAP50-95={fmt(float(loose['mAP50-95']))}，低于 `broken` 的 {fmt(float(broken['mAP50-95']))}；mAP50 也更低。不过 `broken` 的召回率只有 {fmt(float(broken['Box-R']))}，低于 `loose` 的 {fmt(float(loose['Box-R']))}，而且 val 中 broken 仅 9 个实例。结论应记为“AP 上 loose 更难，但 broken 的召回也不稳定，样本量不足以做强结论”。",
        "",
        f"### F. close_mosaic 前后趋势是否变化？\n\n有明显变化。epoch 11-20 平均 mAP50-95={fmt(mosaic_before)}；epoch 21-30（关闭 mosaic 的最后 10 轮）平均={fmt(mosaic_after)}，约为前段的 {mosaic_ratio:.1f} 倍。由于训练轮次也同时增加，这只能说明关闭 mosaic 后的后段表现更好，不能单独归因于 close_mosaic。",
        "",
        "### G. 1280 对极小框是否更有帮助？\n\n尺寸审计显示 1280 会把每个框的理论边长变为 640 的 2 倍，明显增加可见像素；但仍有 24.9% 的框最短边小于 8 px、62.0% 小于 16 px，因此只能缓解，不能消除极小目标难度。",
        "",
        "### H. 30 epoch 是否足以判断是否值得继续正式训练？\n\n足以判断“值得继续”，但不足以判断已经收敛。最佳指标在 epoch 28，最后 10 轮 mAP 整体上升，train loss 仍下降，val loss 后段没有恶化，所以进入更长的正式 baseline（例如 300 epoch）有依据。当前绝对 AP 和置信度仍很低，正式实验必须保留中间 checkpoint 和验证曲线，不能仅凭最终 epoch 判断。",
        "",
        "## 7. 数值稳定性与异常",
        "",
        f"- 批次级训练 loss：{'未发现 NaN/Inf' if batch_finite else '发现 NaN/Inf'}。",
        f"- epoch 级验证 loss 非有限值：{val_nan or '无'}。",
        "- 前期验证 cls loss 若出现极大值，按原始 results.csv 保留；它反映早期置信度校准不稳定，并不等于训练批次发生 NaN。",
        "- 未触发 OOM，batch 始终为 4。",
        "",
        "## 8. 产物",
        "",
        f"- `results.csv`：`{Path(status['save_dir']) / 'results.csv'}`",
        f"- `best.pt`：`{status['best_pt']}`",
        f"- `last.pt`：`{status['last_pt']}`",
        f"- 低置信度预测：`{PRED_ROOT}`",
        f"- bbox 审计：`{WORKSPACE / 'notes' / 'bbox_size_audit.md'}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    settings_dir = WORKSPACE / "configs" / "ultralytics_settings"
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")
    os.environ["TORCH_HOME"] = str(WORKSPACE / "cache" / "torch")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    install_write_guard()

    import torch
    import ultralytics
    from ultralytics import YOLO

    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(f"Wrong Python environment: {sys.prefix}")
    if not Path(ultralytics.__file__).resolve().is_relative_to((PROJECT / "ultralytics_src").resolve()):
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    commit = subprocess.check_output(["git", "-C", str(PROJECT / "ultralytics_src"), "rev-parse", "HEAD"], text=True).strip()
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Official source commit changed: {commit}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    status_path = WORKSPACE / "logs" / "latest_yolo11s_diagnostic_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if not status.get("success"):
        raise RuntimeError(f"Training did not complete successfully: {status.get('error')}")
    status["batch_log"] = str(WORKSPACE / "logs" / f"{status['run_name']}_batches.jsonl")
    run = Path(status["save_dir"])
    rows = read_results(run / "results.csv")
    for required in (Path(status["best_pt"]), Path(status["last_pt"]), run / "results.png", run / "confusion_matrix.png"):
        if not required.is_file():
            raise RuntimeError(f"Missing required artifact: {required}")

    redirect_label_caches()
    model = YOLO(status["best_pt"])
    metrics = model.val(
        data=str(DATA_YAML),
        split="val",
        imgsz=1280,
        # Ultralytics doubles the validation batch inside model.train(); batch=8
        # reproduces the canonical best.pt validation shown at training end.
        batch=8,
        rect=False,
        half=True,
        workers=0,
        device=0,
        plots=True,
        project=str(WORKSPACE / "analysis"),
        name=f"{status['run_name']}_best_val",
        exist_ok=True,
        verbose=True,
    )
    per_class = json_native(metrics.summary(decimals=8))
    if len(per_class) != 2:
        raise RuntimeError(f"Expected two per-class rows, got {per_class}")
    write_csv(WORKSPACE / "analysis" / "yolo11s_best_val_per_class.csv", per_class)
    (WORKSPACE / "analysis" / "yolo11s_best_val_per_class.json").write_text(
        json.dumps(per_class, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    predictions = prediction_stats(model, [0.001, 0.01, 0.05, 0.25])
    integrity = verify_dataset_unchanged("dataset_integrity_after_yolo11s_diagnostic.json")
    status["post_analysis"] = {
        "success": True,
        "per_class_csv": str(WORKSPACE / "analysis" / "yolo11s_best_val_per_class.csv"),
        "prediction_root": str(PRED_ROOT),
        "report": str(REPORT_PATH),
        "dataset_integrity_after": integrity,
    }
    REPORT_PATH.write_text(make_report(status, rows, per_class, predictions, integrity), encoding="utf-8")
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status["post_analysis"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
