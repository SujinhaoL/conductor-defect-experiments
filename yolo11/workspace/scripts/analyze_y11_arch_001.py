r"""Analyze Y11-ARCH-001 against Y11-DIAG-001 using val only."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from check_dataset import snapshot
from readonly_dataset import DATASET, PROJECT, PROTECTED_ROOT, WORKSPACE, install_write_guard, redirect_label_caches

EXPECTED_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
BASE_RUN = WORKSPACE / "runs" / "yolo11s_diag_1280_e30_20260916_104603"
P2_RUN = WORKSPACE / "runs" / "yolo11s_p2_diag_1280_e30_20260916_202431"
BASE_BEST = BASE_RUN / "weights" / "best.pt"
P2_BEST = P2_RUN / "weights" / "best.pt"
DATA_YAML = WORKSPACE / "configs" / "conductor_bl_v1.yaml"
STATUS_PATH = WORKSPACE / "logs" / "yolo11s_p2_diag_1280_e30_20260916_202431_status.json"
BASE_STATUS_PATH = WORKSPACE / "logs" / "yolo11s_diag_1280_e30_20260916_104603_status.json"
CLASS_CSV = WORKSPACE / "analysis" / "Y11-ARCH-001_per_class.csv"
PRED_CSV = WORKSPACE / "analysis" / "Y11-ARCH-001_prediction_summary.csv"
REPORT = WORKSPACE / "notes" / "p3_vs_p2_structure_report.md"
LOWCONF_IMAGE = P2_RUN / "val_lowconf_pred.jpg"
ANALYSIS_ROOT = WORKSPACE / "analysis" / "Y11-ARCH-001_validation"

METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def read_results(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [{key.strip(): float(value) for key, value in raw.items()} for raw in csv.DictReader(stream)]
    if len(rows) != 30 or [int(row["epoch"]) for row in rows] != list(range(1, 31)):
        raise RuntimeError(f"Expected epochs 1..30 in {path}")
    return rows


def best_row(rows: list[dict[str, float]]) -> dict[str, float]:
    finite = [row for row in rows if math.isfinite(row[METRICS["map50_95"]])]
    return max(finite, key=lambda row: (row[METRICS["map50_95"]], row[METRICS["map50"]], -row["epoch"]))


def extremum(rows: list[dict[str, float]], field: str) -> tuple[float, int]:
    row = max(rows, key=lambda item: (item[field], -item["epoch"]))
    return row[field], int(row["epoch"])


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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def delta(new: float, old: float) -> str:
    return f"{new - old:+.6f}"


def pct(new: float, old: float) -> str:
    if old == 0:
        return "NA"
    return f"{(new / old - 1) * 100:+.2f}%"


def validate_model(model_path: Path, label: str):
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import model_info

    model = YOLO(str(model_path))
    info = model_info(model.model, verbose=True, imgsz=1280)
    if info is None:
        raise RuntimeError(f"Unable to obtain model info for {label}")
    metrics = model.val(
        data=str(DATA_YAML), split="val", imgsz=1280, batch=8, workers=0, device=0,
        rect=False, half=True, plots=False, save_json=False,
        project=str(ANALYSIS_ROOT), name=f"{label}_best_val", exist_ok=True, verbose=True,
    )
    summary = native(metrics.summary(decimals=10))
    results = {key: float(value) for key, value in metrics.results_dict.items()}
    return model, info, summary, results


def class_rows(label: str, architecture: str, summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in summary:
        if row["Class"] == "all":
            continue
        output.append({
            "experiment_id": label,
            "architecture": architecture,
            "class_name": row["Class"],
            "images": row["Images"],
            "instances": row["Instances"],
            "precision": row["Box-P"],
            "recall": row["Box-R"],
            "map50": row["mAP50"],
            "map50_95": row["mAP50-95"],
            "delta_precision_vs_p3": None,
            "delta_recall_vs_p3": None,
            "delta_map50_vs_p3": None,
            "delta_map50_95_vs_p3": None,
        })
    return output


def val_images() -> list[Path]:
    images = sorted(
        path for path in (DATASET / "images" / "val").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    )
    if len(images) != 23:
        raise RuntimeError(f"Expected 23 val images, got {len(images)}")
    return images


def summarize_predictions(model, experiment_id: str, architecture: str, images: list[Path]) -> list[dict[str, Any]]:
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
            "experiment_id": experiment_id,
            "architecture": architecture,
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


def make_lowconf_grid(model, images: list[Path]) -> None:
    results = model.predict(
        source=[str(path) for path in images], conf=0.01, imgsz=1280, batch=4,
        device=0, save=False, save_txt=False, save_conf=False, verbose=False,
    )
    ranked = sorted(
        results,
        key=lambda result: float(result.boxes.conf.max().item()) if len(result.boxes) else -1.0,
        reverse=True,
    )[:8]
    thumb_w, thumb_h, header = 640, 450, 34
    canvas = Image.new("RGB", (thumb_w * 2, (thumb_h + header) * 4), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, result in enumerate(ranked):
        plotted = Image.fromarray(result.plot(line_width=2, font_size=12)[:, :, ::-1])
        plotted.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        x = (index % 2) * thumb_w + (thumb_w - plotted.width) // 2
        y0 = (index // 2) * (thumb_h + header)
        y = y0 + header + (thumb_h - plotted.height) // 2
        canvas.paste(plotted, (x, y))
        max_conf = float(result.boxes.conf.max().item()) if len(result.boxes) else math.nan
        label = f"{Path(result.path).name} | boxes={len(result.boxes)} | max={fmt(max_conf, 4)} | conf=0.01"
        draw.text((index % 2 * thumb_w + 8, y0 + 9), label, fill=(255, 255, 255), font=font)
    canvas.save(LOWCONF_IMAGE, quality=94, subsampling=0)


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

    controlled = (
        "model", "pretrained", "epochs", "imgsz", "batch", "nbs", "workers", "device", "seed",
        "deterministic", "patience", "cache", "amp", "optimizer", "lr0", "lrf", "momentum",
        "weight_decay", "warmup_epochs", "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v",
        "degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr", "mosaic",
        "mixup", "cutmix", "copy_paste", "close_mosaic", "resume", "split",
    )
    base_args = yaml.safe_load((BASE_RUN / "args.yaml").read_text(encoding="utf-8-sig"))
    p2_args = yaml.safe_load((P2_RUN / "args.yaml").read_text(encoding="utf-8-sig"))
    differences = {key: [base_args.get(key), p2_args.get(key)] for key in controlled if base_args.get(key) != p2_args.get(key)}
    if set(differences) != {"model"}:
        raise RuntimeError(f"Single-variable condition violated: {differences}")

    required = (
        "results.csv", "results.png", "args.yaml", "weights/best.pt", "weights/last.pt",
        "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png", "confusion_matrix.png",
        "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    )
    missing = [str(P2_RUN / item) for item in required if not (P2_RUN / item).is_file()]
    if missing:
        raise RuntimeError(f"Missing outputs: {missing}")

    base_rows = read_results(BASE_RUN / "results.csv")
    p2_rows = read_results(P2_RUN / "results.csv")
    base_best = best_row(base_rows)
    p2_best = best_row(p2_rows)
    base_max_p, base_max_p_epoch = extremum(base_rows, METRICS["precision"])
    base_max_r, base_max_r_epoch = extremum(base_rows, METRICS["recall"])
    p2_max_p, p2_max_p_epoch = extremum(p2_rows, METRICS["precision"])
    p2_max_r, p2_max_r_epoch = extremum(p2_rows, METRICS["recall"])

    redirect_label_caches()
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    base_model, base_info, base_summary, base_revalidation = validate_model(BASE_BEST, "Y11-DIAG-001")
    p2_model, p2_info, p2_summary, p2_revalidation = validate_model(P2_BEST, "Y11-ARCH-001")

    base_class = class_rows("Y11-DIAG-001", "P3/P4/P5", base_summary)
    p2_class = class_rows("Y11-ARCH-001", "P2/P3/P4/P5", p2_summary)
    base_by_class = {row["class_name"]: row for row in base_class}
    for row in p2_class:
        old = base_by_class[row["class_name"]]
        for metric in ("precision", "recall", "map50", "map50_95"):
            row[f"delta_{metric}_vs_p3"] = row[metric] - old[metric]
    write_csv(CLASS_CSV, base_class + p2_class)

    images = val_images()
    base_predictions = summarize_predictions(base_model, "Y11-DIAG-001", "P3/P4/P5", images)
    p2_predictions = summarize_predictions(p2_model, "Y11-ARCH-001", "P2/P3/P4/P5", images)
    write_csv(PRED_CSV, base_predictions + p2_predictions)
    make_lowconf_grid(p2_model, images)

    base_status = json.loads(BASE_STATUS_PATH.read_text(encoding="utf-8"))
    p2_status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    transfer = p2_status["weight_transfer"]
    base_pred_by_conf = {row["confidence_threshold"]: row for row in base_predictions}
    p2_pred_by_conf = {row["confidence_threshold"]: row for row in p2_predictions}
    base_class_by_name = {row["class_name"]: row for row in base_class}
    p2_class_by_name = {row["class_name"]: row for row in p2_class}

    decision = (
        "情况 C：P2 的整体 mAP50-95 明显下降；broken 未检出，loose 虽然 mAP50 上升，但 mAP50-95 明显下降。"
        "当前证据说明仅增加 P2 不足以解决问题；下一步应优先分析切片/局部放大训练与推理，以及目标上下文，而不继续堆叠注意力模块。"
    )
    conf_comment = (
        f"conf=0.001 时总候选框 {base_pred_by_conf[0.001]['total_prediction_boxes']}→"
        f"{p2_pred_by_conf[0.001]['total_prediction_boxes']}，最高 confidence "
        f"{fmt(base_pred_by_conf[0.001]['confidence_max'])}→{fmt(p2_pred_by_conf[0.001]['confidence_max'])}；"
        f"conf=0.25 时有预测图片数 {base_pred_by_conf[0.25]['images_with_predictions']}→"
        f"{p2_pred_by_conf[0.25]['images_with_predictions']}。"
    )

    lines = [
        "# Y11-DIAG-001 与 Y11-ARCH-001：P3/P4/P5 vs P2/P3/P4/P5",
        "",
        "> 本报告仅使用 train/val。test 未用于训练、验证、选模、预测诊断或阈值选择。",
        "",
        "## 1. 研究问题与数据依据",
        "",
        "- Mosaic ON 尺寸审计：全部目标 min_side 中位数约 10.510 px，broken 约 17.928 px，loose 约 9.357 px。",
        "- 全部目标中约 34.99% 的 min_side <8 px，约 7.35% <4 px；loose 中约 41.24% <8 px。",
        "- 标准 YOLO11s 最浅检测层为 P3/8。新增 P2/4 后，一格对应阈值从 8 px 降至 4 px，因此本实验检验空间分辨率是否是核心瓶颈。",
        "",
        "## 2. 单变量与结构验证",
        "",
        "- 主动变量仅为模型结构：`P3/P4/P5 → P2/P3/P4/P5`；两次训练的 epochs、imgsz、batch、nbs、seed、增强、loss 权重与优化器协议一致。",
        f"- 实测 Detect strides：`{p2_status['training_strides']}`；结构验证通过。",
        f"- 官方权重迁移 {transfer['total_transferred_items']}/{transfer['destination_state_items']} state items：同名同 shape {transfer['exact_transferred_items']}，Detect 重映射 {transfer['remapped_detect_items']}。",
        f"- nc=80→2 正常差异 {len(transfer['category_A_nc80_to_nc2_detection_classification_keys'])} 项；P2 新参数 {len(transfer['category_B_new_p2_path_and_p2_head_keys'])} 项；非预期缺失 {len(transfer['category_C_unexpected_unloaded_keys'])} 项。",
        f"- optimizer=auto 实际解析为 {p2_status['optimizer_actual']['name']}，lr={p2_status['optimizer_actual']['lr0']}，momentum={p2_status['optimizer_actual']['momentum']}。",
        "",
        "## 3. 整体结果",
        "",
        "best epoch 按最大 mAP50-95 选择；完全并列时比较 mAP50。P/R/mAP 均来自同一 best epoch。",
        "",
        "|实验|结构|best epoch|P|R|mAP50|mAP50-95|参数量|GFLOPs@1280|峰值显存 allocated|训练时间|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"|Y11-DIAG-001|P3/P4/P5|{int(base_best['epoch'])}|{fmt(base_best[METRICS['precision']])}|{fmt(base_best[METRICS['recall']])}|{fmt(base_best[METRICS['map50']])}|{fmt(base_best[METRICS['map50_95']])}|{base_info[1]:,}|{base_info[3]:.3f}|{base_status['peak_allocated_MiB']:.2f} MiB|{base_status['elapsed_seconds']:.2f} s|",
        f"|Y11-ARCH-001|P2/P3/P4/P5|{int(p2_best['epoch'])}|{fmt(p2_best[METRICS['precision']])}|{fmt(p2_best[METRICS['recall']])}|{fmt(p2_best[METRICS['map50']])}|{fmt(p2_best[METRICS['map50_95']])}|{p2_info[1]:,}|{p2_info[3]:.3f}|{p2_status['peak_allocated_MiB']:.2f} MiB|{p2_status['elapsed_seconds']:.2f} s|",
        "",
        f"- P2 相对 P3：mAP50 {delta(p2_best[METRICS['map50']], base_best[METRICS['map50']])}（{pct(p2_best[METRICS['map50']], base_best[METRICS['map50']])}）；mAP50-95 {delta(p2_best[METRICS['map50_95']], base_best[METRICS['map50_95']])}（{pct(p2_best[METRICS['map50_95']], base_best[METRICS['map50_95']])}）。",
        f"- max precision（独立记录，不冒充 best epoch P）：P3={fmt(base_max_p)}@epoch {base_max_p_epoch}；P2={fmt(p2_max_p)}@epoch {p2_max_p_epoch}。",
        f"- max recall（独立记录）：P3={fmt(base_max_r)}@epoch {base_max_r_epoch}；P2={fmt(p2_max_r)}@epoch {p2_max_r_epoch}。",
        "",
        "## 4. 统一设置 best.pt 逐类别复验",
        "",
        "两模型均使用 val、imgsz=1280、batch=8、half=True、rect=False 重新验证。",
        "",
        "|类别|P3 mAP50|P2 mAP50|变化|P3 mAP50-95|P2 mAP50-95|变化|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("broken", "loose"):
        old, new = base_class_by_name[name], p2_class_by_name[name]
        lines.append(
            f"|{name}|{fmt(old['map50'])}|{fmt(new['map50'])}|{delta(new['map50'], old['map50'])}|"
            f"{fmt(old['map50_95'])}|{fmt(new['map50_95'])}|{delta(new['map50_95'], old['map50_95'])}|"
        )
    lines += [
        "",
        f"1. loose 呈混合结果，不能判为明确改善：mAP50 {fmt(base_class_by_name['loose']['map50'])}→{fmt(p2_class_by_name['loose']['map50'])}，但主指标 mAP50-95 {fmt(base_class_by_name['loose']['map50_95'])}→{fmt(p2_class_by_name['loose']['map50_95'])}，且 recall 下降。",
        f"2. broken 未改善：mAP50 {fmt(base_class_by_name['broken']['map50'])}→{fmt(p2_class_by_name['broken']['map50'])}，mAP50-95 {fmt(base_class_by_name['broken']['map50_95'])}→{fmt(p2_class_by_name['broken']['map50_95'])}。",
        "3. 提升没有集中在尺寸更小的 loose；P2 的 loose 指标也低于 baseline。",
        "",
        "## 5. 低置信度预测",
        "",
        "|实验|conf|有预测图片/23|总框|broken|loose|confidence 中位数|最高 confidence|",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in base_predictions + p2_predictions:
        lines.append(
            f"|{row['experiment_id']}|{row['confidence_threshold']}|{row['images_with_predictions']}|"
            f"{row['total_prediction_boxes']}|{row['broken_boxes']}|{row['loose_boxes']}|"
            f"{fmt(row['confidence_median'])}|{fmt(row['confidence_max'])}|"
        )
    lines += [
        "",
        f"- {conf_comment}",
        "- `val_lowconf_pred.jpg` 由 P2 best.pt 在 conf=0.01 的预测生成，按最高置信度选取 8 张 val 图组成接触表。",
        "- 人工检查：train_batch0.jpg 的 Mosaic 与标注框渲染正常；val_batch0_pred.jpg 在默认阈值下无预测框，与 conf=0.25 的 0 个预测一致。低阈值接触表显示大量候选沿导线密集分布，主要是重复或误检，尚未形成可靠定位。",
        "",
        "## 6. 稳定性与完整性",
        "",
        f"- 共检查 {p2_status['batches_checked']} 个训练 batch，全部 loss 有限：{p2_status['all_batch_losses_finite']}；epoch 非有限值：{p2_status['epoch_stability']['nonfinite'] or '无'}。",
        f"- test_used={p2_status['test_used']}。训练前后数据集完整性均为 unchanged={p2_status['dataset_integrity_before']['unchanged'] and p2_status['dataset_integrity_after']['unchanged']}。",
        "- 官方源码 commit 与工作树保持锁定。",
        "",
        "## 7. 决策",
        "",
        f"**{decision}**",
        "",
        "本轮只完成 Y11-ARCH-001，不自动启动第二 seed、100/150 epoch、其他模型/分辨率、增强变化、注意力、BiFPN、ASFF、loss、SAHI、tiling、crop 或 test。",
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
        "structure": {
            "strides": p2_status["training_strides"],
            "baseline_model_info": {"layers": base_info[0], "parameters": base_info[1], "gflops_1280": base_info[3]},
            "p2_model_info": {"layers": p2_info[0], "parameters": p2_info[1], "gflops_1280": p2_info[3]},
        },
        "best_epoch": int(p2_best["epoch"]),
        "best_epoch_metrics": {short: p2_best[field] for short, field in METRICS.items()},
        "baseline_best_epoch": int(base_best["epoch"]),
        "baseline_best_epoch_metrics": {short: base_best[field] for short, field in METRICS.items()},
        "max_precision_any_epoch": p2_max_p,
        "epoch_of_max_precision": p2_max_p_epoch,
        "max_recall_any_epoch": p2_max_r,
        "epoch_of_max_recall": p2_max_r_epoch,
        "best_pt_revalidation": {"baseline": base_revalidation, "p2": p2_revalidation},
        "per_class": base_class + p2_class,
        "prediction_summary": native(base_predictions + p2_predictions),
        "confidence_comment": conf_comment,
        "decision": decision,
        "report": str(REPORT),
        "per_class_csv": str(CLASS_CSV),
        "prediction_csv": str(PRED_CSV),
        "lowconf_visualization": str(LOWCONF_IMAGE),
        "dataset_integrity": integrity_after,
        "git_status_after": git_after,
    }
    p2_status["post_analysis"] = native(analysis)
    STATUS_PATH.write_text(json.dumps(p2_status, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(native(analysis), ensure_ascii=False, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
