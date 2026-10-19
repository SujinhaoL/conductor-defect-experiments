r"""Audit the locked Ultralytics Mosaic pipeline without training or model inference.

The central dataset is protected by the project write guard.  This script uses
YOLODataset's official transforms from the locked editable source and only adds
external probes around Mosaic and RandomPerspective to count retained targets.
"""

from __future__ import annotations

import csv
import heapq
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from check_dataset import snapshot
from readonly_dataset import (
    BLOCKED_WRITES,
    DATASET,
    PROJECT,
    PROTECTED_ROOT,
    WORKSPACE,
    install_write_guard,
    redirect_label_caches,
)

EXPECTED_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
CLASS_NAMES = {0: "broken", 1: "loose"}
SEEDS = tuple(range(10))
IMGSZ = 1280
ANALYSIS = WORKSPACE / "analysis" / "mosaic_audit"
SAMPLES_CSV = ANALYSIS / "mosaic_bbox_samples.csv"
SUMMARY_CSV = ANALYSIS / "mosaic_bbox_summary.csv"
ON_SAMPLES = ANALYSIS / "mosaic_on_samples.jpg"
OFF_SAMPLES = ANALYSIS / "mosaic_off_samples.jpg"
SMALLEST = ANALYSIS / "mosaic_on_smallest_boxes.jpg"
REPORT = WORKSPACE / "notes" / "mosaic_effect_audit.md"
STATUS = WORKSPACE / "logs" / "mosaic_audit_status.json"

HYPERPARAMETERS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "copy_paste_mode": "flip",
}


def class_counts(labels: dict[str, Any]) -> dict[str, int]:
    """Return broken/loose/all counts from an official transform label dict."""
    values = labels.get("cls")
    if values is None:
        flat: list[int] = []
    elif hasattr(values, "detach"):
        flat = values.detach().cpu().numpy().reshape(-1).astype(int).tolist()
    else:
        flat = np.asarray(values).reshape(-1).astype(int).tolist()
    counts = Counter(flat)
    return {"broken": counts[0], "loose": counts[1], "all": len(flat)}


class SampleRecorder:
    """Collect target counts around official transforms for one augmented sample."""

    def __init__(self) -> None:
        self.active: dict[str, Any] = {}

    def start(self, condition: str, seed: int, sample_index: int, dataset_index: int, source_image: str) -> None:
        self.active = {
            "condition": condition,
            "seed": seed,
            "sample_index": sample_index,
            "dataset_index": dataset_index,
            "source_image": source_image,
        }

    def set_counts(self, stage: str, counts: dict[str, int]) -> None:
        for class_name, value in counts.items():
            self.active[f"{stage}_{class_name}"] = int(value)

    def set_source_layout(self, layout: list[dict[str, Any]]) -> None:
        counts = Counter()
        for item in layout:
            patch_counts = class_counts(item["labels_patch"])
            counts["broken"] += patch_counts["broken"]
            counts["loose"] += patch_counts["loose"]
            counts["all"] += patch_counts["all"]
        self.set_counts("source_candidates", dict(counts))

    def finish(self, final_labels: dict[str, Any]) -> dict[str, Any]:
        self.set_counts("final_targets", class_counts(final_labels))
        for class_name in ("broken", "loose", "all"):
            post_rp = self.active[f"post_random_perspective_{class_name}"]
            final = self.active[f"final_targets_{class_name}"]
            if post_rp != final:
                raise RuntimeError(
                    f"Unexpected count change after RandomPerspective for {class_name}: {post_rp} -> {final}"
                )
        return dict(self.active)


class TransformProbe:
    """Transparent wrapper around an official transform instance."""

    def __init__(self, transform: Any, stage: str, recorder: SampleRecorder) -> None:
        self.transform = transform
        self.stage = stage
        self.recorder = recorder

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        if self.stage == "mosaic":
            # This default is exact for mosaic=0.  For mosaic=1, get_params()
            # replaces it with the four-source layout count before any clipping.
            self.recorder.set_counts("source_candidates", class_counts(labels))
            output = self.transform(labels)
            self.recorder.set_counts("post_mosaic", class_counts(output))
            return output
        if self.stage == "random_perspective":
            self.recorder.set_counts("pre_random_perspective", class_counts(labels))
            output = self.transform(labels)
            self.recorder.set_counts("post_random_perspective", class_counts(output))
            return output
        raise ValueError(self.stage)


def configure_official_probes(dataset: Any, recorder: SampleRecorder) -> None:
    """Insert count-only wrappers without changing official transform behavior."""
    augment_compose = dataset.transforms.transforms[0]
    transforms = augment_compose.transforms
    mosaic_index = next(i for i, item in enumerate(transforms) if item.__class__.__name__ == "Mosaic")
    perspective_index = next(
        i for i, item in enumerate(transforms) if item.__class__.__name__ == "RandomPerspective"
    )
    mosaic = transforms[mosaic_index]
    original_get_params = mosaic.get_params

    def audited_get_params(labels: dict[str, Any]) -> dict[str, Any]:
        params = original_get_params(labels)
        recorder.set_source_layout(params["layout"])
        return params

    # Assigning to this one instance is an external audit hook; no source file is modified.
    mosaic.get_params = audited_get_params
    transforms[mosaic_index] = TransformProbe(mosaic, "mosaic", recorder)
    transforms[perspective_index] = TransformProbe(
        transforms[perspective_index], "random_perspective", recorder
    )


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataset(condition: str, seed: int, recorder: SampleRecorder):
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset

    seed_everything(seed)
    mosaic = 1.0 if condition == "MOSAIC_ON" else 0.0
    hyp = get_cfg(overrides={**HYPERPARAMETERS, "mosaic": mosaic})
    dataset = YOLODataset(
        img_path=str(DATASET / "images" / "train"),
        imgsz=IMGSZ,
        batch_size=4,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        data={"names": CLASS_NAMES, "nc": 2},
        task="detect",
        prefix=f"{condition} seed={seed}: ",
    )
    if len(dataset) != 107:
        raise RuntimeError(f"Expected 107 train images, got {len(dataset)}")
    configure_official_probes(dataset, recorder)
    return dataset


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else math.nan


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {key: math.nan for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    return {
        "min": min(values),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "p50": percentile(values, 50),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "max": max(values),
    }


def font(size: int):
    for candidate in (
        WORKSPACE / "configs" / "ultralytics_settings" / "Ultralytics" / "Arial.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
    ):
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def tensor_to_image(tensor: Any) -> Image.Image:
    array = tensor.detach().cpu().numpy()
    array = array.transpose(1, 2, 0)
    # Ultralytics Format currently returns uint8 CHW tensors. Keep this guard so
    # the visual audit also remains correct if a future compatible Format emits
    # normalized floats.
    if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0.0)) <= 1.5:
        array = array * 255.0
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def sample_tile(image: Image.Image, boxes: np.ndarray, classes: np.ndarray, caption: str) -> Image.Image:
    tile_size, caption_height = 300, 45
    resized = image.resize((tile_size, tile_size), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (tile_size, tile_size + caption_height), "white")
    canvas.paste(resized, (0, 0))
    draw = ImageDraw.Draw(canvas)
    colors = {0: (255, 60, 60), 1: (0, 210, 255)}
    sx, sy = tile_size / image.width, tile_size / image.height
    for box, class_id in zip(boxes, classes):
        cx, cy, width, height = box
        x1, y1 = (cx - width / 2) * sx, (cy - height / 2) * sy
        x2, y2 = (cx + width / 2) * sx, (cy + height / 2) * sy
        color = colors[int(class_id)]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=2)
        draw.text((max(0, x1), max(0, y1 - 16)), CLASS_NAMES[int(class_id)], fill=color, font=font(13))
    draw.text((6, tile_size + 4), caption, fill="black", font=font(14))
    draw.rectangle((0, 0, tile_size - 1, tile_size + caption_height - 1), outline=(150, 150, 150))
    return canvas


def create_contact_sheet(tiles: list[Image.Image], output: Path) -> None:
    if len(tiles) < 20:
        raise RuntimeError(f"Expected at least 20 visualization tiles, got {len(tiles)}")
    columns = 5
    rows = math.ceil(len(tiles) / columns)
    canvas = Image.new("RGB", (columns * tiles[0].width, rows * tiles[0].height), "white")
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
    canvas.save(output, quality=94, subsampling=0)


def smallest_crop(image: Image.Image, row: dict[str, Any]) -> dict[str, Any]:
    cx, cy = row["bbox_center_x_px"], row["bbox_center_y_px"]
    bw, bh = row["bbox_width_px"], row["bbox_height_px"]
    context = min(480.0, max(160.0, bw * 14.0, bh * 14.0))
    left = max(0.0, min(cx - context / 2, image.width - context))
    top = max(0.0, min(cy - context / 2, image.height - context))
    right, bottom = min(float(image.width), left + context), min(float(image.height), top + context)
    crop = image.crop((round(left), round(top), round(right), round(bottom)))
    return {
        "image": crop,
        "box_in_crop": (cx - bw / 2 - left, cy - bh / 2 - top, cx + bw / 2 - left, cy + bh / 2 - top),
        "row": dict(row),
    }


def smallest_contact_sheet(items: list[dict[str, Any]]) -> None:
    ordered = sorted(items, key=lambda item: (item["row"]["bbox_min_side_px"], item["row"]["bbox_area_px"]))[:30]
    columns, tile_width, image_height, caption_height = 5, 320, 240, 78
    canvas = Image.new(
        "RGB", (columns * tile_width, math.ceil(len(ordered) / columns) * (image_height + caption_height)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    colors = {0: (255, 60, 60), 1: (0, 180, 230)}
    for index, item in enumerate(ordered):
        crop = item["image"]
        scale = min(tile_width / crop.width, image_height / crop.height)
        resized = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        ox = (index % columns) * tile_width + (tile_width - resized.width) // 2
        oy = (index // columns) * (image_height + caption_height) + (image_height - resized.height) // 2
        canvas.paste(resized, (ox, oy))
        x1, y1, x2, y2 = item["box_in_crop"]
        x1, y1, x2, y2 = ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale
        row = item["row"]
        color = colors[row["class_id"]]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        draw.line((center_x - 10, center_y, center_x + 10, center_y), fill=color, width=2)
        draw.line((center_x, center_y - 10, center_x, center_y + 10), fill=color, width=2)
        caption_y = (index // columns) * (image_height + caption_height) + image_height + 3
        draw.text(
            ((index % columns) * tile_width + 5, caption_y),
            f"#{index + 1} {row['class_name']}  seed={row['seed']} sample={row['sample_index']}",
            fill=color,
            font=font(14),
        )
        draw.text(
            ((index % columns) * tile_width + 5, caption_y + 23),
            f"{row['bbox_width_px']:.2f} x {row['bbox_height_px']:.2f}px  min={row['bbox_min_side_px']:.2f}",
            fill="black",
            font=font(14),
        )
        draw.text(
            ((index % columns) * tile_width + 5, caption_y + 46),
            Path(row["source_image"]).name,
            fill="black",
            font=font(13),
        )
    canvas.save(SMALLEST, quality=95, subsampling=0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows for {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_summary(bbox_rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for condition in ("MOSAIC_ON", "MOSAIC_OFF"):
        condition_samples = [row for row in samples if row["condition"] == condition]
        for class_name in ("broken", "loose", "all"):
            selected = [
                row for row in bbox_rows
                if row["condition"] == condition and (class_name == "all" or row["class_name"] == class_name)
            ]
            record: dict[str, Any] = {
                "condition": condition,
                "class_name": class_name,
                "sample_count": len(condition_samples),
                "bbox_count": len(selected),
            }
            for metric, column in (
                ("width", "bbox_width_px"),
                ("height", "bbox_height_px"),
                ("min_side", "bbox_min_side_px"),
                ("area", "bbox_area_px"),
            ):
                stats = describe([row[column] for row in selected])
                record.update({f"{metric}_{name}": value for name, value in stats.items()})
            min_sides = [row["bbox_min_side_px"] for row in selected]
            areas = [row["bbox_area_px"] for row in selected]
            for threshold in (2, 4, 8, 16, 32):
                count = sum(value < threshold for value in min_sides)
                record[f"min_side_lt_{threshold}_count"] = count
                record[f"min_side_lt_{threshold}_ratio"] = count / len(min_sides) if min_sides else math.nan
            for side in (4, 8, 16, 32):
                count = sum(value < side * side for value in areas)
                record[f"area_lt_{side}x{side}_count"] = count
                record[f"area_lt_{side}x{side}_ratio"] = count / len(areas) if areas else math.nan
            source = sum(row[f"source_candidates_{class_name}"] for row in condition_samples)
            post_mosaic = sum(row[f"post_mosaic_{class_name}"] for row in condition_samples)
            post_rp = sum(row[f"post_random_perspective_{class_name}"] for row in condition_samples)
            final = sum(row[f"final_targets_{class_name}"] for row in condition_samples)
            record.update(
                source_candidates=source,
                post_mosaic_candidates=post_mosaic,
                post_random_perspective_targets=post_rp,
                final_targets=final,
                mosaic_clip_retention=post_mosaic / source if source else math.nan,
                random_perspective_retention=post_rp / post_mosaic if post_mosaic else math.nan,
                total_retention=final / source if source else math.nan,
                samples_with_zero_final_targets=sum(row[f"final_targets_{class_name}"] == 0 for row in condition_samples),
            )
            output.append(record)
    return output


def pct(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.2%}"


def write_report(summary: list[dict[str, Any]], bbox_rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> None:
    def get(condition: str, class_name: str) -> dict[str, Any]:
        return next(row for row in summary if row["condition"] == condition and row["class_name"] == class_name)

    on_all, off_all = get("MOSAIC_ON", "all"), get("MOSAIC_OFF", "all")
    on_broken, off_broken = get("MOSAIC_ON", "broken"), get("MOSAIC_OFF", "broken")
    on_loose, off_loose = get("MOSAIC_ON", "loose"), get("MOSAIC_OFF", "loose")

    seed_directions: dict[str, dict[int, float]] = defaultdict(dict)
    for condition in ("MOSAIC_ON", "MOSAIC_OFF"):
        for seed in SEEDS:
            values = [
                row["bbox_min_side_px"]
                for row in bbox_rows
                if row["condition"] == condition and row["seed"] == seed
            ]
            seed_directions[condition][seed] = sum(value < 8 for value in values) / len(values)
    on_greater_lt8 = sum(
        seed_directions["MOSAIC_ON"][seed] > seed_directions["MOSAIC_OFF"][seed] for seed in SEEDS
    )

    median_ratio = on_all["min_side_p50"] / off_all["min_side_p50"]
    tiny4_ratio = on_all["min_side_lt_4_ratio"]
    enough_for_ablation = (
        median_ratio <= 0.80
        or on_all["min_side_lt_8_ratio"] - off_all["min_side_lt_8_ratio"] >= 0.05
        or off_all["total_retention"] - on_all["total_retention"] >= 0.05
    )
    broken_median_ratio = on_broken["min_side_p50"] / off_broken["min_side_p50"]
    loose_median_ratio = on_loose["min_side_p50"] / off_loose["min_side_p50"]
    broken_lt8_delta = on_broken["min_side_lt_8_ratio"] - off_broken["min_side_lt_8_ratio"]
    loose_lt8_delta = on_loose["min_side_lt_8_ratio"] - off_loose["min_side_lt_8_ratio"]
    lines = [
        "# Mosaic 对极小目标尺寸的只读审计",
        "",
        f"> 锁定源码 commit：`{EXPECTED_COMMIT}`。只读取 train=107；未读取 val/test 目标用于策略判断；没有模型 forward、loss 或训练。",
        "",
        "## 审计设计",
        "",
        "- 两组均直接使用当前 Ultralytics `YOLODataset` 官方训练 transforms；外层探针只计数，不改图像、bbox 或随机过程。",
        f"- `MOSAIC_ON` 与 `MOSAIC_OFF` 各使用种子 0-9，每个种子完整遍历 107 张训练图，共各 {len(SEEDS) * 107} 个最终增强样本。",
        "- 唯一条件差异为 `mosaic=1.0` 与 `mosaic=0.0`；imgsz、RandomPerspective、HSV、翻转等参数完全一致。",
        "- retention 可可靠追踪为：进入 Mosaic 的源候选数 → Mosaic 裁剪后 → RandomPerspective 过滤后；这是类别总量追踪，不宣称逐框身份匹配。",
        "",
        "## 核心统计",
        "",
        "|范围|ON min_side P50|OFF min_side P50|ON <2px|OFF <2px|ON <4px|OFF <4px|ON <8px|OFF <8px|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, on, off in (
        ("all", on_all, off_all),
        ("broken", on_broken, off_broken),
        ("loose", on_loose, off_loose),
    ):
        lines.append(
            f"|{label}|{on['min_side_p50']:.3f}|{off['min_side_p50']:.3f}|"
            f"{pct(on['min_side_lt_2_ratio'])}|{pct(off['min_side_lt_2_ratio'])}|"
            f"{pct(on['min_side_lt_4_ratio'])}|{pct(off['min_side_lt_4_ratio'])}|"
            f"{pct(on['min_side_lt_8_ratio'])}|{pct(off['min_side_lt_8_ratio'])}|"
        )
    lines += [
        "",
        "## 目标保留",
        "",
        "|范围|条件|源候选|Mosaic 后|RandomPerspective 后|Mosaic clip retention|RP retention|总 retention|",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("all", "broken", "loose"):
        for condition in ("MOSAIC_ON", "MOSAIC_OFF"):
            row = get(condition, label)
            lines.append(
                f"|{label}|{condition}|{row['source_candidates']}|{row['post_mosaic_candidates']}|"
                f"{row['post_random_perspective_targets']}|{pct(row['mosaic_clip_retention'])}|"
                f"{pct(row['random_perspective_retention'])}|{pct(row['total_retention'])}|"
            )
    lines += [
        "",
        "## A-H 回答",
        "",
        f"### A. 全部目标 min_side 中位数\n\nMosaic ON={on_all['min_side_p50']:.3f}px；OFF={off_all['min_side_p50']:.3f}px。ON/OFF={median_ratio:.3f}。",
        "",
        f"### B. broken\n\nON={on_broken['min_side_p50']:.3f}px；OFF={off_broken['min_side_p50']:.3f}px。",
        "",
        f"### C. loose\n\nON={on_loose['min_side_p50']:.3f}px；OFF={off_loose['min_side_p50']:.3f}px。",
        "",
        "### D. Mosaic ON 是否增加 <2/<4/<8 px？",
        "",
        "描述性统计显示：<4px 与 <8px 比例明显增加，<2px 没有增加。"
        f"全部目标：<2px {pct(off_all['min_side_lt_2_ratio'])}→{pct(on_all['min_side_lt_2_ratio'])}；"
        f"<4px {pct(off_all['min_side_lt_4_ratio'])}→{pct(on_all['min_side_lt_4_ratio'])}；"
        f"<8px {pct(off_all['min_side_lt_8_ratio'])}→{pct(on_all['min_side_lt_8_ratio'])}。"
        f"10 个配对种子中有 {on_greater_lt8}/10 个种子的 <8px 比例在 ON 下更高。这里报告描述性差异，不把重复增强框当作独立样本做伪显著性检验。",
        "",
        "### E. loose 是否更严重？\n\n按实际落入极小尺寸区间的风险看，是："
        f"loose 的 <8px 比例从 {pct(off_loose['min_side_lt_8_ratio'])} 增至 {pct(on_loose['min_side_lt_8_ratio'])} "
        f"（+{loose_lt8_delta:.2%}），broken 从 {pct(off_broken['min_side_lt_8_ratio'])} 增至 "
        f"{pct(on_broken['min_side_lt_8_ratio'])}（+{broken_lt8_delta:.2%}）。"
        f"但若只比较各类别自身的中位数相对缩小，broken 略大：broken ON/OFF={broken_median_ratio:.3f}，"
        f"loose={loose_median_ratio:.3f}。因此结论是 loose 的绝对极小框暴露更严重，而不是其相对缩小倍数更大。",
        "",
        f"### F. 是否大量只剩 1-3 像素？\n\n{'是' if tiny4_ratio >= 0.10 else '未达到大量'}。"
        f"Mosaic ON 中 min_side<4px 为 {on_all['min_side_lt_4_count']}/{on_all['bbox_count']} ({pct(tiny4_ratio)})，"
        f"其中 <2px 为 {on_all['min_side_lt_2_count']} ({pct(on_all['min_side_lt_2_ratio'])})。",
        "",
        "### G. 是否观察到目标丢失/裁剪？",
        "",
        f"Mosaic ON 的源候选→Mosaic 后保留 {on_all['post_mosaic_candidates']}/{on_all['source_candidates']} "
        f"({pct(on_all['mosaic_clip_retention'])})；再经 RandomPerspective 后为 "
        f"{on_all['post_random_perspective_targets']}/{on_all['post_mosaic_candidates']} "
        f"({pct(on_all['random_perspective_retention'])})。OFF 的总 retention={pct(off_all['total_retention'])}。",
        "",
        f"### H. 是否有充分依据做 mosaic 1→0 单变量训练？\n\n{'有' if enough_for_ablation else '目前证据不足'}。"
        "该判断只基于上面的尺寸、极小框比例和保留率；它支持或不支持启动一次严格单变量消融，不预先断言训练指标一定改善。",
        "",
        "## 产物与限制",
        "",
        "- `mosaic_bbox_samples.csv`：最终增强 bbox 逐框数据。",
        "- `mosaic_bbox_summary.csv`：两条件 × broken/loose/all 的分位数、阈值比例和 retention 汇总。",
        "- contact sheets 只用于人工复核；结论使用 CSV 数值。",
        "- 两组使用相同种子集合和相同样本遍历顺序；Mosaic ON 会额外消耗随机数，因此后续 RandomPerspective 不是逐样本共享同一随机矩阵，本报告比较的是重复采样后的条件分布。",
        "- 本审计模拟 Mosaic 开启阶段；原 30 epoch 设置的最后 10 epoch 会由 `close_mosaic=10` 关闭 Mosaic。",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_environment() -> dict[str, Any]:
    import ultralytics

    source = PROJECT / "ultralytics_src"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if commit != EXPECTED_COMMIT or git_status.strip():
        raise RuntimeError(f"Official source mismatch: commit={commit}, status={git_status!r}")
    if not Path(ultralytics.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(f"Wrong environment: {sys.prefix}")
    audit = json.loads((WORKSPACE / "logs" / "dataset_check.json").read_text(encoding="utf-8"))
    if not audit["passed"] or audit["splits"]["train"]["images"] != 107:
        raise RuntimeError("Locked train split audit does not match 107 images")
    return {"commit": commit, "git_status": git_status, "ultralytics": ultralytics.__version__, "ultralytics_file": ultralytics.__file__}


def main() -> None:
    started = time.monotonic()
    os.environ["YOLO_CONFIG_DIR"] = str(WORKSPACE / "configs" / "ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    install_write_guard()
    redirect_label_caches()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    environment = verify_environment()
    before = snapshot()
    bbox_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    contact_tiles: dict[str, list[Image.Image]] = {"MOSAIC_ON": [], "MOSAIC_OFF": []}
    smallest_heap: list[tuple[float, float, int, dict[str, Any]]] = []
    unique_bbox = 0

    for condition in ("MOSAIC_ON", "MOSAIC_OFF"):
        for seed in SEEDS:
            recorder = SampleRecorder()
            dataset = build_dataset(condition, seed, recorder)
            order = np.random.default_rng(seed).permutation(len(dataset)).tolist()
            for sample_index, dataset_index in enumerate(order, 1):
                source_image = dataset.im_files[dataset_index]
                recorder.start(condition, seed, sample_index, dataset_index, source_image)
                output = dataset[dataset_index]
                sample_record = recorder.finish(output)
                sample_rows.append(sample_record)
                image = tensor_to_image(output["img"])
                if image.size != (IMGSZ, IMGSZ):
                    raise RuntimeError(f"Unexpected final image size: {image.size}")
                bboxes_norm = output["bboxes"].detach().cpu().numpy().reshape(-1, 4)
                classes = output["cls"].detach().cpu().numpy().reshape(-1).astype(int)
                bboxes_px = bboxes_norm.copy()
                bboxes_px[:, [0, 2]] *= image.width
                bboxes_px[:, [1, 3]] *= image.height
                if seed == 0 and len(contact_tiles[condition]) < 20:
                    contact_tiles[condition].append(
                        sample_tile(image, bboxes_px, classes, f"seed=0 sample={sample_index} boxes={len(classes)}")
                    )
                for bbox_index, (box, class_id) in enumerate(zip(bboxes_px, classes), 1):
                    unique_bbox += 1
                    cx, cy, width, height = map(float, box)
                    row: dict[str, Any] = {
                        "condition": condition,
                        "seed": seed,
                        "sample_index": sample_index,
                        "dataset_index": dataset_index,
                        "source_image": source_image,
                        "bbox_index": bbox_index,
                        "class_id": int(class_id),
                        "class_name": CLASS_NAMES[int(class_id)],
                        "final_image_width": image.width,
                        "final_image_height": image.height,
                        "bbox_center_x_px": cx,
                        "bbox_center_y_px": cy,
                        "bbox_width_px": width,
                        "bbox_height_px": height,
                        "bbox_min_side_px": min(width, height),
                        "bbox_area_px": width * height,
                    }
                    row.update({key: value for key, value in sample_record.items() if key not in row})
                    bbox_rows.append(row)
                    if condition == "MOSAIC_ON":
                        item = smallest_crop(image, row)
                        entry = (-row["bbox_min_side_px"], -row["bbox_area_px"], unique_bbox, item)
                        if len(smallest_heap) < 30:
                            heapq.heappush(smallest_heap, entry)
                        elif entry[:2] > smallest_heap[0][:2]:
                            heapq.heapreplace(smallest_heap, entry)
            print(
                f"AUDIT_PROGRESS condition={condition} seed={seed} samples={len(order)} "
                f"total_bbox_rows={len(bbox_rows)}",
                flush=True,
            )

    summary = make_summary(bbox_rows, sample_rows)
    write_csv(SAMPLES_CSV, bbox_rows)
    write_csv(SUMMARY_CSV, summary)
    create_contact_sheet(contact_tiles["MOSAIC_ON"], ON_SAMPLES)
    create_contact_sheet(contact_tiles["MOSAIC_OFF"], OFF_SAMPLES)
    smallest_contact_sheet([entry[3] for entry in smallest_heap])
    write_report(summary, bbox_rows, sample_rows)

    after = snapshot()
    changes = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    git_after = subprocess.check_output(
        ["git", "-C", str(PROJECT / "ultralytics_src"), "status", "--porcelain"], text=True
    )
    if changes or git_after.strip() or BLOCKED_WRITES:
        raise RuntimeError(
            f"Integrity failure: dataset_changes={changes[:10]}, git={git_after!r}, blocked={BLOCKED_WRITES}"
        )
    status = {
        "success": True,
        "training_started": False,
        "model_forward_used": False,
        "test_used": False,
        "val_used_for_strategy": False,
        "conditions": ["MOSAIC_ON", "MOSAIC_OFF"],
        "seeds": list(SEEDS),
        "samples_per_condition": len(SEEDS) * 107,
        "bbox_rows": len(bbox_rows),
        "source": environment,
        "dataset_integrity": {"unchanged": True, "entries_checked": len(after)},
        "git_status_after": git_after,
        "blocked_dataset_writes": list(BLOCKED_WRITES),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "outputs": [str(SAMPLES_CSV), str(SUMMARY_CSV), str(ON_SAMPLES), str(OFF_SAMPLES), str(SMALLEST), str(REPORT)],
    }
    STATUS.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
