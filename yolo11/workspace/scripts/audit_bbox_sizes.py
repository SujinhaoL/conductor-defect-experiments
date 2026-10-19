"""Read-only train/val bounding-box size audit and smallest-target contact sheet."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import quantiles

from PIL import Image, ImageDraw, ImageFont

from readonly_dataset import DATASET, WORKSPACE, install_write_guard

INPUT_SIZES = (640, 960, 1280)
CLASS_NAMES = {0: "broken", 1: "loose"}
ANALYSIS = WORKSPACE / "analysis"
ALL_CSV = ANALYSIS / "bbox_size_all.csv"
SUMMARY_CSV = ANALYSIS / "bbox_size_summary.csv"
CONTACT_SHEET = ANALYSIS / "smallest_bbox_contact_sheet.jpg"
REPORT = WORKSPACE / "notes" / "bbox_size_audit.md"


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile matching NumPy's default method."""
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def descriptive(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p10": percentile(values, 0.10),
        "p25": percentile(values, 0.25),
        "p50": percentile(values, 0.50),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def load_rows() -> list[dict]:
    rows = []
    for split in ("train", "val"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            lines = [line.split() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            for label_line, tokens in enumerate(lines, 1):
                class_float, center_x, center_y, norm_width, norm_height = map(float, tokens)
                class_id = int(class_float)
                if class_float != class_id or class_id not in CLASS_NAMES:
                    raise ValueError(f"Unexpected class in {label_path}:{label_line}: {class_float}")
                pixel_width = norm_width * image_width
                pixel_height = norm_height * image_height
                row = {
                    "image_filename": image_path.name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_line": label_line,
                    "split": split,
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id],
                    "image_width": image_width,
                    "image_height": image_height,
                    "center_x_norm": center_x,
                    "center_y_norm": center_y,
                    "bbox_width_norm": norm_width,
                    "bbox_height_norm": norm_height,
                    "bbox_width_original_px": pixel_width,
                    "bbox_height_original_px": pixel_height,
                    "bbox_min_side_original_px": min(pixel_width, pixel_height),
                    "bbox_area_original_px2": pixel_width * pixel_height,
                    "bbox_area_ratio": norm_width * norm_height,
                }
                for size in INPUT_SIZES:
                    scale = min(size / image_width, size / image_height)
                    scaled_width = pixel_width * scale
                    scaled_height = pixel_height * scale
                    row.update(
                        {
                            f"letterbox_scale_{size}": scale,
                            f"bbox_width_{size}_px": scaled_width,
                            f"bbox_height_{size}_px": scaled_height,
                            f"bbox_min_side_{size}_px": min(scaled_width, scaled_height),
                            f"bbox_area_{size}_px2": scaled_width * scaled_height,
                        }
                    )
                rows.append(row)
    if not rows:
        raise RuntimeError("No train/val boxes found.")
    return rows


def write_all_csv(rows: list[dict]) -> None:
    with ALL_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict]) -> list[dict]:
    summary = []
    scopes = {"train": lambda row: row["split"] == "train", "val": lambda row: row["split"] == "val", "train+val": lambda row: True}
    classes = {"broken": lambda row: row["class_id"] == 0, "loose": lambda row: row["class_id"] == 1, "all": lambda row: True}
    sizes: tuple[str | int, ...] = ("original", *INPUT_SIZES)
    for scope_name, scope_filter in scopes.items():
        for class_name, class_filter in classes.items():
            selected = [row for row in rows if scope_filter(row) and class_filter(row)]
            for size in sizes:
                suffix = "original" if size == "original" else str(size)
                values = {
                    "width": [row[f"bbox_width_{suffix}_px"] for row in selected],
                    "height": [row[f"bbox_height_{suffix}_px"] for row in selected],
                    "min_side": [row[f"bbox_min_side_{suffix}_px"] for row in selected],
                    "area": [row[f"bbox_area_{suffix}_px2"] for row in selected],
                }
                record = {"scope": scope_name, "class": class_name, "input_size": size, "bbox_count": len(selected)}
                for metric, metric_values in values.items():
                    record.update({f"{metric}_{stat}": value for stat, value in descriptive(metric_values).items()})
                if size != "original":
                    for threshold in (4, 8, 16):
                        count = sum(value < threshold for value in values["min_side"])
                        record[f"min_side_lt_{threshold}_count"] = count
                        record[f"min_side_lt_{threshold}_ratio"] = count / len(selected)
                    for threshold in (16 * 16, 32 * 32):
                        count = sum(value < threshold for value in values["area"])
                        record[f"area_lt_{threshold}_count"] = count
                        record[f"area_lt_{threshold}_ratio"] = count / len(selected)
                else:
                    for threshold in (4, 8, 16):
                        record[f"min_side_lt_{threshold}_count"] = ""
                        record[f"min_side_lt_{threshold}_ratio"] = ""
                    for threshold in (16 * 16, 32 * 32):
                        record[f"area_lt_{threshold}_count"] = ""
                        record[f"area_lt_{threshold}_ratio"] = ""
                summary.append(record)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        WORKSPACE / "configs" / "ultralytics_settings" / "Ultralytics" / "Arial.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def contact_sheet(rows: list[dict]) -> list[dict]:
    smallest = sorted(rows, key=lambda row: (row["bbox_min_side_640_px"], row["bbox_area_640_px2"], row["image_filename"], row["label_line"]))[:30]
    columns, tile_width, image_height, caption_height = 5, 360, 230, 95
    rows_count = math.ceil(len(smallest) / columns)
    canvas = Image.new("RGB", (columns * tile_width, rows_count * (image_height + caption_height)), "white")
    draw = ImageDraw.Draw(canvas)
    small_font, tiny_font = font(18), font(15)
    colors = {0: (40, 90, 255), 1: (0, 190, 190)}
    for index, row in enumerate(smallest):
        with Image.open(row["image_path"]) as source:
            source = source.convert("RGB")
            width, height = source.size
            cx = row["center_x_norm"] * width
            cy = row["center_y_norm"] * height
            bw = row["bbox_width_original_px"]
            bh = row["bbox_height_original_px"]
            context_width = min(width, max(256.0, bw * 8.0, bh * 3.0))
            context_height = min(height, max(192.0, bh * 8.0, bw * 3.0))
            left = max(0.0, min(cx - context_width / 2, width - context_width))
            top = max(0.0, min(cy - context_height / 2, height - context_height))
            crop_box = (round(left), round(top), round(left + context_width), round(top + context_height))
            crop = source.crop(crop_box)
        scale = min(tile_width / crop.width, image_height / crop.height)
        resized = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
        tile_x = (index % columns) * tile_width
        tile_y = (index // columns) * (image_height + caption_height)
        image_x = tile_x + (tile_width - resized.width) // 2
        image_y = tile_y + (image_height - resized.height) // 2
        canvas.paste(resized, (image_x, image_y))
        bbox_left = image_x + ((cx - bw / 2) - crop_box[0]) * scale
        bbox_top = image_y + ((cy - bh / 2) - crop_box[1]) * scale
        bbox_right = image_x + ((cx + bw / 2) - crop_box[0]) * scale
        bbox_bottom = image_y + ((cy + bh / 2) - crop_box[1]) * scale
        color = colors[row["class_id"]]
        draw.rectangle((bbox_left, bbox_top, bbox_right, bbox_bottom), outline=color, width=4)
        # Crosshairs keep sub-pixel/tiny boxes visible after resizing.
        center_x = (bbox_left + bbox_right) / 2
        center_y = (bbox_top + bbox_bottom) / 2
        draw.line((center_x - 10, center_y, center_x + 10, center_y), fill=color, width=2)
        draw.line((center_x, center_y - 10, center_x, center_y + 10), fill=color, width=2)
        caption_y = tile_y + image_height + 4
        draw.text((tile_x + 6, caption_y), f"#{index + 1} {row['split']}  {row['image_filename']}  L{row['label_line']}", fill="black", font=tiny_font)
        draw.text((tile_x + 6, caption_y + 24), row["class_name"], fill=color, font=small_font)
        draw.text((tile_x + 6, caption_y + 49), f"orig {row['bbox_width_original_px']:.1f} x {row['bbox_height_original_px']:.1f} px", fill="black", font=tiny_font)
        draw.text((tile_x + 6, caption_y + 70), f"640 {row['bbox_width_640_px']:.1f} x {row['bbox_height_640_px']:.1f} | 1280 {row['bbox_width_1280_px']:.1f} x {row['bbox_height_1280_px']:.1f}", fill="black", font=tiny_font)
        draw.rectangle((tile_x, tile_y, tile_x + tile_width - 1, tile_y + image_height + caption_height - 1), outline=(180, 180, 180), width=1)
    canvas.save(CONTACT_SHEET, quality=95, subsampling=0)
    return smallest


def format_stats(stats: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.2f}" for key, value in stats.items())


def write_report(rows: list[dict], summary: list[dict], smallest: list[dict]) -> None:
    combined = [record for record in summary if record["scope"] == "train+val"]
    counts = Counter(row["class_name"] for row in rows)
    lines = [
        "# Train + Val 目标尺寸只读审计",
        "",
        "本审计只读取 train 和 val；没有读取 test 的标签或目标尺寸，也没有修改、重存任何原始图片。",
        "",
        f"共审计 {len(rows)} 个框：broken={counts['broken']}，loose={counts['loose']}。",
        "",
        "Letterbox 理论尺寸采用 `scale=min(input/image_width, input/image_height)`；padding 只改变位置，不改变框宽高。数值未取整，代表 resize 后的理论连续像素尺寸。",
        "",
        "## Train + Val 合并统计",
        "",
    ]
    for size in INPUT_SIZES:
        lines.extend([f"### 输入 {size}", ""])
        for class_name in ("broken", "loose", "all"):
            record = next(item for item in combined if item["class"] == class_name and item["input_size"] == size)
            lines.append(
                f"- {class_name}（n={record['bbox_count']}）：min_side P10/P50/P90="
                f"{record['min_side_p10']:.2f}/{record['min_side_p50']:.2f}/{record['min_side_p90']:.2f}px；"
                f"<4px {record['min_side_lt_4_count']} ({record['min_side_lt_4_ratio']:.1%})，"
                f"<8px {record['min_side_lt_8_count']} ({record['min_side_lt_8_ratio']:.1%})，"
                f"<16px {record['min_side_lt_16_count']} ({record['min_side_lt_16_ratio']:.1%})；"
                f"area<16² {record['area_lt_256_count']} ({record['area_lt_256_ratio']:.1%})，"
                f"area<32² {record['area_lt_1024_count']} ({record['area_lt_1024_ratio']:.1%})。"
            )
        lines.append("")
    lines.extend(
        [
            "## 最小目标人工检查图",
            "",
            f"按 640 输入下理论 min_side 排序，选取前 {len(smallest)} 个框。图中的框和十字线仅画在 workspace 内生成的裁剪图上。",
            "",
            f"最小框：{smallest[0]['split']}/{smallest[0]['image_filename']} 第 {smallest[0]['label_line']} 行，"
            f"{smallest[0]['class_name']}，原始 {smallest[0]['bbox_width_original_px']:.2f}×{smallest[0]['bbox_height_original_px']:.2f}px，"
            f"640 后 {smallest[0]['bbox_width_640_px']:.2f}×{smallest[0]['bbox_height_640_px']:.2f}px，"
            f"1280 后 {smallest[0]['bbox_width_1280_px']:.2f}×{smallest[0]['bbox_height_1280_px']:.2f}px。",
            "",
            "完整逐框数据见 `workspace/analysis/bbox_size_all.csv`，全部分位数与阈值统计见 `bbox_size_summary.csv`。",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    install_write_guard()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    (WORKSPACE / "notes").mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    write_all_csv(rows)
    summary = write_summary(rows)
    smallest = contact_sheet(rows)
    write_report(rows, summary, smallest)
    result = {
        "boxes": len(rows),
        "classes": Counter(row["class_name"] for row in rows),
        "all_csv": str(ALL_CSV),
        "summary_csv": str(SUMMARY_CSV),
        "contact_sheet": str(CONTACT_SHEET),
        "report": str(REPORT),
        "test_used": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()
