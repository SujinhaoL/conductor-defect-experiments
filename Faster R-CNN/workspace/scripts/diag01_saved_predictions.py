"""DIAG01：只读历史预测，解释低分框与定位；不导入/运行检测模型。

在 Faster R-CNN 文件夹下用专属 .venv 执行。默认新建唯一诊断目录；
--verify-existing 只核验已有成果，不重新生成图片，不训练、不推理。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / "workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115"
DATA = Path(r"D:\小论文\conductor_defect_dataset\03_coco_dataset")
AUDIT = PROJECT / "workspace/audit/FRCNN-DIAG-SAVED1280-001"
LABELS = {1: "broken（断股）", 2: "loose（松股）"}
GREEN, ORANGE = "#008a46", "#c45000"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    require(bool(rows), f"不能写空表：{path}")
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# 代码导读①：COCO 的 bbox 是 [左上x, 左上y, 宽, 高]，不是两个角坐标。
def xyxy(box):
    x, y, width, height = box
    return [x, y, x + width, y + height]


# 代码导读②：IoU 衡量位置重合，和模型的 score（置信度）是两件事。
def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def best_overlap(gt, predictions, same_class=True):
    candidates = [p for p in predictions
                  if not same_class or p["category_id"] == gt["category_id"]]
    if not candidates:
        return None, 0.0
    # 最大重合，不是最高分；同一预测可能成为多个GT的最佳候选。
    best = max(candidates, key=lambda p: (iou(gt["bbox"], p["bbox"]), p["score"]))
    return best, iou(gt["bbox"], best["bbox"])


def check_prediction(p, images):
    require(p["image_id"] in images and p["category_id"] in LABELS, "图片/类别ID异常")
    b = p["bbox"]
    require(len(b) == 4 and all(math.isfinite(v) for v in b), "框坐标非有限")
    require(math.isfinite(p["score"]) and .001 < p["score"] <= 1, "历史score范围异常")
    im = images[p["image_id"]]
    require(b[2] > 0 and b[3] > 0 and b[0] >= -.001 and b[1] >= -.001
            and b[0] + b[2] <= im["width"] + .001
            and b[1] + b[3] <= im["height"] + .001, "框越界或零面积；停止，不修正")


def crop_panel(image, gt, best, overlap, case_title, filename):
    """展示GT及同类最大IoU候选。只选框用于解释，不用于计算AP。"""
    boxes = [xyxy(gt["bbox"])] + ([xyxy(best["bbox"])] if best else [])
    x1, y1 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x2, y2 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    margin = max(40, .35 * max(x2 - x1, y2 - y1))
    region = (max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin)),
              min(image.width, math.ceil(x2 + margin)), min(image.height, math.ceil(y2 + margin)))
    crop = image.crop(region)
    ratio = min(940 / crop.width, 550 / crop.height)
    crop = crop.resize((max(1, round(crop.width * ratio)), max(1, round(crop.height * ratio))))
    panel = Image.new("RGB", (1000, 740), "#f6f8fa")
    ox, oy = (1000 - crop.width) // 2, 130 + (550 - crop.height) // 2
    panel.paste(crop, (ox, oy))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 22)
    small = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 19)
    draw.text((24, 12), case_title + " | " + filename, font=font, fill="#102536")
    draw.text((24, 46), f"GT {gt['id']}：{LABELS[gt['category_id']]}  |  绿色=人工标注；橙色=同类最大重合候选", font=small, fill="#102536")
    text = f"候选分数={best['score']:.6f}；IoU={overlap:.6f}" if best else "该图保存结果中没有同类候选框"
    draw.text((24, 78), text + "（IoU不是置信度）", font=small, fill="#102536")
    for box, color in [(gt["bbox"], GREEN)] + ([(best["bbox"], ORANGE)] if best else []):
        a, b, c, d = xyxy(box)
        draw.rectangle((ox + (a - region[0]) * ratio, oy + (b - region[1]) * ratio,
                        ox + (c - region[0]) * ratio, oy + (d - region[1]) * ratio), outline=color, width=3)
    draw.text((24, 702), "为看清位置而放大显示；不改变模型输入。候选覆盖不等于一对一正确检出。", font=small, fill="#334155")
    return panel


def verify_existing(folder):
    folder = Path(folder).resolve()
    manifest = read(folder / "manifest.json")
    for key in ("sources_sha256", "outputs_sha256"):
        for path, digest in manifest[key].items():
            require(sha(path) == digest, f"文件变更：{path}")
    summary = read(folder / "summary.json")
    for case in summary["cases"]:
        print(f"{case['key']}: images={case['images']} GT={case['gt']} "
              f"saved={case['saved_predictions']} above0.30={case['above_030']} "
              f"same_class_IoU50={case['gt_same_class_iou50']}/{case['gt']}")
    print("DIAG01_VERIFY_PASS: saved predictions only; no training/inference/test")
    print(f"REPORT={folder / 'index.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", type=Path)
    args = parser.parse_args()
    expected = PROJECT / ".venv/Scripts/python.exe"
    require(Path(sys.executable).resolve() == expected.resolve(), f"请使用Faster专属解释器：{expected}")
    if args.verify_existing:
        verify_existing(args.verify_existing)
        return

    # 一次独立整理使用唯一目录，历史训练/预测目录只读。
    out = AUDIT / ("diag01_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    (out / "crops").mkdir()
    (out / "code").mkdir()
    print(f"RESULT_DIR={out}", flush=True)
    sources = {}

    def record(path):
        path = Path(path).resolve()
        sources[str(path)] = sha(path)
        return path

    acceptance = read(record(PROJECT / "workspace/audit/step11_review/acceptance.json"))
    require(acceptance["status"] == "PASS", "缺少训练验收")
    selected = read(record(RUN / "selected_checkpoint.json"))
    require(selected["selected_epoch"] == 5, "固定选中轮次变化")
    for kind in ("best_checkpoint", "last_checkpoint"):
        path = record(selected[kind])  # 只读文件计算哈希，不加载模型。
        require(sources[str(path)] == selected[kind + "_sha256"], "权重哈希变化")
    config = record(RUN / "resolved_scale1280_val_001.py")
    require(sources[str(config)] == acceptance["source_files_sha256"][str(config)], "冻结配置变化")
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1
    assert iou([0, 0, 10, 10], [10, 0, 10, 10]) == 0
    assert abs(iou([0, 0, 10, 10], [5, 0, 10, 10]) - 1 / 3) < 1e-12

    # 代码导读③：先在同一epoch100比较train/val，再单独解释已选epoch5。
    cases = [
        ("train8_epoch100", "第100轮 · 训练子集8图", "instances_debug8.json",
         "epoch_100/predictions_coco.json", "train8_final", 8, 15, 33),
        ("val23_epoch100", "第100轮 · 验证集23图", "instances_val.json",
         "validation/epoch_100/predictions.bbox.json", "final_val_metrics", 23, 37, 78),
        ("val23_epoch005", "第5轮 · 验证集23图（已固定最佳轮）", "instances_val.json",
         "validation/epoch_005/predictions.bbox.json", "selected_metrics", 23, 37, 2300),
    ]
    summaries, records, prediction_rows, gt_rows = [], [], [], []
    image_cache = {}
    for key, title, annotation, predfile, metric_key, ni, ng, npred in cases:
        coco = read(record(DATA / "annotations" / annotation))
        predictions = read(record(RUN / "artifacts" / predfile))
        images = {im["id"]: im for im in coco["images"]}
        require(len(images) == ni and len(coco["annotations"]) == ng and len(predictions) == npred,
                f"历史数量发生变化：{key}")
        require({c['id']: c['name'] for c in coco['categories']} == {1: 'broken', 2: 'loose'}, "类别映射变化")
        for p in predictions:
            check_prediction(p, images)
        case_gt, counts = [], []
        for iid, im in images.items():
            p_list = sorted([p for p in predictions if p["image_id"] == iid], key=lambda p: -p["score"])
            require(len(p_list) <= 100, "超过历史每图100框上限")
            counts.append(len(p_list))
            # rank为该图所有保存框的置信度降序名次，不因网页显示阈值改变。
            p_list = [dict(p, rank=i + 1) for i, p in enumerate(p_list)]
            g_list = [g for g in coco["annotations"] if g["image_id"] == iid]
            imagepath = DATA / "images" / im["file_name"]
            if iid not in image_cache:
                record(imagepath)
                with Image.open(imagepath) as src:
                    original = src.convert("RGB")
                require(original.size == (im["width"], im["height"]), "原图尺寸与标注不符")
                thumbnail = original.copy()
                thumbnail.thumbnail((1440, 1000))
                thumbnail.save(out / "images" / f"{iid}.jpg", quality=93)
                image_cache[iid] = str(out / "images" / f"{iid}.jpg")
            else:
                with Image.open(imagepath) as src:
                    original = src.convert("RGB")
            for p in p_list:
                prediction_rows.append({"case": key, "image_id": iid, "file_name": im["file_name"],
                                        "rank": p["rank"], "category_id": p["category_id"],
                                        "class_name": LABELS[p["category_id"]], "score": p["score"],
                                        **dict(zip(("x", "y", "width", "height"), p["bbox"]))})
            cards = []
            for gt in g_list:
                same, same_iou = best_overlap(gt, p_list)
                anyclass, any_iou = best_overlap(gt, p_list, same_class=False)
                high = [p for p in p_list if p["score"] > .30]
                _, high_iou = best_overlap(gt, high)
                state = ("有超过0.30的同类重合候选" if high_iou >= .5 else
                         "仅低分同类候选达到IoU0.50" if same_iou >= .5 else
                         "保存框中无IoU0.50同类候选")
                row = {"case": key, "image_id": iid, "file_name": im["file_name"], "gt_id": gt["id"],
                       "gt_class": LABELS[gt["category_id"]], "best_same_iou": same_iou,
                       "best_same_score": same["score"] if same else None,
                       "best_same_rank": same["rank"] if same else None,
                       "best_any_iou": any_iou,
                       "best_any_class": LABELS[anyclass["category_id"]] if anyclass else None,
                       "best_any_score": anyclass["score"] if anyclass else None,
                       "best_same_iou_above_030": high_iou, "finding": state}
                case_gt.append(row)
                gt_rows.append(row)
                cropname = f"crops/{key}_gt{gt['id']}.jpg"
                crop_panel(original, gt, same, same_iou, title, Path(im["file_name"]).name).save(out / cropname, quality=94)
                cards.append(dict(row, crop=cropname))
            records.append({"case": key, "image": im, "thumbnail": f"images/{iid}.jpg",
                            "predictions": p_list, "gt": g_list, "cards": cards})
        summary = {"key": key, "title": title, "images": ni, "gt": ng, "saved_predictions": len(predictions),
                   "above_030": sum(p["score"] > .30 for p in predictions),
                   "images_empty": counts.count(0), "images_at_100": counts.count(100),
                   "min_score": min(p["score"] for p in predictions), "max_score": max(p["score"] for p in predictions),
                   "gt_same_class_iou50": sum(r["best_same_iou"] >= .5 for r in case_gt),
                   "gt_same_class_iou50_above030": sum(r["best_same_iou_above_030"] >= .5 for r in case_gt),
                   "states": dict(Counter(r["finding"] for r in case_gt)),
                   "mAP50": acceptance[metric_key]["mAP50"],
                   "mAP50_95": acceptance[metric_key]["mAP50_95"],
                   "metrics_source": str(PROJECT / "workspace/audit/step11_review/acceptance.json")}
        summaries.append(summary)
        print(f"CASE_PASS {key}: {len(predictions)} saved boxes; {len(case_gt)} GT", flush=True)

    require(summaries[0]["gt_same_class_iou50"] == acceptance["train8_gt_with_same_class_candidate_iou_ge_0_5"], "训练几何复算不一致")
    require(summaries[2]["gt_same_class_iou50"] == acceptance["selected_val_gt_with_same_class_candidate_iou_ge_0_5"], "验证几何复算不一致")
    # GT123与另一条历史诊断独立比对；并非拿RPN当作本次最终预测。
    previous = record(PROJECT / "workspace/runs/FRCNN-DIAG-RPNROI-TRAIN8/step12_24867_6686/scale1280/per_gt.csv")
    with previous.open(encoding="utf-8-sig", newline="") as f:
        previous_rows = list(csv.DictReader(f))
    old123 = next(r for r in previous_rows if int(r["annotation_id"]) == 123)
    new123 = next(r for r in gt_rows if r["case"] == "train8_epoch100" and r["gt_id"] == 123)
    require(abs(new123["best_same_iou"] - float(old123["final_same_class_max_iou"])) < 1e-6, "GT123与旧诊断不一致")
    summary = {"status": "OFFLINE_GENERATION_PASS", "experiment": "FRCNN-DIAG-SAVED1280-001",
               "training_run": str(RUN), "cases": summaries,
               "detector_forward": False, "training": False, "test_accessed": False,
               "coverage_definition": "逐GT独立寻找同类最大IoU，不做一对一匹配；不是普通Recall或AP。",
               "metric_note": "AP引用已验收历史精确结果，不重新选checkpoint。所有新统计只使用保存预测。",
               "boundary": "保存结果已过score>0.001、NMS0.5、每图最多100；显示阈值0不能恢复之前丢弃的框。"}
    write(out / "summary.json", summary)
    write_csv(out / "all_saved_predictions_sorted.csv", prediction_rows)
    write_csv(out / "per_gt_localization.csv", gt_rows)
    write_csv(out / "case_summary.csv", [{k: v for k, v in s.items() if k != "states"} for s in summaries])
    payload = {"summary": summary, "records": records}
    write(out / "viewer_data.json", payload)
    template = record(Path(__file__).with_name("diag01_viewer.html"))
    html = template.read_text(encoding="utf-8").replace("__DIAG_DATA__", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    (out / "index.html").write_text(html, encoding="utf-8")
    for source in (Path(__file__), template):
        record(source)
        shutil.copy2(source, out / "code" / source.name)
    # 所有输入在分析结束后再校验，防止过程中读到了变化中的文件。
    for path, digest in sources.items():
        require(sha(path) == digest, f"分析期间输入变化：{path}")
    outputs = {str(p.resolve()): sha(p) for p in out.rglob("*") if p.is_file()}
    write(out / "manifest.json", {"created": datetime.now().astimezone().isoformat(),
          "python": sys.executable, "command": sys.argv, "sources_sha256": sources,
          "outputs_sha256": outputs, "status": "OFFLINE_GENERATION_PASS"})
    print("DIAG01_OFFLINE_PASS: 54 image-cases; 89 GT-crops; no model forward", flush=True)
    print(f"REPORT={out / 'index.html'}")
    verify_existing(out)


if __name__ == "__main__":
    main()
