"""Audit and convert the existing split to COCO. No model or training is run."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(r"D:\小论文\conductor_defect_dataset\02_yolo11_dataset")
DEST = Path(r"D:\小论文\conductor_defect_dataset\03_coco_dataset")
SPLITS = ("train", "val", "test")
CLASSES = ("broken", "loose")
EXPECTED = {"train": (107, 39, 129), "val": (23, 9, 28), "test": (23, 8, 28)}
PLAN = ROOT / "workspace/configs/step04_conversion_plan.json"
CATEGORIES = [{"id": 1, "name": "broken", "supercategory": "conductor_defect"},
              {"id": 2, "name": "loose", "supercategory": "conductor_defect"}]
MAPPING = [{"name": name, "yolo_id": i, "coco_id": i + 1, "model_label": i}
           for i, name in enumerate(CLASSES)]


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value, exclusive=False):
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def identity():
    runpy.run_path(str(ROOT / "workspace/scripts/step01_check_cuda.py"))["check_identity"]()
    require(SOURCE.resolve() != DEST.resolve() and DEST.parent == SOURCE.parent,
            "Unexpected source/destination")


def audit_source():
    import mmcv
    from PIL import Image
    import yaml
    issues, warnings, items = [], [], []
    files = {name: sha(SOURCE / name) for name in ("data.yaml", "split_manifest.csv", "dataset_report.txt")}
    config = yaml.safe_load((SOURCE / "data.yaml").read_text(encoding="utf-8-sig"))
    require(config["names"] == {0: "broken", 1: "loose"}, "Unexpected YOLO class mapping")
    require(Path(config["path"]).resolve() == SOURCE.resolve(), "YOLO YAML root mismatch")
    for split in SPLITS:
        require(config[split].replace("\\", "/") == f"images/{split}", "YOLO split path mismatch")
    with (SOURCE / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"split", "final_name", "image_id", "sha256", "width", "height", "split_group_id", "broken_boxes", "loose_boxes"}
    require(rows and required.issubset(rows[0]), "Missing required manifest columns")
    manifest = {}
    for row in rows:
        key = (row["split"], row["final_name"])
        if key in manifest:
            issues.append({"type": "duplicate_manifest_row", "key": key})
        if key[0] not in SPLITS or Path(key[1]).name != key[1] or not key[1]:
            issues.append({"type": "invalid_manifest_path", "key": key})
        if row.get("new_name") != key[1] or row.get("final_image_id") != Path(key[1]).stem:
            issues.append({"type": "manifest_name_disagreement", "key": key})
        manifest[key] = row
    present, groups, hashes = set(), defaultdict(list), defaultdict(list)
    for split in SPLITS:
        image_dir, label_dir = SOURCE / "images" / split, SOURCE / "labels" / split
        image_paths = sorted(image_dir.iterdir(), key=lambda p: p.name.lower())
        label_paths = sorted(label_dir.iterdir(), key=lambda p: p.name.lower())
        used_labels = set()
        for image_path in image_paths:
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                issues.append({"type": "unexpected_image_entry", "path": str(image_path)})
                continue
            key = (split, image_path.name)
            present.add(key)
            row = manifest.get(key)
            if row is None:
                issues.append({"type": "image_missing_in_manifest", "key": key})
                continue
            label = label_dir / (image_path.stem + ".txt")
            if not label.is_file() or label.name.lower() in used_labels:
                issues.append({"type": "missing_or_ambiguous_label", "path": str(label)})
                continue
            used_labels.add(label.name.lower())
            try:
                image_hash = sha(image_path)
                require(image_hash == row["sha256"], "Image SHA256 differs from manifest")
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()  # Decode all pixels; no EXIF transpose or source modification.
                    width, height = image.size
                    orientation = image.getexif().get(274)
                    mode = image.mode
                require((width, height) == (int(row["width"]), int(row["height"])), "Manifest/image dimensions differ")
                require(orientation in (None, 1), f"Nontrivial EXIF orientation {orientation}; manual decision needed")
                require(mode == "RGB", f"Image mode {mode} needs review")
                # MMDetection's default file loader uses MMCV/OpenCV. Check that
                # decoder's dimensions as well, including its EXIF behavior.
                decoded = mmcv.imfrombytes(image_path.read_bytes(), flag="color", backend="cv2")
                require(decoded is not None and decoded.shape[:2] == (height, width), "MMCV decoding/dimensions mismatch")
                del decoded
                boxes, counts = [], Counter()
                for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
                    if not line.strip():
                        warnings.append({"type": "blank_label_line", "path": str(label), "line": line_no})
                        continue
                    fields = line.split()
                    require(len(fields) == 5 and fields[0] in ("0", "1"), f"Invalid label format/class on line {line_no}")
                    values = [Decimal(token) for token in fields[1:]]
                    require(all(v.is_finite() for v in values), f"Non-finite box on line {line_no}")
                    xc, yc, bw, bh = values
                    require(0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1,
                            f"Invalid normalized coordinates/zero area on line {line_no}")
                    x1, y1, x2, y2 = xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2
                    require(x1 >= 0 and y1 >= 0 and x2 <= 1 and y2 <= 1,
                            f"Out-of-bounds box on line {line_no}: {[str(v) for v in (x1,y1,x2,y2)]}")
                    bbox = [float(x1 * width), float(y1 * height), float(bw * width), float(bh * height)]
                    require(bbox[2] >= 1 and bbox[3] >= 1,
                            f"Subpixel width/height on line {line_no}; MMDetection would discard it")
                    cls = int(fields[0])
                    boxes.append({"line": line_no, "yolo_class": cls, "normalized": [float(v) for v in values], "bbox": bbox})
                    counts[CLASSES[cls]] += 1
                require(all(counts[name] == int(row[name + "_boxes"]) for name in CLASSES), "Manifest/YOLO per-image class counts differ")
                item = {"split": split, "filename": image_path.name, "source_image_id": row["image_id"],
                        "image_path": str(image_path), "label_path": str(label), "image_sha256": image_hash,
                        "label_sha256": sha(label), "width": width, "height": height, "mode": mode,
                        "exif_orientation": orientation, "split_group_id": row["split_group_id"],
                        "scene_id": row.get("scene_id", ""), "boxes": boxes}
                items.append(item)
                hashes[image_hash].append(key)
                groups[row["split_group_id"]].append(key)
                if len(items) % 25 == 0:
                    print(f"AUDITED_IMAGES={len(items)}/{len(rows)}", flush=True)
            except Exception as error:
                issues.append({"type": "image_or_label_validation", "key": key, "error": str(error)})
        actual_labels = {p.name.lower() for p in label_paths if p.is_file()}
        if actual_labels != used_labels or any(not p.is_file() for p in label_paths):
            issues.append({"type": "orphan_or_unexpected_labels", "split": split,
                           "unused": sorted(actual_labels - used_labels)})
    if present != set(manifest):
        issues.append({"type": "manifest_file_set_mismatch", "missing_images": sorted(set(manifest) - present)})
    repeated_hashes = [value for value in hashes.values() if len(value) > 1]
    if repeated_hashes:
        issues.append({"type": "identical_image_files", "groups": repeated_hashes})
    for group, members in groups.items():
        if not group or len({split for split, _ in members}) > 1:
            issues.append({"type": "missing_or_cross_split_group", "group": group, "members": members})
    statistics = {}
    for split, expected in EXPECTED.items():
        selected = [x for x in items if x["split"] == split]
        counts = Counter(CLASSES[b["yolo_class"]] for x in selected for b in x["boxes"])
        statistics[split] = {"images": len(selected), "boxes": sum(counts.values()),
                             **{name: counts[name] for name in CLASSES}}
        if (len(selected), counts["broken"], counts["loose"]) != expected:
            issues.append({"type": "unexpected_split_counts", "split": split, "actual": statistics[split], "expected": expected})
    fingerprint_data = {"metadata": files, "files": [(x["split"], x["filename"], x["image_sha256"], x["label_sha256"]) for x in items]}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_data, sort_keys=True).encode("utf-8")).hexdigest()
    summary = {"status": "PASS" if not issues else "FAIL", "source": str(SOURCE), "destination": str(DEST),
               "source_fingerprint": fingerprint, "metadata_sha256": files, "statistics": statistics,
               "total_images": len(items), "total_boxes": sum(len(x["boxes"]) for x in items),
               "split_group_count": len(groups), "multi_image_groups": {k: v for k, v in groups.items() if len(v) > 1},
               "singleton_groups": sum(len(v) == 1 for v in groups.values()),
               "empty_scene_ids": sum(not x["scene_id"] for x in items),
               "exif_orientations": dict(Counter(str(x["exif_orientation"]) for x in items)),
               "min_box_width_px": min((b["bbox"][2] for x in items for b in x["boxes"]), default=None),
               "min_box_height_px": min((b["bbox"][3] for x in items for b in x["boxes"]), default=None),
               "issues": issues, "warnings": warnings, "class_mapping": MAPPING,
               "group_scope": "Reuse previously reviewed split_group_id. No inferred scene IDs, near-duplicate rescreening, or resplitting.",
               "training_started": False}
    return summary, items


def build_coco(items):
    images, annotations, provenance = [], [], []
    for image_id, item in enumerate(items, 1):
        images.append({"id": image_id, "file_name": f'{item["split"]}/{item["filename"]}',
                       "width": item["width"], "height": item["height"]})
        for box in item["boxes"]:
            ann_id = len(annotations) + 1
            x, y, w, h = box["bbox"]
            annotations.append({"id": ann_id, "image_id": image_id, "category_id": box["yolo_class"] + 1,
                                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
            reverse = [(x + w / 2) / item["width"], (y + h / 2) / item["height"],
                       w / item["width"], h / item["height"]]
            error = max(abs(a - b) for a, b in zip(reverse, box["normalized"]))
            require(error <= 1e-12, "YOLO/COCO coordinate round-trip mismatch")
            provenance.append({"coco_annotation_id": ann_id, "coco_image_id": image_id,
                               "split": item["split"], "image": item["filename"], "label_file": item["label_path"],
                               "label_line": box["line"], "yolo_class": box["yolo_class"], "coco_category_id": box["yolo_class"] + 1,
                               "model_label": box["yolo_class"], "x": x, "y": y, "width": w, "height": h,
                               "normalized_roundtrip_max_error": error})
    datasets = {}
    for name in (*SPLITS, "trainval"):
        included = {"train", "val"} if name == "trainval" else {name}
        subset = [image for image in images if image["file_name"].split("/")[0] in included]
        ids = {image["id"] for image in subset}
        datasets[name] = {"info": {"description": "Conductor defects; original image-level split preserved", "version": "1.0"},
                          "licenses": [], "images": subset,
                          "annotations": [ann for ann in annotations if ann["image_id"] in ids],
                          "categories": CATEGORIES}
    return datasets, provenance


def verify_snapshot(summary, items):
    for name, expected in summary["metadata_sha256"].items():
        require(sha(SOURCE / name) == expected, "Source metadata changed during processing")
    for item in items:
        require(sha(Path(item["image_path"])) == item["image_sha256"] and sha(Path(item["label_path"])) == item["label_sha256"],
                "Source file changed during processing: " + item["filename"])


def verify_coco(datasets, items):
    from mmdet.datasets import CocoDataset
    original = {(x["split"], x["filename"]): x for x in items}
    results = {}
    for split, expected_data in datasets.items():
        path = DEST / "annotations" / f"instances_{split}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        require(data == expected_data, "Serialized COCO content differs")
        images = {x["id"]: x for x in data["images"]}
        require(len(images) == len(data["images"]), "Duplicate image IDs")
        require(len({x["id"] for x in data["annotations"]}) == len(data["annotations"]), "Duplicate annotation IDs")
        require(len({x["file_name"].lower() for x in data["images"]}) == len(data["images"]), "Duplicate file names")
        dataset = CocoDataset(data_root=str(DEST), ann_file=str(path), data_prefix={"img": "images/"},
                              metainfo={"classes": CLASSES}, pipeline=[], test_mode=True, serialize_data=False)
        require(dataset.cat2label == {1: 0, 2: 1}, "Unexpected MMDetection class mapping")
        require(len(dataset) == len(images), "MMDetection image count differs")
        count, max_error = 0, 0.0
        for index in range(len(dataset)):
            loaded = dataset.get_data_info(index)
            info = images[loaded["img_id"]]
            src_split, filename = info["file_name"].split("/", 1)
            item = original[(src_split, filename)]
            require(Path(loaded["img_path"]).resolve() == (DEST / "images" / src_split / filename).resolve(), "MMDetection image path mismatch")
            require(len(loaded["instances"]) == len(item["boxes"]), "MMDetection silently dropped annotations")
            for instance, box in zip(loaded["instances"], item["boxes"]):
                x, y, w, h = box["bbox"]
                error = max(abs(a - b) for a, b in zip(instance["bbox"], (x, y, x + w, y + h)))
                max_error = max(max_error, error)
                require(error <= 1e-8 and instance["bbox_label"] == box["yolo_class"] and instance["ignore_flag"] == 0,
                        "MMDetection box/class/ignore mismatch")
                count += 1
        require(count == len(data["annotations"]), "MMDetection box total differs")
        counts = Counter(a["category_id"] for a in data["annotations"])
        results[split] = {"images": len(dataset), "boxes": count, "broken": counts[1], "loose": counts[2],
                          "max_mmdet_coordinate_error_px": max_error, "cat2label": dataset.cat2label,
                          "annotation_sha256": sha(path)}
        expected = (130, 48, 157) if split == "trainval" else EXPECTED[split]
        require((len(dataset), counts[1], counts[2]) == expected, "Converted split count mismatch")
    return results


def draw_annotations(datasets):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    colors = {1: (240, 60, 60), 2: (20, 210, 230)}
    font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    small_font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else ImageFont.load_default()
    drawn = 0
    for split in SPLITS:
        data = datasets[split]
        grouped = defaultdict(list)
        for ann in data["annotations"]:
            grouped[ann["image_id"]].append(ann)
        target = DEST / "audit/ground_truth" / split
        target.mkdir(parents=True)
        sheet = None
        for index, info in enumerate(data["images"]):
            if index % 16 == 0:
                sheet = Image.new("RGB", (1600, 1280), (245, 245, 245))
            with Image.open(DEST / "images" / info["file_name"]) as source:
                image = source.convert("RGB")
            drawer = ImageDraw.Draw(image)
            font_size = max(18, min(64, round(max(image.size) / 80)))
            font = ImageFont.truetype(str(font_path), font_size) if font_path.exists() else small_font
            line_width = max(3, round(max(image.size) / 1200))
            for ann in grouped[info["id"]]:
                x, y, w, h = ann["bbox"]
                color = colors[ann["category_id"]]
                drawer.rectangle((x, y, x + w, y + h), outline=color, width=line_width)
                text = f'{CLASSES[ann["category_id"] - 1]} #{ann["id"]}'
                text_box = drawer.textbbox((0, 0), text, font=font)
                tw, th = text_box[2] + 8, text_box[3] + 8
                tx, ty = max(0, min(x, image.width - tw)), max(0, y - th)
                drawer.rectangle((tx, ty, tx + tw, ty + th), fill=color)
                drawer.text((tx + 4, ty + 2), text, font=font, fill=(0, 0, 0))
            image.save(target / (Path(info["file_name"]).stem + ".jpg"), quality=92)
            thumbnail = ImageOps.contain(image, (392, 284))
            col, row = index % 4, (index % 16) // 4
            sheet.paste(thumbnail, (col * 400 + (400 - thumbnail.width) // 2, row * 320 + 28 + (284 - thumbnail.height) // 2))
            ImageDraw.Draw(sheet).text((col * 400 + 5, row * 320 + 3), f'{split} / {Path(info["file_name"]).stem}', font=small_font, fill=(0, 0, 0))
            drawn += 1
            if index % 16 == 15 or index == len(data["images"]) - 1:
                sheet.save(DEST / "audit/contact_sheets" / f"{split}_{index // 16 + 1:02d}.jpg", quality=90)
            if drawn % 25 == 0:
                print(f"GROUND_TRUTH_OVERLAYS={drawn}/153", flush=True)
    return drawn


def convert(summary, items, logs):
    require(PLAN.is_file(), "Conversion plan missing; request preparation first")
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    require(plan["source_fingerprint"] == summary["source_fingerprint"], "Source changed since preparation; inspect before proceeding")
    require(not DEST.exists() or (DEST.is_dir() and not any(DEST.iterdir())),
            "Destination is not empty. Stop for inspection; existing files will not be overwritten.")
    datasets, provenance = build_coco(items)
    DEST.mkdir(exist_ok=True)
    for directory in ("annotations", "audit", "audit/contact_sheets", "images/train", "images/val", "images/test"):
        (DEST / directory).mkdir(parents=True, exist_ok=True)
    save_json(DEST / "audit/conversion_status.json", {"status": "RUNNING", "log_dir": str(logs)})
    save_json(DEST / "audit/source_audit.json", summary)
    save_json(DEST / "audit/source_inventory.json", items)
    save_json(DEST / "audit/class_mapping.json", {"num_classes": 2, "background_is_target_class": False, "mapping": MAPPING})
    for index, item in enumerate(items, 1):
        destination = DEST / "images" / item["split"] / item["filename"]
        with Path(item["image_path"]).open("rb") as source, destination.open("xb") as output:
            shutil.copyfileobj(source, output, 1024 * 1024)
        require(sha(destination) == item["image_sha256"], "Copied image hash differs")
        if index % 25 == 0:
            print(f"COPIED_IMAGES_HASH_VERIFIED={index}/153", flush=True)
    for split, data in datasets.items():
        save_json(DEST / "annotations" / f"instances_{split}.json", data, exclusive=True)
    write_csv(DEST / "audit/annotation_mapping.csv", provenance)
    readback = verify_coco(datasets, items)
    print("COCO_AND_MMDETECTION_READBACK_PASS", flush=True)
    overlays = draw_annotations(datasets)
    verify_snapshot(summary, items)
    report = {"status": "PASS", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": str(SOURCE), "destination": str(DEST), "source_fingerprint": summary["source_fingerprint"],
              "original_split_preserved": True, "source_files_unchanged": True, "copied_images_sha256_match": True,
              "class_mapping": MAPPING, "before": summary["statistics"], "after": readback,
              "max_normalized_roundtrip_error": max(x["normalized_roundtrip_max_error"] for x in provenance),
              "ground_truth_overlays": overlays, "annotations_dropped_clipped_or_repaired": 0,
              "trainval_copies_created": 0, "training_started": False,
              "converter_sha256": sha(Path(__file__)), "log_dir": str(logs)}
    save_json(DEST / "audit/conversion_report.json", report)
    save_json(logs / "conversion_report.json", report)
    text = ["YOLO to COCO conversion: PASS", "No training performed.",
            "Class mapping: YOLO 0 -> COCO 1 -> model label 0 = broken",
            "Class mapping: YOLO 1 -> COCO 2 -> model label 1 = loose", "Model num_classes=2; background is not a third target class.",
            "Use data_root=03_coco_dataset and data_prefix=dict(img='images/') for every split.",
            "COCO file_name includes train/, val/ or test/. trainval references train/ and val/ without copying again.",
            "Source images and labels are unchanged. Copied images are byte-identical.",
            "Ground-truth overlays are separate JPEG previews at original pixel dimensions; source image copies are unchanged.",
            "Existing split_group_id reused; scene_id remains unknown. No new scene attribution or repartitioning."]
    text += [f"{name}: images={r['images']}, boxes={r['boxes']}, broken={r['broken']}, loose={r['loose']}" for name, r in readback.items()]
    (DEST / "audit/conversion_report.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    save_json(DEST / "audit/conversion_status.json", {"status": "PASS", "log_dir": str(logs)})
    for name, result in readback.items():
        print(f"{name}: images={result['images']} boxes={result['boxes']} broken={result['broken']} loose={result['loose']}", flush=True)


class Tee:
    def __init__(self, console, file):
        self.console, self.file = console, file

    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / "workspace"), "Logs must stay inside this workspace")
    logs.mkdir(parents=True, exist_ok=True)
    console = sys.stdout
    with (logs / "step04_console.txt").open("w", encoding="utf-8") as output:
        sys.stdout = Tee(console, output)
        try:
            identity()
            print("Auditing original split, full image decoding, EXIF and every YOLO box.", flush=True)
            summary, items = audit_source()
            save_json(logs / "source_audit.json", summary)
            save_json(logs / "source_inventory.json", items)
            require(summary["status"] == "PASS", "Source audit failed; see source_audit.json. No conversion was performed.")
            print("SOURCE_AUDIT_PASS", flush=True)
            # Test numerical conversion in memory in both modes, without creating a dataset elsewhere.
            _, provenance = build_coco(items)
            print("COORDINATE_ROUNDTRIP_PASS", flush=True)
            if args.audit_only:
                print(json.dumps(summary["statistics"], ensure_ascii=False), flush=True)
                print("SOURCE_FINGERPRINT=" + summary["source_fingerprint"], flush=True)
                print("STEP04_AUDIT_ONLY_PASS; destination not created or modified.", flush=True)
            else:
                convert(summary, items, logs)
                print("STEP04_COCO_CONVERSION_PASS", flush=True)
            return 0
        except BaseException:
            detail = traceback.format_exc()
            print(detail, flush=True)
            save_json(logs / "step04_failure.json", {"status": "FAIL", "traceback": detail, "training_started": False})
            # Preserve all files on failure; do not delete, repair or rerun automatically.
            status = DEST / "audit/conversion_status.json"
            if not args.audit_only and status.is_file():
                previous = json.loads(status.read_text(encoding="utf-8"))
                if previous.get("status") == "RUNNING" and previous.get("log_dir") == str(logs):
                    save_json(status, {"status": "FAIL", "log_dir": str(logs)})
            print("STEP04_FAIL; send terminal output and LOG_DIR. No training was started.", flush=True)
            return 1
        finally:
            sys.stdout = console


if __name__ == "__main__":
    raise SystemExit(main())
