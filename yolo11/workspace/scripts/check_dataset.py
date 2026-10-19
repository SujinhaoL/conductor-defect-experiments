"""Validate all splits without changing or repairing any source dataset file."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

from readonly_dataset import DATASET, PROTECTED_ROOT, WORKSPACE, install_write_guard


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot():
    result = {}
    for path in sorted(PROTECTED_ROOT.rglob("*")):
        stat = path.stat()
        entry = {"kind": "dir" if path.is_dir() else "file", "mtime_ns": stat.st_mtime_ns}
        if path.is_file():
            entry["size"] = stat.st_size
            if path.is_relative_to(DATASET):
                entry["sha256"] = sha256(path)
        result[path.relative_to(PROTECTED_ROOT).as_posix()] = entry
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-unchanged", action="store_true")
    args = parser.parse_args()
    install_write_guard()
    logdir = WORKSPACE / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    before_path = logdir / "dataset_snapshot_before.json"
    if args.verify_unchanged:
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = snapshot()
        changes = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
        report = {"unchanged": not changes, "changed_paths": changes,
                  "entries_checked": len(after), "sha256_scope": str(DATASET),
                  "metadata_scope": str(PROTECTED_ROOT)}
        (logdir / "dataset_integrity_after.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(bool(changes))

    from PIL import Image
    import yaml

    before = snapshot()
    if before_path.exists():
        raise RuntimeError("Initial dataset snapshot already exists; refusing to replace it.")
    before_path.write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    errors, warnings, stats = [], [], {}
    original_yaml = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8-sig"))
    expected_names = {0: "broken", 1: "loose"}
    if original_yaml.get("names") != expected_names:
        errors.append({"path": str(DATASET / "data.yaml"), "reason": "Expected names {0: broken, 1: loose}"})
    if Path(original_yaml.get("path", "")).resolve() != DATASET.resolve():
        warnings.append("Original data.yaml path differs; the project configuration must be used.")
    for split in ("train", "val", "test"):
        if original_yaml.get(split) != f"images/{split}":
            warnings.append(f"Original YAML {split} differs from images/{split}.")

    with (DATASET / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    manifest_map = {(r["split"], r["final_name"]): r for r in manifest}
    if len(manifest_map) != len(manifest):
        errors.append({"reason": "Duplicate manifest split/name entries"})
    groups, hashes, actual_keys = defaultdict(set), defaultdict(set), set()
    for row in manifest:
        groups[row["split_group_id"]].add(row["split"])
    for group, splits in groups.items():
        if len(splits) > 1:
            errors.append({"reason": f"Split group leakage: {group}: {sorted(splits)}"})

    for split in ("train", "val", "test"):
        image_dir, label_dir = DATASET / "images" / split, DATASET / "labels" / split
        for directory in (image_dir, label_dir):
            if not directory.is_dir():
                errors.append({"path": str(directory), "reason": "Missing required directory"})
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
        labels = sorted(label_dir.rglob("*.txt"))
        counts, positive_images = Counter(), Counter()
        empty_labels = 0
        expected_labels = set()
        for image_path in images:
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            expected_labels.add(label_path)
            context = {"image": str(image_path), "label": str(label_path)}
            try:
                with Image.open(image_path) as image:
                    size, fmt = image.size, image.format
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
                if min(size) < 10:
                    raise ValueError(f"Image too small: {size}")
                if fmt == "JPEG":
                    with image_path.open("rb") as stream:
                        stream.seek(-2, 2)
                        if stream.read() != b"\xff\xd9":
                            raise ValueError("Missing JPEG end marker; automatic repair is forbidden")
                digest = before[image_path.relative_to(PROTECTED_ROOT).as_posix()]["sha256"]
                hashes[digest].add(split)
                key = (split, image_path.name)
                actual_keys.add(key)
                row = manifest_map.get(key)
                if row is None or row["sha256"] != digest:
                    raise ValueError("Image missing from manifest or SHA256 mismatch")
                if size != (int(row["width"]), int(row["height"])):
                    raise ValueError("Image dimensions disagree with manifest")
                rows = [line.split() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
                if not rows:
                    empty_labels += 1
                    raise ValueError("Empty label: unexpected pure negative image for this dataset")
                seen, image_counts = set(), Counter()
                for number, tokens in enumerate(rows, 1):
                    if len(tokens) != 5:
                        raise ValueError(f"Line {number}: detection label must have 5 columns")
                    cls, x, y, width, height = map(float, tokens)
                    if not all(math.isfinite(v) for v in (cls, x, y, width, height)):
                        raise ValueError(f"Line {number}: non-finite value")
                    if cls not in (0., 1.):
                        raise ValueError(f"Line {number}: invalid class {cls}")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                        raise ValueError(f"Line {number}: invalid normalized coordinates")
                    if min(x-width/2, y-height/2) < -1e-6 or max(x+width/2, y+height/2) > 1+1e-6:
                        raise ValueError(f"Line {number}: box extends outside image")
                    values = (cls, x, y, width, height)
                    if values in seen:
                        raise ValueError(f"Line {number}: duplicate label")
                    seen.add(values)
                    image_counts[int(cls)] += 1
                if (image_counts[0], image_counts[1]) != (int(row["broken_boxes"]), int(row["loose_boxes"])):
                    raise ValueError("Label class counts disagree with manifest")
                counts.update(image_counts)
                positive_images.update(image_counts.keys())
            except Exception as exc:
                errors.append({**context, "reason": f"{type(exc).__name__}: {exc}"})
        for orphan in set(labels) - expected_labels:
            errors.append({"label": str(orphan), "reason": "Orphan label"})
        stats[split] = {"images": len(images), "labels": len(labels), "boxes": dict(counts),
                        "positive_images": dict(positive_images), "empty_labels": empty_labels}
        print(f"{split}: {stats[split]}", flush=True)
    if actual_keys != set(manifest_map):
        errors.append({"reason": "Manifest and image file sets differ"})
    for digest, splits in hashes.items():
        if len(splits) > 1:
            errors.append({"reason": f"Identical image hash appears across splits: {digest}"})
    result = {"passed": not errors, "dataset": str(DATASET), "names": expected_names,
              "original_yaml": original_yaml, "splits": stats, "manifest_rows": len(manifest),
              "errors": errors, "warnings": warnings,
              "checks": ["image verify and decode", "JPEG end markers", "label classes and box bounds",
                         "image/label pairing", "manifest hashes, dimensions and box counts",
                         "no cross-split identical images", "no cross-split groups"]}
    (logdir / "dataset_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
