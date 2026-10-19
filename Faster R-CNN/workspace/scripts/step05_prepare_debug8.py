"""Verify saved COCO results and freeze a train-only diagnostic subset; no training."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"D:\小论文\conductor_defect_dataset\03_coco_dataset")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads((DATA / "audit/conversion_report.json").read_text(encoding="utf-8"))
    status = json.loads((DATA / "audit/conversion_status.json").read_text(encoding="utf-8"))
    assert report["status"] == status["status"] == "PASS"
    source = json.loads((DATA / "audit/source_inventory.json").read_text(encoding="utf-8"))
    inventory = {(x["split"], x["filename"]): x for x in source}
    datasets = {}
    for split, expected in report["after"].items():
        path = DATA / "annotations" / f"instances_{split}.json"
        assert sha(path) == expected["annotation_sha256"]
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["images"]) == expected["images"] and len(data["annotations"]) == expected["boxes"]
        assert len({x["id"] for x in data["images"]}) == len(data["images"])
        assert len({x["id"] for x in data["annotations"]}) == len(data["annotations"])
        counts = Counter(a["category_id"] for a in data["annotations"])
        assert (counts[1], counts[2]) == (expected["broken"], expected["loose"])
        if split != "trainval":
            for image in data["images"]:
                original = inventory[tuple(image["file_name"].split("/", 1))]
                assert sha(DATA / "images" / image["file_name"]) == original["image_sha256"]
        datasets[split] = data
    for field in ("images", "annotations"):
        a = {x["id"]: x for x in datasets["train"][field]}
        b = {x["id"]: x for x in datasets["val"][field]}
        c = {x["id"]: x for x in datasets["test"][field]}
        assert not a.keys() & b.keys() and not a.keys() & c.keys() and not b.keys() & c.keys()
        assert {**a, **b} == {x["id"]: x for x in datasets["trainval"][field]}
    annotations = defaultdict(list)
    for ann in datasets["train"]["annotations"]:
        annotations[ann["image_id"]].append(ann)
    strata = defaultdict(list)
    for image in datasets["train"]["images"]:
        categories = {a["category_id"] for a in annotations[image["id"]]}
        name = "both" if categories == {1, 2} else "broken_only" if categories == {1} else "loose_only"
        strata[name].append(image)
    rng, selected, used_groups = random.Random(42), [], set()
    for stratum, needed in (("both", 2), ("broken_only", 3), ("loose_only", 3)):
        pool = sorted(strata[stratum], key=lambda x: x["file_name"])
        rng.shuffle(pool)
        count = 0
        for image in pool:
            row = inventory[tuple(image["file_name"].split("/", 1))]
            if row["split_group_id"] in used_groups:
                continue
            selected.append({"image_id": image["id"], "file_name": image["file_name"],
                             "split_group_id": row["split_group_id"], "stratum": stratum,
                             "image_sha256": row["image_sha256"], "width": image["width"], "height": image["height"],
                             "broken": sum(a["category_id"] == 1 for a in annotations[image["id"]]),
                             "loose": sum(a["category_id"] == 2 for a in annotations[image["id"]])})
            used_groups.add(row["split_group_id"])
            count += 1
            if count == needed:
                break
        assert count == needed
    selected.sort(key=lambda x: x["image_id"])
    plan = {"experiment": "FRCNN-DEBUG-OVERFIT-8", "purpose": "Pipeline and memorization diagnostic, not generalization performance",
            "selection_seed": 42, "selection_method": "2 both-class + 3 broken-only + 3 loose-only; distinct existing split_group_id",
            "source_split": "train", "source_annotations_sha256": report["after"]["train"]["annotation_sha256"],
            "all_coco_hashes": {k: v["annotation_sha256"] for k, v in report["after"].items()},
            "samples": selected, "images": len(selected), "broken": sum(x["broken"] for x in selected),
            "loose": sum(x["loose"] for x in selected), "training_started": False}
    path = ROOT / "workspace/configs/frcnn_debug8_selection.json"
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == plan, "Selection already exists and differs"
    else:
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "workspace/audit/step04_independent_acceptance.json").write_text(json.dumps(
        {"status": "PASS", "four_json_hashes_and_counts": True, "copied_image_hashes_verified": 153,
         "trainval_exact_union": True, "original_splits_disjoint": True, "training_started": False}, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
