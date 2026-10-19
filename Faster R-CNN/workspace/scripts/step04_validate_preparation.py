"""Check conversion math and MMDetection parsing in memory; no dataset is written."""
import json
from pathlib import Path
import runpy
from collections import Counter
from mmdet.datasets import CocoDataset

ROOT = Path(__file__).resolve().parents[2]
module = runpy.run_path(str(ROOT / "workspace/scripts/step04_convert_coco.py"))
summary = json.loads((ROOT / "workspace/audit/step04_preflight/source_audit.json").read_text(encoding="utf-8"))
items = json.loads((ROOT / "workspace/audit/step04_preflight/source_inventory.json").read_text(encoding="utf-8"))
plan = json.loads((ROOT / "workspace/configs/step04_conversion_plan.json").read_text(encoding="utf-8"))
assert summary["status"] == "PASS" and summary["source_fingerprint"] == plan["source_fingerprint"]
datasets, provenance = module["build_coco"](items)
parser = CocoDataset(metainfo={"classes": ("broken", "loose")}, pipeline=[], lazy_init=True,
                     data_prefix={"img": str(module["DEST"] / "images")}, serialize_data=False)
parser.cat_ids, parser.cat2label = [1, 2], {1: 0, 2: 1}
source = {(x["split"], x["filename"]): x for x in items}
ids = {}
for split, data in datasets.items():
    # Exercise JSON numeric round-trip as well as real MMDetection bbox parsing.
    loaded = json.loads(json.dumps(data, allow_nan=False))
    assert loaded == data
    assert len({x["id"] for x in data["images"]}) == len(data["images"])
    assert len({a["id"] for a in data["annotations"]}) == len(data["annotations"])
    ids[split] = {x["id"] for x in data["images"]}
    count = Counter()
    for image in loaded["images"]:
        row = dict(image, img_id=image["id"])
        annotations = [a for a in loaded["annotations"] if a["image_id"] == image["id"]]
        parsed = parser.parse_data_info({"raw_img_info": row, "raw_ann_info": annotations})
        src = source[tuple(image["file_name"].split("/", 1))]
        assert len(parsed["instances"]) == len(src["boxes"])
        for result, original in zip(parsed["instances"], src["boxes"]):
            x, y, w, h = original["bbox"]
            assert result["bbox"] == [x, y, x + w, y + h]
            assert result["bbox_label"] == original["yolo_class"] and result["ignore_flag"] == 0
            count[("broken", "loose")[result["bbox_label"]]] += 1
    expected = plan["expected"][split]
    assert (len(loaded["images"]), sum(count.values()), count["broken"], count["loose"]) == (
        expected["images"], expected["boxes"], expected["broken"], expected["loose"])
    print(f"IN_MEMORY_MMDET_PARSE_PASS={split}: {len(loaded['images'])} images, {sum(count.values())} boxes")
assert not ids["train"] & ids["val"] and not ids["train"] & ids["test"] and not ids["val"] & ids["test"]
assert ids["trainval"] == ids["train"] | ids["val"]
assert not ids["trainval"] & ids["test"]
report = {"status": "PASS", "source_fingerprint": summary["source_fingerprint"],
          "source_full_decode_audit_pass": True, "in_memory_coco_and_mmdet_parser_pass": True,
          "disjoint_split_ids_pass": True, "trainval_union_pass": True,
          "max_normalized_roundtrip_error": max(x["normalized_roundtrip_max_error"] for x in provenance),
          "destination_created": False, "training_started": False,
          "scope": "Source audit plus in-memory conversion and official MMDetection parsing. Full saved-file readback and rendering occur in the user-run conversion."}
module["save_json"](ROOT / "workspace/audit/step04_preflight/preparation_validation.json", report)
print("STEP04_PREPARATION_VALIDATION_PASS")
