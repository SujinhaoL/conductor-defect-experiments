"""Focused CPU-only smoke check of the new static checker. No training."""
import contextlib
import copy
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "workspace/audit/step05_preflight"
AUDIT.mkdir(exist_ok=True)
module = runpy.run_path(str(ROOT / "workspace/scripts/step05_check_debug8.py"))
plan, cfg, provenance = module["check_preparation"]()
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS, MODELS
from mmengine.dataset import pseudo_collate
import torch
import numpy as np
register_all_modules(init_default_scope=True)
train = json.loads((module["DATA"] / "annotations/instances_train.json").read_text(encoding="utf-8"))
selected = {x["image_id"] for x in plan["samples"]}
dataset_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
dataset_cfg.ann_file = "annotations/instances_train.json"
dataset_cfg.indices = [i for i, x in enumerate(train["images"]) if x["id"] in selected]
dataset = DATASETS.build(dataset_cfg)
assert len(dataset) == 8 and dataset.cat2label == {1: 0, 2: 1}
pair = [dataset[0], dataset[1]]
preprocessor = MODELS.build(cfg.model.data_preprocessor).cpu()
with torch.no_grad():
    result = preprocessor(pseudo_collate(copy.deepcopy(pair)), training=False)
for i, packed in enumerate(pair):
    sample = packed["data_samples"]
    boxes = sample.gt_instances.bboxes.tensor
    raw = np.array([x["bbox"] for x in dataset.get_data_info(i)["instances"]])
    sx, sy = sample.scale_factor
    assert np.abs(boxes.numpy() - raw * [sx, sy, sx, sy]).max() < 1e-3
    h, w = sample.img_shape
    rgb = result["inputs"][i] * preprocessor.std + preprocessor.mean
    assert (rgb[:, :h, :w] - packed["inputs"][[2, 1, 0]].float()).abs().max() < 1e-4
    assert torch.equal(result["data_samples"][i].gt_instances.bboxes, boxes)
    rgb_view = packed["inputs"][[2, 1, 0]].permute(1, 2, 0).numpy()
    module["draw"](rgb_view, boxes.tolist(), sample.gt_instances.labels.tolist(), AUDIT / f"resized_smoke_{i + 1}.jpg")
print("TWO_IMAGE_CPU_PIPELINE_SMOKE_PASS")
with (AUDIT / "weight_load_smoke.txt").open("w", encoding="utf-8") as log:
    with contextlib.redirect_stdout(log):
        weights = module["check_weights"](cfg, provenance, AUDIT)
assert weights["loaded_tensors"] == 344 and len(weights["expected_shape_changes"]) == 4
module["save_json"](AUDIT / "preparation_validation.json", {
    "status": "PASS", "pipeline_smoke_images": 2, "selected_images": 8,
    "matched_pretrained_tensors": 344, "expected_output_shape_changes": 4,
    "training_started": False, "detector_forward_performed": False,
    "debug_dataset_created": False, "scope": "CPU preparation check only; user stage will generate all eight image previews and permanent debug annotations."})
print("STEP05_PREPARATION_VALIDATION_PASS: no training, no detector forward")
