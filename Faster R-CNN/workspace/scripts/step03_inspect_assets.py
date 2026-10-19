"""Preparation-only read checks. Does not instantiate or run a detector."""
from pathlib import Path
import json
import torch
from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
weight = ROOT / "workspace/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
checkpoint = torch.load(weight, map_location="cpu", weights_only=True)
state = checkpoint["state_dict"]
cfg = Config.fromfile(str(ROOT / "workspace/configs/step03_official_coco_demo.py"))
report = {"weights_only_cpu_read": "PASS", "num_state_tensors": len(state),
          "top_level_keys": list(checkpoint), "meta_keys": list(checkpoint.get("meta", {})),
          "state_modules": sorted({key.split(".")[0] for key in state}),
          "output_layers": {k: list(v.shape) for k, v in state.items() if ".fc_cls." in k or ".fc_reg." in k},
          "config_classes": cfg.model.roi_head.bbox_head.num_classes,
          "test_pipeline": [dict(stage) for stage in cfg.test_dataloader.dataset.pipeline],
          "test_cfg": dict(cfg.model.test_cfg), "training_started": False,
          "note": "File and config inspection only. No model created, loaded or executed."}
(ROOT / "workspace/audit/step03_assets_read_check.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
