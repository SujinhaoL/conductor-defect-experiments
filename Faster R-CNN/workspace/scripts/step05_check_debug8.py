"""User-run 8-image transform and 2-class weight audit. Never starts training."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import runpy
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r"D:\小论文\conductor_defect_dataset\03_coco_dataset")
CONFIG = ROOT / "workspace/configs/frcnn_debug8_pipeline_check.py"
SELECTION = ROOT / "workspace/configs/frcnn_debug8_selection.json"
HELPERS = runpy.run_path(str(ROOT / "workspace/scripts/step04_convert_coco.py"))
require, sha, save_json, Tee = (HELPERS[n] for n in ("require", "sha", "save_json", "Tee"))
CLASSES = ("broken", "loose")


def check_preparation():
    from mmengine.config import Config
    runpy.run_path(str(ROOT / "workspace/scripts/step01_check_cuda.py"))["check_identity"]()
    plan = json.loads(SELECTION.read_text(encoding="utf-8"))
    for split, expected in plan["all_coco_hashes"].items():
        require(sha(DATA / "annotations" / f"instances_{split}.json") == expected, "Original COCO annotation file changed")
    require(len(plan["samples"]) == 8 and len({x["image_id"] for x in plan["samples"]}) == 8, "Expected eight unique images")
    for sample in plan["samples"]:
        require(sample["file_name"].startswith("train/"), "Debug subset must use only original train")
        require(sha(DATA / "images" / sample["file_name"]) == sample["image_sha256"], "Selected image changed")
    cfg = Config.fromfile(str(CONFIG))
    require(cfg.model.roi_head.bbox_head.num_classes == 2, "Expected two target classes")
    require(cfg.model.backbone.init_cfg is None and cfg.train_cfg is None, "Static-check config unexpectedly enables initialization download or training")
    require([x.type for x in cfg.train_pipeline] == ["LoadImageFromFile", "LoadAnnotations", "Resize", "PackDetInputs"],
            "Unexpected/random image transform")
    require(tuple(cfg.train_pipeline[2].scale) == (1333, 800) and cfg.train_pipeline[2].keep_ratio, "Resize settings changed")
    require(cfg.train_dataloader.batch_size == 2 and not cfg.train_dataloader.sampler.shuffle, "Unexpected batch/order")
    provenance = json.loads((ROOT / "workspace/configs/step03_preparation.json").read_text(encoding="utf-8"))
    require(sha(Path(cfg.load_from)) == provenance["checkpoint"]["sha256"], "Official checkpoint hash changed")
    print("STEP05_PREPARATION_CHECK_PASS", flush=True)
    return plan, cfg, provenance


def ensure_debug_annotations(plan):
    original = json.loads((DATA / "annotations/instances_train.json").read_text(encoding="utf-8"))
    ids = {s["image_id"] for s in plan["samples"]}
    subset = {"info": {"description": "FRCNN-DEBUG-OVERFIT-8; train-only pipeline/memorization diagnostic"},
              "licenses": original.get("licenses", []), "categories": original["categories"],
              "images": [x for x in original["images"] if x["id"] in ids],
              "annotations": [a for a in original["annotations"] if a["image_id"] in ids]}
    require([x["file_name"] for x in subset["images"]] == [x["file_name"] for x in plan["samples"]], "Subset selection differs")
    counts = Counter(a["category_id"] for a in subset["annotations"])
    require((counts[1], counts[2]) == (plan["broken"], plan["loose"]), "Subset class counts differ")
    path = DATA / "annotations/instances_debug8.json"
    if path.exists():
        require(json.loads(path.read_text(encoding="utf-8")) == subset, "Existing debug JSON differs; left untouched")
    else:
        save_json(path, subset, exclusive=True)
    print(f'DEBUG8_SUBSET_PASS: images=8 broken={counts[1]} loose={counts[2]} boxes={sum(counts.values())}', flush=True)
    return subset


def draw(rgb, boxes, labels, path, boundary=None):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(rgb)
    canvas = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", max(16, min(48, image.width // 60)))
    for box, label in zip(boxes, labels):
        color = ((240, 60, 60), (20, 210, 230))[int(label)]
        coords = [float(x) for x in box]
        canvas.rectangle(coords, outline=color, width=max(2, image.width // 600))
        text = CLASSES[int(label)]
        x, y = coords[:2]
        bounds = canvas.textbbox((0, 0), text, font=font)
        tw, th = bounds[2] + 6, bounds[3] + 6
        x, y = max(0, min(x, image.width - tw)), max(0, y - th)
        canvas.rectangle((x, y, x + tw, y + th), fill=color)
        canvas.text((x + 3, y + 1), text, font=font, fill=(0, 0, 0))
    if boundary:
        h, w = boundary
        canvas.rectangle((0, 0, w - 1, h - 1), outline=(255, 220, 30), width=2)
    image.save(path, quality=95)


def check_pipeline(cfg, subset, audit):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    from mmengine.dataset import pseudo_collate
    from mmdet.registry import DATASETS, MODELS

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    require(len(dataset) == 8 and dataset.cat2label == {1: 0, 2: 1}, "Dataset length/class mapping differs")
    batches, records, original_boxes = [], [], []
    for key in ("original", "resized", "padded"):
        (audit / key).mkdir()
    for index in range(8):
        raw = dataset.get_data_info(index)
        packed = dataset[index]
        again = dataset[index]
        sample = packed["data_samples"]
        tensor = getattr(sample.gt_instances.bboxes, "tensor", sample.gt_instances.bboxes)
        again_boxes = getattr(again["data_samples"].gt_instances.bboxes, "tensor", again["data_samples"].gt_instances.bboxes)
        require(torch.equal(packed["inputs"], again["inputs"]) and torch.equal(tensor, again_boxes), "Pipeline is not deterministic")
        labels = sample.gt_instances.labels
        require(len(tensor) == len(raw["instances"]), "Pipeline discarded boxes")
        require(labels.tolist() == [x["bbox_label"] for x in raw["instances"]], "Pipeline changed labels")
        require(len(sample.ignored_instances) == 0, "Unexpected ignored instances")
        sx, sy = sample.scale_factor
        xyxy = np.array([x["bbox"] for x in raw["instances"]], dtype=np.float64)
        expected = xyxy * np.array([sx, sy, sx, sy])
        error = float(np.max(np.abs(tensor.numpy() - expected)))
        require(error < 1e-3, "Resize coordinates differ or were clipped")
        h, w = sample.img_shape[:2]
        require(packed["inputs"].shape == (3, h, w), "Resized tensor shape differs")
        require(max(h, w) <= 1333 and min(h, w) <= 800, "Resize bounds violated")
        name = Path(raw["img_path"]).stem
        with Image.open(raw["img_path"]) as image:
            draw(np.array(image.convert("RGB")), xyxy, labels, audit / "original" / f"{name}.jpg")
        rgb = packed["inputs"][[2, 1, 0]].permute(1, 2, 0).numpy()
        draw(rgb, tensor.tolist(), labels.tolist(), audit / "resized" / f"{name}.jpg")
        records.append({"image_id": raw["img_id"], "name": name, "original_hw": list(sample.ori_shape),
                        "resized_hw": [h, w], "scale_factor_xy": [float(sx), float(sy)],
                        "boxes": len(tensor), "class_labels": labels.tolist(), "resize_max_error_px": error,
                        "smallest_resized_width": float((tensor[:, 2] - tensor[:, 0]).min()),
                        "smallest_resized_height": float((tensor[:, 3] - tensor[:, 1]).min())})
        batches.append(packed)
        original_boxes.append(tensor.clone())
        print(f'DEBUG8_IMAGE={index + 1}/8 {name}: {raw["width"]}x{raw["height"]} -> {w}x{h}; boxes={len(tensor)}', flush=True)
    preprocessor = MODELS.build(cfg.model.data_preprocessor).cpu()
    require(preprocessor.batch_augments is None, "Unexpected batch augmentation")
    for offset in range(0, 8, 2):
        pair = copy.deepcopy(batches[offset:offset + 2])
        # Pixel preprocessing only: there is no detector forward, loss or optimizer.
        with torch.no_grad():
            output = preprocessor(pseudo_collate(pair), training=False)
        pixels = output["inputs"]
        require(pixels.shape[0] == 2 and pixels.shape[-2] % 32 == 0 and pixels.shape[-1] % 32 == 0, "Invalid batch padding")
        for local in range(2):
            idx = offset + local
            row = records[idx]
            h, w = row["resized_hw"]
            normalized = pixels[local]
            rgb = normalized * preprocessor.std + preprocessor.mean
            original_rgb = batches[idx]["inputs"][[2, 1, 0]].float()
            error = float((rgb[:, :h, :w] - original_rgb).abs().max())
            require(error < 1e-4, "Color conversion/normalization failed round-trip")
            require(torch.count_nonzero(normalized[:, h:, :]).item() == 0 and torch.count_nonzero(normalized[:, :, w:]).item() == 0,
                    "Unexpected normalized padding values")
            tensor = output["data_samples"][local].gt_instances.bboxes
            require(torch.equal(tensor, original_boxes[idx]), "Preprocessor changed boxes")
            restored = rgb.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
            draw(restored, tensor.tolist(), row["class_labels"], audit / "padded" / f'{row["name"]}.jpg', boundary=(h, w))
            row.update(batch_number=offset // 2 + 1, padded_batch_hw=list(pixels.shape[-2:]),
                       denormalization_max_pixel_error=error, pad_normalized_value=0)
    # An overview is for navigation; full-resolution views are retained above.
    contact = Image.new("RGB", (1800, 8 * 350), (245, 245, 245))
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
    for index, row in enumerate(records):
        for col, stage in enumerate(("original", "resized", "padded")):
            with Image.open(audit / stage / f'{row["name"]}.jpg') as img:
                thumbnail = ImageOps.contain(img, (590, 315))
            contact.paste(thumbnail, (col * 600 + (600 - thumbnail.width) // 2, index * 350 + 30))
            ImageDraw.Draw(contact).text((col * 600 + 5, index * 350 + 5), f'{row["name"]} / {stage}', font=font, fill=(0, 0, 0))
    contact.save(audit / "pipeline_contact_sheet.jpg", quality=95)
    save_json(audit / "pipeline_report.json", {"status": "PASS", "images": records,
              "resize_scale_wh": [1333, 800], "keep_ratio": True, "random_image_augmentation": False,
              "batch_size": 2, "workers": 0, "batch_padding_divisor": 32,
              "mean_rgb": [123.675, 116.28, 103.53], "std_rgb": [58.395, 57.12, 57.375],
              "padding_preview": "Denormalized for viewing. Yellow rectangle marks resized-image boundary. Normalized padding 0 becomes mean RGB after inverse normalization.",
              "training_started": False})
    print("DEBUG8_RESIZE_BOXES_PASS", flush=True)
    print("DEBUG8_NORMALIZE_AND_PADDING_PASS", flush=True)
    return records


def check_weights(cfg, provenance, audit):
    import logging
    import torch
    from mmengine.logging import MMLogger
    from mmengine.runner import load_checkpoint, set_random_seed
    from mmdet.registry import MODELS
    logger = MMLogger.get_instance("step05_weights", log_file=str(audit / "model_initialization_and_load.log"))
    # Keep hundreds of routine initialization messages in the file. Warnings
    # remain visible in the terminal, including the four expected shape changes.
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.WARNING)
    expected_bad = {"roi_head.bbox_head.fc_cls.weight", "roi_head.bbox_head.fc_cls.bias",
                    "roi_head.bbox_head.fc_reg.weight", "roi_head.bbox_head.fc_reg.bias"}
    set_random_seed(42, deterministic=True)
    model = MODELS.build(cfg.model).cpu()
    model.init_weights()
    model.eval()
    checkpoint = torch.load(cfg.load_from, map_location="cpu", weights_only=True)
    state = checkpoint["state_dict"]
    before = model.state_dict()
    missing, unexpected = sorted(before.keys() - state.keys()), sorted(state.keys() - before.keys())
    bad = {key for key in before.keys() & state.keys() if before[key].shape != state[key].shape}
    require(not missing and not unexpected and bad == expected_bad, "Unexpected checkpoint key/shape mismatch")
    initial_head = {key: before[key].clone() for key in bad}
    shape_changes = {key: {"coco": list(state[key].shape), "two_class": list(before[key].shape)} for key in sorted(bad)}
    print("EXPECTED: exactly four COCO output tensors have different shapes after changing 80 to 2 classes.", flush=True)
    load_checkpoint(model, cfg.load_from, map_location="cpu", strict=False, logger=logger)
    after = model.state_dict()
    unequal = [key for key in state.keys() - bad if not torch.equal(after[key], state[key])]
    require(not unequal, "Some compatible pretrained tensors did not load")
    require(all(torch.equal(after[key], initial_head[key]) for key in bad), "Mismatched output layer was not left at its new initialization")
    counts = Counter(key.split(".")[0] for key in state.keys() - bad)
    require(sum(counts.values()) == 344, "Unexpected compatible tensor count")
    report = {"status": "PASS", "checkpoint": provenance["checkpoint"], "num_classes": 2,
              "missing_keys": missing, "unexpected_keys": unexpected, "expected_shape_changes": shape_changes,
              "loaded_tensors": sum(counts.values()), "loaded_tensors_by_module": dict(counts),
              "unequal_loaded_tensors": unequal, "new_output_layer_tensors": sorted(bad),
              "explanation": "fc_cls has 2 target classes plus background; fc_reg has 2*4 coordinates. Only these output weights/biases are newly initialized; backbone/FPN/RPN/shared ROI layers load COCO weights.",
              "random_seed": 42, "training_started": False, "detector_forward_performed": False}
    save_json(audit / "two_class_checkpoint_audit.json", report)
    print("TWO_CLASS_OFFICIAL_CHECKPOINT_LOAD_PASS: matched=344 expected_new_output_tensors=4", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-preparation", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args()
    if args.check_preparation:
        check_preparation()
        return 0
    require(args.log_dir is not None, "--log-dir required")
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / "workspace/logs"), "Unexpected log directory")
    logs.mkdir(parents=True, exist_ok=True)
    audit = DATA / "audit/debug8" / logs.name
    console = sys.stdout
    with (logs / "step05_console.txt").open("w", encoding="utf-8") as output:
        sys.stdout = Tee(console, output)
        try:
            from mmdet.utils import register_all_modules
            register_all_modules(init_default_scope=True)
            plan, cfg, provenance = check_preparation()
            require(not audit.exists(), "This audit directory already exists")
            audit.mkdir(parents=True)
            subset = ensure_debug_annotations(plan)
            save_json(audit / "selection.json", plan)
            cfg.dump(str(logs / "resolved_pipeline_config.py"))
            records = check_pipeline(cfg, subset, audit)
            weight_audit = check_weights(cfg, provenance, audit)
            check_preparation()
            report = {"status": "PASS", "experiment": "FRCNN-DEBUG-OVERFIT-8", "images": 8,
                      "broken": plan["broken"], "loose": plan["loose"], "boxes": sum(x["boxes"] for x in records),
                      "audit_dir": str(audit), "debug_annotation_sha256": sha(DATA / "annotations/instances_debug8.json"),
                      "selection_sha256": sha(SELECTION), "config_sha256": sha(CONFIG),
                      "loaded_weight_tensors": weight_audit["loaded_tensors"], "new_output_tensors": 4,
                      "training_started": False, "detector_forward_performed": False,
                      "note": "Static pretraining check only. No metrics, loss, optimizer, backward, or training checkpoint was produced."}
            save_json(logs / "step05_summary.json", report)
            save_json(audit / "step05_summary.json", report)
            print("AUDIT_DIR=" + str(audit), flush=True)
            print("STEP05_DEBUG8_STATIC_PASS", flush=True)
            return 0
        except BaseException:
            detail = traceback.format_exc()
            print(detail, flush=True)
            save_json(logs / "step05_failure.json", {"status": "FAIL", "traceback": detail, "training_started": False})
            print("STEP05_FAIL; send complete output and LOG_DIR. No training was started.", flush=True)
            return 1
        finally:
            sys.stdout = console


if __name__ == "__main__":
    raise SystemExit(main())
