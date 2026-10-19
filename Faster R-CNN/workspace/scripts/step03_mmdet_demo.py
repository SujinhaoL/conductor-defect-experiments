"""User-run MMDetection installation and one official-image inference; no training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "mmdetection"
COMMIT = "44ebd17b145c2372c4b700bfb9cb20dbd28ab64a"
MANIFEST = ROOT / "workspace/configs/step03_preparation.json"
LOCK = ROOT / "workspace/configs/step03_requirements.lock.txt"
BASE = ROOT / "workspace/configs/step03_base_constraints.txt"
CONFIG = ROOT / "workspace/configs/step03_official_coco_demo.py"
VIS_THRESHOLD = 0.3
HELPERS = runpy.run_path(str(ROOT / "workspace/scripts/step02c_build_mmcv.py"))
require, digest, run = (HELPERS[n] for n in ("require", "digest", "run"))
GIT = ["git", "-c", "safe.directory=" + SOURCE.as_posix()]


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def check_base():
    runpy.run_path(str(ROOT / "workspace/scripts/step01_check_cuda.py"))["check_identity"]()
    for line in BASE.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", 1)
            require(md.version(name) == version, "Previously accepted package changed: " + name)
    print("PREVIOUS_ENVIRONMENT_VERSIONS_PASS", flush=True)


def check_preparation():
    check_base()
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(data["mmdet_commit"] == COMMIT, "Manifest commit changed")
    head = subprocess.check_output(GIT + ["rev-parse", "HEAD"], cwd=SOURCE, text=True).strip()
    require(head == COMMIT, "Official source commit changed")
    require(not subprocess.check_output(GIT + ["diff", "HEAD", "--name-only"], cwd=SOURCE),
            "Official tracked source has changes")
    for path, expected in [(LOCK, data["lock_sha256"]), (BASE, data["base_constraints_sha256"]),
                           (Path(data["config"]), data["config_sha256"]),
                           (Path(data["demo_image"]), data["demo_image_sha256"]),
                           (Path(data["checkpoint"]["path"]), data["checkpoint"]["sha256"])]:
        require(digest(path) == expected, "Hash mismatch: " + str(path))
    for record in data["dependency_wheels"]:
        filename = record["filename"]
        require(Path(filename).name == filename, "Invalid wheel filename")
        require(digest(ROOT / "workspace/wheels/step03_dependencies" / filename) == record["sha256"],
                "Dependency hash mismatch: " + filename)
    print("STEP03_PREPARATION_CHECK_PASS", flush=True)
    return data


def demo(logs):
    # Heavy imports happen only in the inference subprocess, never in the preparation check.
    import numpy as np
    from PIL import Image
    import torch
    import mmcv
    import mmengine
    import mmdet
    from mmengine.config import Config
    from mmdet.apis import init_detector, inference_detector
    from mmdet.visualization import DetLocalVisualizer

    data = check_preparation()
    require(mmdet.__version__ == "3.3.0", "Wrong MMDetection version")
    require(Path(mmdet.__file__).resolve().is_relative_to(SOURCE), "MMDetection imported from another source")
    require(torch.cuda.is_available(), "CUDA unavailable")
    checkpoint_path = Path(data["checkpoint"]["path"])
    # No unsafe pickle fallback. Confirm this official checkpoint loads in weights-only mode.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = {re.sub(r"^module\.", "", k): v for k, v in checkpoint["state_dict"].items()}
    print("[3/4] Loading the complete official 80-class COCO detector checkpoint.", flush=True)
    model = init_detector(str(CONFIG), str(checkpoint_path), device="cpu", palette="coco")
    require(not model.training and model.roi_head.bbox_head.num_classes == 80,
            "Expected an evaluation-mode 80-class COCO model")
    # Audit exact post-load tensors, not just a reassuring loader message.
    actual = model.state_dict()
    missing = sorted(set(actual) - set(state))
    unexpected = sorted(set(state) - set(actual))
    shape_mismatch = [k for k in state.keys() & actual.keys() if state[k].shape != actual[k].shape]
    unequal = [k for k in state.keys() & actual.keys() if k not in shape_mismatch
               and not torch.equal(state[k], actual[k])]
    batch_counters = [k for k in missing if k.endswith(".num_batches_tracked")]
    substantive_missing = sorted(set(missing) - set(batch_counters))
    counts = Counter(k.split(".")[0] for k in state.keys() & actual.keys())
    audit = {"checkpoint": data["checkpoint"], "model_num_classes": 80,
             "matched_state_tensors_by_module": dict(counts), "missing_keys": missing,
             "missing_legacy_batchnorm_counters": batch_counters, "unexpected_keys": unexpected,
             "shape_mismatch_keys": shape_mismatch, "unequal_loaded_tensor_keys": unequal,
             "note": "Legacy BatchNorm num_batches_tracked buffers may be absent; they are not learned detector weights.",
             "meta_keys": sorted(checkpoint.get("meta", {}).keys()),
             "class_names": list(model.dataset_meta["classes"])}
    write_json(logs / "checkpoint_load_audit.json", audit)
    require(not substantive_missing and not unexpected and not shape_mismatch and not unequal,
            "Checkpoint did not fully match; inspect checkpoint_load_audit.json")
    require(all(counts.get(name, 0) > 0 for name in ("backbone", "neck", "rpn_head", "roi_head")),
            "Checkpoint must cover backbone, FPN, RPN and ROI head")
    print("OFFICIAL_COCO_FULL_CHECKPOINT_LOAD_PASS", flush=True)
    print("MATCHED_TENSORS_BY_MODULE=" + json.dumps(dict(counts)), flush=True)
    del checkpoint, state, actual
    Config.fromfile(data["config"]).dump(str(logs / "official_config_resolved.py"))
    model.cfg.dump(str(logs / "inference_config_resolved.py"))
    with Image.open(data["demo_image"]) as source_image:
        source_image.load()
        rgb = np.array(source_image.convert("RGB"))
    # Passing an array avoids Windows OpenCV imread limitations with Chinese paths.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    model.to("cuda:0")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    print("[4/4] Running official demo image on CUDA; no training.", flush=True)
    prediction = inference_detector(model, bgr)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - started
    pred = prediction.pred_instances.cpu()
    require(torch.isfinite(pred.bboxes).all().item() and torch.isfinite(pred.scores).all().item(),
            "Prediction has non-finite values")
    require(len(pred) > 0, "No returned detections on the official demo image; inspect inference log")
    classes = list(model.dataset_meta["classes"])
    records = [{"bbox_xyxy": box, "label_index": int(label), "class_name": classes[int(label)],
                "score": float(score)} for box, label, score in
               zip(pred.bboxes.tolist(), pred.labels.tolist(), pred.scores.tolist())]
    write_json(logs / "demo_predictions.json", {"image": data["demo_image"],
               "coordinate_space": "original image pixels, xyxy", "predictions": records,
               "note": "All model-returned predictions after score filtering, NMS and max_per_img; visualization threshold is not applied to this JSON."})
    visualizer = DetLocalVisualizer(name="step03_official_coco")
    visualizer.dataset_meta = model.dataset_meta
    visualizer.add_datasample("official_demo", rgb, prediction.cpu(), draw_gt=False,
                              show=False, pred_score_thr=VIS_THRESHOLD)
    visualization = logs / "demo_prediction.jpg"
    Image.fromarray(visualizer.get_image()).save(visualization, quality=95)
    summary = {"status": "PASS", "training_started": False, "conductor_dataset_accessed": False,
               "mmdet_version": mmdet.__version__, "mmdet_commit": COMMIT,
               "python": sys.version, "python_executable": sys.executable,
               "torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "mmcv": mmcv.__version__, "mmengine": mmengine.__version__,
               "gpu": torch.cuda.get_device_name(0), "gpu_capability": torch.cuda.get_device_capability(0),
               "checkpoint_sha256": data["checkpoint"]["sha256"], "num_classes": 80,
               "original_hw": list(rgb.shape[:2]), "resized_hw": list(prediction.img_shape),
               "scale_factor": list(prediction.scale_factor), "resize_scale_wh": [1333, 800],
               "keep_ratio": True, "model_score_threshold": float(model.test_cfg.rcnn.score_thr),
               "nms_iou_threshold": float(model.test_cfg.rcnn.nms.iou_threshold),
               "max_per_img": int(model.test_cfg.rcnn.max_per_img),
               "visualization_score_threshold": VIS_THRESHOLD, "returned_predictions": len(records),
               "visible_predictions": sum(r["score"] > VIS_THRESHOLD for r in records),
               "ap_evaluation_performed": False, "single_cold_inference_seconds": round(elapsed, 3),
               "peak_allocated_cuda_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
               "visualization": str(visualization)}
    write_json(logs / "demo_summary.json", summary)
    print("CUDA_OFFICIAL_IMAGE_INFERENCE_PASS", flush=True)
    print(f'RETURNED_PREDICTIONS={summary["returned_predictions"]}', flush=True)
    print(f'VISIBLE_PREDICTIONS_ABOVE_0.3={summary["visible_predictions"]}', flush=True)
    print("VISUALIZATION=" + str(visualization), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--check-preparation", action="store_true")
    parser.add_argument("--demo-only", action="store_true")
    args = parser.parse_args()
    if args.check_preparation:
        check_preparation()
        return 0
    require(args.log_dir is not None, "--log-dir required")
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / "workspace/logs"), "Logs must stay in this workspace")
    logs.mkdir(parents=True, exist_ok=True)
    if args.demo_only:
        demo(logs)
        return 0
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL",
              "training_started": False, "mmdet_commit": COMMIT}
    try:
        data = check_preparation()
        env = os.environ.copy()
        for key in list(env):
            if key.upper() in {"PYTHONHOME", "PYTHONPATH", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}:
                del env[key]
        env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONNOUSERSITE="1",
                   PIP_CONFIG_FILE="NUL", PIP_NO_INDEX="1", NO_PROXY="*", MPLBACKEND="Agg",
                   MPLCONFIGDIR=str(ROOT / "workspace/cache/matplotlib"))
        python = [sys.executable, "-I", "-X", "utf8"]
        pip = python + ["-m", "pip", "--isolated", "--disable-pip-version-check"]
        print("[1/4] Installing six pinned runtime dependencies from verified local wheels.", flush=True)
        run(pip + ["install", "--no-index", "--no-deps", "--require-hashes", "--only-binary=:all:",
                   "--find-links", ROOT / "workspace/wheels/step03_dependencies", "-r", LOCK,
                   "--report", logs / "dependencies_install.json"], ROOT, env, logs / "dependencies_install.txt")
        print("[2/4] Installing official MMDetection 3.3.0 in this environment only.", flush=True)
        run(pip + ["install", "--no-index", "--no-deps", "--no-build-isolation", "--use-pep517",
                   "--editable", SOURCE, "--report", logs / "mmdet_install.json"], ROOT, env, logs / "mmdet_install.txt")
        run(pip + ["check"], ROOT, env, logs / "pip_check.txt")
        check_base()
        (logs / "requirements_freeze.txt").write_bytes(subprocess.check_output(pip + ["freeze", "--all"], env=env))
        run(python + [Path(__file__), "--demo-only", "--log-dir", logs], ROOT, env, logs / "official_demo.txt")
        report["checkpoint"] = data["checkpoint"]
        report["status"] = "PASS"
    except BaseException:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    write_json(logs / "step03_manifest.json", report)
    print("STEP03_MMDET_DEMO_" + report["status"], flush=True)
    if report["status"] != "PASS":
        print("Send complete terminal output and LOG_DIR. Do not start training.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
