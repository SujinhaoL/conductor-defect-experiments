"""Check real MMCV CPU/CUDA operators on synthetic tensors, without training."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
from pathlib import Path
import runpy
import traceback

ROOT = Path(__file__).resolve().parents[2]


def check(report, emit):
    identity = runpy.run_path(str(ROOT / "workspace/scripts/step01_check_cuda.py"))["check_identity"]
    report["identity"] = identity()
    emit("ENVIRONMENT_IDENTITY_PASS")
    import numpy
    import torch
    import torchvision
    import mmengine
    import mmcv
    import mmcv._ext
    from mmcv.ops import nms, batched_nms, roi_align, get_compiler_version, get_compiling_cuda_version
    from torchvision.ops import roi_align as tv_roi_align

    expected = {"numpy": "1.26.4", "torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128",
                "mmengine": "0.10.7", "mmcv": "2.1.0"}
    for name, module in [("numpy", numpy), ("torch", torch), ("torchvision", torchvision), ("mmengine", mmengine), ("mmcv", mmcv)]:
        if module.__version__ != expected[name]:
            raise RuntimeError(f"Unexpected {name} version: {module.__version__}")
        if not Path(module.__file__).resolve().is_relative_to(ROOT / ".venv"):
            raise RuntimeError(f"{name} came from another environment")
    ext_path = Path(mmcv._ext.__file__).resolve()
    if not ext_path.is_relative_to(ROOT / ".venv"):
        raise RuntimeError("MMCV native extension came from outside the dedicated environment")
    for conflicting in ("mmcv-lite", "mmcv-full"):
        try:
            metadata.version(conflicting)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError("Conflicting package installed: " + conflicting)
    report["versions"] = expected
    report["extension"] = str(ext_path)
    report["compiling_cuda"] = get_compiling_cuda_version()
    report["compiler"] = get_compiler_version()
    emit("MMCV_VERSION=" + mmcv.__version__)
    emit("MMENGINE_VERSION=" + mmengine.__version__)
    emit("MMCV_EXTENSION=" + str(ext_path))
    emit("MMCV_COMPILING_CUDA=" + report["compiling_cuda"])
    emit("MMCV_COMPILER=" + report["compiler"])
    if not report["compiling_cuda"].startswith("12.8"):
        raise RuntimeError("Expected a CUDA 12.8 MMCV extension")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0):
        raise RuntimeError("Expected the previously verified CUDA sm_120 device")
    report["gpu"] = torch.cuda.get_device_name(0)

    def passed(name):
        torch.cuda.synchronize()
        report["passed_checks"].append(name)
        emit(name + "_PASS")

    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32)
    scores = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)
    for device, label in [("cpu", "CPU"), ("cuda:0", "CUDA")]:
        dets, indices = nms(boxes.to(device), scores.to(device), 0.5)
        if indices.cpu().tolist() != [0, 2]:
            raise RuntimeError(f"{label} NMS failed its known-answer test")
        torch.testing.assert_close(dets.cpu(), torch.cat((boxes[[0, 2]], scores[[0, 2], None]), dim=1))
        passed("MMCV_" + label + "_NMS")

    batch_boxes = torch.cat((boxes, boxes[:1]), dim=0).cuda()
    batch_scores = torch.tensor([0.9, 0.8, 0.7, 0.85], device="cuda:0")
    labels = torch.tensor([0, 0, 0, 1], device="cuda:0")
    _, indices = batched_nms(batch_boxes, batch_scores, labels, dict(type="nms", iou_threshold=0.5))
    if indices.cpu().tolist() != [0, 3, 2]:
        raise RuntimeError("Class-aware CUDA NMS failed its known-answer test")
    passed("MMCV_CUDA_BATCHED_NMS")

    # Single operator gradient checks, not a model/optimizer/training loop.
    rois = torch.tensor([[0, 0, 0, 3, 3], [0, 0, 0, 4, 4]], dtype=torch.float32)
    reference_input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).requires_grad_()
    reference = tv_roi_align(reference_input, rois, (2, 2), spatial_scale=1.0, sampling_ratio=2, aligned=True)
    reference.sum().backward()
    for device, label in [("cpu", "CPU"), ("cuda:0", "CUDA")]:
        feature = reference_input.detach().to(device).clone().requires_grad_()
        output = roi_align(feature, rois.to(device), (2, 2), 1.0, 2, "avg", True)
        torch.testing.assert_close(output.detach().cpu(), reference.detach(), rtol=1e-5, atol=1e-5)
        passed("MMCV_" + label + "_ROIALIGN_FORWARD")
        output.sum().backward()
        torch.testing.assert_close(feature.grad.cpu(), reference_input.grad, rtol=1e-5, atol=1e-5)
        passed("MMCV_" + label + "_ROIALIGN_BACKWARD")
    report["scope"] = "Synthetic operator forward/gradient checks only. No model, dataset, optimizer, or training."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not output.is_relative_to(ROOT / "workspace/logs"):
        raise RuntimeError("Reports must stay within this workspace's logs")
    output.mkdir(parents=True, exist_ok=True)
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "status": "FAIL", "passed_checks": []}
    try:
        check(report, lambda line: print(line, flush=True))
        report["status"] = "PASS"
        print("STEP02C_MMCV_OPS_PASS", flush=True)
    except Exception:
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
        print("STEP02C_MMCV_OPS_FAIL", flush=True)
    finally:
        (output / "mmcv_ops_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
