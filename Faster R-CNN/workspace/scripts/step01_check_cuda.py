"""Validate only the dedicated environment and synthetic CUDA operations.

No dataset, model checkpoint, optimizer, backward pass, or training loop is used.
Torchvision checks do not replace the later MMCV CUDA operator acceptance tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import importlib.metadata
import json
from pathlib import Path
import platform
import site
import sys
import traceback

PROJECT = Path(__file__).resolve().parents[2]
EXPECTED_ENV = PROJECT / ".venv"
EXPECTED_PYTHON = "3.10.20"


def require(condition, explanation):
    if not condition:
        raise RuntimeError(explanation)


def check_identity():
    require(Path(sys.prefix).resolve() == EXPECTED_ENV.resolve(),
            f"Wrong environment: {sys.prefix}; expected {EXPECTED_ENV}")
    require(Path(sys.executable).resolve() == (EXPECTED_ENV / "Scripts/python.exe").resolve(),
            f"Wrong interpreter: {sys.executable}")
    require(platform.python_version() == EXPECTED_PYTHON,
            f"Wrong Python version: {platform.python_version()}; expected {EXPECTED_PYTHON}")
    cfg = (EXPECTED_ENV / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    require("include-system-site-packages = false" in cfg, "System packages must not be shared")
    require(not site.ENABLE_USER_SITE, "User site-packages must not be enabled")
    require(sys.prefix != sys.base_prefix, "This must be a virtual environment")
    return {"python": platform.python_version(), "executable": sys.executable,
            "prefix": sys.prefix, "base_prefix": sys.base_prefix,
            "user_site_enabled": site.ENABLE_USER_SITE, "sys_path": sys.path}


def run_checks(report, emit):
    report["identity"] = check_identity()
    emit("ENVIRONMENT_IDENTITY_PASS")
    emit("Python: " + platform.python_version())
    emit("Interpreter: " + sys.executable)
    import numpy as np
    import torch
    import torchvision
    from torch.nn import functional as F
    from torchvision.ops import nms, roi_align

    expected = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "numpy": "1.26.4"}
    for name, module in (("torch", torch), ("torchvision", torchvision), ("numpy", np)):
        require(module.__version__ == expected[name], f"Unexpected {name} version: {module.__version__}")
        require(Path(module.__file__).resolve().is_relative_to(EXPECTED_ENV.resolve()),
                f"{name} was loaded from another environment: {module.__file__}")
    report["versions"] = {**expected, "torch_cuda": torch.version.cuda,
                          "cudnn": torch.backends.cudnn.version()}
    emit("PyTorch: " + torch.__version__)
    emit("torchvision: " + torchvision.__version__)
    emit("NumPy: " + np.__version__)
    emit("PyTorch CUDA runtime: " + str(torch.version.cuda))
    require(torch.version.cuda == "12.8", "The CUDA 12.8 build is required")
    report["cuda_available"] = torch.cuda.is_available()
    emit("CUDA available: " + str(report["cuda_available"]))
    require(report["cuda_available"], "CUDA is unavailable")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    report["gpu"] = {"name": props.name, "capability": list(torch.cuda.get_device_capability(device)),
                     "memory_mib": props.total_memory // 2**20,
                     "compiled_architectures": torch.cuda.get_arch_list()}
    emit("GPU: " + props.name)
    emit("Compute capability: " + str(tuple(report["gpu"]["capability"])))
    emit("Compiled architectures: " + str(report["gpu"]["compiled_architectures"]))
    require(report["gpu"]["capability"] == [12, 0], "Unexpected GPU; inspect before continuing")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    def passed(name):
        torch.cuda.synchronize()
        report["passed_checks"].append(name)
        emit(name + "_PASS")

    with torch.no_grad():
        a = torch.arange(4096, dtype=torch.float32).reshape(64, 64) / 4096
        gpu = a.to(device)
        torch.testing.assert_close((gpu * 2 + 1).cpu(), a * 2 + 1)
        passed("CUDA_TENSOR")
        torch.testing.assert_close((gpu @ gpu.T).cpu(), a @ a.T, rtol=1e-4, atol=1e-4)
        passed("CUDA_MATMUL")

        inp = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16) / 768
        weight = torch.arange(4 * 3 * 3 * 3, dtype=torch.float32).reshape(4, 3, 3, 3) / 108
        expected_conv = F.conv2d(inp, weight, padding=1)
        actual_conv = F.conv2d(inp.to(device), weight.to(device), padding=1).cpu()
        torch.testing.assert_close(actual_conv, expected_conv, rtol=1e-4, atol=1e-4)
        passed("CUDA_CONV2D")

        boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]], dtype=torch.float32, device=device)
        scores = torch.tensor([0.9, 0.8, 0.7], device=device)
        require(nms(boxes, scores, 0.5).cpu().tolist() == [0, 2], "Unexpected CUDA NMS result")
        passed("TORCHVISION_CUDA_NMS")

        feat = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        rois = torch.tensor([[0, 0, 0, 3, 3]], dtype=torch.float32)
        ref_roi = roi_align(feat, rois, (2, 2), sampling_ratio=2, aligned=True)
        got_roi = roi_align(feat.to(device), rois.to(device), (2, 2), sampling_ratio=2, aligned=True).cpu()
        torch.testing.assert_close(got_roi, ref_roi, rtol=1e-4, atol=1e-4)
        passed("TORCHVISION_CUDA_ROIALIGN")

    report["packages"] = sorted(
        ({"name": d.metadata["Name"], "version": d.version} for d in importlib.metadata.distributions()),
        key=lambda d: (d["name"] or "").lower())
    report["torch_build_config"] = torch.__config__.show()
    report["peak_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.identity_only:
        check_identity()
        print("ENVIRONMENT_IDENTITY_PASS")
        return 0
    parser.error("--output-dir is required") if args.output_dir is None else None
    out = args.output_dir.resolve()
    require(out.is_relative_to((PROJECT / "workspace/logs").resolve()), "Reports must remain under workspace/logs")
    out.mkdir(parents=True, exist_ok=True)
    report = {"timestamp_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(),
              "status": "FAIL", "passed_checks": [],
              "scope": "Synthetic forward operations only; no training; not MMCV validation."}
    lines = []

    def emit(line):
        lines.append(line)
        print(line, flush=True)

    try:
        run_checks(report, emit)
        report["status"] = "PASS"
        emit("STEP01_CUDA_PASS")
    except Exception:
        report["traceback"] = traceback.format_exc()
        emit(report["traceback"])
        emit("STEP01_CUDA_FAIL")
    finally:
        emit("Report: " + str(out / "cuda_check.json"))
        (out / "cuda_check.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (out / "cuda_check.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
