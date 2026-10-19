"""Read-only environment inspection; writes reports only inside workspace."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT / "workspace"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="environment_before.txt")
    parser.add_argument("--gpu-test", action="store_true")
    args = parser.parse_args()
    for name in ("configs", "scripts", "weights", "runs", "logs", "notes"):
        (WORKSPACE / name).mkdir(parents=True, exist_ok=True)
    settings_dir = WORKSPACE / "configs" / "ultralytics_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["PYTHONUTF8"] = "1"
    lines = [f"Windows: {platform.platform()}", f"Python: {sys.version}",
             f"Python executable: {sys.executable}", f"prefix: {sys.prefix}",
             f"base_prefix: {sys.base_prefix}"]
    for command in (["git", "--version"], ["py", "-0p"], ["where", "python"],
                    [sys.executable, "-m", "pip", "--version"], ["nvidia-smi"],
                    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"]):
        lines.append("\n> " + subprocess.list2cmdline(command))
        try:
            p = subprocess.run(command, capture_output=True, timeout=60)
            lines.append((p.stdout + p.stderr).decode("utf-8", errors="replace"))
            lines.append(f"exit_code={p.returncode}")
        except Exception as exc:
            lines.append(f"{type(exc).__name__}: {exc}")
    for module in ("torch", "ultralytics"):
        spec = importlib.util.find_spec(module)
        lines.append(f"\n{module} installed: {spec is not None}")
        if spec is None:
            continue
        try:
            imported = __import__(module)
            lines.extend([f"{module}.__version__: {imported.__version__}",
                          f"{module}.__file__: {imported.__file__}"])
            if module == "torch":
                torch = imported
                lines.extend([f"torch.version.cuda: {torch.version.cuda}",
                              f"torch.cuda.is_available(): {torch.cuda.is_available()}"])
                if torch.cuda.is_available():
                    lines.extend([f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}",
                                  f"device capability: {torch.cuda.get_device_capability(0)}",
                                  f"compiled architectures: {torch.cuda.get_arch_list()}"])
                    if args.gpu_test:
                        import torchvision
                        x = torch.randn(2, 3, 64, 64, device="cuda", requires_grad=True)
                        conv = torch.nn.Conv2d(3, 8, 3).cuda()
                        loss = conv(x).square().mean()
                        loss.backward()
                        boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device="cuda")
                        kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
                        torch.cuda.synchronize()
                        assert torch.isfinite(loss) and torch.isfinite(x.grad).all()
                        lines.extend([f"torchvision: {torchvision.__version__}",
                                      f"CUDA forward/backward + torchvision NMS: PASS; loss={loss.item()}; kept={kept.tolist()}"])
        except Exception as exc:
            lines.append(f"{module} ERROR: {type(exc).__name__}: {exc}")
    report = "\n".join(lines) + "\n"
    (WORKSPACE / "logs" / args.output).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
