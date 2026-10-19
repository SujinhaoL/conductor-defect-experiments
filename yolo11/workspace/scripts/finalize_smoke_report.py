"""Verify saved artifacts and freeze the environment, without further training."""
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from readonly_dataset import PROJECT, WORKSPACE, install_write_guard


def main():
    install_write_guard()
    os.environ["YOLO_CONFIG_DIR"] = str(WORKSPACE / "configs" / "ultralytics_settings")
    os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")
    import torch
    from ultralytics import YOLO
    from PIL import Image

    logs = WORKSPACE / "logs"
    status = json.loads((logs / "latest_smoke_status.json").read_text(encoding="utf-8"))
    if not status["success"]:
        raise RuntimeError("Latest smoke did not pass; inspect its status and console before finalizing.")
    run = Path(status["save_dir"])
    validation = {"checkpoints": {}, "plots": {}, "git_clean": False}
    for name in ("best", "last"):
        path = run / "weights" / f"{name}.pt"
        model = YOLO(str(path))
        assert model.task == "detect" and model.names == {0: "broken", 1: "loose"}
        finite = all(bool(torch.isfinite(p).all()) for p in model.model.parameters())
        assert finite
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        validation["checkpoints"][name] = {"path": str(path), "bytes": path.stat().st_size,
                                           "sha256": digest, "loadable": True, "finite_parameters": finite,
                                           "names": model.names}
    for path in sorted(run.iterdir()):
        if path.suffix.lower() in {".jpg", ".png"}:
            with Image.open(path) as image:
                dimensions = image.size
                image.verify()
            with Image.open(path) as image:
                image.load()
            validation["plots"][path.name] = dimensions
    source = PROJECT / "ultralytics_src"
    git_status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], encoding="utf-8")
    validation["git_clean"] = not git_status.strip()
    (logs / "git_status_after.txt").write_text(git_status or "Clean: no tracked modifications or untracked files.\n", encoding="utf-8")
    assert validation["git_clean"]
    pip_check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, encoding="utf-8")
    (logs / "pip_check.txt").write_text(pip_check.stdout + pip_check.stderr, encoding="utf-8")
    pip_check.check_returncode()
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], encoding="utf-8")
    (logs / "requirements_freeze.txt").write_text(freeze, encoding="utf-8")
    subprocess.run([sys.executable, "-X", "utf8", str(WORKSPACE / "scripts" / "check_dataset.py"), "--verify-unchanged"], check=True)
    integrity = json.loads((logs / "dataset_integrity_after.json").read_text(encoding="utf-8"))
    validation["dataset_unchanged"] = integrity["unchanged"]
    (logs / "artifact_verification.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    all_console = "\n".join(p.read_text(encoding="utf-8") for p in sorted(logs.glob("smoke_console_*.log")))
    all_console = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", all_console)
    warning_lines = list(dict.fromkeys(line for line in all_console.splitlines() if "warning" in line.lower()))
    (logs / "training_warnings.txt").write_text("\n".join(warning_lines) + ("\n" if warning_lines else "No Warning/UserWarning lines in training console logs.\n"), encoding="utf-8")
    with (run / "results.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = {"status": status, "artifacts_verified": len(validation["plots"]),
               "dataset_integrity": integrity, "first_epoch": rows[0], "last_epoch": rows[-1],
               "warning_lines": warning_lines}
    (logs / "final_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
