r"""Planned Y11-TUNE-001: Y11-DIAG-001 with only mosaic changed to 0.0.

This file is prepared for a future authorized run. It is intentionally guarded
against accidental execution. It never evaluates the test split.
"""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from check_dataset import snapshot
from readonly_dataset import (
    BLOCKED_WRITES,
    DATASET,
    PROJECT,
    PROTECTED_ROOT,
    WORKSPACE,
    install_write_guard,
    redirect_label_caches,
)

EXPECTED_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
EXPECTED_NAMES = {0: "broken", 1: "loose"}
FIXED_BATCH = 4
FIXED_NBS = 64


def verify_dataset_unchanged(output_name: str) -> dict:
    baseline = json.loads((WORKSPACE / "logs" / "dataset_snapshot_before.json").read_text(encoding="utf-8"))
    current = snapshot()
    changes = [key for key in sorted(baseline.keys() | current.keys()) if baseline.get(key) != current.get(key)]
    report = {
        "unchanged": not changes,
        "changed_paths": changes,
        "entries_checked": len(current),
        "sha256_scope": str(DATASET),
        "metadata_scope": str(PROTECTED_ROOT),
    }
    (WORKSPACE / "logs" / output_name).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if changes:
        raise RuntimeError(f"Protected dataset changed: {changes[:10]}")
    return report


def nvidia_memory() -> str:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.free,memory.total,utilization.gpu,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    return (process.stdout + process.stderr).strip()


def read_epoch_stability(results_csv: Path) -> dict:
    """Summarize epoch-level NaN/Inf without changing training behavior."""
    fields = (
        "train/box_loss", "train/cls_loss", "train/dfl_loss",
        "val/box_loss", "val/cls_loss", "val/dfl_loss",
        "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
    )
    with results_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    nonfinite = []
    for row in rows:
        bad = [field for field in fields if not math.isfinite(float(row[field]))]
        if bad:
            nonfinite.append({"epoch": int(row["epoch"]), "fields": bad})
    return {
        "epochs_recorded": len(rows),
        "nonfinite": nonfinite,
        "nonfinite_at_or_after_epoch_10": [entry for entry in nonfinite if entry["epoch"] >= 10],
    }


def main() -> None:
    if os.environ.get("RUN_Y11_TUNE_001") != "YES":
        raise RuntimeError(
            "Y11-TUNE-001 is prepared but not authorized in this stage. "
            "A future authorized run must set RUN_Y11_TUNE_001=YES explicitly."
        )

    settings_dir = WORKSPACE / "configs" / "ultralytics_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")
    os.environ["TORCH_HOME"] = str(WORKSPACE / "cache" / "torch")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    install_write_guard()

    import torch
    import ultralytics
    from ultralytics import YOLO

    source = PROJECT / "ultralytics_src"
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    if commit != EXPECTED_COMMIT or git_status.strip():
        raise RuntimeError(f"Official source mismatch: commit={commit}, status={git_status!r}")
    if not Path(ultralytics.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(f"Wrong Python environment: {sys.prefix}")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070 Ti":
        raise RuntimeError("Expected NVIDIA GeForce RTX 5070 Ti on CUDA:0")
    audit = json.loads((WORKSPACE / "logs" / "dataset_check.json").read_text(encoding="utf-8"))
    if (
        not audit["passed"]
        or audit["splits"]["train"]["images"] != 107
        or audit["splits"]["val"]["images"] != 23
        or audit["names"] != {"0": "broken", "1": "loose"}
    ):
        raise RuntimeError("Dataset audit does not match the locked project data")

    integrity_before = verify_dataset_unchanged("dataset_integrity_before_y11_tune_001.json")
    redirect_label_caches()
    (WORKSPACE / "weights").mkdir(parents=True, exist_ok=True)
    os.chdir(WORKSPACE / "weights")

    run_name = "yolo11s_tune_1280_e30_mosaic0_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    status_path = WORKSPACE / "logs" / f"{run_name}_status.json"
    batch_log = WORKSPACE / "logs" / f"{run_name}_batches.jsonl"
    parameters = {
        "data": str(WORKSPACE / "configs" / "conductor_bl_v1.yaml"),
        "epochs": 30,
        "imgsz": 1280,
        "batch": FIXED_BATCH,
        "nbs": FIXED_NBS,
        "workers": 0,
        "device": 0,
        "seed": 42,
        "deterministic": True,
        "patience": 0,
        "cache": False,
        "plots": True,
        "amp": True,
        "save": True,
        "save_period": 10,
        "project": str(WORKSPACE / "runs"),
        "name": run_name,
        "exist_ok": False,
        "val": True,
        "split": "val",
        "resume": False,
        "optimizer": "auto",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3.0,
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 10,
    }
    status = {
        "experiment_id": "Y11-TUNE-001",
        "parent_experiment": "Y11-DIAG-001",
        "changed_variable": "mosaic: 1.0 -> 0.0",
        "success": False,
        "run_name": run_name,
        "parameters": parameters,
        "source_commit": commit,
        "source_status_before": git_status,
        "dataset_integrity_before": integrity_before,
        "console_log": os.environ.get("Y11_TUNE_001_CONSOLE_LOG", "NA"),
        "gpu_before": nvidia_memory(),
        "batches_checked": 0,
        "all_batch_losses_finite": True,
    }
    started = time.monotonic()

    def verify_train_start(trainer) -> None:
        expected = {
            key: parameters[key]
            for key in (
                "nbs", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
                "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
                "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "cutmix", "copy_paste",
                "close_mosaic",
            )
        }
        actual = {key: getattr(trainer.args, key) for key in expected}
        if actual != expected:
            raise RuntimeError(f"A controlled parameter changed: expected={expected}, actual={actual}")
        if trainer.device.type != "cuda" or trainer.data["names"] != EXPECTED_NAMES:
            raise RuntimeError(f"Unexpected device/classes: {trainer.device}/{trainer.data['names']}")
        if trainer.batch_size != FIXED_BATCH or trainer.args.batch != FIXED_BATCH:
            raise RuntimeError(f"Batch must remain {FIXED_BATCH}")
        if len(trainer.train_loader.dataset) != 107 or len(trainer.test_loader.dataset) != 23:
            raise RuntimeError("Trainer did not load exactly 107 train and 23 val images")
        status.update(save_dir=str(trainer.save_dir), controlled_parameters=actual)

    def disable_automatic_batch_reduction(trainer) -> None:
        trainer._oom_retries = 3
        if trainer.batch_size != FIXED_BATCH or trainer.args.nbs != FIXED_NBS:
            raise RuntimeError("Controlled batch or nbs changed during training")

    def verify_batch(trainer) -> None:
        losses = {name: float(value.detach().float().cpu()) for name, value in trainer.loss_items.items()}
        finite = all(math.isfinite(value) for value in losses.values()) and bool(torch.isfinite(trainer.loss).all())
        status["batches_checked"] += 1
        status["all_batch_losses_finite"] &= finite
        with batch_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": trainer.epoch + 1, "losses": losses, "finite": finite}) + "\n")
        if not finite:
            raise FloatingPointError(f"NaN/Inf loss: {losses}")

    try:
        model = YOLO("yolo11s.pt")
        pretrained = WORKSPACE / "weights" / "yolo11s.pt"
        with pretrained.open("rb") as stream:
            status["pretrained_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        if model.task != "detect" or len(model.names) != 80:
            raise RuntimeError("Expected official 80-class COCO YOLO11s detection weights")
        model.add_callback("on_train_start", verify_train_start)
        model.add_callback("on_train_epoch_start", disable_automatic_batch_reduction)
        model.add_callback("on_train_batch_end", verify_batch)
        metrics = model.train(**parameters)
        trainer = model.trainer
        run = Path(trainer.save_dir)
        status.update(
            success=True,
            save_dir=str(run),
            best_pt=str(trainer.best),
            last_pt=str(trainer.last),
            results_csv=str(run / "results.csv"),
            final_metrics={key: float(value) for key, value in metrics.results_dict.items()},
            epoch_stability=read_epoch_stability(run / "results.csv"),
            peak_allocated_MiB=round(torch.cuda.max_memory_allocated() / 2**20, 2),
            peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2**20, 2),
            gpu_after=nvidia_memory(),
        )
    except Exception as exc:
        message = str(exc).lower()
        status.update(
            error=traceback.format_exc(),
            oom=("out of memory" in message or "cublas_status_alloc_failed" in message),
        )
        raise
    finally:
        status["elapsed_seconds"] = round(time.monotonic() - started, 2)
        status["dataset_integrity_after"] = verify_dataset_unchanged("dataset_integrity_after_y11_tune_001.json")
        status["blocked_dataset_writes"] = list(BLOCKED_WRITES)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
