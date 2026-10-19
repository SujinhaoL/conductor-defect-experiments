r"""Run Y11-DIAG-002: YOLO11s 1280 batch=4 nbs=16 for 30 epochs.

The central dataset and official source are protected. OOM is re-raised without
allowing Ultralytics to reduce the batch. This script never evaluates test.
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
from readonly_dataset import BLOCKED_WRITES, DATASET, PROJECT, PROTECTED_ROOT, WORKSPACE, install_write_guard, redirect_label_caches

EXPECTED_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
EXPECTED_NAMES = {0: "broken", 1: "loose"}
FIXED_BATCH = 4
FIXED_NBS = 16
ACCUMULATION_LOG = WORKSPACE / "logs" / "yolo11s_nbs16_accumulation.csv"


def verify_dataset_unchanged(output_name: str) -> dict:
    before = json.loads((WORKSPACE / "logs" / "dataset_snapshot_before.json").read_text(encoding="utf-8"))
    after = snapshot()
    changes = [key for key in sorted(before.keys() | after.keys()) if before.get(key) != after.get(key)]
    report = {
        "unchanged": not changes,
        "changed_paths": changes,
        "entries_checked": len(after),
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
    nonfinite = []
    with results_csv.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        fields = {key: float(row[key]) for key in ("val/box_loss", "val/cls_loss", "val/dfl_loss")}
        bad = [key for key, value in fields.items() if not math.isfinite(value)]
        if bad:
            json_fields = {
                key: value if math.isfinite(value) else ("NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity"))
                for key, value in fields.items()
            }
            nonfinite.append({"epoch": int(row["epoch"]), "fields": bad, "values": json_fields})
    return {
        "epochs_recorded": len(rows),
        "nonfinite_val_losses": nonfinite,
        "nonfinite_at_or_after_epoch_10": [entry for entry in nonfinite if entry["epoch"] >= 10],
    }


def main() -> None:
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
    if commit != EXPECTED_COMMIT:
        raise RuntimeError(f"Official source commit changed: {commit}")
    if git_status.strip():
        raise RuntimeError(f"Official source is not clean:\n{git_status}")
    if not Path(ultralytics.__file__).resolve().is_relative_to(source.resolve()):
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(f"Wrong Python environment: {sys.prefix}")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5070 Ti":
        raise RuntimeError("Expected NVIDIA GeForce RTX 5070 Ti on CUDA:0")
    audit = json.loads((WORKSPACE / "logs" / "dataset_check.json").read_text(encoding="utf-8"))
    if not audit["passed"] or audit["splits"]["train"]["images"] != 107 or audit["splits"]["val"]["images"] != 23:
        raise RuntimeError("Dataset audit does not match the locked project data.")

    integrity_before = verify_dataset_unchanged("dataset_integrity_before_yolo11s_nbs16.json")
    redirect_label_caches()
    (WORKSPACE / "weights").mkdir(parents=True, exist_ok=True)
    os.chdir(WORKSPACE / "weights")
    if ACCUMULATION_LOG.exists():
        raise RuntimeError(f"Refusing to overwrite accumulation log: {ACCUMULATION_LOG}")
    with ACCUMULATION_LOG.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerow(
            ["epoch", "epoch_batch_index", "batch_index", "accumulate", "learning_rate", "lr_min", "lr_max", "lr_pg0", "lr_pg1", "lr_pg2"]
        )

    run_name = "yolo11s_diag_1280_e30_nbs16_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    }
    status = {
        "experiment_id": "Y11-DIAG-002",
        "success": False,
        "run_name": run_name,
        "parameters": parameters,
        "expected_batch": FIXED_BATCH,
        "expected_nbs": FIXED_NBS,
        "accumulation_log": str(ACCUMULATION_LOG),
        "console_log": os.environ.get("NBS16_CONSOLE_LOG", "NA"),
        "ultralytics": ultralytics.__version__,
        "ultralytics_file": ultralytics.__file__,
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "driver_and_memory_before": nvidia_memory(),
        "dataset": str(DATASET),
        "dataset_integrity_before": integrity_before,
        "git_commit": commit,
        "git_status_before": git_status,
        "model": "yolo11s.pt",
        "batches_checked": 0,
        "all_batch_losses_finite": True,
    }
    started = time.monotonic()
    print(f"SOURCE_IMPORT={ultralytics.__file__}", flush=True)
    print(f"DIAGNOSTIC_RUN={run_name}", flush=True)
    print(f"GPU_BEFORE={status['driver_and_memory_before']}", flush=True)

    def verify_train_start(trainer) -> None:
        if trainer.device.type != "cuda":
            raise RuntimeError(f"Training unexpectedly uses {trainer.device}")
        if trainer.data["names"] != EXPECTED_NAMES:
            raise RuntimeError(f"Unexpected classes: {trainer.data['names']}")
        if trainer.batch_size != FIXED_BATCH or trainer.args.batch != FIXED_BATCH:
            raise RuntimeError(f"Batch must remain {FIXED_BATCH}, got {trainer.batch_size}/{trainer.args.batch}")
        if trainer.args.nbs != FIXED_NBS:
            raise RuntimeError(f"nbs must be {FIXED_NBS}, got {trainer.args.nbs}")
        invariant = {
            "optimizer": "auto", "lr0": 0.01, "lrf": 0.01, "momentum": 0.937, "weight_decay": 0.0005,
            "warmup_epochs": 3.0, "box": 7.5, "cls": 0.5, "dfl": 1.5, "hsv_h": 0.015, "hsv_s": 0.7,
            "hsv_v": 0.4, "degrees": 0.0, "translate": 0.1, "scale": 0.5, "shear": 0.0,
            "perspective": 0.0, "flipud": 0.0, "fliplr": 0.5, "mosaic": 1.0, "mixup": 0.0,
            "cutmix": 0.0, "copy_paste": 0.0, "close_mosaic": 10,
        }
        actual = {key: getattr(trainer.args, key) for key in invariant}
        if actual != invariant:
            raise RuntimeError(f"A controlled parameter changed: expected={invariant}, actual={actual}")
        defaults = trainer.optimizer.defaults
        momentum = defaults.get("momentum")
        if momentum is None and defaults.get("betas"):
            momentum = defaults["betas"][0]
        status.update(
            train_images_loaded=len(trainer.train_loader.dataset),
            val_images_loaded=len(trainer.test_loader.dataset),
            training_device=str(trainer.device),
            save_dir=str(trainer.save_dir),
            actual_batch=trainer.batch_size,
            actual_nbs=trainer.args.nbs,
            initial_accumulate=int(trainer.accumulate),
            controlled_parameters=actual,
            optimizer_actual={
                "name": f"{type(trainer.optimizer).__name__} (optimizer=auto)",
                "lr0": max(
                    float(group.get("initial_lr", group["lr"]))
                    for group in trainer.optimizer.param_groups
                ),
                "momentum": float(momentum),
                "weight_decay": max(float(group.get("weight_decay", 0.0)) for group in trainer.optimizer.param_groups),
            },
        )
        if status["train_images_loaded"] != 107 or status["val_images_loaded"] != 23:
            raise RuntimeError("Trainer did not load exactly 107 train and 23 val images.")
        print("TRAIN_START_CHECK=" + json.dumps(status, ensure_ascii=False), flush=True)

    def disable_automatic_batch_reduction(trainer) -> None:
        trainer._oom_retries = 3
        if trainer.batch_size != FIXED_BATCH or trainer.args.batch != FIXED_BATCH or trainer.args.nbs != FIXED_NBS:
            raise RuntimeError("Controlled batch or nbs changed during training.")

    def record_batch(trainer) -> None:
        if trainer.batch_size != FIXED_BATCH or trainer.args.batch != FIXED_BATCH or trainer.args.nbs != FIXED_NBS:
            raise RuntimeError("Controlled batch or nbs changed during training.")
        losses = {name: float(value.detach().float().cpu()) for name, value in trainer.loss_items.items()}
        finite = all(math.isfinite(value) for value in losses.values()) and bool(torch.isfinite(trainer.loss).all())
        status["batches_checked"] += 1
        status["all_batch_losses_finite"] &= finite
        lrs = [float(group["lr"]) for group in trainer.optimizer.param_groups]
        epoch_batch_index = (status["batches_checked"] - 1) % len(trainer.train_loader) + 1
        with ACCUMULATION_LOG.open("a", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerow(
                [trainer.epoch + 1, epoch_batch_index, status["batches_checked"], int(trainer.accumulate), lrs[0], min(lrs), max(lrs), *lrs[:3]]
            )
        entry = {
            "epoch": trainer.epoch + 1,
            "epoch_batch_index": epoch_batch_index,
            "batch_index": status["batches_checked"],
            "accumulate": int(trainer.accumulate),
            "learning_rates": lrs,
            "loss_items": losses,
            "finite": finite,
            "loss_device": str(trainer.loss.device),
            "batch_size": trainer.batch_size,
            "allocated_MiB": round(torch.cuda.memory_allocated() / 2**20, 2),
            "reserved_MiB": round(torch.cuda.memory_reserved() / 2**20, 2),
        }
        with batch_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        if not finite:
            raise FloatingPointError(f"NaN/Inf training loss: {entry}")

    try:
        model = YOLO("yolo11s.pt")
        pretrained = WORKSPACE / "weights" / "yolo11s.pt"
        with pretrained.open("rb") as stream:
            status["pretrained_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        status["pretrained_classes"] = len(model.names)
        if model.task != "detect" or len(model.names) != 80:
            raise RuntimeError("Expected official 80-class COCO YOLO11s detection weights.")
        model.add_callback("on_train_start", verify_train_start)
        model.add_callback("on_train_epoch_start", disable_automatic_batch_reduction)
        model.add_callback("on_train_batch_end", record_batch)
        metrics = model.train(**parameters)
        trainer = model.trainer
        run = Path(trainer.save_dir)
        stability = read_epoch_stability(run / "results.csv")
        with ACCUMULATION_LOG.open(encoding="utf-8-sig", newline="") as stream:
            accumulation_rows = list(csv.DictReader(stream))
        tail = accumulation_rows[-min(100, len(accumulation_rows)):]
        steady = max({int(row["accumulate"]) for row in tail}, key=lambda value: sum(int(row["accumulate"]) == value for row in tail))
        status.update(
            success=True,
            save_dir=str(run),
            best_pt=str(trainer.best),
            last_pt=str(trainer.last),
            actual_batch=trainer.batch_size,
            actual_nbs=trainer.args.nbs,
            steady_accumulate=steady,
            accumulation_values=sorted({int(row["accumulate"]) for row in accumulation_rows}),
            accumulation_rows=len(accumulation_rows),
            peak_allocated_MiB=round(torch.cuda.max_memory_allocated() / 2**20, 2),
            peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2**20, 2),
            final_metrics={key: float(value) for key, value in metrics.results_dict.items()},
            final_names=model.names,
            epoch_stability=stability,
            driver_and_memory_after=nvidia_memory(),
        )
    except Exception as exc:
        message = str(exc).lower()
        status.update(
            error=traceback.format_exc(),
            oom=("out of memory" in message or "cublas_status_alloc_failed" in message or "cudnn_status_internal_error" in message),
            peak_allocated_MiB=round(torch.cuda.max_memory_allocated() / 2**20, 2),
            peak_reserved_MiB=round(torch.cuda.max_memory_reserved() / 2**20, 2),
            nvidia_memory_at_error=nvidia_memory(),
        )
        if status["oom"]:
            (WORKSPACE / "notes" / "yolo11s_nbs16_oom_report.md").write_text(
                "# Y11-DIAG-002 OOM 报告\n\n"
                f"- 堆栈：\n```text\n{status['error']}\n```\n"
                f"- peak allocated: {status['peak_allocated_MiB']} MiB\n"
                f"- peak reserved: {status['peak_reserved_MiB']} MiB\n"
                f"- nvidia-smi: {status['nvidia_memory_at_error']}\n"
                "- 脚本未自动降低 batch；建议由用户决定是否改为 batch=2。\n",
                encoding="utf-8",
            )
        raise
    finally:
        status["elapsed_seconds"] = round(time.monotonic() - started, 2)
        status["blocked_dataset_writes"] = list(BLOCKED_WRITES)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        (WORKSPACE / "logs" / "latest_yolo11s_nbs16_status.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("NBS16_STATUS=" + json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
