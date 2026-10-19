r"""Official YOLO11n COCO-pretrained baseline: exactly five smoke-test epochs.

Run from cmd with the project's interpreter:
    .venv\Scripts\python.exe -X utf8 workspace\scripts\train_yolo11.py
The original dataset is read-only. Only label cache destinations are adapted.
"""
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

from readonly_dataset import BLOCKED_WRITES, DATASET, PROJECT, WORKSPACE, install_write_guard, redirect_label_caches


def main():
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

    expected_source = PROJECT / "ultralytics_src"
    if not Path(ultralytics.__file__).resolve().is_relative_to(expected_source.resolve()):
        raise RuntimeError(f"Wrong Ultralytics import: {ultralytics.__file__}")
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError(f"Use the project .venv interpreter, current prefix is {sys.prefix}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; stop and inspect environment instead of silently training on CPU.")
    audit = json.loads((WORKSPACE / "logs" / "dataset_check.json").read_text(encoding="utf-8"))
    if not audit["passed"]:
        raise RuntimeError("Dataset read-only checks did not pass.")
    # Verify hashes/metadata before each run so a stale check cannot authorize changed data.
    subprocess.run([sys.executable, "-X", "utf8", str(WORKSPACE / "scripts" / "check_dataset.py"),
                    "--verify-unchanged"], check=True, cwd=PROJECT)
    redirect_label_caches()
    os.chdir(WORKSPACE / "weights")

    run_name = "yolo11n_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    status_path = WORKSPACE / "logs" / f"{run_name}_status.json"
    observations_path = WORKSPACE / "logs" / f"{run_name}_batches.jsonl"
    parameters = dict(data=str(WORKSPACE / "configs" / "conductor_bl_v1.yaml"),
                      epochs=5, imgsz=640, batch=4, workers=0, device=0,
                      seed=42, deterministic=True, plots=True, cache=False,
                      project=str(WORKSPACE / "runs"), name=run_name, exist_ok=False,
                      save=True, val=True, split="val", resume=False)
    status = {"success": False, "run_name": run_name, "parameters": parameters,
              "ultralytics": ultralytics.__version__, "ultralytics_file": ultralytics.__file__,
              "python": sys.version, "python_executable": sys.executable,
              "torch": torch.__version__, "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(0), "dataset": str(DATASET),
              "git_commit": subprocess.check_output(["git", "-C", str(expected_source), "rev-parse", "HEAD"], text=True).strip(),
              "model": "yolo11n.pt", "batches_checked": 0, "all_batch_losses_finite": True}
    started = time.monotonic()
    print(f"SOURCE_IMPORT={ultralytics.__file__}", flush=True)
    print(f"SMOKE_RUN={run_name}", flush=True)

    def verify_train_start(trainer):
        if trainer.device.type != "cuda":
            raise RuntimeError(f"Training unexpectedly uses {trainer.device}")
        if trainer.data["names"] != {0: "broken", 1: "loose"}:
            raise RuntimeError(f"Unexpected training classes: {trainer.data['names']}")
        status.update(train_images_loaded=len(trainer.train_loader.dataset),
                      val_images_loaded=len(trainer.test_loader.dataset),
                      training_device=str(trainer.device), save_dir=str(trainer.save_dir),
                      actual_batch=trainer.batch_size)
        if (status["train_images_loaded"], status["val_images_loaded"]) != (
            audit["splits"]["train"]["images"], audit["splits"]["val"]["images"]
        ):
            raise RuntimeError("Trainer image counts differ from the validated dataset.")
        print("TRAIN_START_CHECK=" + json.dumps(status, ensure_ascii=False), flush=True)

    def verify_batch(trainer):
        values = {name: float(value.detach().float().cpu()) for name, value in trainer.loss_items.items()}
        finite = all(math.isfinite(value) for value in values.values()) and bool(torch.isfinite(trainer.loss).all())
        status["batches_checked"] += 1
        status["all_batch_losses_finite"] &= finite
        entry = {"epoch": trainer.epoch + 1, "batch_index": status["batches_checked"],
                 "loss_items": values, "finite": finite, "loss_device": str(trainer.loss.device),
                 "allocated_MiB": round(torch.cuda.memory_allocated() / 2**20, 2)}
        with observations_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        if not finite:
            raise FloatingPointError(f"NaN/Inf loss encountered: {entry}")

    model = None
    try:
        model = YOLO("yolo11n.pt")  # Official COCO pretrained weights, never a prior run checkpoint.
        with (WORKSPACE / "weights" / "yolo11n.pt").open("rb") as stream:
            status["pretrained_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        status["pretrained_classes"] = len(model.names)
        if len(model.names) != 80 or model.task != "detect":
            raise RuntimeError("Expected the official 80-class COCO detection checkpoint.")
        model.add_callback("on_train_start", verify_train_start)
        model.add_callback("on_train_batch_end", verify_batch)
        metrics = model.train(**parameters)
        trainer = model.trainer
        run = Path(trainer.save_dir)
        status.update(save_dir=str(run), best_pt=str(trainer.best), last_pt=str(trainer.last),
                      peak_allocated_MiB=round(torch.cuda.max_memory_allocated()/2**20, 2),
                      peak_reserved_MiB=round(torch.cuda.max_memory_reserved()/2**20, 2),
                      metrics={key: float(value) for key, value in metrics.results_dict.items()},
                      final_names=model.names)
        with (run / "results.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        status["epochs_recorded"] = len(rows)
        status["all_csv_values_finite"] = all(math.isfinite(float(v)) for row in rows for v in row.values())
        status["artifacts"] = {"best_pt": Path(trainer.best).is_file(), "last_pt": Path(trainer.last).is_file(),
                               "training_batch_plots": bool(list(run.glob("train_batch*.jpg"))),
                               "validation_predictions": bool(list(run.glob("val_batch*_pred.jpg"))),
                               "validation_labels": bool(list(run.glob("val_batch*_labels.jpg"))),
                               "results_plot": (run / "results.png").is_file()}
        if len(rows) != 5 or not status["all_csv_values_finite"] or not all(status["artifacts"].values()):
            raise RuntimeError(f"Smoke result verification failed: {status}")
        status["success"] = True
    except Exception:
        status["error"] = traceback.format_exc()
        raise
    finally:
        status["elapsed_seconds"] = round(time.monotonic() - started, 2)
        status["blocked_dataset_writes"] = list(BLOCKED_WRITES)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        (WORKSPACE / "logs" / "latest_smoke_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print("SMOKE_STATUS=" + json.dumps(status, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
