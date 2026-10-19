from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any


EXPERIMENT_ID = "RTD-BASE-001-R1"
EXPECTED_ULTRALYTICS_VERSION = "8.4.152"
EXPECTED_GIT_COMMIT = "9aa41d679d121afb8b4cc512d7a162f4f36886c3"
EXPECTED_OFFICIAL_WEIGHT_SHA256 = "6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f"
EXPECTED_NAMES = {0: "broken", 1: "loose"}
EXPECTED_COUNTS = {"train": 107, "val": 23, "test": 23}
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}

PROJECT_ROOT = Path(r"D:\小论文\model_projects\RT-DETR")
WORKSPACE = PROJECT_ROOT / "workspace"
CONFIG_PATH = WORKSPACE / "configs" / "conductor_bl_v1.yaml"
WEIGHTS_DIR = WORKSPACE / "weights"
OFFICIAL_WEIGHT = WEIGHTS_DIR / "rtdetr-l.pt"
RUNS_ROOT = WORKSPACE / "runs"
LOGS_DIR = WORKSPACE / "logs"
NOTES_DIR = WORKSPACE / "notes"
ANALYSIS_DIR = WORKSPACE / "analysis"

DATASET_ROOT = Path(r"D:\小论文\conductor_defect_dataset\02_yolo11_dataset")
CENTRAL_CONFIG = DATASET_ROOT / "data.yaml"
YOLO_CONFIG = Path(r"D:\小论文\model_projects\yolo11\workspace\configs\conductor_bl_v1.yaml")
ULTRALYTICS_SOURCE = Path(r"D:\小论文\model_projects\yolo11\ultralytics_src")
PREVIOUS_FAILED_STATUS = ANALYSIS_DIR / "RTD-BASE-001_status.json"


class TeeStream:
    def __init__(self, primary: Any, log_file: Any) -> None:
        self.primary = primary
        self.log_file = log_file

    def write(self, text: str) -> int:
        primary_result = self.primary.write(text)
        self.primary.flush()
        self.log_file.write(text)
        self.log_file.flush()
        return primary_result if isinstance(primary_result, int) else len(text)

    def flush(self) -> None:
        self.primary.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.primary, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self.primary.fileno()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_git(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(ULTRALYTICS_SOURCE), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def normalize_names(raw: Any) -> dict[int, str]:
    if isinstance(raw, list):
        return {index: str(value) for index, value in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(key): str(value) for key, value in raw.items()}
    raise ValueError(f"Unsupported names definition: {raw!r}")


def load_and_resolve_dataset_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {path}")
    base = Path(str(raw.get("path", path.parent)))
    if not base.is_absolute():
        base = path.parent / base
    base = base.resolve()
    resolved = {
        "yaml": str(path.resolve()),
        "path": str(base),
        "train": str((base / str(raw["train"])).resolve()),
        "val": str((base / str(raw["val"])).resolve()),
        "test": str((base / str(raw["test"])).resolve()),
        "names": normalize_names(raw["names"]),
    }
    return resolved


def inspect_dataset() -> dict[str, Any]:
    expected_root = DATASET_ROOT.resolve()
    if not expected_root.is_dir():
        raise PermissionError(f"Central dataset is not readable: {expected_root}")

    configs = {
        "central": load_and_resolve_dataset_yaml(CENTRAL_CONFIG),
        "yolo11": load_and_resolve_dataset_yaml(YOLO_CONFIG),
        "rtdetr": load_and_resolve_dataset_yaml(CONFIG_PATH),
    }
    expected_paths = {
        "path": str(expected_root),
        "train": str((expected_root / "images" / "train").resolve()),
        "val": str((expected_root / "images" / "val").resolve()),
        "test": str((expected_root / "images" / "test").resolve()),
    }
    for label, config in configs.items():
        for key, expected in expected_paths.items():
            if config[key] != expected:
                raise RuntimeError(f"{label} YAML {key} mismatch: {config[key]} != {expected}")
        if config["names"] != EXPECTED_NAMES:
            raise RuntimeError(f"{label} YAML class mismatch: {config['names']} != {EXPECTED_NAMES}")

    split_details: dict[str, Any] = {}
    for split, expected_count in EXPECTED_COUNTS.items():
        images_dir = expected_root / "images" / split
        labels_dir = expected_root / "labels" / split
        images = sorted(path for path in images_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(labels_dir.rglob("*.txt"))
        if len(images) != expected_count or len(labels) != expected_count:
            raise RuntimeError(
                f"{split} count mismatch: images={len(images)}, labels={len(labels)}, expected={expected_count}"
            )
        image_keys = {path.relative_to(images_dir).with_suffix("").as_posix() for path in images}
        label_keys = {path.relative_to(labels_dir).with_suffix("").as_posix() for path in labels}
        if image_keys != label_keys:
            missing_labels = sorted(image_keys - label_keys)
            missing_images = sorted(label_keys - image_keys)
            raise RuntimeError(
                f"{split} image/label pairing mismatch; missing_labels={missing_labels}, missing_images={missing_images}"
            )
        class_histogram = {0: 0, 1: 0}
        object_count = 0
        for label_path in labels:
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                fields = stripped.split()
                if len(fields) < 5:
                    raise RuntimeError(f"Malformed label: {label_path}:{line_number}")
                class_id = int(float(fields[0]))
                if class_id not in EXPECTED_NAMES:
                    raise RuntimeError(f"Unexpected class id {class_id}: {label_path}:{line_number}")
                coordinates = [float(value) for value in fields[1:]]
                if not all(math.isfinite(value) for value in coordinates):
                    raise RuntimeError(f"Non-finite label value: {label_path}:{line_number}")
                class_histogram[class_id] += 1
                object_count += 1
        split_details[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": object_count,
            "class_histogram": {EXPECTED_NAMES[key]: value for key, value in class_histogram.items()},
        }
    return {"configs": configs, "splits": split_details}


def snapshot_dataset() -> dict[str, Any]:
    """Snapshot only formal inputs: images and YOLO .txt labels. Framework *.cache files are excluded."""
    entries: list[dict[str, Any]] = []
    formal_files = [
        path
        for path in (DATASET_ROOT / "images").rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    formal_files.extend(
        path for path in (DATASET_ROOT / "labels").rglob("*.txt") if path.is_file()
    )
    for path in sorted(formal_files, key=lambda item: item.as_posix()):
        stat = path.stat()
        entries.append(
            {
                "relative_path": path.relative_to(DATASET_ROOT).as_posix(),
                "size": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "root": str(DATASET_ROOT.resolve()),
        "captured_at": now_iso(),
        "scope": "images/**/* supported image files and labels/**/*.txt; excludes Ultralytics *.cache",
        "file_count": len(entries),
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": entries,
    }


def list_label_caches() -> dict[str, dict[str, Any]]:
    labels_root = (DATASET_ROOT / "labels").resolve()
    caches: dict[str, dict[str, Any]] = {}
    for path in sorted(labels_root.rglob("*.cache"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        resolved = path.resolve()
        try:
            resolved.relative_to(labels_root)
        except ValueError as error:
            raise RuntimeError(f"Cache path escaped labels directory: {resolved}") from error
        caches[path.relative_to(DATASET_ROOT).as_posix()] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return caches


def cleanup_new_label_caches(before: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current = list_label_caches()
    generated = sorted(set(current) - set(before))
    removed: list[dict[str, Any]] = []
    labels_root = (DATASET_ROOT / "labels").resolve()
    for relative_path in generated:
        path = (DATASET_ROOT / relative_path).resolve()
        try:
            path.relative_to(labels_root)
        except ValueError as error:
            raise RuntimeError(f"Refusing to remove cache outside labels directory: {path}") from error
        if path.suffix.lower() != ".cache" or not path.is_file():
            raise RuntimeError(f"Refusing to remove uncertain cache target: {path}")
        record = {"relative_path": relative_path, **current[relative_path]}
        path.unlink()
        record["removed"] = not path.exists()
        removed.append(record)
    remaining = list_label_caches()
    return {
        "present_before": before,
        "generated_during_run": [{"relative_path": item, **current[item]} for item in generated],
        "removed": removed,
        "remaining_after_cleanup": remaining,
    }


def list_source_pt_files() -> set[str]:
    return {
        path.relative_to(ULTRALYTICS_SOURCE).as_posix()
        for path in ULTRALYTICS_SOURCE.rglob("*.pt")
        if path.is_file()
    }


def relocate_new_source_pt_files(before: set[str], timestamp: str) -> list[dict[str, str]]:
    moved: list[dict[str, str]] = []
    for relative_path in sorted(list_source_pt_files() - before):
        source = (ULTRALYTICS_SOURCE / relative_path).resolve()
        try:
            source.relative_to(ULTRALYTICS_SOURCE.resolve())
        except ValueError as error:
            raise RuntimeError(f"Source PT path escaped locked source: {source}") from error
        destination = WEIGHTS_DIR / source.name
        if destination.exists():
            destination = WEIGHTS_DIR / f"amp_{timestamp}_{source.name}"
        source.replace(destination)
        moved.append({"source": str(source), "destination": str(destination)})
    return moved


def dataset_differences(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    before_map = {entry["relative_path"]: entry for entry in before["entries"]}
    after_map = {entry["relative_path"]: entry for entry in after["entries"]}
    differences: list[dict[str, Any]] = []
    for relative_path in sorted(set(before_map) | set(after_map)):
        if before_map.get(relative_path) != after_map.get(relative_path):
            differences.append(
                {
                    "relative_path": relative_path,
                    "before": before_map.get(relative_path),
                    "after": after_map.get(relative_path),
                }
            )
    return differences


def assert_dataset_unchanged(before: dict[str, Any], label: str, save_path: Path) -> dict[str, Any]:
    after = snapshot_dataset()
    write_json(save_path, after)
    differences = dataset_differences(before, after)
    if differences:
        write_json(ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_changes_{label}.json", differences)
        raise RuntimeError(f"Formal images or .txt labels changed during {label}; differences={len(differences)}")
    return after


def check_source_and_environment() -> dict[str, Any]:
    import torch
    import ultralytics

    if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"Ultralytics version mismatch: {ultralytics.__version__} != {EXPECTED_ULTRALYTICS_VERSION}"
        )
    import_path = Path(ultralytics.__file__).resolve()
    try:
        import_path.relative_to(ULTRALYTICS_SOURCE.resolve())
    except ValueError as error:
        raise RuntimeError(f"Ultralytics import path is outside locked source: {import_path}") from error
    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain=v1", "--untracked-files=no")
    if commit != EXPECTED_GIT_COMMIT:
        raise RuntimeError(f"Git commit mismatch: {commit} != {EXPECTED_GIT_COMMIT}")
    if status:
        raise RuntimeError(f"Tracked Ultralytics source files are not clean:\n{status}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "ultralytics": ultralytics.__version__,
        "ultralytics_import_path": str(import_path),
        "git_commit": commit,
        "git_tracked_status_clean": True,
    }


def extract_metrics(metrics: Any) -> dict[str, Any]:
    overall = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    classes: dict[str, dict[str, float]] = {}
    class_indices = [int(value) for value in metrics.box.ap_class_index]
    for position, class_id in enumerate(class_indices):
        precision, recall, map50, map50_95 = metrics.box.class_result(position)
        class_name = str(metrics.names[class_id])
        classes[class_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "map50": float(map50),
            "map50_95": float(map50_95),
        }
    for class_name in EXPECTED_NAMES.values():
        if class_name not in classes:
            raise RuntimeError(f"Formal evaluation did not return metrics for class: {class_name}")
    bundle = {
        "overall": overall,
        "classes": classes,
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
        "save_dir": str(Path(metrics.save_dir).resolve()),
    }
    for scope_name, scope_metrics in [("overall", overall), *classes.items()]:
        for metric_name, value in scope_metrics.items():
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite formal metric: {scope_name}.{metric_name}={value}")
    return bundle


def read_training_results(results_csv: Path) -> list[dict[str, str]]:
    if not results_csv.is_file():
        raise FileNotFoundError(results_csv)
    with results_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 100:
        raise RuntimeError(f"Training did not complete exactly 100 epochs; results rows={len(rows)}")
    return rows


def determine_best_epoch(best_pt: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    import torch

    checkpoint = torch.load(best_pt, map_location="cpu", weights_only=False)
    train_metrics = checkpoint.get("train_metrics") or {}
    best_fitness = train_metrics.get("fitness")
    if best_fitness is None:
        best_fitness = train_metrics.get("metrics/mAP50-95(B)")
    map_key = next((key for key in rows[0] if key.strip() == "metrics/mAP50-95(B)"), None)
    if map_key is None:
        raise RuntimeError(f"mAP50-95 column missing from {best_pt.parent.parent / 'results.csv'}")
    csv_maps = [float(row[map_key]) for row in rows]
    if best_fitness is None:
        target_value = max(csv_maps)
    else:
        target_value = float(best_fitness)
    minimum_difference = min(abs(value - target_value) for value in csv_maps)
    candidate_indices = [
        index for index, value in enumerate(csv_maps) if abs(abs(value - target_value) - minimum_difference) <= 1e-12
    ]
    best_index = candidate_indices[-1]
    best_epoch = int(float(rows[best_index]["epoch"]))
    return {
        "best_epoch": best_epoch,
        "best_fitness_checkpoint": target_value,
        "best_map50_95_csv": csv_maps[best_index],
        "selection_basis": "Ultralytics detection fitness = mAP50-95; matched stripped best.pt train_metrics to results.csv",
    }


def format_metric_row(label: str, values: dict[str, float]) -> str:
    return (
        f"| {label} | {values['precision']:.6f} | {values['recall']:.6f} | "
        f"{values['map50']:.6f} | {values['map50_95']:.6f} |"
    )


def write_success_report(summary: dict[str, Any]) -> None:
    val = summary["formal_evaluations"]["val"]
    test = summary["formal_evaluations"]["test"]
    lines = [
        "# RT-DETR single-run report",
        "",
        f"- Experiment: {EXPERIMENT_ID}",
        "- Stage: FormalBaseline",
        "- Model: RT-DETR-L (official Ultralytics implementation)",
        f"- Pretrained weights: {OFFICIAL_WEIGHT}",
        f"- Pretrained weights SHA256: {summary['pretrained_weight_sha256']}",
        "- Network structure modified: No",
        "- Epochs: 100",
        "- Image size: 1280",
        "- Batch: 4",
        "- Seed: 42",
        "- Deterministic: False",
        f"- Best epoch: {summary['best_epoch']['best_epoch']}",
        f"- Training runtime (wall seconds): {summary['runtime']['training_wall_seconds']:.3f}",
        f"- Total runtime (wall seconds): {summary['runtime']['total_wall_seconds']:.3f}",
        f"- Peak CUDA reserved VRAM (GiB): {summary['peak_vram']['max_reserved_gib']:.3f}",
        f"- Peak CUDA allocated VRAM (GiB): {summary['peak_vram']['max_allocated_gib']:.3f}",
        f"- Parameters: {summary['model_profile']['parameters']}",
        f"- GFLOPs at imgsz=1280: {summary['model_profile']['gflops_at_1280']}",
        f"- best.pt: {summary['best_pt']['path']}",
        f"- best.pt SHA256: {summary['best_pt']['sha256']}",
        f"- Run directory: {summary['run_dir']}",
        "- Test executions: 1",
        f"- Git tracked source status clean after experiment: {summary['source_after']['git_tracked_status_clean']}",
        f"- Formal images and .txt labels unchanged: {summary['dataset_unchanged']}",
        f"- Generated label caches: {', '.join(item['relative_path'] for item in summary['label_cache_audit']['generated_during_run']) or 'None'}",
        f"- Generated label caches cleaned: {not bool(summary['label_cache_audit']['remaining_after_cleanup'])}",
        "",
        "| split/class | Precision | Recall | mAP50 | mAP50-95 |",
        "| --- | ---: | ---: | ---: | ---: |",
        format_metric_row("val overall", val["overall"]),
        format_metric_row("val broken", val["classes"]["broken"]),
        format_metric_row("val loose", val["classes"]["loose"]),
        format_metric_row("test overall", test["overall"]),
        format_metric_row("test broken", test["classes"]["broken"]),
        format_metric_row("test loose", test["classes"]["loose"]),
        "",
    ]
    (NOTES_DIR / "rtdetr_single_run_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_failure_report(timestamp: str, phase: str, error_text: str, run_dir: Path | None) -> None:
    lines = [
        "# RT-DETR single-run failure report",
        "",
        f"- Experiment: {EXPERIMENT_ID}",
        f"- Timestamp: {timestamp}",
        f"- Failed phase: {phase}",
        f"- Run directory: {run_dir if run_dir is not None else 'not created'}",
        "- Automatic parameter changes: None",
        "- Automatic retry/resume: None",
        "- Test evaluation attempted: See status JSON",
        "",
        "## Error",
        "",
        "```text",
        error_text.rstrip(),
        "```",
        "",
    ]
    (NOTES_DIR / f"{EXPERIMENT_ID}_failure_{timestamp}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    for directory in [WEIGHTS_DIR, RUNS_ROOT, LOGS_DIR, NOTES_DIR, ANALYSIS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    os.chdir(WEIGHTS_DIR)

    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_name = f"{EXPERIMENT_ID}_{timestamp}"
    expected_run_dir = RUNS_ROOT / run_name
    if expected_run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {expected_run_dir}")
    log_path = LOGS_DIR / f"{run_name}.log"
    status_path = ANALYSIS_DIR / f"{EXPERIMENT_ID}_status.json"
    dataset_before_path = ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_manifest_before.json"
    dataset_after_path = ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_manifest_after.json"

    log_handle = log_path.open("x", encoding="utf-8", buffering=1)
    sys.stdout = TeeStream(sys.__stdout__, log_handle)
    sys.stderr = TeeStream(sys.__stderr__, log_handle)

    status: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "timestamp": timestamp,
        "status": "running",
        "phase": "preflight",
        "log_path": str(log_path),
        "run_dir": str(expected_run_dir),
        "test_execution_count": 0,
        "updated_at": now_iso(),
    }
    write_json(status_path, status)
    dataset_before: dict[str, Any] | None = None
    label_caches_before: dict[str, dict[str, Any]] = {}
    source_pt_before: set[str] = set()
    cleanup_baselines_ready = False
    experiment_start = time.perf_counter()

    try:
        print(f"[{now_iso()}] Starting {EXPERIMENT_ID}")
        print("Authorized fixed controls: epochs=100, imgsz=1280, batch=4, workers=0, device=0, seed=42")
        print("RT-DETR mechanism control: deterministic=False")
        print("AMP, optimizer, LR, augmentations, loss and architecture parameters are not overridden.")
        print(f"Process working directory: {Path.cwd()}")

        if not PREVIOUS_FAILED_STATUS.is_file():
            raise FileNotFoundError(f"Previous failed-run status is missing: {PREVIOUS_FAILED_STATUS}")
        previous_status = json.loads(PREVIOUS_FAILED_STATUS.read_text(encoding="utf-8"))
        if previous_status.get("test_execution_count") != 0:
            raise RuntimeError(
                f"Previous RTD-BASE-001 test count must remain 0, found {previous_status.get('test_execution_count')}"
            )

        dataset_inspection = inspect_dataset()
        dataset_before = snapshot_dataset()
        write_json(dataset_before_path, dataset_before)
        if dataset_before["file_count"] != sum(EXPECTED_COUNTS.values()) * 2:
            raise RuntimeError(f"Unexpected formal input file count: {dataset_before['file_count']} != 306")
        label_caches_before = list_label_caches()
        source_pt_before = list_source_pt_files()
        cleanup_baselines_ready = True
        environment = check_source_and_environment()
        print(
            json.dumps(
                {
                    "dataset": dataset_inspection["splits"],
                    "formal_input_manifest_sha256": dataset_before["manifest_sha256"],
                    "formal_input_file_count": dataset_before["file_count"],
                    "label_caches_present_before": label_caches_before,
                    "previous_test_execution_count": previous_status["test_execution_count"],
                    "environment": environment,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        from ultralytics import RTDETR
        from ultralytics.utils.torch_utils import get_flops
        import torch

        status.update(phase="official_weight", updated_at=now_iso())
        write_json(status_path, status)
        download_required = not OFFICIAL_WEIGHT.exists()
        model = RTDETR(str(OFFICIAL_WEIGHT))
        if not OFFICIAL_WEIGHT.is_file():
            raise FileNotFoundError(f"Official weight was not created: {OFFICIAL_WEIGHT}")
        if model.task != "detect":
            raise RuntimeError(f"Unexpected task in rtdetr-l.pt: {model.task}")
        pretrained_weight_sha256 = sha256_file(OFFICIAL_WEIGHT)
        if pretrained_weight_sha256 != EXPECTED_OFFICIAL_WEIGHT_SHA256:
            raise RuntimeError(
                f"Official rtdetr-l.pt SHA256 mismatch: {pretrained_weight_sha256} != {EXPECTED_OFFICIAL_WEIGHT_SHA256}"
            )
        print(
            json.dumps(
                {
                    "official_weight": str(OFFICIAL_WEIGHT),
                    "downloaded_now": download_required,
                    "sha256": pretrained_weight_sha256,
                    "sha256_verified": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        status.update(phase="training", updated_at=now_iso())
        write_json(status_path, status)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        training_start = time.perf_counter()
        model.train(
            data=str(CONFIG_PATH),
            epochs=100,
            imgsz=1280,
            batch=4,
            workers=0,
            device=0,
            seed=42,
            pretrained=True,
            cache=False,
            plots=True,
            deterministic=False,
            project=str(RUNS_ROOT),
            name=run_name,
            exist_ok=False,
        )
        training_wall_seconds = time.perf_counter() - training_start
        actual_run_dir = Path(model.trainer.save_dir).resolve()
        if actual_run_dir != expected_run_dir.resolve():
            raise RuntimeError(f"Unexpected run directory: {actual_run_dir} != {expected_run_dir.resolve()}")
        results_csv = actual_run_dir / "results.csv"
        rows = read_training_results(results_csv)
        official_training_seconds = float(rows[-1].get("time", "nan"))
        args_path = actual_run_dir / "args.yaml"
        import yaml

        actual_args = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        expected_args = {
            "epochs": 100,
            "imgsz": 1280,
            "batch": 4,
            "workers": 0,
            "device": "0",
            "seed": 42,
            "pretrained": True,
            "cache": False,
            "plots": True,
            "deterministic": False,
            "exist_ok": False,
        }
        mismatches = {key: {"actual": actual_args.get(key), "expected": value} for key, value in expected_args.items() if actual_args.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Saved training arguments do not match fixed controls: {mismatches}")

        best_pt = actual_run_dir / "weights" / "best.pt"
        last_pt = actual_run_dir / "weights" / "last.pt"
        if not best_pt.is_file() or not last_pt.is_file():
            raise FileNotFoundError(f"Missing required checkpoint(s): best={best_pt.exists()}, last={last_pt.exists()}")
        best_sha256 = sha256_file(best_pt)
        best_epoch = determine_best_epoch(best_pt, rows)
        peak_reserved = torch.cuda.max_memory_reserved(0)
        peak_allocated = torch.cuda.max_memory_allocated(0)
        assert_dataset_unchanged(
            dataset_before,
            "training",
            ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_manifest_after_training.json",
        )
        source_mid = check_source_and_environment()

        status.update(phase="formal_val", best_pt=str(best_pt), best_pt_sha256=best_sha256, updated_at=now_iso())
        write_json(status_path, status)
        best_model = RTDETR(str(best_pt))
        parameters = sum(parameter.numel() for parameter in best_model.model.parameters())
        try:
            gflops_at_1280 = float(get_flops(best_model.model, imgsz=1280))
        except Exception as profile_error:
            print(f"GFLOPs profiling unavailable: {profile_error}")
            gflops_at_1280 = 0.0
        val_metrics_object = best_model.val(
            data=str(CONFIG_PATH),
            split="val",
            imgsz=1280,
            device=0,
            conf=0.001,
            iou=0.7,
            max_det=300,
            project=str(actual_run_dir),
            name="formal_val",
            exist_ok=False,
        )
        val_metrics = extract_metrics(val_metrics_object)
        write_json(actual_run_dir / "formal_val_metrics.json", val_metrics)
        partial_metrics = {
            "experiment_id": EXPERIMENT_ID,
            "status": "val_complete_test_not_started",
            "best_pt": {"path": str(best_pt), "sha256": best_sha256},
            "formal_evaluations": {"val": val_metrics},
        }
        write_json(ANALYSIS_DIR / "rtdetr_single_run_metrics.json", partial_metrics)
        assert_dataset_unchanged(
            dataset_before,
            "formal_val",
            ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_manifest_after_val.json",
        )
        check_source_and_environment()

        status.update(phase="formal_test", test_execution_count=1, updated_at=now_iso())
        write_json(status_path, status)
        test_metrics_object = best_model.val(
            data=str(CONFIG_PATH),
            split="test",
            imgsz=1280,
            device=0,
            conf=0.001,
            iou=0.7,
            max_det=300,
            project=str(actual_run_dir),
            name="formal_test",
            exist_ok=False,
        )
        test_metrics = extract_metrics(test_metrics_object)
        write_json(actual_run_dir / "formal_test_metrics.json", test_metrics)

        peak_reserved = max(peak_reserved, torch.cuda.max_memory_reserved(0))
        peak_allocated = max(peak_allocated, torch.cuda.max_memory_allocated(0))
        dataset_after_evaluation = assert_dataset_unchanged(
            dataset_before,
            "formal_test_before_cache_cleanup",
            ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_manifest_after_evaluation.json",
        )
        status.update(phase="cache_cleanup", updated_at=now_iso())
        write_json(status_path, status)
        label_cache_audit = cleanup_new_label_caches(label_caches_before)
        write_json(ANALYSIS_DIR / f"{EXPERIMENT_ID}_label_cache_audit.json", label_cache_audit)
        relocated_source_pt_files = relocate_new_source_pt_files(source_pt_before, timestamp)
        dataset_after = assert_dataset_unchanged(
            dataset_before,
            "cache_cleanup",
            dataset_after_path,
        )
        source_after = check_source_and_environment()
        total_wall_seconds = time.perf_counter() - experiment_start

        summary = {
            "experiment_id": EXPERIMENT_ID,
            "stage": "FormalBaseline",
            "status": "success",
            "model": "RT-DETR-L",
            "official_ultralytics_implementation": True,
            "network_structure_modified": False,
            "pretrained_weight": str(OFFICIAL_WEIGHT),
            "pretrained_weight_downloaded_by_ultralytics": download_required,
            "pretrained_weight_sha256": pretrained_weight_sha256,
            "training_controls": expected_args,
            "epochs_completed": len(rows),
            "dataset": dataset_inspection,
            "dataset_manifest_before_sha256": dataset_before["manifest_sha256"],
            "dataset_manifest_after_evaluation_sha256": dataset_after_evaluation["manifest_sha256"],
            "dataset_manifest_after_sha256": dataset_after["manifest_sha256"],
            "dataset_unchanged": dataset_before["manifest_sha256"] == dataset_after["manifest_sha256"],
            "label_cache_audit": label_cache_audit,
            "relocated_amp_auxiliary_weights": relocated_source_pt_files,
            "environment": environment,
            "source_mid": source_mid,
            "source_after": source_after,
            "model_profile": {
                "parameters": parameters,
                "gflops_at_1280": gflops_at_1280 if gflops_at_1280 > 0 else None,
            },
            "runtime": {
                "training_wall_seconds": training_wall_seconds,
                "training_official_csv_seconds": official_training_seconds,
                "total_wall_seconds": total_wall_seconds,
            },
            "peak_vram": {
                "max_reserved_bytes": peak_reserved,
                "max_reserved_gib": peak_reserved / (1024**3),
                "max_allocated_bytes": peak_allocated,
                "max_allocated_gib": peak_allocated / (1024**3),
            },
            "best_epoch": best_epoch,
            "best_pt": {"path": str(best_pt), "sha256": best_sha256},
            "last_pt": {"path": str(last_pt), "sha256": sha256_file(last_pt)},
            "run_dir": str(actual_run_dir),
            "formal_evaluations": {"val": val_metrics, "test": test_metrics},
            "test_execution_count": 1,
            "console_log": str(log_path),
            "completed_at": now_iso(),
        }
        write_json(ANALYSIS_DIR / "rtdetr_single_run_metrics.json", summary)
        write_success_report(summary)
        status.update(status="success", phase="complete", summary_path=str(ANALYSIS_DIR / "rtdetr_single_run_metrics.json"), updated_at=now_iso())
        write_json(status_path, status)
        print(f"[{now_iso()}] {EXPERIMENT_ID} completed successfully")
        return 0
    except BaseException:
        error_text = traceback.format_exc()
        print(error_text, file=sys.stderr)
        phase = str(status.get("phase", "unknown"))
        dataset_check: dict[str, Any] | None = None
        cache_cleanup_at_failure: dict[str, Any] | None = None
        relocated_source_pt_at_failure: list[dict[str, str]] = []
        if cleanup_baselines_ready:
            try:
                cache_cleanup_at_failure = cleanup_new_label_caches(label_caches_before)
                write_json(ANALYSIS_DIR / f"{EXPERIMENT_ID}_label_cache_audit_at_failure.json", cache_cleanup_at_failure)
            except Exception as cache_error:
                cache_cleanup_at_failure = {"error": repr(cache_error)}
            try:
                relocated_source_pt_at_failure = relocate_new_source_pt_files(source_pt_before, timestamp)
            except Exception as relocation_error:
                relocated_source_pt_at_failure = [{"error": repr(relocation_error)}]
        if dataset_before is not None:
            try:
                current_snapshot = snapshot_dataset()
                differences = dataset_differences(dataset_before, current_snapshot)
                dataset_check = {
                    "unchanged": not differences,
                    "difference_count": len(differences),
                    "current_manifest_sha256": current_snapshot["manifest_sha256"],
                }
                write_json(ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_manifest_at_failure.json", current_snapshot)
                if differences:
                    write_json(ANALYSIS_DIR / f"{EXPERIMENT_ID}_dataset_changes_at_failure.json", differences)
            except Exception as snapshot_error:
                dataset_check = {"unchanged": None, "error": repr(snapshot_error)}
        try:
            git_commit = run_git("rev-parse", "HEAD")
            git_status = run_git("status", "--porcelain=v1", "--untracked-files=no")
            source_check = {
                "git_commit": git_commit,
                "git_tracked_status_clean": not bool(git_status),
                "git_tracked_status": git_status,
            }
        except Exception as source_error:
            source_check = {"error": repr(source_error)}
        status.update(
            status="failed",
            phase=phase,
            error=error_text,
            dataset_check_at_failure=dataset_check,
            label_cache_cleanup_at_failure=cache_cleanup_at_failure,
            relocated_amp_auxiliary_weights_at_failure=relocated_source_pt_at_failure,
            source_check_at_failure=source_check,
            updated_at=now_iso(),
        )
        write_json(status_path, status)
        write_failure_report(timestamp, phase, error_text, expected_run_dir if expected_run_dir.exists() else None)
        return 1
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            log_handle.flush()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
