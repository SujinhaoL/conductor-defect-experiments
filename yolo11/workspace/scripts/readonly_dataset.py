"""Project-side dataset write guard and cache location adapter; no source edits."""
import hashlib
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT / "workspace"
PROTECTED_ROOT = Path("D:/小论文/conductor_defect_dataset").resolve()
DATASET = PROTECTED_ROOT / "02_yolo11_dataset"
_installed = False
BLOCKED_WRITES = []


def protected(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    return Path(os.fsdecode(path)).resolve().is_relative_to(PROTECTED_ROOT)


def install_write_guard():
    """Block Python filesystem mutations under the entire protected data project."""
    global _installed
    if _installed:
        return

    def guard(event, args):
        paths = []
        if event == "open":
            path, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            ):
                paths = [path]
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime", "os.truncate"}:
            paths = [args[0]]
        elif event in {"os.rename", "os.link", "os.symlink"}:
            paths = list(args[:2])
        for path in paths:
            if protected(path):
                message = f"Read-only dataset: blocked {event}: {path}"
                BLOCKED_WRITES.append(message)
                raise PermissionError(message)

    sys.addaudithook(guard)
    _installed = True


def redirect_label_caches():
    """Change only cache I/O destinations at runtime, keeping training logic intact."""
    import ultralytics.data.dataset as dataset_module
    import ultralytics.data.utils as utils_module

    cache_root = WORKSPACE / "cache" / "labels"
    cache_root.mkdir(parents=True, exist_ok=True)
    original_load = utils_module.load_dataset_cache_file
    original_save = utils_module.save_dataset_cache_file

    def location(path):
        path = Path(path)
        if not protected(path):
            return path
        digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
        return cache_root / f"{path.stem}_{digest}.cache"

    def load_cache(path):
        return original_load(location(path))

    def save_cache(prefix, path, x, version):
        destination = location(path)
        print(f"Dataset label cache redirected: {path} -> {destination}", flush=True)
        return original_save(prefix, destination, x, version)

    for module in (utils_module, dataset_module):
        module.load_dataset_cache_file = load_cache
        module.save_dataset_cache_file = save_cache
