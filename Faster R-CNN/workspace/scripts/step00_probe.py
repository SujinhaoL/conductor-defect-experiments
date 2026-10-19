"""Read-only environment probe. Standard library only; never trains or installs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import ctypes
from datetime import datetime, timezone, timedelta
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "conductor_defect_dataset" / "02_yolo11_dataset"
PACKAGES = {"torch", "torchvision", "mmcv", "mmcv-lite", "mmcv-full", "mmengine", "mmdet", "numpy", "ultralytics", "pillow", "pip"}


def decode(raw):
    if b"\x00" in raw:
        return raw.decode("utf-16-le", errors="replace").lstrip("\ufeff")
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace")


def command(args):
    try:
        r = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return {"command": args, "returncode": r.returncode, "output": decode(r.stdout).strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": args, "error": str(exc)}


def sha256(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def package_metadata(prefix):
    # Inspect files only: do not import or execute another model's packages.
    result = []
    for meta in sorted((prefix / "Lib" / "site-packages").glob("*.dist-info/METADATA")):
        name = version = None
        for line in meta.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Name: "):
                name = line[6:]
            elif line.startswith("Version: "):
                version = line[9:]
            if name and version:
                break
        if name and name.lower() in PACKAGES:
            result.append({"name": name, "version": version, "metadata": str(meta)})
    return result


def environment():
    env_files = list(ROOT.glob("*/.venv/pyvenv.cfg")) + list(ROOT.glob("*/venv/pyvenv.cfg"))
    prefixes = {Path(sys.prefix), Path("D:/anaconda3")}
    prefixes.update(p for p in Path("D:/anaconda3/envs").glob("*") if p.is_dir())
    prefixes.update(p.parent for p in env_files)
    conda_registry = Path.home() / ".conda/environments.txt"
    if conda_registry.exists():
        prefixes.update(Path(s) for s in conda_registry.read_text(encoding="utf-8").splitlines() if s)
    result = {
        "probe_interpreter": sys.executable,
        "probe_python": sys.version,
        "platform": platform.platform(),
        "isolated_mode": bool(sys.flags.isolated),
        "terminal_context": {k: os.environ.get(k) for k in ("CONDA_PREFIX", "CONDA_DEFAULT_ENV", "VIRTUAL_ENV", "CUDA_PATH", "CUDA_HOME", "VSINSTALLDIR", "VSCMD_VER", "PYTHONPATH")},
        "path_tools": {n: shutil.which(n) for n in ("python", "conda", "uv", "git", "nvcc", "cl", "cmake", "ninja", "wsl")},
        "environments": [],
        "commands": [],
        "directory_inventory": {str(p): sorted(x.name for x in p.iterdir()) if p.exists() else None for p in (ROOT / "Faster R-CNN", ROOT / "RT-DETR", DATA, DATA.parent / "03_coco_dataset")},
    }
    installed_kib = ctypes.c_ulonglong()
    if ctypes.windll.kernel32.GetPhysicallyInstalledSystemMemory(ctypes.byref(installed_kib)):
        result["installed_ram_gib"] = round(installed_kib.value / 2**20, 2)
    result["cpu"] = platform.processor()
    for p in sorted(prefixes):
        cfg = p / "pyvenv.cfg"
        python_metadata = []
        for meta in (p / "conda-meta").glob("python-*.json"):
            content = json.loads(meta.read_text(encoding="utf-8"))
            if content.get("name") == "python":
                python_metadata.append(content.get("version"))
        result["environments"].append({"prefix": str(p), "python_exists": (p / "python.exe").exists() or (p / "Scripts/python.exe").exists(), "python_version_from_conda_metadata": python_metadata, "pyvenv_cfg": cfg.read_text(encoding="utf-8") if cfg.exists() else None, "packages_from_metadata_only": package_metadata(p)})
    cmd = os.environ.get("COMSPEC", "C:/Windows/System32/cmd.exe")
    for args in ([cmd, "/d", "/c", "ver"], ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap", "--format=csv"], ["git", "--version"], ["wsl.exe", "--status"], ["wsl.exe", "-l", "-v"]):
        result["commands"].append(command(args))
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    result["vswhere_exists"] = vswhere.exists()
    if vswhere.exists():
        result["commands"].append(command([str(vswhere), "-all", "-products", "*", "-format", "json", "-utf8"]))
        result["commands"].append(command([str(vswhere), "-all", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]))
    cuda_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    result["cuda_toolkits"] = [str(p) for p in cuda_root.glob("v*")]
    for compiler in cuda_root.glob("v*/bin/nvcc.exe"):
        result["commands"].append(command([str(compiler), "--version"]))
    result["disk_free_gib"] = round(shutil.disk_usage(ROOT).free / 2**30, 2)
    return result


def image_size(p):
    # Header inspection only. Full decode and EXIF orientation audit remain pending.
    with p.open("rb") as f:
        sig = f.read(24)
        if sig[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", sig[16:24])
        if sig[:2] != b"\xff\xd8":
            raise ValueError("Unsupported image format for header audit")
        f.seek(2)
        while True:
            byte = f.read(1)
            if not byte:
                break
            if byte != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if not marker or marker in (b"\xd9", b"\xda"):
                break
            if marker[0] in (0, 1, 0xD8) or 0xD0 <= marker[0] <= 0xD7:
                continue
            length = struct.unpack(">H", f.read(2))[0]
            if marker[0] in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                _, h, w = struct.unpack(">BHH", f.read(5))
                return w, h
            if length < 2:
                raise ValueError("Invalid JPEG marker length")
            f.seek(length - 2, 1)
    raise ValueError("Image size header not found")


def data_audit():
    rows = list(csv.DictReader((DATA / "split_manifest.csv").open(encoding="utf-8-sig", newline="")))
    errors, records = [], []
    expected = defaultdict(list)
    for row in rows:
        expected[(row["split"], row["new_name"])].append(row)
    for key, values in expected.items():
        if len(values) != 1:
            errors.append({"kind": "duplicate_manifest_key", "key": key})
    actual = set()
    splits = {}
    for split in ("train", "val", "test"):
        images = sorted(p for p in (DATA / "images" / split).iterdir() if p.is_file())
        labels = {p.name: p for p in (DATA / "labels" / split).glob("*.txt")}
        counts, used = Counter(), set()
        empty_labels = []
        stems = Counter(p.stem for p in images)
        for stem, n in stems.items():
            if n > 1:
                errors.append({"kind": "duplicate_image_stem", "split": split, "stem": stem})
        for p in images:
            key = (split, p.name)
            actual.add(key)
            rec = {"split": split, "name": p.name, "sha256": sha256(p)}
            try:
                rec["width"], rec["height"] = image_size(p)
            except Exception as exc:
                errors.append({"kind": "image_header", "image": str(p), "error": str(exc)})
            lab = labels.get(p.stem + ".txt")
            per_image = Counter()
            if lab is None:
                errors.append({"kind": "missing_label", "image": str(p)})
            else:
                used.add(lab.name)
                rec["label_sha256"] = sha256(lab)
                lines = lab.read_text(encoding="utf-8-sig").splitlines()
                if not any(s.strip() for s in lines):
                    empty_labels.append(p.name)
                seen = set()
                for n, line in enumerate(lines, 1):
                    if not line.strip():
                        continue
                    try:
                        parts = line.split()
                        if len(parts) != 5 or parts[0] not in ("0", "1"):
                            raise ValueError("Expected class 0/1 and four coordinates")
                        vals = tuple(Decimal(v) for v in parts[1:])
                        if not all(v.is_finite() for v in vals):
                            raise ValueError("Non-finite coordinate")
                        x, y, w, h = vals
                        if w <= 0 or h <= 0:
                            raise ValueError("Non-positive box area")
                        if min(x-w/2, y-h/2) < 0 or max(x+w/2, y+h/2) > 1:
                            raise ValueError("Out-of-bounds box; not clipped")
                        box = (parts[0], *vals)
                        if box in seen:
                            errors.append({"kind": "duplicate_box", "label": str(lab), "line": n})
                        seen.add(box)
                        per_image[parts[0]] += 1
                    except Exception as exc:
                        errors.append({"kind": "label", "path": str(lab), "line": n, "raw": line, "error": str(exc)})
            rec["broken"] = per_image["0"]
            rec["loose"] = per_image["1"]
            counts.update(per_image)
            matched = expected.get(key, [])
            if len(matched) == 1:
                row = matched[0]
                for col in ("sha256", "width", "height"):
                    if str(rec.get(col)) != row[col]:
                        errors.append({"kind": "manifest_mismatch", "image": p.name, "field": col, "actual": rec.get(col), "manifest": row[col]})
                for cls in ("broken", "loose"):
                    if rec[cls] != int(row[cls + "_boxes"]):
                        errors.append({"kind": "manifest_box_mismatch", "image": p.name, "class": cls})
                rec.update({k: row[k] for k in ("split_group_id", "scene_id", "source_folder", "original_name")})
            records.append(rec)
        for name in sorted(set(labels) - used):
            errors.append({"kind": "orphan_label", "split": split, "name": name})
        splits[split] = {"images": len(images), "labels": len(labels), "broken": counts["0"], "loose": counts["1"], "boxes": sum(counts.values()), "empty_labels": empty_labels}
    for key in sorted(set(expected) - actual):
        errors.append({"kind": "manifest_image_missing", "key": key})
    for key in sorted(actual - set(expected)):
        errors.append({"kind": "image_not_in_manifest", "key": key})
    duplicate_groups = {}
    cross_split = {}
    for field in ("sha256", "split_group_id", "scene_id", "original_name"):
        groups = defaultdict(list)
        for rec in records:
            if rec.get(field):
                value = rec[field]
                if field == "original_name":
                    value = rec["source_folder"] + "/" + value
                groups[value].append({"split": rec["split"], "name": rec["name"]})
        duplicate_groups[field] = {k: v for k, v in groups.items() if len(v) > 1}
        cross_split[field] = {k: v for k, v in groups.items() if len({i["split"] for i in v}) > 1}
    return {"source": str(DATA), "manifest_rows": len(rows), "splits": splits, "totals": {k: sum(s[k] for s in splits.values()) for k in ("images", "labels", "broken", "loose", "boxes")}, "errors": errors, "cross_split_collisions": cross_split, "duplicate_groups": duplicate_groups, "scene_id_nonempty": sum(bool(r.get("scene_id")) for r in records), "group_count": len({r.get("split_group_id") for r in records}), "records": records, "limitations": ["Header dimensions only; full image decoding and EXIF audit are pending.", "SHA256 detects exact byte duplicates only. Group checks do not prove absence of unrecorded related scenes or near duplicates.", "No COCO conversion, drawing, annotation changes, or split changes performed."]}


def previous_configs():
    runs = [("yolo11", "Y11-BASE-001-R1_20260917_203044_883451"), ("RT-DETR", "RTD-BASE-001-R1_20260918_172120"), ("yolo11", "Y11-SENS-IMGSZ1600-001"), ("RT-DETR", "RTD-SENS-IMGSZ1600-001")]
    keys = {"model", "data", "epochs", "batch", "imgsz", "workers", "seed", "deterministic", "optimizer", "amp", "lr0", "lrf", "pretrained", "conf", "iou", "max_det", "split", "nbs", "patience", "save_period"}
    result = []
    for model, run in runs:
        p = ROOT / model / "workspace/runs" / run / "args.yaml"
        item = {"path": str(p), "exists": p.exists()}
        if p.exists():
            item["sha256"] = sha256(p)
            item["selected_raw_yaml_lines"] = [line for line in p.read_text(encoding="utf-8-sig").splitlines() if line.split(":", 1)[0] in keys]
        result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="store_true", help="Read and audit original images/labels; no source writes")
    args = parser.parse_args()
    now = datetime.now(timezone(timedelta(hours=8)))
    out = WORKSPACE / "audit" / ("step00_" + now.strftime("%Y%m%d_%H%M%S_%f"))
    out.mkdir(parents=True, exist_ok=False)
    print("STEP00_READ_ONLY_PROBE", flush=True)
    env = environment()
    result = {"timestamp_asia_shanghai": now.isoformat(), "environment": env, "previous_configs": previous_configs()}
    if args.dataset:
        result["dataset"] = data_audit()
    destination = out / "probe.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["STEP00_READ_ONLY_PROBE", "Time: " + now.isoformat(), "Probe interpreter: " + sys.executable, "Probe Python: " + platform.python_version(), "Standard library only; no install, model import, inference or training.", "Report: " + str(destination), "", "ENVIRONMENTS (package metadata only)"]
    for e in env["environments"]:
        lines.append(e["prefix"] + ": " + ", ".join(p["name"] + "=" + p["version"] for p in e["packages_from_metadata_only"]))
        lines.append("Python metadata: " + json.dumps(e["python_version_from_conda_metadata"]))
        if e["pyvenv_cfg"]:
            lines.append(e["pyvenv_cfg"])
    lines += ["", "SYSTEM COMMANDS"]
    for c in env["commands"]:
        lines.append(json.dumps(c, ensure_ascii=False))
    lines += ["", "PATH_TOOLS: " + json.dumps(env["path_tools"], ensure_ascii=False), "CUDA_TOOLKITS: " + str(env["cuda_toolkits"])]
    lines += ["VSWHERE_EXISTS: " + str(env["vswhere_exists"]), "RAM_GIB: " + str(env.get("installed_ram_gib")), "CPU: " + env["cpu"], "DISK_FREE_GIB: " + str(env["disk_free_gib"]), "TERMINAL_CONTEXT: " + json.dumps(env["terminal_context"], ensure_ascii=False), "DIRECTORY_INVENTORY: " + json.dumps(env["directory_inventory"], ensure_ascii=False)]
    lines += ["", "PREVIOUS ARGS (literal values, not proof of runtime optimizer)"]
    for config in result["previous_configs"]:
        lines.append(config["path"])
        lines.extend(config.get("selected_raw_yaml_lines", ["MISSING"]))
    if args.dataset:
        data = result["dataset"]
        lines += ["", "DATASET"]
        for split, counts in data["splits"].items():
            lines.append(split + ": " + json.dumps(counts))
        lines.append("TOTALS: " + json.dumps(data["totals"]))
        lines.append("DATA_ERRORS: " + str(len(data["errors"])))
        lines.append("CROSS_SPLIT_COLLISIONS: " + json.dumps({k: len(v) for k, v in data["cross_split_collisions"].items()}))
        lines.append("EXACT_DUPLICATE_GROUPS_ALL_SPLITS: " + str(len(data["duplicate_groups"]["sha256"])))
        lines.append("SPLIT_GROUPS: " + str(data["group_count"]))
        lines.append("NONEMPTY_SCENE_IDS: " + str(data["scene_id_nonempty"]))
        lines.extend(data["limitations"])
    lines.append("STEP00_PROBE_FINISHED (not a CUDA/MMCV acceptance test)")
    summary = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
