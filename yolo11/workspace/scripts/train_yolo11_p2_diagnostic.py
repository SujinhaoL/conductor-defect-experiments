r"""Y11-ARCH-001: single-variable P2/4 architecture diagnostic on train/val only."""

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
from typing import Any

import yaml

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
P2_YAML = WORKSPACE / "configs" / "yolo11s_p2.yaml"
REFERENCE_ARGS = WORKSPACE / "runs" / "yolo11s_diag_1280_e30_20260916_104603" / "args.yaml"
ARCH_NOTE = WORKSPACE / "notes" / "Y11-ARCH-001_architecture_preflight.md"
DIFF_NOTE = WORKSPACE / "notes" / "Y11-ARCH-001_preflight_diff.md"
TRANSFER_JSON = WORKSPACE / "logs" / "Y11-ARCH-001_weight_transfer.json"

PREFLIGHT_FIELDS = (
    "epochs", "imgsz", "batch", "nbs", "workers", "device", "seed", "deterministic", "patience",
    "cache", "amp", "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "box", "cls", "dfl", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
    "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "cutmix", "copy_paste",
    "close_mosaic",
)


def verify_dataset_unchanged(output_name: str) -> dict[str, Any]:
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


def read_epoch_stability(results_csv: Path) -> dict[str, Any]:
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


def build_parameters(run_name: str) -> dict[str, Any]:
    return {
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
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "close_mosaic": 10,
    }


def audit_and_transfer_weights(official, p2) -> dict[str, Any]:
    """Load all exact matches plus remapped P3/P4/P5 box-regression branches into the shifted P2 head."""
    import torch

    source = {key: value.detach().cpu() for key, value in official.model.state_dict().items()}
    destination = p2.model.state_dict()
    exact = {
        key: value
        for key, value in source.items()
        if key in destination and tuple(value.shape) == tuple(destination[key].shape)
    }
    remapped: dict[str, torch.Tensor] = {}
    mismatches = []
    for source_branch, target_branch in ((0, 1), (1, 2), (2, 3)):
        for head_name in ("cv2", "cv3"):
            source_prefix = f"model.23.{head_name}.{source_branch}."
            for source_key, value in source.items():
                if not source_key.startswith(source_prefix):
                    continue
                target_key = source_key.replace(
                    f"model.23.{head_name}.{source_branch}.",
                    f"model.26.{head_name}.{target_branch}.",
                    1,
                )
                if target_key in destination and tuple(value.shape) == tuple(destination[target_key].shape):
                    remapped[target_key] = value
                elif target_key in destination:
                    mismatches.append({
                        "source_key": source_key,
                        "target_key": target_key,
                        "source_shape": list(value.shape),
                        "target_shape": list(destination[target_key].shape),
                    })
    source_dfl = source.get("model.23.dfl.conv.weight")
    target_dfl = "model.26.dfl.conv.weight"
    if source_dfl is not None and target_dfl in destination and tuple(source_dfl.shape) == tuple(destination[target_dfl].shape):
        remapped[target_dfl] = source_dfl

    transferred = {**exact, **remapped}
    p2.model.load_state_dict(transferred, strict=False)
    missing = sorted(key for key in destination if key not in transferred)
    p2_new = sorted(
        key for key in missing
        if key.startswith(("model.24.", "model.25.", "model.26.cv2.0.", "model.26.cv3.0."))
    )
    nc_head = sorted(
        key for key in missing
        if key.startswith(("model.26.cv3.1.", "model.26.cv3.2.", "model.26.cv3.3."))
    )
    allowed = set(p2_new) | set(nc_head)
    unexpected = sorted(key for key in missing if key not in allowed)
    backbone_unloaded = sorted(key for key in missing if key.startswith(tuple(f"model.{i}." for i in range(23))))
    if unexpected or backbone_unloaded:
        raise RuntimeError(
            f"Unexpected pretrained-weight gaps: unexpected={unexpected[:10]}, backbone={backbone_unloaded[:10]}"
        )
    audit = {
        "source_state_items": len(source),
        "destination_state_items": len(destination),
        "exact_transferred_items": len(exact),
        "remapped_detect_items": len(remapped),
        "total_transferred_items": len(transferred),
        "transferred_target_keys": sorted(transferred),
        "category_A_nc80_to_nc2_detection_classification_keys": nc_head,
        "category_B_new_p2_path_and_p2_head_keys": p2_new,
        "category_C_unexpected_unloaded_keys": unexpected,
        "shape_mismatch_mappings": mismatches,
        "backbone_or_standard_neck_unloaded_keys": backbone_unloaded,
        "source_detect_index": 23,
        "target_detect_index": 26,
        "manual_branch_mapping": {"source_0_P3": "target_1_P3", "source_1_P4": "target_2_P4", "source_2_P5": "target_3_P5"},
    }
    return audit


def inspect_forward(model, imgsz: int = 1280) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run one CUDA FP16 dummy forward and record top-level layer and Detect-input shapes."""
    import torch

    layer_rows: list[dict[str, Any]] = []
    layer_outputs: dict[int, dict[str, Any]] = {}
    detect_inputs: list[dict[str, Any]] = []
    handles = []

    def first_4d(value):
        if isinstance(value, torch.Tensor) and value.ndim == 4:
            return value
        if isinstance(value, (list, tuple)):
            for item in value:
                found = first_4d(item)
                if found is not None:
                    return found
        return None

    for index, module in enumerate(model.model):
        def hook(_module, _inputs, output, idx=index):
            tensor = first_4d(output)
            if tensor is not None:
                layer_outputs[idx] = {
                    "channels": int(tensor.shape[1]),
                    "height": int(tensor.shape[2]),
                    "width": int(tensor.shape[3]),
                    "stride": imgsz / int(tensor.shape[2]),
                }
        handles.append(module.register_forward_hook(hook))

    detect = model.model[-1]

    def detect_pre_hook(_module, inputs):
        features = inputs[0]
        for index, tensor in zip(detect.f, features):
            detect_inputs.append({
                "layer_index": int(index),
                "channels": int(tensor.shape[1]),
                "height": int(tensor.shape[2]),
                "width": int(tensor.shape[3]),
                "stride": imgsz / int(tensor.shape[2]),
            })

    handles.append(detect.register_forward_pre_hook(detect_pre_hook))
    original_training = model.training
    model.cuda().half().eval()
    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, imgsz, imgsz, device="cuda", dtype=torch.float16))
    finally:
        for handle in handles:
            handle.remove()
        model.cpu().float()
        model.train(original_training)
        torch.cuda.empty_cache()

    for index, module in enumerate(model.model):
        info = layer_outputs.get(index, {})
        layer_rows.append({
            "index": index,
            "from": module.f,
            "module": module.type.replace("ultralytics.nn.modules.", ""),
            "channels": info.get("channels", "Detect output"),
            "spatial": f"{info['height']}x{info['width']}" if info else "multiple outputs",
            "stride": info.get("stride", "multiple"),
        })
    return layer_rows, detect_inputs


def write_preflight_notes(
    parameters: dict[str, Any], run_name: str, official_rows: list[dict[str, Any]], official_detect: list[dict[str, Any]],
    p2_rows: list[dict[str, Any]], p2_detect: list[dict[str, Any]], official_info, p2_info, transfer: dict[str, Any]
) -> None:
    reference = yaml.safe_load(REFERENCE_ARGS.read_text(encoding="utf-8-sig"))

    def comparable(field: str, value):
        return str(value) if field == "device" else value

    differences = {
        field: [reference[field], parameters[field]]
        for field in PREFLIGHT_FIELDS
        if comparable(field, reference[field]) != comparable(field, parameters[field])
    }
    if differences or parameters["resume"] is not False or parameters["split"] != "val":
        raise RuntimeError(f"Training-protocol preflight failed: {differences}")

    diff_lines = [
        "# Y11-ARCH-001 单变量训练前核对",
        "",
        f"- 参考：`{REFERENCE_ARGS}`",
        "- 允许变化：检测尺度 `P3/P4/P5 → P2/P3/P4/P5`、由此产生的参数量/GFLOPs/权重迁移数量，以及独立输出路径。",
        "- 训练策略字段差异：**无**。",
        "- optimizer=auto 必须在第一个 batch 前解析为 AdamW、lr=0.001667、momentum=0.9；训练回调会强制检查。",
        "- `resume=False`；仅 train/val，test 封存。",
        f"- 计划运行名：`{run_name}`。",
        "",
        "|训练字段|Y11-DIAG-001|Y11-ARCH-001|判定|",
        "|---|---:|---:|---|",
    ]
    for field in PREFLIGHT_FIELDS:
        diff_lines.append(f"|{field}|`{reference[field]}`|`{parameters[field]}`|一致|")
    diff_lines += [
        "",
        "|架构字段|Y11-DIAG-001|Y11-ARCH-001|判定|",
        "|---|---|---|---|",
        f"|model|`yolo11s.pt`|`{P2_YAML}`|允许的唯一架构变化|",
        "|detection_scales|P3/8, P4/16, P5/32|P2/4, P3/8, P4/16, P5/32|允许的唯一架构变化|",
        "",
        f"机器核验训练策略差异：`{json.dumps(differences, ensure_ascii=False, sort_keys=True)}`",
    ]
    DIFF_NOTE.write_text("\n".join(diff_lines) + "\n", encoding="utf-8")

    lines = [
        "# Y11-ARCH-001 官方结构与 P2 架构预检查",
        "",
        "## 1. 锁定依据",
        "",
        f"- 官方源码 commit：`{EXPECTED_COMMIT}`，工作树 clean。",
        "- 官方 YAML：`ultralytics_src/ultralytics/cfg/models/11/yolo11.yaml`。",
        "- 官方定义的 Detect 输入为 layer `[16, 19, 22]`，对应 P3/8、P4/16、P5/32。",
        "",
        "## 2. 数据动机",
        "",
        "- Mosaic ON 后全部目标 min_side median ≈ 10.510 px；broken ≈ 17.928 px；loose ≈ 9.357 px。",
        "- 全部目标 `<8 px` ≈ 34.99%，`<4 px` ≈ 7.35%；loose `<8 px` ≈ 41.24%。",
        "- 标准 P3 的一格对应 8 px；P2 的一格对应 4 px。因此 P2 是有数据依据的待验证假设，并非预设有效。",
        "",
        "## 3. 标准 YOLO11s 实际逐层结构（dummy 1280×1280）",
        "",
        f"- 官方 nc=80 模型：{official_info[1]:,} parameters，{official_info[3]:.3f} GFLOPs @ 1280。",
        "",
        "|index|from|module|输出通道|空间尺寸|对应 stride|",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in official_rows:
        lines.append(f"|{row['index']}|`{row['from']}`|{row['module']}|{row['channels']}|{row['spatial']}|{row['stride']}|")
    lines += [
        "",
        "标准 Detect 实际输入：",
        "",
        "|layer|channels|shape|stride|",
        "|---:|---:|---:|---:|",
    ]
    for row in official_detect:
        lines.append(f"|{row['layer_index']}|{row['channels']}|{row['height']}×{row['width']}|{row['stride']}|")
    lines += [
        "",
        "## 4. 最小 P2 改动与结构验证",
        "",
        "- 官方 backbone 和标准 P3/P4/P5 neck layers 0–22 原样保留。",
        "- 只从标准 P3 layer 16 上采样，与 backbone P2 layer 2 拼接，再经一个 C3k2 形成 layer 25 的 P2/4 特征。",
        "- 新 Detect layer 26 输入 `[25, 16, 19, 22]`。",
        f"- P2 nc=2 模型：{p2_info[1]:,} parameters，{p2_info[3]:.3f} GFLOPs @ 1280。",
        "- dummy 1280×1280 forward 成功。",
        "",
        "P2 Detect 实际输入：",
        "",
        "|layer|channels|shape|stride|",
        "|---:|---:|---:|---:|",
    ]
    for row in p2_detect:
        lines.append(f"|{row['layer_index']}|{row['channels']}|{row['height']}×{row['width']}|{row['stride']}|")
    lines += [
        "",
        "新增/变化层：",
        "",
        "|index|from|module|输出通道|空间尺寸|stride|",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in p2_rows[23:]:
        lines.append(f"|{row['index']}|`{row['from']}`|{row['module']}|{row['channels']}|{row['spatial']}|{row['stride']}|")
    lines += [
        "",
        "## 5. 官方预训练权重迁移",
        "",
        f"- 官方 state items：{transfer['source_state_items']}；P2 state items：{transfer['destination_state_items']}。",
        f"- 同名同 shape 直接迁移：{transfer['exact_transferred_items']}。",
        f"- 因 Detect index 23→26 而人工重映射的兼容回归/DFL items：{transfer['remapped_detect_items']}。",
        f"- 总迁移：{transfer['total_transferred_items']} / {transfer['destination_state_items']}。",
        f"- A（nc=80→2 检测分类分支差异）：{len(transfer['category_A_nc80_to_nc2_detection_classification_keys'])} items。",
        f"- B（新增 P2 路径和 P2 head）：{len(transfer['category_B_new_p2_path_and_p2_head_keys'])} items，随机初始化。",
        f"- C（非预期未加载）：{len(transfer['category_C_unexpected_unloaded_keys'])} items。",
        f"- backbone/标准 neck 未加载：{len(transfer['backbone_or_standard_neck_unloaded_keys'])} items。",
        "",
        "### A. nc=80→2 导致的分类分支未加载 key",
        "",
    ]
    lines.extend(f"- `{key}`" for key in transfer["category_A_nc80_to_nc2_detection_classification_keys"])
    lines += ["", "### B. 新增 P2 路径/分支未加载 key", ""]
    lines.extend(f"- `{key}`" for key in transfer["category_B_new_p2_path_and_p2_head_keys"])
    lines += ["", "### shape mismatch 映射记录", ""]
    lines.extend(
        f"- `{item['source_key']}` {item['source_shape']} → `{item['target_key']}` {item['target_shape']}"
        for item in transfer["shape_mismatch_mappings"]
    )
    lines += [
        "",
        "## 6. 结论",
        "",
        "- 结构验证通过：四个 Detect 输入 strides 为 4/8/16/32。",
        "- 官方 backbone 和标准 P3/P4/P5 neck 权重全部迁移；没有非预期缺失。",
        "- P2 是否改善极小目标仍需由本次单变量训练验证。",
    ]
    ARCH_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    TRANSFER_JSON.write_text(json.dumps(transfer, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_model(parameters: dict[str, Any], run_name: str):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import model_info

    official = YOLO("yolo11s.pt")
    p2 = YOLO(str(P2_YAML), task="detect")
    p2.load("yolo11s.pt")  # sets ckpt so train() seeds its rebuilt trainer model from this in-memory model
    transfer = audit_and_transfer_weights(official, p2)
    official_info = model_info(official.model, verbose=True, imgsz=1280)
    p2_info = model_info(p2.model, verbose=True, imgsz=1280)
    official_rows, official_detect = inspect_forward(official.model, 1280)
    p2_rows, p2_detect = inspect_forward(p2.model, 1280)
    strides = [float(item["stride"]) for item in p2_detect]
    expected_shapes = [(320, 320), (160, 160), (80, 80), (40, 40)]
    actual_shapes = [(item["height"], item["width"]) for item in p2_detect]
    if strides != [4.0, 8.0, 16.0, 32.0] or actual_shapes != expected_shapes:
        raise RuntimeError(f"P2 structure validation failed: strides={strides}, shapes={actual_shapes}")
    if [int(x) for x in p2.model.stride.tolist()] != [4, 8, 16, 32]:
        raise RuntimeError(f"Model stride tensor mismatch: {p2.model.stride.tolist()}")
    write_preflight_notes(
        parameters, run_name, official_rows, official_detect, p2_rows, p2_detect,
        official_info, p2_info, transfer,
    )
    signature_keys = (
        "model.0.conv.weight",
        "model.16.cv1.conv.weight",
        "model.26.cv2.1.0.conv.weight",
        "model.26.dfl.conv.weight",
    )
    signature = {key: p2.model.state_dict()[key].detach().float().cpu().clone() for key in signature_keys}
    torch.cuda.empty_cache()
    return p2, transfer, {
        "layers": p2_info[0], "parameters": p2_info[1], "gradients": p2_info[2], "gflops_1280": p2_info[3],
        "detect_inputs": p2_detect, "strides": strides,
    }, signature


def main() -> None:
    if os.environ.get("RUN_Y11_ARCH_001") != "YES":
        raise RuntimeError("Y11-ARCH-001 requires explicit authorization via RUN_Y11_ARCH_001=YES.")

    settings_dir = WORKSPACE / "configs" / "ultralytics_settings"
    settings_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_dir)
    os.environ["MPLCONFIGDIR"] = str(WORKSPACE / "cache" / "matplotlib")
    os.environ["TORCH_HOME"] = str(WORKSPACE / "cache" / "torch")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    install_write_guard()

    import torch
    import ultralytics

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
        not audit["passed"] or audit["splits"]["train"]["images"] != 107
        or audit["splits"]["val"]["images"] != 23 or audit["names"] != {"0": "broken", "1": "loose"}
    ):
        raise RuntimeError("Dataset audit does not match the locked project data")

    integrity_before = verify_dataset_unchanged("dataset_integrity_before_y11_arch_001.json")
    redirect_label_caches()
    (WORKSPACE / "weights").mkdir(parents=True, exist_ok=True)
    os.chdir(WORKSPACE / "weights")
    run_name = "yolo11s_p2_diag_1280_e30_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    parameters = build_parameters(run_name)
    model, transfer, structure, signature = prepare_model(parameters, run_name)
    if "--preflight-only" in sys.argv:
        print(json.dumps({
            "preflight": "PASS", "strides": structure["strides"], "parameters": structure["parameters"],
            "gflops_1280": structure["gflops_1280"], "transferred": transfer["total_transferred_items"],
            "destination_items": transfer["destination_state_items"], "unexpected": transfer["category_C_unexpected_unloaded_keys"],
        }, ensure_ascii=False, indent=2), flush=True)
        return

    status_path = WORKSPACE / "logs" / f"{run_name}_status.json"
    batch_log = WORKSPACE / "logs" / f"{run_name}_batches.jsonl"
    status = {
        "experiment_id": "Y11-ARCH-001",
        "stage": "ArchitectureDiagnostic",
        "parent_experiment": "Y11-DIAG-001",
        "changed_variable": "detection_scales: P3/P4/P5 -> P2/P3/P4/P5",
        "success": False,
        "run_name": run_name,
        "parameters": parameters,
        "architecture": structure,
        "weight_transfer": transfer,
        "source_commit": commit,
        "source_status_before": git_status,
        "dataset_integrity_before": integrity_before,
        "console_log": os.environ.get("Y11_ARCH_001_CONSOLE_LOG", "NA"),
        "gpu_before": nvidia_memory(),
        "batches_checked": 0,
        "all_batch_losses_finite": True,
        "test_used": False,
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
        if trainer.batch_size != FIXED_BATCH or trainer.args.batch != FIXED_BATCH or trainer.args.nbs != FIXED_NBS:
            raise RuntimeError("Controlled batch/nbs changed")
        if len(trainer.train_loader.dataset) != 107 or len(trainer.test_loader.dataset) != 23:
            raise RuntimeError("Trainer did not load exactly 107 train and 23 val images")
        strides = [int(value) for value in trainer.model.stride.detach().cpu().tolist()]
        if strides != [4, 8, 16, 32]:
            raise RuntimeError(f"Training model lost P2 strides: {strides}")
        for key, expected_tensor in signature.items():
            actual_tensor = trainer.model.state_dict()[key].detach().float().cpu()
            if not torch.equal(actual_tensor, expected_tensor):
                raise RuntimeError(f"Pretrained transfer was not preserved for {key}")
        optimizer = trainer.optimizer
        optimizer_name = optimizer.__class__.__name__
        initial_lr = float(optimizer.param_groups[0].get("initial_lr", optimizer.defaults.get("lr")))
        momentum = float(optimizer.defaults.get("betas", (math.nan, math.nan))[0])
        status["optimizer_actual"] = {"name": optimizer_name, "lr0": initial_lr, "momentum": momentum}
        if optimizer_name != "AdamW" or not math.isclose(initial_lr, 0.001667, rel_tol=0, abs_tol=5e-7) or not math.isclose(momentum, 0.9, rel_tol=0, abs_tol=1e-12):
            raise RuntimeError(f"optimizer=auto mismatch: {status['optimizer_actual']}")
        status.update(save_dir=str(trainer.save_dir), controlled_parameters=actual, training_strides=strides)

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

    model.add_callback("on_train_start", verify_train_start)
    model.add_callback("on_train_epoch_start", disable_automatic_batch_reduction)
    model.add_callback("on_train_batch_end", verify_batch)
    try:
        pretrained = WORKSPACE / "weights" / "yolo11s.pt"
        with pretrained.open("rb") as stream:
            status["pretrained_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
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
        status.update(error=traceback.format_exc(), oom=("out of memory" in message or "cublas_status_alloc_failed" in message))
        raise
    finally:
        status["elapsed_seconds"] = round(time.monotonic() - started, 2)
        status["dataset_integrity_after"] = verify_dataset_unchanged("dataset_integrity_after_y11_arch_001.json")
        status["blocked_dataset_writes"] = list(BLOCKED_WRITES)
        status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
