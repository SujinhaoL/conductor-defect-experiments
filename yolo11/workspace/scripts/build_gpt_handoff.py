"""Package existing smoke-test evidence for review. Does not train or edit inputs."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE = PROJECT / "workspace"
OUTPUT = WORKSPACE / "handoff_to_gpt" / "smoke_review_20260915"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    status = json.loads((WORKSPACE / "logs" / "latest_smoke_status.json").read_text(encoding="utf-8"))
    run = Path(status["save_dir"])
    dataset_check = json.loads((WORKSPACE / "logs" / "dataset_check.json").read_text(encoding="utf-8"))
    commit = status["git_commit"]
    source = PROJECT / "ultralytics_src"
    assert subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() == commit
    assert not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()

    # Read only manifest metadata; do not copy the central dataset into the project.
    with (Path(status["dataset"]) / "split_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dimensions = {}
    for split in ("train", "val", "test"):
        counts = Counter(f"{row['width']}x{row['height']}" for row in rows if row["split"] == split)
        dimensions[split] = {"images": sum(counts.values()), "resolution_counts": dict(counts.most_common())}
    (OUTPUT / "image_dimensions_summary.json").write_text(json.dumps(dimensions, ensure_ascii=False, indent=2), encoding="utf-8")

    handoff = f"""YOLO11 baseline 阶段交接：请先审核，再制定下一步 Codex 提示词

一、当前做到哪里
已完成 yolo11 项目、官方 git clone、独立 .venv、editable 源码安装、workspace、中央数据集只读检查、YOLO11n 5 epoch GPU Smoke Test，以及训练产物复核。
当前阶段停止；正式 baseline 尚未开始。没有改 YOLO11 结构、loss 或官方数据增强策略，没有启动 300 epoch，没有用 test 集做模型评估或调参。

二、源码与环境
项目：{PROJECT}
官方仓库：https://github.com/ultralytics/ultralytics
分支：main；commit：{commit}
官方源码浅克隆 --depth 1；源码完整、历史仅当前提交；未 git pull；本次交接前 git status 再检查为空。
Python：3.12.13；解释器：{status['python_executable']}
PyTorch：{status['torch']}；torchvision：0.23.0+cu128；CUDA 运行时：{status['cuda']}
Ultralytics：{status['ultralytics']}，editable 安装。
实际导入：{status['ultralytics_file']}
GPU：{status['gpu']}，16303 MiB 显存，NVIDIA 驱动 591.86，计算能力 12.0。
GPU 前向、反向和 torchvision NMS 测试通过。没有修改系统驱动或系统 CUDA。

三、数据集
{status['dataset']}
train=107，val=23，test=23；类别仅 0=broken、1=loose。
框数量：train broken=39/loose=129，val broken=9/loose=28，test broken=8/loose=28。
没有 rust、normal 类或纯负样本。数据集图片/标签/清单匹配、坐标、图片 SHA256 和分组划分检查均通过。
图像包含 5472×3648 等多种分辨率，不能把所有图片视为固定尺寸；实际分辨率计数见 image_dimensions_summary.json。
整个 D:/小论文/conductor_defect_dataset 均必须只读。本次训练前后完整性复核无变化。
项目自建 conductor_bl_v1.yaml 指向中央数据集；没有复制数据集。

四、实际 smoke 设置与结果
模型初始化：YOLO("yolo11n.pt")，官方 COCO 预训练检测权重，非 last.pt 续训。
epochs=5，imgsz=640，batch=4，workers=0，device=0，seed=42，deterministic=True，plots=True，cache=False。
保留官方默认 optimizer=auto，日志显示本次选 AdamW(lr=0.001667, momentum=0.9)。args.yaml 记录初始化参数，实际运行需结合日志阅读。
本次官方默认增强值包括 mosaic=1.0、close_mosaic=10、scale=0.5、fliplr=0.5；没有为 5 epoch 人工修改增强。
成功运行：{run.name}
135 个训练 batch 全部 loss 为有限值；没有 NaN/Inf 或 OOM。实际设备 cuda:0；batch 保持 4。
训练脚本主体耗时约 {status['elapsed_seconds']} 秒。
PyTorch 峰值 allocated={status['peak_allocated_MiB']} MiB，reserved={status['peak_reserved_MiB']} MiB。
这只是 YOLO11n / 640 / batch4 的实测，不能直接推算 s/m 或 960/1280 的可用 batch；也不代表整卡总占用。
best.pt、last.pt 可重新加载、类别正确、参数无 NaN/Inf；训练 batch、验证标签/预测、曲线均已生成。

重要：最终 best.pt 的 val Precision、Recall、mAP50、mAP50-95 都为 0；首批验证预测图没有可见预测框。
train/box_loss：4.41567 -> 4.08762；train/cls_loss：20.5507 -> 12.1559；train/dfl_loss：2.25546 -> 1.57553。
所以结论仅是“软件/训练管线通过 Smoke Test”，不是“已经训练出了有效检测器”。
不能仅凭 epoch 少就确定 mAP=0 的原因；请先检查脚本、实际配置、预训练迁移、标签图、预测图、日志和指标。

五、必须了解的实施细节
1. train_yolo11.py 依赖 readonly_dataset.py 和 check_dataset.py。请一起审查，不能只看单个入口。
2. 项目侧 Python 审计钩子限制数据写入；缓存 I/O 在运行时重定向到 workspace/cache/labels。没有修改任何官方源文件，也未替换训练循环、loss 或增强逻辑。
3. 官方日志中的 labels/train.cache、val.cache 是逻辑路径，实际缓存位于 workspace/cache/labels；请结合重定向代码和完整性证据检查。
4. 脚本中的 trainer.test_loader 是官方验证 dataloader 属性名；本次加载的是 val=23 张，并未使用 test split。
5. 官方 AMP 自检自动下载 yolo26n.pt，只用于自检，训练模型始终是 YOLO11n。
6. 初次运行 yolo11n_smoke_20260915_163655 因项目 loss 记录回调把 dict 当 Tensor 使用，在首个 batch 后中止。修正项目回调后从 COCO 权重重新跑满 5 epoch；不是 resume。失败日志保留，仅用于说明修复经过。
7. 本版本官方 loss_items 为 dict；官方默认 C2PSA 和自定义两类检测头适配均保留，未人为增加 attention。
8. 正常结束时官方移除了 best/last 中的优化器状态；本次已完成的 last.pt 不应被当作完整中断续训快照。
9. 成功训练控制台没有 Warning/UserWarning 行；没有屏蔽 warnings。安装/clone 的已处理提示见 smoke_test_report.md。

六、原始结果位置
结果目录：{run}
best.pt：{status['best_pt']}
last.pt：{status['last_pt']}
报告：{WORKSPACE / 'notes' / 'smoke_test_report.md'}

七、请 GPT 接下来做什么
第一件：审查 train_yolo11.py、其依赖脚本、conductor_bl_v1.yaml、实际 args.yaml、完整日志、results.csv 和图像。
请明确区分：已证实的错误、需要进一步验证的疑点，以及仅因 smoke epoch 少而尚不能下结论的现象。
第二件：审核后，再结合 RTX 5070 Ti 16GB、高清图与小缺陷特征，规划第一组正式 baseline 的 n/s/m、640/960/1280、batch、epochs、数据增强和统一评估规则。
当前并未为这些候选组合测量显存或正式比较效果；如需补充证据，先给出有边界的只读分析或小规模试跑提示词。
本次先不要直接执行正式训练，也不要更改 YOLO11 或中央数据集。待我把你的下一步 Codex 提示词发给 Codex 后，再按新的范围操作。
"""
    (OUTPUT / "给GPT的交接说明.txt").write_text(handoff, encoding="utf-8")

    text_files = [
        *[(f"scripts/{name}", WORKSPACE / "scripts" / name) for name in (
            "train_yolo11.py", "readonly_dataset.py", "check_dataset.py", "run_smoke.py",
            "configure_settings.py", "inspect_environment.py", "verify_write_guard.py", "finalize_smoke_report.py")],
        ("configs/conductor_bl_v1.yaml", WORKSPACE / "configs" / "conductor_bl_v1.yaml"),
        ("configs/torch_constraints.txt", WORKSPACE / "configs" / "torch_constraints.txt"),
        ("outputs/args.yaml", run / "args.yaml"),
        ("outputs/results.csv", run / "results.csv"),
        *[(f"logs/{name}", WORKSPACE / "logs" / name) for name in (
            "git_info.txt", "git_status_after.txt", "environment_before.txt", "environment_after.txt",
            "requirements_freeze.txt", "pip_check.txt", "dataset_check.json", "dataset_integrity_after.json",
            "write_guard_check.json", "latest_smoke_status.json", "artifact_verification.json", "training_warnings.txt",
            "gpu_during_smoke.csv", "smoke_console_20260915_163738.log", "smoke_console_20260915_163652.log",
            "yolo11n_smoke_20260915_163741_batches.jsonl")],
        ("notes/smoke_test_report.md", WORKSPACE / "notes" / "smoke_test_report.md"),
        ("notes/setup_decisions.md", WORKSPACE / "notes" / "setup_decisions.md"),
        ("source_reference/default.yaml", source / "ultralytics" / "cfg" / "default.yaml"),
        ("source_reference/yolo11.yaml", source / "ultralytics" / "cfg" / "models" / "11" / "yolo11.yaml"),
    ]
    sections = [handoff, "\n以下各节为实际文件内容；源文件副本在 originals/ 内。控制台合并文本仅去除 ANSI 颜色控制符。\n"]
    manifest = []
    for relative, original in text_files:
        destination = OUTPUT / "originals" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        content = original.read_text(encoding="utf-8-sig")
        if original.suffix == ".log":
            content = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", content)
        sections.append(f"\n{'=' * 72}\nFILE: {relative}\nORIGINAL: {original}\n{'=' * 72}\n{content}\n")
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        manifest.append({"attachment": str(destination.relative_to(OUTPUT)), "original": str(original), "sha256": digest})
    sections.append("\nIMAGE_DIMENSIONS_SUMMARY\n" + json.dumps(dimensions, ensure_ascii=False, indent=2))
    (OUTPUT / "GPT审查材料_脚本配置日志.txt").write_text("\n".join(sections), encoding="utf-8")

    primary_images = ["results.png", "train_batch0.jpg", "train_batch1.jpg", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]
    for name in primary_images:
        shutil.copyfile(run / name, OUTPUT / name)
    for original in sorted(run.iterdir()):
        if original.suffix in {".jpg", ".png"} and original.name not in primary_images:
            destination = OUTPUT / "more_images" / original.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
    (OUTPUT / "attachment_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = OUTPUT.parent / "yolo11_smoke_review_20260915.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                zipped.write(path, arcname=path.relative_to(OUTPUT).as_posix())
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        count = len(zipped.infolist())
    print(json.dumps({"folder": str(OUTPUT), "zip": str(archive), "zip_bytes": archive.stat().st_size,
                      "archive_files": count, "primary_upload_text": str(OUTPUT / "GPT审查材料_脚本配置日志.txt"),
                      "primary_text_bytes": (OUTPUT / "GPT审查材料_脚本配置日志.txt").stat().st_size,
                      "primary_images": primary_images}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
