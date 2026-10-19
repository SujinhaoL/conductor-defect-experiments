# 官方 YOLO11n baseline：5 epoch Smoke Test 报告

日期：2026-09-15（北京时间）

## 结论

**训练流程 Smoke Test 成功。正式检测效果尚未建立。**

成功运行 5 个 epoch（135 个训练 batch），数据加载、CUDA 计算、反向传播、验证、权重保存和可视化输出均跑通。所有 batch 的 loss 以及 results.csv 中的数值均为有限值，没有 NaN/Inf，没有显存不足，batch 未下调。

**最终 best.pt 在 val 上的 Precision、Recall、mAP50、mAP50-95 均为 0。** 验证预测图已生成，人工查看 `val_batch0_pred.jpg` 没有可见预测框。这次只能证明软件流程可运行，不能用于宣称模型已经收敛或具有有效检测能力，也不应直接作为论文正式实验成绩。仅凭本次 smoke test 不能确定低指标的具体原因。

原数据工程保持只读；官方源码未修改；没有启动正式长训练；没有用 test 集做推理评估或调参。

## 1. 源码与环境

| 项目 | 实际值 |
|---|---|
| 项目根目录 | `D:/小论文/model_projects/yolo11` |
| 官方仓库 | https://github.com/ultralytics/ultralytics |
| 分支 | `main` |
| git commit | `9aa41d679d121afb8b4cc512d7a162f4f36886c3` |
| Ultralytics 版本 | `8.4.152` |
| 安装方式 | 在项目 `.venv` 中 `pip install -e ultralytics_src` |
| 实际导入文件 | `D:/小论文/model_projects/yolo11/ultralytics_src/ultralytics/__init__.py` |
| 克隆方式 | 官方仓库浅克隆，`--depth 1`，完整当前源码、单提交历史 |
| git status | 克隆后、训练后均干净，未执行 git pull |
| Windows | Windows 11，内部版本 `10.0.26200.9445` |
| Git | `2.54.0.windows.1` |
| Python | `3.12.13`，64 bit |
| Python 实际路径 | `D:/小论文/model_projects/yolo11/.venv/Scripts/python.exe` |
| pip | `26.2.1` |
| PyTorch | `2.8.0+cu128` |
| torchvision | `0.23.0+cu128` |
| PyTorch CUDA 运行时 | `12.8` |
| torch.cuda.is_available() | `True` |
| GPU | `NVIDIA GeForce RTX 5070 Ti` |
| 总显存 | `16303 MiB`，约 16 GB |
| NVIDIA Driver | `591.86` |
| 计算能力 / wheel 架构 | `(12, 0)` / 含 `sm_120` |
| 依赖检查 | `pip check` 通过 |

`nvidia-smi` 的 CUDA 13.1 是驱动支持能力；PyTorch 实际运行时是 12.8，两者不同正常。选型依据为 [PyTorch 官方版本配对](https://pytorch.org/get-started/previous-versions/#v280)、[Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)及 [NVIDIA CUDA 12.8 驱动要求](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)。未安装、卸载或修改系统 CUDA/显卡驱动。

GPU 证据包括：环境中的 CUDA 卷积前向/反向传播和 torchvision NMS 测试通过；训练设备为 `cuda:0`；逐 batch loss 位于 `cuda:0`。训练进程 PyTorch 峰值 allocated 为 `655.32 MiB`，reserved 为 `724 MiB`。这些数值不含所有桌面程序和驱动占用，不能当作整卡总占用。

## 2. 数据集只读检查

原数据集：`D:/小论文/conductor_defect_dataset/02_yolo11_dataset`。

项目配置：`D:/小论文/model_projects/yolo11/workspace/configs/conductor_bl_v1.yaml`。

| split | 图片 | 标签文件 | broken 框 | loose 框 | 空标签 |
|---|---:|---:|---:|---:|---:|
| train | 107 | 107 | 39 | 129 | 0 |
| val | 23 | 23 | 9 | 28 | 0 |
| test | 23 | 23 | 8 | 28 | 0 |
| 合计 | 153 | 153 | 56 | 185 | 0 |

类别固定为 `0=broken`、`1=loose`，没有 rust、normal 或 background 目标类别。

通过的检查：六个图片/标签目录存在；原 data.yaml 正确；所有图片可验证和完整解码；JPEG 结束标记完整；标签为检测格式五列；类别正确；坐标有限且归一化、边界有效；图片和标签配对；无重复标签；与 split_manifest.csv 的 SHA256、尺寸和框数量一致；无跨 split 相同图片；无跨 split 分组。

test 只进行了文件与标签完整性检查，没有用于模型评估、选模或调参。

### 数据保护证据

- 没有复制数据集到项目，也没有修改原文件的权限或内容。
- Python 审计钩子保护整个 `D:/小论文/conductor_defect_dataset`；8 类合成写入事件验证通过，验证本身不执行文件变更。
- 官方标签缓存的读写在项目脚本中重定向到 `workspace/cache/labels`，`cache=False` 禁用图片缓存。不修改官方缓存函数源文件，也不改变标签含义或增强策略。
- train/val 缓存实际为 `train_62db7ced5494.cache`、`val_b59bf0f8134a.cache`。官方扫描日志可能仍显示原始逻辑缓存路径，实际读写已重定向。
- 训练期间被阻止的数据写入次数为 0。
- 训练前后比较了整个数据工程的 **1749 个文件/目录条目**，文件清单、大小、修改时间均无变化；对 `02_yolo11_dataset` 中每个文件另做 SHA256 比较，全部相同。
- 证据：`workspace/logs/dataset_snapshot_before.json`、`dataset_integrity_after.json`、`dataset_check.json`、`write_guard_check.json`。

## 3. 训练设置

| 参数 | 值 |
|---|---|
| model | `YOLO("yolo11n.pt")`，官方 COCO 80 类预训练检测权重 |
| epochs | 5 |
| imgsz | 640 |
| batch | 4，实际保持 4 |
| workers | 0 |
| device | 0，即 CUDA:0 |
| seed | 42 |
| deterministic | True |
| plots | True |
| cache | False |
| resume | False |
| validation split | val |
| optimizer | 官方默认 auto，本次自动选 AdamW，初始 lr=0.001667 |
| 数据增强 | 当前提交官方默认配置，未人工调整 |
| 网络 / loss / trainer | 官方源码保持原样 |

模型读取 COCO 80 类权重后，官方训练过程自动将类别数设为 2，并迁移 448/499 项预训练参数。这是自定义数据集的官方流程，不是人为增加网络模块。官方 YOLO11 原本自带的 C2PSA 等模块也未改动。

权重下载地址：https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt

预训练权重位置：`D:/小论文/model_projects/yolo11/workspace/weights/yolo11n.pt`。

预训练 SHA256：`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`。

## 4. 成功运行结果

成功运行名称：`yolo11n_smoke_20260915_163741`。

训练循环及每轮验证约 81 秒；包含初始化、自检、保存和最终验证的脚本主体约 91.34 秒。

结果目录：`D:/小论文/model_projects/yolo11/workspace/runs/yolo11n_smoke_20260915_163741`。

best.pt：`D:/小论文/model_projects/yolo11/workspace/runs/yolo11n_smoke_20260915_163741/weights/best.pt`。

last.pt：`D:/小论文/model_projects/yolo11/workspace/runs/yolo11n_smoke_20260915_163741/weights/last.pt`。

两份权重均已重新加载验证：检测任务正确、类别正确、参数无 NaN/Inf。结果目录中 17 张图片均通过文件解码验证；人工查看了训练 batch 和验证预测图。

| 检查项 | 结果 |
|---|---|
| 实际加载数据 | 107 train + 23 val，无背景、无损坏 |
| 实际训练轮数 | 5 |
| loss 批次检查 | 135/135 通过 |
| results.csv | 5 行 epoch 记录，所有数值有限 |
| best.pt / last.pt | 均生成、可加载 |
| 训练可视化 | train_batch0.jpg / 1 / 2 |
| 验证标签与预测图 | val_batch0/1/2_labels.jpg 和对应 pred.jpg |
| 其他图 | results.png、labels.jpg、PR/P/R/F1 曲线、混淆矩阵 |
| 最终 val mAP50 / mAP50-95 | 0 / 0 |
| test 模型评估 | 未运行 |

| loss | epoch 1 | epoch 5 |
|---|---:|---:|
| train/box_loss | 4.41567 | 4.08762 |
| train/cls_loss | 20.5507 | 12.1559 |
| train/dfl_loss | 2.25546 | 1.57553 |

训练 loss 总体有所下降，但这不意味着已经收敛；最终 val 指标为 0。第 3 轮曾出现接近零的 mAP50（CSV 为 0.00002），不具备正式效果解释价值。

## 5. 重要提示、warning 与已修复问题

1. 完整历史 git clone 停滞，已保留未完成目录到 `workspace/logs/incomplete_clone_20260915_163254`，随后使用官方浅克隆成功。未删除已有数据，未 git pull。
2. `py -0p` 不可用，因为 Windows Python Launcher 未安装；直接使用项目虚拟环境解释器解决。
3. uv 跨磁盘硬链接提示：自动改为复制依赖文件，环境可正常运行。
4. PyTorch 安装过程中曾提前探测，torchvision 尚未完成安装。安装结束后前向、反向与 NMS 检查全部通过；最终以 `environment_after.txt` 为准。
5. 第一次训练尝试 `yolo11n_smoke_20260915_163655` 在第一个 batch 后因项目 loss 记录回调误将字典当 Tensor 使用而停止。只修复 `workspace/scripts/train_yolo11.py` 的记录逻辑；随后从原始 `yolo11n.pt` 重新训练完整 5 epoch。不是 resume，没有改官方源码、数据集或超参数。失败日志和结果目录保留。
6. 官方 AMP 自检会另外下载 `yolo26n.pt`。它只用于混合精度兼容性检查；本次训练模型和初始化权重始终是 YOLO11n。
7. `Overriding model.yaml nc=80 with nc=2`、预训练参数部分迁移、冻结 `dfl.conv.weight`、auto optimizer 选择提示均来自官方默认训练流程。
8. 成功训练日志没有 Warning/UserWarning 行，没有 OOM、NaN 或 Inf 异常；未屏蔽 warnings。
9. 官方结束训练时会移除 best/last 中的优化器状态。本次已完成的 last.pt 不应被当作一个仍可原样断点续训的中断快照。未来真正中断的训练才使用其保留状态的 last.pt 和 `resume=True`。
10. 部分官方 Windows 控制台行会省略中文字符；实际导入路径、文件位置和 args.yaml 均指向正确的 `D:/小论文/...` 目录，已核实。

## 6. 文件与日志

- `workspace/configs/conductor_bl_v1.yaml`：本项目数据配置。
- `workspace/configs/torch_constraints.txt`：PyTorch/torchvision 版本约束。
- `workspace/configs/ultralytics_settings/Ultralytics/settings.json`：项目独立设置。
- `workspace/scripts/train_yolo11.py`：固定 5 epoch 的训练入口，有 Windows main 保护。
- `workspace/scripts/run_smoke.py`、项目根 `run_smoke.cmd`：运行并完整记录控制台输出。
- `workspace/scripts/readonly_dataset.py`、`verify_write_guard.py`：只读保护、缓存路径适配及验证。
- `workspace/scripts/check_dataset.py`：数据审计与前后完整性比较。
- `workspace/scripts/inspect_environment.py`、`configure_settings.py`、`prepare_source.py`：环境、项目设置与源码记录。
- `workspace/scripts/finalize_smoke_report.py`：产物、权重、环境冻结与数据完整性复核，**不训练**。
- `workspace/logs/git_info.txt`、`git_status_after.txt`：源码提交及干净状态。
- `workspace/logs/environment_before.txt`、`environment_after.txt`：安装前后环境。
- `workspace/logs/install_torch.log`、`install_ultralytics.log`、`requirements_freeze.txt`、`pip_check.txt`：安装和依赖记录。
- `workspace/logs/smoke_console_20260915_163738.log`：成功运行完整日志。
- `workspace/logs/yolo11n_smoke_20260915_163741_batches.jsonl`：逐 batch 的 CUDA、loss、有限值检查。
- `workspace/logs/latest_smoke_status.json`、`artifact_verification.json`、`final_summary.json`：运行与产物验证。
- `workspace/notes/setup_decisions.md`：兼容性选择及安装命令。
- `workspace/notes/smoke_test_report.md`：本报告。
- `.venv`、`ultralytics_src`、`workspace/weights`、`workspace/cache`、`workspace/runs`：独立环境、源码、官方权重、项目缓存与结果。

## 7. 下一步

先查看训练 batch 中的标注位置，以及 val 的真实标签图与预测图，确认对数据和可视化的理解。保留本次源码提交、环境锁定记录和数据划分作为后续对照的基础。

下一阶段应先确定正式 baseline 的训练轮数、模型大小和统一评估规则，再进行充分训练并观察 val 上的收敛与错误案例。原始高分辨率图像经 640 缩放后有较小的缺陷，后续可以先做只读尺寸分析来决定正式实验方案；本阶段未改变 imgsz、增强或模型结构。test 保留到最终方案确定后做一次正式评估。

手动再次执行同样的 5 epoch smoke test（会创建新的时间戳结果目录）：

```bat
cd /d D:\小论文\model_projects\yolo11
run_smoke.cmd
```

该命令每次都从官方 COCO `yolo11n.pt` 开始；没有自动正式训练计划。
