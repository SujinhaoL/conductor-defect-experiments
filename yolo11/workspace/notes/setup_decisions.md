# 环境与只读数据设计记录

- 项目：`D:/小论文/model_projects/yolo11`。
- 原数据工程 `D:/小论文/conductor_defect_dataset` 只读；不复制数据，不调整权限，不修复原图片或标签。
- 默认 Anaconda base 为 Python 3.13.5，未安装 torch/ultralytics。已有 test 环境为 Python 3.9.24，含 torch 2.8.0+cu128，但不将其用于本项目，也不修改它。
- 使用机器上已有的 uv 管理的 Python 3.12.13，创建项目 `.venv`，无系统 Python/驱动修改。
- GPU：RTX 5070 Ti，16303 MiB，驱动 591.86，计算能力 12.0。
- 选择官方稳定组合 torch 2.8.0+cu128 / torchvision 0.23.0+cu128；不是 nightly，不追求最新版本。wheel 含 sm_120 架构。
- `nvidia-smi` 显示 CUDA 13.1 代表驱动支持能力；`torch.version.cuda=12.8` 代表本次 PyTorch 使用的运行时，两者不同属正常情况。
- [官方 PyTorch 版本配对与安装索引](https://pytorch.org/get-started/previous-versions/#v280)
- [PyTorch Blackwell / CUDA 12.8 支持说明](https://pytorch.org/blog/pytorch-2-7/)
- [CUDA 12.8 Windows 驱动要求](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/)：CUDA 12.8 GA 对应 Windows 驱动至少 570.65，当前 591.86 满足。
- 安装索引仅使用 `https://download.pytorch.org/whl/cu128`；Ultralytics editable 安装使用当前官方 clone。
- `py -0p` 不可用：未安装 Windows Python Launcher。直接使用 `.venv/Scripts/python.exe`。
- uv 在 C: 缓存与 D: 项目之间无法硬链接，已自动改为复制依赖文件；不影响运行，也不是复制数据集。
- `environment_torch.txt` 是安装过程中提前进行的探测，torchvision 当时尚未安装完成；以安装完成后的 `environment_torch_verified.txt` 和最终 `environment_after.txt` 为准。
- 初次完整 git clone 长时间停滞，主动中止后使用官方仓库浅克隆（`--depth 1`），源码完整，历史仅保存当前提交；不执行 git pull。
- 训练脚本仅通过项目侧 I/O 适配将标签 `.cache` 定位到 `workspace/cache/labels`，不改变模型、loss、增强策略或官方源码。`cache=False` 用于避免图片磁盘缓存。
- Python 审计钩子阻止向整个原数据工程的写入、删除、重命名等文件操作。训练前检查 JPEG 完整性，避免触发官方损坏图片自动修复。
- 完成后复核整个原数据工程的文件清单、文件大小和修改时间，并逐文件比较 `02_yolo11_dataset` 的 SHA256；所有校验记录只写入 workspace。

## 实际使用的 cmd 命令

```bat
cd /d D:\小论文\model_projects\yolo11
uv venv --python 3.12.13 --seed .venv
.venv\Scripts\python.exe -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
git -c http.version=HTTP/1.1 clone --depth 1 https://github.com/ultralytics/ultralytics.git ultralytics_src
.venv\Scripts\python.exe -m pip install -e ultralytics_src -c workspace\configs\torch_constraints.txt
```
