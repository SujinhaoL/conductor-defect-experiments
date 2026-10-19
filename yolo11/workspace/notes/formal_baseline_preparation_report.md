# FormalBaseline 执行系统准备报告

核验时间：2026-09-17T10:57:33.583432（本机北京时间）。项目：`D:\小论文\model_projects\yolo11`。

**结果：执行系统已建立，三个实验的最终 preflight / dry-run 均通过。没有启动任何正式训练。等待用户/GPT 审阅后，第一轮才执行 Y11-BASE-001-S42。**

## A. 实际参考运行

`D:\小论文\model_projects\yolo11\workspace\runs\yolo11s_diag_1280_e30_20260916_104603`

该路径从唯一登记表 Runs 中的 `Y11-DIAG-001` 行读取，没有按名称猜测。其 stage 仍是 Diagnostic。

## B. 实际读取的原始证据

- args：`D:\小论文\model_projects\yolo11\workspace\runs\yolo11s_diag_1280_e30_20260916_104603\args.yaml`
- results：`D:\小论文\model_projects\yolo11\workspace\runs\yolo11s_diag_1280_e30_20260916_104603\results.csv`
- status：`D:\小论文\model_projects\yolo11\workspace\logs\yolo11s_diag_1280_e30_20260916_104603_status.json`
- 控制台：`D:\小论文\model_projects\yolo11\workspace\logs\yolo11s_diagnostic_console_20260916_104555.log`
- 登记表：[experiment_registry.xlsx](D:/小论文/model_projects/yolo11/workspace/experiment_tracking/experiment_registry.xlsx)；完整读取 65 列、FieldDefinitions 和既有 6 个实验。
- 参考 CSV 核对：30 epochs，best epoch=28，P=0.007310、R=0.035710、mAP50=0.021410、mAP50-95=0.009640。
- 原 args SHA256：`1e5ab8721ab9de426a2fd32ff3f4a44f3acc6ac3565ec63b1b3acfcad87328b4`。

## C. 从 args.yaml 冻结的参数

协议：[formal_baseline_protocol.yaml](D:/小论文/model_projects/yolo11/workspace/configs/formal_baseline_protocol.yaml)。自动提取并冻结 **112 项**，没有人工重建短参数清单。
仅豁免输出身份字段 `project`、`name`、`save_dir`；timestamp 只用于输出目录名，不属于训练参数。其余字段（包括暂不参与检测训练的通用配置项）均保留并参加 diff。

| 参数 | 冻结值 |
|---|---|
| model / pretrained / resume | `yolo11s.pt` / `True` / `False` |
| data | `D:\小论文\model_projects\yolo11\workspace\configs\conductor_bl_v1.yaml`，沿用原配置 |
| imgsz / batch / nbs / epochs | 1280 / 4 / 64 / 30 |
| workers / device / deterministic | 0 / `0` / `True` |
| seed | 母版 42；ID 白名单分别固定 42、3407、2026 |
| optimizer | `auto` |
| lr0 / lrf / momentum（调用参数） | 0.01 / 0.01 / 0.937 |
| weight_decay / warmup_epochs | 0.0005 / 3.0 |
| warmup_momentum / warmup_bias_lr（调用参数） | 0.8 / 0.1 |
| mosaic / close_mosaic | 1.0 / 10 |
| mixup / cutmix / copy_paste | 0.0 / 0.0 / 0.0 |
| fliplr / flipud / translate / scale | 0.5 / 0.0 / 0.1 / 0.5 |
| hsv_h / hsv_s / hsv_v | 0.015 / 0.7 / 0.4 |
| degrees / shear / perspective | 0.0 / 0.0 / 0.0 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| patience / amp / cache / rect / cos_lr | 0 / True / False / False / False |
| save / save_period / plots / split / exist_ok | True / 10 / True / val / False |

**auto 优化器口径：**母版日志实际为 AdamW、lr=0.001667、momentum=0.9、衰减组 weight_decay=0.0005；冻结 YAML 仍保留原调用值 lr0=0.01、momentum=0.937。正式运行开始时会实测并核对实际优化器，不能把调用前数值写成实际值。

**官方运行时行为：**锁定源码 `ultralytics/engine/trainer.py:1146` 在 `build_optimizer(auto)` 中将运行时 warmup_bias_lr 设为 0.0，而 args.yaml 保存发生在该步骤前。因此冻结输入保持 0.1；运行时另作严格检查并记录这一个官方派生变化。没有把 0.0 写回协议，没有扩大调用参数的 seed-only 豁免范围。

预训练文件：`D:\小论文\model_projects\yolo11\workspace\weights\yolo11s.pt`。SHA256：`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`，与 DIAG-001 状态记录一致。
补充 CPU 反序列化检查确认该现有权重为 DetectionModel，scale=s，COCO 80 类，Detect strides=[8,16,32]，backbone/head 与锁定官方 YAML 相同。没有调用 forward、predict、val 或 train。三个 dry-run 本身不构造模型。

## D. 新建文件与关键实现

| 文件 | 作用 |
|---|---|
| [formal_baseline_protocol.yaml](D:/小论文/model_projects/yolo11/workspace/configs/formal_baseline_protocol.yaml) | 原 args 全字段冻结、参考证据哈希、环境、权重和三 ID/seed 映射 |
| [run_formal_baseline.py](D:/小论文/model_projects/yolo11/workspace/scripts/run_formal_baseline.py) | 单 ID 执行入口；dry-run/preflight、防漂移、只读保护、固定官方训练、仅 val 后处理与 status |
| [formal_baseline_registry.py](D:/小论文/model_projects/yolo11/workspace/scripts/formal_baseline_registry.py) | 在原 Excel 中安全追加/更新当前行，先备份，保留历史格式，同步 CSV |
| [summarize_formal_baseline.py](D:/小论文/model_projects/yolo11/workspace/scripts/summarize_formal_baseline.py) | 三个 Completed 且证据一致后才生成 individual + mean/std；ddof=1 |
| [test_formal_baseline.py](D:/小论文/model_projects/yolo11/workspace/scripts/test_formal_baseline.py) | 15 项安全回归测试；临时副本和合成事件，不训练、不改真实登记表 |
| [formal_baseline_changes.patch](D:/小论文/model_projects/yolo11/workspace/notes/formal_baseline_changes.patch) | 上述 5 个新增配置/脚本的完整 unified diff，供 GPT/人工审阅 |
| [formal_baseline_preparation_report.md](D:/小论文/model_projects/yolo11/workspace/notes/formal_baseline_preparation_report.md) | 本准备报告 |
| [audit_summary.json](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_system_audit_20260917_105717/audit_summary.json) | 最终系统审计及三次 dry-run 证据路径 |

没有修改既有训练/分析脚本、PROJECT_STATE.md、旧报告、真实 registry.xlsx / registry.csv 或历史 run。没有创建第二套登记系统。测试用的工作簿是临时副本，测试结束后由临时目录管理器清理。

已复用 `check_dataset.snapshot()`、`readonly_dataset.install_write_guard()` 和 `redirect_label_caches()`；没有运行会覆盖旧审计日志的 check_dataset.main()。现有 build_experiment_registry.mjs 会重建工作簿，因此不用于本次安全追加。

项目环境新增 `openpyxl==3.1.5` 与其依赖 `et_xmlfile==2.0.0`，以满足明确要求的 openpyxl 写入方式。Python/PyTorch/CUDA/Ultralytics 核心版本未改，pip check 通过。

### 执行边界与登记流程

- CLI 只接受白名单 experiment-id 和可选 --dry-run，没有 seed、epochs、model、resume 或其他覆盖参数入口。每次只处理一个 ID，不连续训练三次。
- preflight 同时比较冻结字典、官方 get_cfg 解析后的完整字典；真正 model.train() 前再次比较精确调用字典。未经允许的差异会停止并输出具体字段，不自动修正。
- 全局独占锁阻止并发运行；已有登记 ID、已有同 ID 的 run、已存在的目标目录均拒绝。新 run 目录用 mkdir(exist_ok=False) 原子预留，保持 exist_ok=False。
- 中央整个目录写入保护和项目内缓存适配沿用成熟代码，另增加官方源码写入保护及模型阶段 test 图像/标签读取阻断。
- 训练启动前向原表追加一个 Running 行，备份 XLSX 和 CSV；正常结束只更新这同一行。可捕获异常或 Ctrl+C 记为 Failed，中断细节写入 notes/status；不改 FieldDefinitions。
- 对新增行保留既有样式并扩展 RunsTable。先保存到临时文件，重新读取核对历史单元格、格式、schema 和 CSV 对应关系，再替换；CSV 替换失败时恢复原 XLSX。
- run 内保留原生权重/args/CSV/曲线/混淆矩阵和 frozen_protocol_diff.txt；日志保存在 workspace/logs/<ID_timestamp>/，包括带 ID+timestamp 的 console.log、status.json、batch 日志。
- best epoch 按登记定义先取 mAP50-95，并列按 mAP50；完全相同取最早轮次。P/R/AP 取同一行，独立最高 P/R 另列。
- 使用 best.pt 仅对固定 val 复验，固定 imgsz=1280、batch=8、half=True、rect=False、conf=0.001、iou=0.7、max_det=300。训练 batch 始终为 4；这是单独的验证协议，不是训练参数变化。
- 记录实测 optimizer/初始 lr/momentum/衰减参数组、accumulate、参数量、1280 GFLOPs（无法可靠取得则 NA）、训练峰值显存和耗时，并完成训练后完整性检查才标记 Completed。
- stage=FormalBaseline、test_used=False、include_in_paper=Candidate；parent_experiment=Y11-DIAG-001；changed_variable 严格按要求写入。
- 汇总需三 ID 均 Completed，CSV/status/权重哈希/协议/登记信息完整一致；缺失或 NA 不冒充 0，也不选最佳 seed。

### 关键代码位置

- [assert_protocol_diff](D:/小论文/model_projects/yolo11/workspace/scripts/run_formal_baseline.py:131)
- [assert_effective_training_args](D:/小论文/model_projects/yolo11/workspace/scripts/run_formal_baseline.py:141)
- [preflight](D:/小论文/model_projects/yolo11/workspace/scripts/run_formal_baseline.py:320)
- [execute_training](D:/小论文/model_projects/yolo11/workspace/scripts/run_formal_baseline.py:404)
- [update_registry](D:/小论文/model_projects/yolo11/workspace/scripts/formal_baseline_registry.py:127)
- [collect_completed](D:/小论文/model_projects/yolo11/workspace/scripts/summarize_formal_baseline.py:40)
- [calculate_statistics](D:/小论文/model_projects/yolo11/workspace/scripts/summarize_formal_baseline.py:27)

## E. 官方源码

**没有修改 ultralytics_src，没有 git pull。** 当前 commit=`9aa41d679d121afb8b4cc512d7a162f4f36886c3`，git status clean。

## F. 中央数据与 test

**没有修改中央数据集、没有重新划分数据、没有新增类别。** 三次最终 preflight 都复用历史快照验证 1749 个条目无变化，并核对 split_manifest、图片哈希、文件名和数量。train/val/test=107/23/23，类别仅 broken/loose。
按本次授权，完整性检查读取了 test 文件的元数据与 SHA256；没有对 test 解码、模型预测、验证、AP 计算或阈值选择。

## G. 最终 preflight 结果

| 实验 | 结果 | 证据 |
|---|---|---|
| Y11-BASE-001-S42 | PASS | [status.json](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-001-S42_20260917_105717_782388/status.json)；[diff](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-001-S42_20260917_105717_782388/frozen_protocol_diff.txt) |
| Y11-BASE-002-S3407 | PASS | [status.json](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-002-S3407_20260917_105725_608290/status.json)；[diff](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-002-S3407_20260917_105725_608290/frozen_protocol_diff.txt) |
| Y11-BASE-003-S2026 | PASS | [status.json](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-003-S2026_20260917_105728_991016/status.json)；[diff](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-003-S2026_20260917_105728_991016/frozen_protocol_diff.txt) |

15 项安全测试全部通过：seed-only 差异、未授权参数/缺字段/新增字段/未知 ID 拒绝、已存在 ID/目录拒绝、best 选取规则、历史行保护、原工作簿副本追加与完成更新、格式/工作表保留、CSV 同步和失败回滚、独占锁、dry-run 不触及训练入口、官方 auto 派生值单独核验、三 seed 样本标准差与缺失阻断、数据/源码写入保护及 test 模型读取保护。
测试没有执行 model.train()。真实 registry.xlsx 与 registry.csv 哈希未变，仍为原 6 个实验；runs 仍是原 7 个目录。
审计过程曾有一次 S2026 dry-run 在 S3407 进程尚未结束时触发独占锁拒绝，未进入训练。该失败审计日志保留；等待前一进程结束后已顺序重跑，并以上表最终三次 PASS 为准。

## H. 三组参数 diff

| 实验 | 相对 Y11-DIAG-001 的训练参数差异 |
|---|---|
| Y11-BASE-001-S42 | `{}`，无差异 |
| Y11-BASE-002-S3407 | 仅 `seed: 42 -> 3407` |
| Y11-BASE-003-S2026 | 仅 `seed: 42 -> 2026` |
输出路径和运行名按新 run 生成，不参与训练参数比较。无 nbs、mosaic、epochs、optimizer、学习率、loss 或结构漂移。

## I. 尚未进行的验证与需要注意的边界

1. **本次没有实际训练。** GPU 训练与真实训练完成后的完整联动流程尚未端到端执行；已完成静态审计、15 项安全测试、临时工作簿副本测试和三个真实环境 dry-run。首轮应在审阅后单独运行 S42，不能把准备通过理解为已有 FormalBaseline 成绩。
2. 如果进程被操作系统强制终止、断电或登记表被 Excel/WPS 锁定，可能留下 Running 状态或锁文件；脚本会拒绝自动重跑，需要根据日志核对后处理。可捕获异常和 Ctrl+C 会尝试记录 Failed。XLSX/CSV 为两个文件，常规替换失败有回滚；断电无法保证跨文件原子性，后续 preflight 会检查两者一致性并停止。
3. 准备报告和 diff 需要交给用户/GPT 审阅。所有命令均是后续操作，本次未执行训练命令。

## J. 后续可执行的 CMD 命令（本次未执行）

每次只运行一条。审核确认后先运行 S42；本报告不自动授权或启动后两次。命令使用绝对路径，可在 CMD 任意目录执行。

### Y11-BASE-001-S42

```bat
"D:\小论文\model_projects\yolo11\.venv\Scripts\python.exe" -B "D:\小论文\model_projects\yolo11\workspace\scripts\run_formal_baseline.py" --experiment-id Y11-BASE-001-S42
```

### Y11-BASE-002-S3407

```bat
"D:\小论文\model_projects\yolo11\.venv\Scripts\python.exe" -B "D:\小论文\model_projects\yolo11\workspace\scripts\run_formal_baseline.py" --experiment-id Y11-BASE-002-S3407
```

### Y11-BASE-003-S2026

```bat
"D:\小论文\model_projects\yolo11\.venv\Scripts\python.exe" -B "D:\小论文\model_projects\yolo11\workspace\scripts\run_formal_baseline.py" --experiment-id Y11-BASE-003-S2026
```

若仅需再次检查，在上述命令末尾加 `--dry-run`，不会训练，也不会登记 FormalBaseline 行。

三次均完整完成后，可单独运行汇总脚本：

```bat
"D:\小论文\model_projects\yolo11\.venv\Scripts\python.exe" -B "D:\小论文\model_projects\yolo11\workspace\scripts\summarize_formal_baseline.py"
```

预期汇总产物（当前尚未生成）：`workspace/analysis/formal_baseline_3seed_summary.csv`、`workspace/analysis/formal_baseline_3seed_metrics.json`、`workspace/notes/formal_baseline_3seed_report.md`。

## 交付文件哈希

| 文件 | SHA256 |
|---|---|
| `workspace/configs/formal_baseline_protocol.yaml` | `367ff7f7f60941ee8b6c34c24d8d36f3382345b04733b534f91ab30db5e39def` |
| `workspace/scripts/run_formal_baseline.py` | `328eb13c8e0aa298b56be3c9bd680c352c0308b38ca63bf81a13697091468705` |
| `workspace/scripts/formal_baseline_registry.py` | `7e2044b4bfed4f15e8dafb6b0698c79f75eb24e80e988503e976907fabb755e5` |
| `workspace/scripts/summarize_formal_baseline.py` | `c22e5d40bb9b00a9ac157771f459778c618ca844a0d15fc07639a308bbe93501` |
| `workspace/scripts/test_formal_baseline.py` | `43fa6d40a24b331dfeba82d524e0b38d8532621193790176f809e9d79c19bf46` |
| 原始 experiment_registry.xlsx（未改） | `05ffb6568a64672aeeca6ac8eed4762270163a4583da1d3136f736021253fc69` |
| 原始 experiment_registry.csv（未改） | `78bd609853f5763f81e2797bc2f1a1433b054a300e634e959729de9dbb6d5204` |

**本次工作到此停止，没有调用 model.train()，没有开始 Y11-BASE-001-S42 或连续三 seed 训练。**
