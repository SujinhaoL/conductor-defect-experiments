# 单次 Y11-BASE-001 准备与 dry-run 报告

核验时间：2026-09-17T19:56:47.527078（北京时间）。本轮修改完成，唯一一次真实环境 dry-run 通过；没有执行训练、val 或 test。

## 当前方案

唯一实验 ID 是 `Y11-BASE-001`，它不是模型文件。参数不再取自 DIAG-001。原 three-seed 方案在正式训练前取消，旧日志、旧准备报告、旧 patch、六项历史实验、历史 PROJECT_STATE.md 均保留。

修改前版本和历史文件哈希清单：[备份目录](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_plan_change_20260917_194422)。除五个经授权的现有配置/脚本外，所有本轮开始前的 workspace 文件 SHA256 均保持不变；没有删除任何旧文件。历史 runs 仍为 7 个目录，登记表仍为 6 项已完成实验，没有新增 Running 行。

## 模型和数据

- 官方预训练文件：`D:\小论文\model_projects\yolo11\workspace\weights\yolo11s.pt`。
- SHA256：`85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`。
- 对经 SHA256 核验的权重作 CPU 元数据检查：DetectionModel、COCO 80 类、scale=s、标准 Detect strides=[8,16,32]，backbone/head 与锁定官方 YAML 一致。没有 forward/train/val/predict。
- 中央 `data.yaml` 与项目 `conductor_bl_v1.yaml` 解析内容完全一致；继续使用项目 YAML，二者都指向固定中央数据。
- train/val/test=107/23/23，类别 0=broken、1=loose。历史快照 1749 个条目核对通过；split_manifest 和图片 SHA256 一致。中央数据未修改，test 只作文件完整性检查。
- 官方源码 commit=`9aa41d679d121afb8b4cc512d7a162f4f36886c3`，git clean；无源码修改、无 git pull、无结构或 loss 修改。
- Python 3.12.13、Ultralytics 8.4.152 editable、PyTorch 2.8.0+cu128、CUDA 12.8、RTX 5070 Ti；本轮未修改依赖。

## 实际 train 调用参数与默认值

调用只传用户明确项及输出身份/禁止覆盖参数。以下是 dry-run 预期值，**不是已训练完成的 args.yaml**：

```json
{
  "model": "D:\\小论文\\model_projects\\yolo11\\workspace\\weights\\yolo11s.pt",
  "data": "D:\\小论文\\model_projects\\yolo11\\workspace\\configs\\conductor_bl_v1.yaml",
  "epochs": 100,
  "imgsz": 1280,
  "batch": 4,
  "workers": 0,
  "device": "0",
  "seed": 42,
  "deterministic": true,
  "pretrained": true,
  "cache": false,
  "plots": true,
  "project": "D:\\小论文\\model_projects\\yolo11\\workspace\\runs",
  "name": "Y11-BASE-001_20260917_195422_484013",
  "save_dir": "D:\\小论文\\model_projects\\yolo11\\workspace\\runs\\Y11-BASE-001_20260917_195422_484013",
  "exist_ok": false
}
```

其余训练参数均由锁定官方 `ultralytics/cfg/default.yaml` 和官方解析器提供；完整解析值保存于 [resolved_training_args.yaml](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-001_20260917_195422_484013/resolved_training_args.yaml)。它用于审计，不会作为完整 kwargs 再传回 train。

| 官方默认参数（项目没有主动覆盖） | 当前解析值 |
|---|---|
| optimizer / nbs | auto / 64 |
| mosaic / close_mosaic | 1.0 / 10 |
| lr0 / lrf / momentum | 0.01 / 0.01 / 0.937 |
| warmup_epochs / warmup_bias_lr | 3.0 / 0.1 |
| box / cls / dfl | 7.5 / 0.5 / 1.5 |
| patience / save_period | 100 / -1 |
| amp / rect / resume | True / False / False |

这些数值与某些旧实验相同不表示从旧实验复制。正式开始时检查 trainer 的全部解析参数并保存官方 args.yaml；实际优化器参数从运行时读取，不将默认调用 lr/momentum 冒充实际值。官方 auto 优化器的 warmup_bias_lr 运行时派生为 0.0，单独记录来源，不改写默认输入。

## 最小修改范围

| 既有文件 | 修改内容 |
|---|---|
| formal_baseline_protocol.yaml | v2 单次方案；仅明确覆盖项、默认配置哈希、环境/权重/完整性证据 |
| run_formal_baseline.py | 唯一 ID 白名单，移除 DIAG 参数母版，保留安全/记录功能；100 轮后固定 best→val→一次 test |
| formal_baseline_registry.py | 保留原 openpyxl 备份/追加/当前行更新逻辑；允许本次最终 test_used=True，增加完成条件检查 |
| summarize_formal_baseline.py | 从单次已保存证据生成结果报告；取消三 seed mean/std，不调用模型 |
| test_formal_baseline.py | 更新默认参数、旧 ID 拒绝、单次 test 条件、失败不重试、历史保护测试 |

新增 [计划变更说明](D:/小论文/model_projects/yolo11/workspace/notes/formal_baseline_plan_change_20260917.md)、本报告和 [本次完整 diff](D:/小论文/model_projects/yolo11/workspace/notes/formal_baseline_single_run_changes_20260917.patch)；原 preparation report 与 formal_baseline_changes.patch 没有覆盖。

## 保留的安全与记录功能

中央整个数据工程写保护、缓存重定向、split/类别/manifest/快照检查、官方结构/权重哈希/源码/环境检查均保留。独占锁和唯一 ID 检查拒绝重复运行，目录原子预留，exist_ok=False。失败、中断保留 ID/日志/产物，不自动重跑；可捕获 OOM、NaN/Inf 直接停止，不改参数恢复。

正常训练完成需确认 100 行连续 epoch 和所有必要产物，保留官方曲线、可视化、best.pt、last.pt、args.yaml、results.csv、runtime、训练峰值 VRAM、参数量和实际优化器。当前版本曲线带 Box 前缀时，保留原文件，并生成用户要求的无前缀同字节副本。

best.pt 由官方训练流程保存，不自行根据 test 选择。原表 best_epoch 字段继续遵循既有 CSV 最大 mAP50-95、并列 mAP50 的定义；同时独立记录官方实际写入 best.pt 的轮次 native_best_checkpoint_epoch，避免四舍五入或同分覆盖时混淆两者。

训练后先固定 best.pt SHA256，再以同一 YAML、imgsz=1280、device=0、split=val 做标准验证。额外只设置输出目录/禁止覆盖，其余由标准默认值提供：当前为 batch=16、workers=8、rect=True（官方 val 方法默认）、conf=0.001（官方解析）、iou=0.7、max_det=300、quantize=None（FP32）。不依据指标调整参数。

val 指标写入文件并记录哈希后，才可消耗一次 test 执行凭据；test 使用同一 best.pt、同一标准验证配置，仅 split 和输出目录不同。执行前再次检查参数一致；凭据在读取 test 模型数据前持久化，失败或中断后也禁止自动重试。test_used 在凭据消耗时保守记录为 True，另用 test_completed/test_state 区分成功与未完整完成。

逐类别 P/R 直接读取官方 summary 的 Box-P/Box-R；缺失为 NA。原登记表没有独立的最终 val overall/test 列，因此不新增或重定义历史列：CSV best-epoch 指标继续按 FieldDefinitions 填写；最终 val/test overall 与逐类指标完整写入当前行 notes（JSON）和 status JSON，并保持 XLSX/CSV 同步。只有 100 epochs、val、一次 test 与完整性检查成功才记 Completed / test_used=True；parent_experiment=NA、changed_variable=NA、architecture_modified=False 写入 notes。

## 核验结果和边界

- **21 项安全回归测试通过**，工作簿测试只修改临时副本；模型流程测试使用模拟对象，没有真实模型评价。
- 唯一真实 dry-run：[status.json](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-001_20260917_195422_484013/status.json)；[preflight.json](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_preflight/Y11-BASE-001_20260917_195422_484013/preflight.json)。
- 最终历史/源码审计：[single_run_audit.json](D:/小论文/model_projects/yolo11/workspace/logs/formal_baseline_plan_change_20260917_194422/single_run_audit.json)。
- 实际训练尚未开始，GPU 训练和训练后联动尚未端到端验证。发生真实错误将停止并保留证据；不会为了完成 100 epochs 悄悄改变协议。
- 当前授权范围按本轮先修改与 dry-run 执行；此前会话明确禁止训练，本轮没有启动正式命令。没有第二实验、第二 seed 或自动任务。

## 本轮对应的 20 项状态

1. 官方 yolo11s.pt：路径、SHA256、COCO 80 类已核验；尚未开始训练。
2. 官方原版 YOLO11s：结构元数据核验通过。
3. 网络结构没有修改。
4. 实际完成 100 epochs：尚未训练，不能声称完成。
5. 实际 args.yaml：尚未生成；完整 dry-run 解析值见上方链接。
6. best epoch：NA，尚未训练。
7. val overall：NA，尚未验证。
8. val broken AP：NA。
9. val loose AP：NA。
10. test overall：NA，尚未测试。
11. test broken AP：NA。
12. test loose AP：NA。
13. training runtime：NA。
14. peak VRAM：NA。
15. best.pt：尚未产生，将位于 workspace/runs/Y11-BASE-001_<timestamp>/weights/best.pt。
16. 正式 run：尚未创建；dry-run 审计目录不属于训练 run。
17. experiment_registry：保持原文件和 6 个历史实验，尚未登记本次正式训练。
18. test 模型使用次数：0；已实现固定 best 和保存 val 后限一次的执行顺序。
19. git status：clean。
20. 中央数据集：保持不变。

## 唯一正式 CMD 命令（本轮未执行）

```bat
"D:\小论文\model_projects\yolo11\.venv\Scripts\python.exe" -B "D:\小论文\model_projects\yolo11\workspace\scripts\run_formal_baseline.py" --experiment-id Y11-BASE-001
```

该命令将执行一次 100 epoch 训练，完成固定 val 和一次最终 test、登记与单次结果报告后退出。任何失败均不自动启动第二次实验。
