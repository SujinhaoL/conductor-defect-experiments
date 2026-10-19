# Y11-BASE-001 正式执行失败报告

执行时间：2026-09-17。实验未完成，状态为 `Failed`；没有自动重试或修改协议。

## 停止原因

Preflight 通过后已经真正调用 `model.train()`。从官方 `workspace/weights/yolo11s.pt` 开始，模型为标准 YOLO11s DetectionModel，训练时将官方 COCO 80 类 Detect head 正常重建为 broken/loose 两类；Detect 层仍为 P3/P4/P5，strides=[8,16,32]，没有结构修改。

第 1、2、3 轮完整结束。第 4 轮训练 batch 完成后，在该轮固定 val 验证阶段检测到验证 loss 为 NaN/Inf，执行系统按已批准规则抛出：

`RuntimeError: 验证 loss 出现 NaN/Inf；停止，不执行自动恢复或最终 test。`

因此没有完成 100 epochs，没有固定正式 best.pt，没有执行训练后的正式 val，也没有读取 test。`test_used=False`、`test_attempts=0`。

## 实际状态

- run：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001_20260917_201125_320591`
- status：`D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001_20260917_201125_320591\status.json`
- console：`D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001_20260917_201125_320591\Y11-BASE-001_20260917_201125_320591_console.log`
- 实际 args：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001_20260917_201125_320591\args.yaml`
- results：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001_20260917_201125_320591\results.csv`，仅包含完整写入的 epoch 1–3
- 中断运行产生的 preliminary best.pt：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001_20260917_201125_320591\weights\best.pt`
- 中断运行产生的 last.pt：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001_20260917_201125_320591\weights\last.pt`
- 完整完成轮数：3/100；第 4 轮未完成验证，因此未写入 results.csv
- 已检查训练 batches：108，训练 batch loss 均有限
- 运行期间实测优化器：AdamW，初始 lr=0.001667，momentum=0.9，非零 weight_decay=0.0005
- 失败运行总耗时：91.25 秒；完整训练 runtime=NA
- peak VRAM：4232.24 MiB
- 正式 best epoch：NA；状态中 `native_best_checkpoint_epoch=3` 仅表示中断前官方暂存 best.pt 的轮次，不是完成后的正式结果
- val overall / broken / loose：NA（训练后正式 val 未执行）
- test overall / broken / loose：NA（test 未使用）

## 记录与完整性

唯一登记表已安全新增并更新 `Y11-BASE-001`，状态为 `Failed`，`epochs_completed=3`，`test_used=False`；XLSX 和 CSV 已同步，旧实验行未覆盖。失败产物和日志全部保留。

训练后复核中央数据工程的 1749 个快照条目全部不变，未发生被阻止或成功的写入；train/val/test 仍为 107/23/23。官方源码仍处于 commit `9aa41d679d121afb8b4cc512d7a162f4f36886c3`，git status clean。

本次到此停止。没有第二实验、第二 seed、自动调参、恢复训练或重跑；也没有根据 test 作任何选择。
