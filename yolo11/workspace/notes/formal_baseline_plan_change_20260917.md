# FormalBaseline 计划变更（2026-09-17）

原 three-seed FormalBaseline 计划在任何正式 `model.train()` 执行前取消：three-seed FormalBaseline plan was abandoned before any formal model.train() execution.

- `Y11-BASE-001-S42`、`Y11-BASE-002-S3407`、`Y11-BASE-003-S2026` 只完成过 dry-run/preflight，没有正式训练。原日志、preparation report 和 `formal_baseline_changes.patch` 全部保留。
- 当前唯一可执行正式实验改为 `Y11-BASE-001`。该名称是实验 ID；模型文件仍是 `workspace/weights/yolo11s.pt`，不是实验 ID。
- 参数来源改为用户明确设置项加锁定 Ultralytics 8.4.152 默认配置；不再从 `Y11-DIAG-001` 复制完整训练参数。明确设置 100 epochs、1280、batch=4、workers=0、device=0、seed=42、deterministic=True、pretrained=True、cache=False、plots=True，以及核验过的 model/data 路径。
- 正式训练完成 100 epochs、固定官方生成的 best.pt SHA256 并保存固定 val 指标后，允许一次最终 test。出现真实错误、NaN/Inf 即停止；不自动重试、改参数、换权重或根据 test 再选结果。
- `Y11-SMOKE-001`、`Y11-DIAG-001`、`Y11-DIAG-002`、`Y11-TUNE-001`、`Y11-TUNE-002`、`Y11-ARCH-001` 全部保留，不删除、不重命名、不改写历史行和结果；其既有 Smoke/Diagnostic/Tuning/ArchitectureDiagnostic 标识保留。
- 修改前的五个配置/脚本保存于 `workspace/logs/formal_baseline_plan_change_20260917_194422/before/`。历史 `PROJECT_STATE.md` 和旧准备报告是当时状态快照；当前执行方案以本文和 protocol_version=2 为准。

本轮执行范围是修改与 `Y11-BASE-001 --dry-run`。当前会话此前明确禁止训练，尚无后续单独启动训练授权，本轮不运行训练、val 或 test。准备完成后给出唯一一条正式 CMD 命令。
