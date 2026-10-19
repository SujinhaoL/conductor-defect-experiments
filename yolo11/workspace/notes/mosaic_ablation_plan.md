# Y11-TUNE-001：Mosaic 单变量消融计划

## 状态

- **仅计划，尚未训练。**
- parent experiment：`Y11-DIAG-001`
- planned changed variable：`mosaic: 1.0 → 0.0`
- `experiment_registry.xlsx` 的 Runs 中不提前登记为 Completed。

## 启动依据

只读增强审计为两组各 10 个种子 × 107 张训练图，共各 1,070 个最终增强样本。关键差异：

- 全部目标 `min_side` P50：ON 10.510 px，OFF 12.274 px，ON/OFF=0.856。
- `<4px`：ON 7.35%，OFF 5.44%，增加 1.91 个百分点。
- `<8px`：ON 34.99%，OFF 27.26%，增加 7.73 个百分点；10/10 个配对种子方向一致。
- loose `<8px`：ON 41.24%，OFF 32.40%，增加 8.84 个百分点。
- 源候选到最终保留率：ON 43.36%，OFF 97.38%。ON 数值反映 4 图 Mosaic 临时画布、最终 1280 裁剪和 RandomPerspective 过滤的组合。
- ON 并未产生 `<2px` 的最终保留框；`<4px` 为 214/2,912（7.35%），因此没有证据称“大量目标只剩 1～3 像素”。

这些数据足以支持一次严格的单变量训练消融，用验证集结果检验关闭 Mosaic 是否改善学习；它们本身不保证指标一定提高。

## 未来固定训练协议

|参数|计划值|
|---|---:|
|model|`yolo11s.pt`|
|imgsz|1280|
|batch|4|
|nbs|64|
|epochs|30|
|workers|0|
|device|0|
|seed|42|
|deterministic|True|
|patience|0|
|amp|True|
|mosaic|**0.0**|
|close_mosaic|10|

其余优化器、学习率、损失权重、HSV、几何增强、保存和验证设置均锁定为 Y11-DIAG-001 的实际设置。`close_mosaic=10` 保持不变，避免同时改第二个变量。

## 未来执行与判断

1. 仅在下一轮得到明确训练指令后，设置脚本的显式运行开关并执行 `train_yolo11_mosaic0_diagnostic.py`。
2. 训练前再次检查官方 commit、干净源码、中央数据集哈希、GPU、train=107、val=23。
3. 只使用 val 比较 Y11-DIAG-001 与 Y11-TUNE-001；test 继续封存。
4. best epoch 继续按最大 mAP50-95、并列时最大 mAP50 选择，再读取同一 epoch 的 P/R/mAP。
5. 重点比较 mAP50-95、mAP50、recall、逐类别 AP、训练稳定性和误检/漏检；不能因单个 epoch 的独立最高 P 或 R 下结论。
