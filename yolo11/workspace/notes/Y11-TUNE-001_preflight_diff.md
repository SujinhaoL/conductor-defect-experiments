# Y11-TUNE-001 训练前单变量核对

## 核对结论

**通过。** Y11-DIAG-001 的真实 `args.yaml` 与 `train_yolo11_mosaic0_diagnostic.py` 的计划训练参数逐字段核对后，唯一主动训练变量差异为：

```text
mosaic: 1.0 -> 0.0
```

允许启动 Y11-TUNE-001。`project/name/save_dir` 只负责写入新的独立运行目录，不是训练策略变化。

## 参数对照

|参数|Y11-DIAG-001 实际值|Y11-TUNE-001 计划值|结论|
|---|---:|---:|---|
|model|`yolo11s.pt`|`YOLO("yolo11s.pt")`|相同，均从官方 COCO 权重开始|
|pretrained weights|`pretrained=True`|`yolo11s.pt`，启动时校验 80 类并记录 SHA256|相同|
|epochs|30|30|相同|
|imgsz|1280|1280|相同|
|batch|4|4|相同|
|nbs|64|64|相同|
|workers|0|0|相同|
|device|0|0|相同|
|seed|42|42|相同|
|deterministic|True|True|相同|
|patience|0|0|相同|
|cache|False|False|相同|
|amp|True|True|相同|
|optimizer|auto|auto|相同|
|lr0|0.01|0.01|相同|
|lrf|0.01|0.01|相同|
|momentum|0.937|0.937|相同|
|weight_decay|0.0005|0.0005|相同|
|warmup_epochs|3.0|3.0|相同|
|box|7.5|7.5|相同|
|cls|0.5|0.5|相同|
|dfl|1.5|1.5|相同|
|hsv_h|0.015|0.015|相同|
|hsv_s|0.7|0.7|相同|
|hsv_v|0.4|0.4|相同|
|degrees|0.0|0.0|相同|
|translate|0.1|0.1|相同|
|scale|0.5|0.5|相同|
|shear|0.0|0.0|相同|
|perspective|0.0|0.0|相同|
|flipud|0.0|0.0|相同|
|fliplr|0.5|0.5|相同|
|mosaic|**1.0**|**0.0**|**唯一主动变化**|
|mixup|0.0|0.0|相同|
|cutmix|0.0|0.0|相同|
|copy_paste|0.0|0.0|相同|
|close_mosaic|10|10|相同|

## 额外安全核对

- `resume=False`，不会加载 Y11-DIAG-001 的 `best.pt` 或任何旧 `last.pt`。
- `val=True`、`split=val`；没有 test 调用。
- 官方源码 commit：`9aa41d679d121afb8b4cc512d7a162f4f36886c3`，训练前 `git status` 为空。
- 中央数据集由写保护钩子保护；训练前后均与初始 SHA256 快照比较。
- 固定 batch=4；OOM 时禁止自动降低 batch。
- 训练脚本需显式设置 `RUN_Y11_TUNE_001=YES` 才能运行。
