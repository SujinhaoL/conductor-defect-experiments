# Y11-ARCH-001 单变量训练前核对

- 参考：`D:\小论文\model_projects\yolo11\workspace\runs\yolo11s_diag_1280_e30_20260916_104603\args.yaml`
- 允许变化：检测尺度 `P3/P4/P5 → P2/P3/P4/P5`、由此产生的参数量/GFLOPs/权重迁移数量，以及独立输出路径。
- 训练策略字段差异：**无**。
- optimizer=auto 必须在第一个 batch 前解析为 AdamW、lr=0.001667、momentum=0.9；训练回调会强制检查。
- `resume=False`；仅 train/val，test 封存。
- 计划运行名：`yolo11s_p2_diag_1280_e30_20260916_202431`。

|训练字段|Y11-DIAG-001|Y11-ARCH-001|判定|
|---|---:|---:|---|
|epochs|`30`|`30`|一致|
|imgsz|`1280`|`1280`|一致|
|batch|`4`|`4`|一致|
|nbs|`64`|`64`|一致|
|workers|`0`|`0`|一致|
|device|`0`|`0`|一致|
|seed|`42`|`42`|一致|
|deterministic|`True`|`True`|一致|
|patience|`0`|`0`|一致|
|cache|`False`|`False`|一致|
|amp|`True`|`True`|一致|
|optimizer|`auto`|`auto`|一致|
|lr0|`0.01`|`0.01`|一致|
|lrf|`0.01`|`0.01`|一致|
|momentum|`0.937`|`0.937`|一致|
|weight_decay|`0.0005`|`0.0005`|一致|
|warmup_epochs|`3.0`|`3.0`|一致|
|box|`7.5`|`7.5`|一致|
|cls|`0.5`|`0.5`|一致|
|dfl|`1.5`|`1.5`|一致|
|hsv_h|`0.015`|`0.015`|一致|
|hsv_s|`0.7`|`0.7`|一致|
|hsv_v|`0.4`|`0.4`|一致|
|degrees|`0.0`|`0.0`|一致|
|translate|`0.1`|`0.1`|一致|
|scale|`0.5`|`0.5`|一致|
|shear|`0.0`|`0.0`|一致|
|perspective|`0.0`|`0.0`|一致|
|flipud|`0.0`|`0.0`|一致|
|fliplr|`0.5`|`0.5`|一致|
|mosaic|`1.0`|`1.0`|一致|
|mixup|`0.0`|`0.0`|一致|
|cutmix|`0.0`|`0.0`|一致|
|copy_paste|`0.0`|`0.0`|一致|
|close_mosaic|`10`|`10`|一致|

|架构字段|Y11-DIAG-001|Y11-ARCH-001|判定|
|---|---|---|---|
|model|`yolo11s.pt`|`D:\小论文\model_projects\yolo11\workspace\configs\yolo11s_p2.yaml`|允许的唯一架构变化|
|detection_scales|P3/8, P4/16, P5/32|P2/4, P3/8, P4/16, P5/32|允许的唯一架构变化|

机器核验训练策略差异：`{}`
