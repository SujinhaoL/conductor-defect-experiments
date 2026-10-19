# Y11-TUNE-002 训练前单变量核对

- 参考参数：`D:\小论文\model_projects\yolo11\workspace\runs\yolo11s_diag_1280_e30_20260916_104603\args.yaml`
- 实验定义：Y11-DIAG-001 仅将 `epochs: 30 → 100`。
- 计划运行名：`yolo11s_tune_1280_e100_20260916_191542`（输出路径变化不属于训练策略变量）。
- 预检查结果：**PASS**
- 权重来源：官方 `YOLO("yolo11s.pt")`；`resume=False`，不加载旧 best.pt/last.pt。
- 数据分割：仅 train/val；`split=val`，test 封存。

|字段|Y11-DIAG-001|Y11-TUNE-002 计划值|判定|
|---|---:|---:|---|
|model|`yolo11s.pt`|`yolo11s.pt`|一致|
|pretrained|`True`|`True`|一致|
|epochs|`30`|`100`|唯一主动变化|
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

## 机器可核验差异

`{"epochs": [30, 100]}`

`project`、`name`、`save_dir` 仅用于隔离输出，不参与单变量判定。
