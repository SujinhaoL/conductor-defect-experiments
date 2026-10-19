# Y11-ARCH-001 官方结构与 P2 架构预检查

## 1. 锁定依据

- 官方源码 commit：`9aa41d679d121afb8b4cc512d7a162f4f36886c3`，工作树 clean。
- 官方 YAML：`ultralytics_src/ultralytics/cfg/models/11/yolo11.yaml`。
- 官方定义的 Detect 输入为 layer `[16, 19, 22]`，对应 P3/8、P4/16、P5/32。

## 2. 数据动机

- Mosaic ON 后全部目标 min_side median ≈ 10.510 px；broken ≈ 17.928 px；loose ≈ 9.357 px。
- 全部目标 `<8 px` ≈ 34.99%，`<4 px` ≈ 7.35%；loose `<8 px` ≈ 41.24%。
- 标准 P3 的一格对应 8 px；P2 的一格对应 4 px。因此 P2 是有数据依据的待验证假设，并非预设有效。

## 3. 标准 YOLO11s 实际逐层结构（dummy 1280×1280）

- 官方 nc=80 模型：9,458,752 parameters，88.820 GFLOPs @ 1280。

|index|from|module|输出通道|空间尺寸|对应 stride|
|---:|---|---|---:|---:|---:|
|0|`-1`|conv.Conv|32|640x640|2.0|
|1|`-1`|conv.Conv|64|320x320|4.0|
|2|`-1`|block.C3k2|128|320x320|4.0|
|3|`-1`|conv.Conv|128|160x160|8.0|
|4|`-1`|block.C3k2|256|160x160|8.0|
|5|`-1`|conv.Conv|256|80x80|16.0|
|6|`-1`|block.C3k2|256|80x80|16.0|
|7|`-1`|conv.Conv|512|40x40|32.0|
|8|`-1`|block.C3k2|512|40x40|32.0|
|9|`-1`|block.SPPF|512|40x40|32.0|
|10|`-1`|block.C2PSA|512|40x40|32.0|
|11|`-1`|torch.nn.modules.upsampling.Upsample|512|80x80|16.0|
|12|`[-1, 6]`|conv.Concat|768|80x80|16.0|
|13|`-1`|block.C3k2|256|80x80|16.0|
|14|`-1`|torch.nn.modules.upsampling.Upsample|256|160x160|8.0|
|15|`[-1, 4]`|conv.Concat|512|160x160|8.0|
|16|`-1`|block.C3k2|128|160x160|8.0|
|17|`-1`|conv.Conv|128|80x80|16.0|
|18|`[-1, 13]`|conv.Concat|384|80x80|16.0|
|19|`-1`|block.C3k2|256|80x80|16.0|
|20|`-1`|conv.Conv|256|40x40|32.0|
|21|`[-1, 10]`|conv.Concat|768|40x40|32.0|
|22|`-1`|block.C3k2|512|40x40|32.0|
|23|`[16, 19, 22]`|head.Detect|Detect output|multiple outputs|multiple|

标准 Detect 实际输入：

|layer|channels|shape|stride|
|---:|---:|---:|---:|
|16|128|160×160|8.0|
|19|256|80×80|16.0|
|22|512|40×40|32.0|

## 4. 最小 P2 改动与结构验证

- 官方 backbone 和标准 P3/P4/P5 neck layers 0–22 原样保留。
- 只从标准 P3 layer 16 上采样，与 backbone P2 layer 2 拼接，再经一个 C3k2 形成 layer 25 的 P2/4 特征。
- 新 Detect layer 26 输入 `[25, 16, 19, 22]`。
- P2 nc=2 模型：9,451,320 parameters，111.342 GFLOPs @ 1280。
- dummy 1280×1280 forward 成功。

P2 Detect 实际输入：

|layer|channels|shape|stride|
|---:|---:|---:|---:|
|25|64|320×320|4.0|
|16|128|160×160|8.0|
|19|256|80×80|16.0|
|22|512|40×40|32.0|

新增/变化层：

|index|from|module|输出通道|空间尺寸|stride|
|---:|---|---|---:|---:|---:|
|23|`16`|torch.nn.modules.upsampling.Upsample|128|320x320|4.0|
|24|`[-1, 2]`|conv.Concat|256|320x320|4.0|
|25|`-1`|block.C3k2|64|320x320|4.0|
|26|`[25, 16, 19, 22]`|head.Detect|Detect output|multiple outputs|multiple|

## 5. 官方预训练权重迁移

- 官方 state items：499；P2 state items：563。
- 同名同 shape 直接迁移：378。
- 因 Detect index 23→26 而人工重映射的兼容回归/DFL items：70。
- 总迁移：448 / 563。
- A（nc=80→2 检测分类分支差异）：51 items。
- B（新增 P2 路径和 P2 head）：64 items，随机初始化。
- C（非预期未加载）：0 items。
- backbone/标准 neck 未加载：0 items。

### A. nc=80→2 导致的分类分支未加载 key

- `model.26.cv3.1.0.1.bn.bias`
- `model.26.cv3.1.0.1.bn.running_mean`
- `model.26.cv3.1.0.1.bn.running_var`
- `model.26.cv3.1.0.1.bn.weight`
- `model.26.cv3.1.0.1.conv.weight`
- `model.26.cv3.1.1.0.bn.bias`
- `model.26.cv3.1.1.0.bn.running_mean`
- `model.26.cv3.1.1.0.bn.running_var`
- `model.26.cv3.1.1.0.bn.weight`
- `model.26.cv3.1.1.0.conv.weight`
- `model.26.cv3.1.1.1.bn.bias`
- `model.26.cv3.1.1.1.bn.running_mean`
- `model.26.cv3.1.1.1.bn.running_var`
- `model.26.cv3.1.1.1.bn.weight`
- `model.26.cv3.1.1.1.conv.weight`
- `model.26.cv3.1.2.bias`
- `model.26.cv3.1.2.weight`
- `model.26.cv3.2.0.1.bn.bias`
- `model.26.cv3.2.0.1.bn.running_mean`
- `model.26.cv3.2.0.1.bn.running_var`
- `model.26.cv3.2.0.1.bn.weight`
- `model.26.cv3.2.0.1.conv.weight`
- `model.26.cv3.2.1.0.bn.bias`
- `model.26.cv3.2.1.0.bn.running_mean`
- `model.26.cv3.2.1.0.bn.running_var`
- `model.26.cv3.2.1.0.bn.weight`
- `model.26.cv3.2.1.0.conv.weight`
- `model.26.cv3.2.1.1.bn.bias`
- `model.26.cv3.2.1.1.bn.running_mean`
- `model.26.cv3.2.1.1.bn.running_var`
- `model.26.cv3.2.1.1.bn.weight`
- `model.26.cv3.2.1.1.conv.weight`
- `model.26.cv3.2.2.bias`
- `model.26.cv3.2.2.weight`
- `model.26.cv3.3.0.1.bn.bias`
- `model.26.cv3.3.0.1.bn.running_mean`
- `model.26.cv3.3.0.1.bn.running_var`
- `model.26.cv3.3.0.1.bn.weight`
- `model.26.cv3.3.0.1.conv.weight`
- `model.26.cv3.3.1.0.bn.bias`
- `model.26.cv3.3.1.0.bn.running_mean`
- `model.26.cv3.3.1.0.bn.running_var`
- `model.26.cv3.3.1.0.bn.weight`
- `model.26.cv3.3.1.0.conv.weight`
- `model.26.cv3.3.1.1.bn.bias`
- `model.26.cv3.3.1.1.bn.running_mean`
- `model.26.cv3.3.1.1.bn.running_var`
- `model.26.cv3.3.1.1.bn.weight`
- `model.26.cv3.3.1.1.conv.weight`
- `model.26.cv3.3.2.bias`
- `model.26.cv3.3.2.weight`

### B. 新增 P2 路径/分支未加载 key

- `model.25.cv1.bn.bias`
- `model.25.cv1.bn.num_batches_tracked`
- `model.25.cv1.bn.running_mean`
- `model.25.cv1.bn.running_var`
- `model.25.cv1.bn.weight`
- `model.25.cv1.conv.weight`
- `model.25.cv2.bn.bias`
- `model.25.cv2.bn.num_batches_tracked`
- `model.25.cv2.bn.running_mean`
- `model.25.cv2.bn.running_var`
- `model.25.cv2.bn.weight`
- `model.25.cv2.conv.weight`
- `model.25.m.0.cv1.bn.bias`
- `model.25.m.0.cv1.bn.num_batches_tracked`
- `model.25.m.0.cv1.bn.running_mean`
- `model.25.m.0.cv1.bn.running_var`
- `model.25.m.0.cv1.bn.weight`
- `model.25.m.0.cv1.conv.weight`
- `model.25.m.0.cv2.bn.bias`
- `model.25.m.0.cv2.bn.num_batches_tracked`
- `model.25.m.0.cv2.bn.running_mean`
- `model.25.m.0.cv2.bn.running_var`
- `model.25.m.0.cv2.bn.weight`
- `model.25.m.0.cv2.conv.weight`
- `model.26.cv2.0.0.bn.bias`
- `model.26.cv2.0.0.bn.num_batches_tracked`
- `model.26.cv2.0.0.bn.running_mean`
- `model.26.cv2.0.0.bn.running_var`
- `model.26.cv2.0.0.bn.weight`
- `model.26.cv2.0.0.conv.weight`
- `model.26.cv2.0.1.bn.bias`
- `model.26.cv2.0.1.bn.num_batches_tracked`
- `model.26.cv2.0.1.bn.running_mean`
- `model.26.cv2.0.1.bn.running_var`
- `model.26.cv2.0.1.bn.weight`
- `model.26.cv2.0.1.conv.weight`
- `model.26.cv2.0.2.bias`
- `model.26.cv2.0.2.weight`
- `model.26.cv3.0.0.0.bn.bias`
- `model.26.cv3.0.0.0.bn.num_batches_tracked`
- `model.26.cv3.0.0.0.bn.running_mean`
- `model.26.cv3.0.0.0.bn.running_var`
- `model.26.cv3.0.0.0.bn.weight`
- `model.26.cv3.0.0.0.conv.weight`
- `model.26.cv3.0.0.1.bn.bias`
- `model.26.cv3.0.0.1.bn.num_batches_tracked`
- `model.26.cv3.0.0.1.bn.running_mean`
- `model.26.cv3.0.0.1.bn.running_var`
- `model.26.cv3.0.0.1.bn.weight`
- `model.26.cv3.0.0.1.conv.weight`
- `model.26.cv3.0.1.0.bn.bias`
- `model.26.cv3.0.1.0.bn.num_batches_tracked`
- `model.26.cv3.0.1.0.bn.running_mean`
- `model.26.cv3.0.1.0.bn.running_var`
- `model.26.cv3.0.1.0.bn.weight`
- `model.26.cv3.0.1.0.conv.weight`
- `model.26.cv3.0.1.1.bn.bias`
- `model.26.cv3.0.1.1.bn.num_batches_tracked`
- `model.26.cv3.0.1.1.bn.running_mean`
- `model.26.cv3.0.1.1.bn.running_var`
- `model.26.cv3.0.1.1.bn.weight`
- `model.26.cv3.0.1.1.conv.weight`
- `model.26.cv3.0.2.bias`
- `model.26.cv3.0.2.weight`

### shape mismatch 映射记录

- `model.23.cv3.0.0.1.conv.weight` [128, 128, 1, 1] → `model.26.cv3.1.0.1.conv.weight` [64, 128, 1, 1]
- `model.23.cv3.0.0.1.bn.weight` [128] → `model.26.cv3.1.0.1.bn.weight` [64]
- `model.23.cv3.0.0.1.bn.bias` [128] → `model.26.cv3.1.0.1.bn.bias` [64]
- `model.23.cv3.0.0.1.bn.running_mean` [128] → `model.26.cv3.1.0.1.bn.running_mean` [64]
- `model.23.cv3.0.0.1.bn.running_var` [128] → `model.26.cv3.1.0.1.bn.running_var` [64]
- `model.23.cv3.0.1.0.conv.weight` [128, 1, 3, 3] → `model.26.cv3.1.1.0.conv.weight` [64, 1, 3, 3]
- `model.23.cv3.0.1.0.bn.weight` [128] → `model.26.cv3.1.1.0.bn.weight` [64]
- `model.23.cv3.0.1.0.bn.bias` [128] → `model.26.cv3.1.1.0.bn.bias` [64]
- `model.23.cv3.0.1.0.bn.running_mean` [128] → `model.26.cv3.1.1.0.bn.running_mean` [64]
- `model.23.cv3.0.1.0.bn.running_var` [128] → `model.26.cv3.1.1.0.bn.running_var` [64]
- `model.23.cv3.0.1.1.conv.weight` [128, 128, 1, 1] → `model.26.cv3.1.1.1.conv.weight` [64, 64, 1, 1]
- `model.23.cv3.0.1.1.bn.weight` [128] → `model.26.cv3.1.1.1.bn.weight` [64]
- `model.23.cv3.0.1.1.bn.bias` [128] → `model.26.cv3.1.1.1.bn.bias` [64]
- `model.23.cv3.0.1.1.bn.running_mean` [128] → `model.26.cv3.1.1.1.bn.running_mean` [64]
- `model.23.cv3.0.1.1.bn.running_var` [128] → `model.26.cv3.1.1.1.bn.running_var` [64]
- `model.23.cv3.0.2.weight` [80, 128, 1, 1] → `model.26.cv3.1.2.weight` [2, 64, 1, 1]
- `model.23.cv3.0.2.bias` [80] → `model.26.cv3.1.2.bias` [2]
- `model.23.cv3.1.0.1.conv.weight` [128, 256, 1, 1] → `model.26.cv3.2.0.1.conv.weight` [64, 256, 1, 1]
- `model.23.cv3.1.0.1.bn.weight` [128] → `model.26.cv3.2.0.1.bn.weight` [64]
- `model.23.cv3.1.0.1.bn.bias` [128] → `model.26.cv3.2.0.1.bn.bias` [64]
- `model.23.cv3.1.0.1.bn.running_mean` [128] → `model.26.cv3.2.0.1.bn.running_mean` [64]
- `model.23.cv3.1.0.1.bn.running_var` [128] → `model.26.cv3.2.0.1.bn.running_var` [64]
- `model.23.cv3.1.1.0.conv.weight` [128, 1, 3, 3] → `model.26.cv3.2.1.0.conv.weight` [64, 1, 3, 3]
- `model.23.cv3.1.1.0.bn.weight` [128] → `model.26.cv3.2.1.0.bn.weight` [64]
- `model.23.cv3.1.1.0.bn.bias` [128] → `model.26.cv3.2.1.0.bn.bias` [64]
- `model.23.cv3.1.1.0.bn.running_mean` [128] → `model.26.cv3.2.1.0.bn.running_mean` [64]
- `model.23.cv3.1.1.0.bn.running_var` [128] → `model.26.cv3.2.1.0.bn.running_var` [64]
- `model.23.cv3.1.1.1.conv.weight` [128, 128, 1, 1] → `model.26.cv3.2.1.1.conv.weight` [64, 64, 1, 1]
- `model.23.cv3.1.1.1.bn.weight` [128] → `model.26.cv3.2.1.1.bn.weight` [64]
- `model.23.cv3.1.1.1.bn.bias` [128] → `model.26.cv3.2.1.1.bn.bias` [64]
- `model.23.cv3.1.1.1.bn.running_mean` [128] → `model.26.cv3.2.1.1.bn.running_mean` [64]
- `model.23.cv3.1.1.1.bn.running_var` [128] → `model.26.cv3.2.1.1.bn.running_var` [64]
- `model.23.cv3.1.2.weight` [80, 128, 1, 1] → `model.26.cv3.2.2.weight` [2, 64, 1, 1]
- `model.23.cv3.1.2.bias` [80] → `model.26.cv3.2.2.bias` [2]
- `model.23.cv3.2.0.1.conv.weight` [128, 512, 1, 1] → `model.26.cv3.3.0.1.conv.weight` [64, 512, 1, 1]
- `model.23.cv3.2.0.1.bn.weight` [128] → `model.26.cv3.3.0.1.bn.weight` [64]
- `model.23.cv3.2.0.1.bn.bias` [128] → `model.26.cv3.3.0.1.bn.bias` [64]
- `model.23.cv3.2.0.1.bn.running_mean` [128] → `model.26.cv3.3.0.1.bn.running_mean` [64]
- `model.23.cv3.2.0.1.bn.running_var` [128] → `model.26.cv3.3.0.1.bn.running_var` [64]
- `model.23.cv3.2.1.0.conv.weight` [128, 1, 3, 3] → `model.26.cv3.3.1.0.conv.weight` [64, 1, 3, 3]
- `model.23.cv3.2.1.0.bn.weight` [128] → `model.26.cv3.3.1.0.bn.weight` [64]
- `model.23.cv3.2.1.0.bn.bias` [128] → `model.26.cv3.3.1.0.bn.bias` [64]
- `model.23.cv3.2.1.0.bn.running_mean` [128] → `model.26.cv3.3.1.0.bn.running_mean` [64]
- `model.23.cv3.2.1.0.bn.running_var` [128] → `model.26.cv3.3.1.0.bn.running_var` [64]
- `model.23.cv3.2.1.1.conv.weight` [128, 128, 1, 1] → `model.26.cv3.3.1.1.conv.weight` [64, 64, 1, 1]
- `model.23.cv3.2.1.1.bn.weight` [128] → `model.26.cv3.3.1.1.bn.weight` [64]
- `model.23.cv3.2.1.1.bn.bias` [128] → `model.26.cv3.3.1.1.bn.bias` [64]
- `model.23.cv3.2.1.1.bn.running_mean` [128] → `model.26.cv3.3.1.1.bn.running_mean` [64]
- `model.23.cv3.2.1.1.bn.running_var` [128] → `model.26.cv3.3.1.1.bn.running_var` [64]
- `model.23.cv3.2.2.weight` [80, 128, 1, 1] → `model.26.cv3.3.2.weight` [2, 64, 1, 1]
- `model.23.cv3.2.2.bias` [80] → `model.26.cv3.3.2.bias` [2]

## 6. 结论

- 结构验证通过：四个 Detect 输入 strides 为 4/8/16/32。
- 官方 backbone 和标准 P3/P4/P5 neck 权重全部迁移；没有非预期缺失。
- P2 是否改善极小目标仍需由本次单变量训练验证。
