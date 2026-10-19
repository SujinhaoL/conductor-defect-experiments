# 当前锁定 Ultralytics 的 Mosaic 训练管线

> 源码依据：官方仓库 commit `9aa41d679d121afb8b4cc512d7a162f4f36886c3`，本说明只描述该 commit 的实际实现。

## 1. 训练增强顺序

检测训练集在 `YOLODataset.build_transforms()` 中调用 `v8_transforms()`，当前顺序是：

1. `Mosaic`
2. `CopyPaste`（本实验概率为 0）
3. `RandomPerspective`
4. `MixUp`（概率为 0）
5. `CutMix`（概率为 0）
6. `Albumentations`
7. `RandomHSV`
8. 垂直翻转（概率为 0）
9. 水平翻转（概率为 0.5）
10. `Format`：转成训练所需的归一化 `xywh` 标签和 CHW 张量

代码位置：`ultralytics/data/augment.py:2812-2853`，`ultralytics/data/dataset.py:304-333`。

非训练增强路径才在 `YOLODataset.build_transforms()` 中直接使用 `LetterBox(new_shape=(imgsz, imgsz), scaleup=False)`。本次训练增强审计走的是上面的 `v8_transforms()`，最终尺寸由 `RandomPerspective(size=(1280, 1280))` 固定；不要把验证阶段的 LetterBox 路径误认为当前训练增强路径。

## 2. `mosaic=1.0` 的含义

它是 Mosaic 变换的应用概率 `p=1.0`，即在 Mosaic 开启阶段每次取样都应用。当前检测配置使用 `n=4`，主图再随机取 3 张图，组成 4 图 Mosaic。它不是把图像尺寸乘以 1，也不是强度系数。

`mosaic=0.0` 表示该变换每次都跳过。后续 `RandomPerspective`、HSV 和水平翻转仍照常执行。

代码位置：`ultralytics/data/augment.py:462-500`、`ultralytics/data/augment.py:2812`。

## 3. `imgsz=1280` 时的尺寸

4 图 Mosaic 先创建 `2560×2560` 的临时画布，因为 `_cat_labels()` 使用 `imgsz * 2`。随后 `RandomPerspective` 的 `size=(1280, 1280)` 把图像和 bbox 变换、裁剪到最终 `1280×1280` 训练输入。

所以最终送入格式化步骤的是 `1280×1280`，不是 `2560×2560`。临时大画布及随机中心布局会让一部分源目标落在最终视野外。

代码位置：`ultralytics/data/augment.py:515-552`、`609-616`、`754-768`、`2812-2823`。

## 4. `scale=0.5` 的真实作用

当前 `RandomPerspective` 对浮点 `scale` 使用：

```text
random.uniform(1 - scale, 1 + scale)
```

因此 `scale=0.5` 每次随机得到 `0.5～1.5` 的仿射缩放因子。它并不表示固定缩小到 50%。缩放与中心平移一起作用；`translate=0.1` 会把输出中心随机放在宽高的 40%～60% 位置。

代码位置：`ultralytics/data/augment.py:1137-1158`。

## 5. RandomPerspective 怎样影响 bbox

本实验 `degrees=0`、`shear=0`、`perspective=0`，但随机缩放 `0.5～1.5` 和平移仍生效。代码用同一个仿射矩阵变换图像与 bbox，再把 bbox 裁到 `1280×1280`。因此目标可能：

- 被缩小；
- 被平移到边缘并部分裁剪；
- 完全移出输出画面；
- 因变换后过小、面积保留过低或宽高比异常而被过滤。

本次审计测得 Mosaic ON 的 6,299 个 Mosaic 后候选框中，2,912 个通过 RandomPerspective；OFF 为 1,680 中保留 1,636。ON 的大幅下降来自“4 图临时画布 + 固定 1280 输出裁剪 + RandomPerspective 候选过滤”的组合，不能只归因于一个阈值。

## 6. 哪些代码会过滤不合格 bbox

有两处关键过滤：

1. Mosaic 合并后，先把框裁到 `2560×2560` 临时画布，并删除零面积框（`augment.py:725-768`）。
2. RandomPerspective 变换后，先裁到最终输出，再由 `box_candidates()` 过滤。检测框要求：变换后宽和高都大于 2 px、保留面积大于原框的 10%、宽高比小于 100（`augment.py:1223-1234`、`1368-1410`）。

注意：最终 CSV 中没有 `<2px` 框，正好与这里的严格 `w2 > 2`、`h2 > 2` 条件一致；约 2.01 px 的框仍可能保留。

## 7. `close_mosaic=10` 在 30 epoch 中的含义

训练器使用从 0 开始的 epoch 索引，并在：

```text
epoch == epochs - close_mosaic
```

时重建训练 transforms。30 epoch、`close_mosaic=10` 会在索引 20，即界面显示的第 21 个 epoch 开始前关闭 Mosaic；第 21～30 epoch 均不再使用 Mosaic。`YOLODataset.close_mosaic()` 同时把 mosaic、copy_paste、mixup、cutmix 置 0 后重建 transforms。

若未来把 `mosaic=0.0`，Mosaic 从第 1 epoch 就已关闭；仍保留 `close_mosaic=10` 只是为了让单变量实验其余参数与 Y11-DIAG-001 完全一致。

代码位置：`ultralytics/engine/trainer.py:470-490`、`1110-1116`，`ultralytics/data/dataset.py:364-374`。
