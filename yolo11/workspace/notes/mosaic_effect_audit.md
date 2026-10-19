# Mosaic 对极小目标尺寸的只读审计

> 锁定源码 commit：`9aa41d679d121afb8b4cc512d7a162f4f36886c3`。只读取 train=107；未读取 val/test 目标用于策略判断；没有模型 forward、loss 或训练。

## 审计设计

- 两组均直接使用当前 Ultralytics `YOLODataset` 官方训练 transforms；外层探针只计数，不改图像、bbox 或随机过程。
- `MOSAIC_ON` 与 `MOSAIC_OFF` 各使用种子 0-9，每个种子完整遍历 107 张训练图，共各 1070 个最终增强样本。
- 唯一条件差异为 `mosaic=1.0` 与 `mosaic=0.0`；imgsz、RandomPerspective、HSV、翻转等参数完全一致。
- retention 可可靠追踪为：进入 Mosaic 的源候选数 → Mosaic 裁剪后 → RandomPerspective 过滤后；这是类别总量追踪，不宣称逐框身份匹配。

## 核心统计

|范围|ON min_side P50|OFF min_side P50|ON <2px|OFF <2px|ON <4px|OFF <4px|ON <8px|OFF <8px|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|all|10.510|12.274|0.00%|0.00%|7.35%|5.44%|34.99%|27.26%|
|broken|17.928|21.061|0.00%|0.00%|0.73%|0.26%|14.54%|10.62%|
|loose|9.357|10.744|0.00%|0.00%|9.37%|7.04%|41.24%|32.40%|

## 目标保留

|范围|条件|源候选|Mosaic 后|RandomPerspective 后|Mosaic clip retention|RP retention|总 retention|
|---|---|---:|---:|---:|---:|---:|---:|
|all|MOSAIC_ON|6716|6299|2912|93.79%|46.23%|43.36%|
|all|MOSAIC_OFF|1680|1680|1636|100.00%|97.38%|97.38%|
|broken|MOSAIC_ON|1585|1521|681|95.96%|44.77%|42.97%|
|broken|MOSAIC_OFF|390|390|386|100.00%|98.97%|98.97%|
|loose|MOSAIC_ON|5131|4778|2231|93.12%|46.69%|43.48%|
|loose|MOSAIC_OFF|1290|1290|1250|100.00%|96.90%|96.90%|

## A-H 回答

### A. 全部目标 min_side 中位数

Mosaic ON=10.510px；OFF=12.274px。ON/OFF=0.856。

### B. broken

ON=17.928px；OFF=21.061px。

### C. loose

ON=9.357px；OFF=10.744px。

### D. Mosaic ON 是否增加 <2/<4/<8 px？

描述性统计显示：<4px 与 <8px 比例明显增加，<2px 没有增加。全部目标：<2px 0.00%→0.00%；<4px 5.44%→7.35%；<8px 27.26%→34.99%。10 个配对种子中有 10/10 个种子的 <8px 比例在 ON 下更高。这里报告描述性差异，不把重复增强框当作独立样本做伪显著性检验。

### E. loose 是否更严重？

按实际落入极小尺寸区间的风险看，是：loose 的 <8px 比例从 32.40% 增至 41.24%（+8.84 个百分点），broken 从 10.62% 增至 14.54%（+3.92 个百分点）。但若只比较各类别自身的中位数相对缩小，broken 略大：broken ON/OFF=0.851，loose=0.871。因此结论是 loose 的绝对极小框暴露更严重，而不是其相对缩小倍数更大。

### F. 是否大量只剩 1-3 像素？

未达到大量。Mosaic ON 中 min_side<4px 为 214/2912 (7.35%)，其中 <2px 为 0 (0.00%)。

### G. 是否观察到目标丢失/裁剪？

Mosaic ON 的源候选→Mosaic 后保留 6299/6716 (93.79%)；再经 RandomPerspective 后为 2912/6299 (46.23%)。OFF 的总 retention=97.38%。

### H. 是否有充分依据做 mosaic 1→0 单变量训练？

有。该判断只基于上面的尺寸、极小框比例和保留率；它支持或不支持启动一次严格单变量消融，不预先断言训练指标一定改善。

## 产物与限制

- `mosaic_bbox_samples.csv`：最终增强 bbox 逐框数据。
- `mosaic_bbox_summary.csv`：两条件 × broken/loose/all 的分位数、阈值比例和 retention 汇总。
- contact sheets 只用于人工复核；结论使用 CSV 数值。
- 两组使用相同种子集合和相同样本遍历顺序；Mosaic ON 会额外消耗随机数，因此后续 RandomPerspective 不是逐样本共享同一随机矩阵，本报告比较的是重复采样后的条件分布。
- 本审计模拟 Mosaic 开启阶段；原 30 epoch 设置的最后 10 epoch 会由 `close_mosaic=10` 关闭 Mosaic。
