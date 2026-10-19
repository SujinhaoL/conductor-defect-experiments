# YOLO11s 与 RT-DETR-L FormalBaseline 横向分析

## 1. 研究对象和当前目标

本分析比较两个已经完成且登记为成功的官方 FormalBaseline：YOLO11s `Y11-BASE-001-R1` 与 RT-DETR-L `RTD-BASE-001-R1`。目标是评估把高清输电导线整图直接输入通用目标检测器时，对 broken/loose 局部缺陷的检测表现，并用 bbox 像素统计描述当前任务的尺寸条件。

本次没有训练模型，没有重新运行 val/test，也没有修改数据集、模型、权重或已有实验结果。

## 2. 数据集基本情况

- 正式数据集：`D:\小论文\conductor_defect_dataset\02_yolo11_dataset`
- 图片总数：153；GT bbox 总数：241
- broken bbox：56；loose bbox：185
- 所有统计直接读取正式图片和 YOLO `.txt` 标签，按每张图实际宽高换算。

## 3. 数据划分

| Split | Images | Purpose |
| --- | --- | --- |
| train | 107 | 训练 |
| val | 23 | 独立验证 |
| test | 23 | 最终测试；本次未重新运行 |

## 4. 标注类别

| class_id | class_name | GT bbox count |
| --- | --- | --- |
| 0 | broken | 56 |
| 1 | loose | 185 |

## 5. YOLO11 FormalBaseline 先决条件

- 官方 Ultralytics YOLO11s 与官方 COCO 预训练 `yolo11s.pt`。
- 未修改网络结构；复用固定 107/23/23 数据划分。
- 正式成功 run 由 experiment registry、status、metrics、args.yaml 与 run 目录交叉确认。

## 6. YOLO11 实验参数

| epochs | imgsz | batch | workers | device | seed | deterministic |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1280 | 4 | 0 | 0 | 42 | True |

## 7. YOLO11 实验结果

| Split/class | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| val overall | 0.012958 | 0.234127 | 0.022262 | 0.004386 |
| val broken | see source | see source | 0.012778 | 0.002556 |
| val loose | see source | see source | 0.031746 | 0.006217 |
| test overall | 0.016253 | 0.205357 | 0.008108 | 0.002288 |
| test broken | see source | see source | 0.008929 | 0.002679 |
| test loose | see source | see source | 0.007288 | 0.001897 |

## 8. RT-DETR FormalBaseline 先决条件

- 官方 Ultralytics RT-DETR-L 与官方 COCO 预训练 `rtdetr-l.pt`。
- 未修改 decoder、backbone、encoder、query 数或其他网络结构；复用固定 107/23/23 数据划分。
- deterministic=False 为已记录的 RT-DETR/CUDA 运行约束。

## 9. RT-DETR 实验参数

| epochs | imgsz | batch | workers | device | seed | deterministic |
| --- | --- | --- | --- | --- | --- | --- |
| 100 | 1280 | 4 | 0 | 0 | 42 | False |

## 10. RT-DETR 实验结果

| Split/class | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| val overall | 0.508274 | 0.017857 | 0.018199 | 0.005388 |
| val broken | see source | see source | 0.000000 | 0.000000 |
| val loose | see source | see source | 0.036397 | 0.010775 |
| test overall | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| test broken | see source | see source | 0.000000 | 0.000000 |
| test loose | see source | see source | 0.000000 | 0.000000 |

## 11. 双模型横向对比

| Item | YOLO11s | RT-DETR-L |
| --- | --- | --- |
| Experiment ID | Y11-BASE-001-R1 | RTD-BASE-001-R1 |
| Model | YOLO11s | RT-DETR-L |
| Official pretrained weight | yolo11s.pt | rtdetr-l.pt |
| Network modified | No | No |
| Ultralytics | 8.4.152 | 8.4.152 |
| git commit | 9aa41d679d121afb8b4cc512d7a162f4f36886c3 | 9aa41d679d121afb8b4cc512d7a162f4f36886c3 |
| Epochs | 100 | 100 |
| imgsz | 1280 | 1280 |
| Batch | 4 | 4 |
| Workers | 0 | 0 |
| Device | 0 | 0 |
| Seed | 42 | 42 |
| Deterministic | Yes | No |
| Parameters | 9,428,566 | 32,810,186 |
| Best epoch | 15 | 71 |
| Training runtime (s) | 1661.422 | 2593.096 |
| Peak VRAM (MiB) | 4526.72 | 11550.00 |
| Val P | 0.012958 | 0.508274 |
| Val R | 0.234127 | 0.017857 |
| Val mAP50 | 0.022262 | 0.018199 |
| Val mAP50-95 | 0.004386 | 0.005388 |
| Val broken AP50 | 0.012778 | 0.000000 |
| Val broken AP50-95 | 0.002556 | 0.000000 |
| Val loose AP50 | 0.031746 | 0.036397 |
| Val loose AP50-95 | 0.006217 | 0.010775 |
| Test P | 0.016253 | 0.000000 |
| Test R | 0.205357 | 0.000000 |
| Test mAP50 | 0.008108 | 0.000000 |
| Test mAP50-95 | 0.002288 | 0.000000 |
| Test broken AP50 | 0.008929 | 0.000000 |
| Test broken AP50-95 | 0.002679 | 0.000000 |
| Test loose AP50 | 0.007288 | 0.000000 |
| Test loose AP50-95 | 0.001897 | 0.000000 |
| best.pt SHA256 | 01f6246e45a0b42fe93762120c8784d8a885dcba21567ab5906d94af2edb1cc9 | 15ba1e62e1170d21e0628753be3eef0a710e06300643b3f9dd421f2c02d53ccd |
| Run directory | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451 | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120 |

结果解读：YOLO11s 在 test 上仍获得非零检测指标，但 Recall 和 mAP 很低；RT-DETR-L 的 test 指标为 0。RT-DETR 的 val Precision 较高并不代表整体更好，因为其 val Recall 仅约 0.018，且 test 无有效检出。Precision 必须与 Recall、AP 和类别结果共同解释。

峰值显存字段来自各正式实验已有记录，但记录口径并不完全相同：YOLO 为正式报告中的 trainer-recorded peak_vram_mib，RT-DETR 为 `torch.cuda.max_memory_reserved`。因此只适合作数量级参考，不应做精细效率排名。

## 12. bbox 尺寸统计

原图 bbox short-side median=52.86px；broken=71.01px，loose=47.56px。完整分位数、split 与类别统计见 `03_bbox_size_summary.md`，逐目标记录见 `04_bbox_object_statistics.csv`。

## 13. 1280 输入后的目标实际像素尺寸

- YOLO 单图 letterbox 参考：short-side median=12.74px；short<8/16/32px=25.31%/63.49%/87.97%。
- RT-DETR 方形拉伸参考：short-side median=17.58px；short<8/16/32px=13.69%/44.40%/83.40%。
- YOLO 保持宽高比并补边；RT-DETR 在当前源码路径中直接变换为方形，因此同一个 bbox 的横纵尺寸变化可能不同。
- 训练期随机增强会继续改变目标尺寸；上述数字是可复核的基础预处理参考。

## 14. 当前实验能支持哪些结论

- 当前数据集规模固定为 107/23/23，只有 broken 与 loose 两类。
- 在 imgsz=1280、batch=4、seed=42 和当前官方实现/权重下，两个 FormalBaseline 的 Recall 与 AP 均较低，存在严重漏检。
- YOLO11s 在本次 test 上保留了少量非零检测能力；RT-DETR-L 在本次 test 上所有汇总与类别 AP 均为 0。
- 大量 GT bbox 经基础预处理后仍处于较小像素尺度；broken 与 loose 的尺寸分布存在可量化差异。

## 15. 当前实验还不能证明哪些结论

- 不能证明所有目标检测器都无法完成该任务。这里只比较两个具体模型、一个 seed 和一组固定条件。
- 不能证明缩放是低性能的唯一原因。bbox 尺寸统计只说明尺度条件，并未隔离样本量、类别形态、背景、标注噪声、增强或优化因素。
- 不能根据 test 结果反向挑选参数，也不能据此宣称某个后续改动一定有效。

## 16. 可能的瓶颈分析

以下是实验提示和需要进一步验证的可能原因：

- 缺陷 bbox 在高清整图中的面积占比较低，基础缩放后可用像素进一步减少。
- 数据规模较小，可能不足以覆盖导线姿态、背景、光照和缺陷形态变化。
- broken/loose 是细粒度局部结构异常，类别间视觉差异可能弱于背景变化。
- 复杂背景、目标遮挡或导线尺度变化可能提高定位与分类难度。
- 训练随机增强会改变小目标尺寸和可见性，其净影响需要受控实验验证。

## 17. 后续值得讨论的方向

- 更高输入分辨率的单变量敏感性实验。bbox 统计显示其具有尺寸层面的实验价值，但不保证 mAP 提升。
- 高分辨率切片/tiling，减少整图缩放对局部像素的压缩。
- 导线 ROI 或两阶段检测，先定位导线区域，再检测局部缺陷。
- 小目标特征增强。
- 数据扩充，以及对类别不平衡和小样本问题的专门分析。
- 其他非 YOLO 检测器，但应采用同一数据划分和独立 test 规则。

这些方向只用于呈现证据与问题，不在本报告中替导师决定论文创新点。
