# FRCNN-R50FPN-VAL-001

目的：官方Faster R-CNN R50-FPN在固定107/23/23导线缺陷数据上的参考基线。

运行编号：`step07_235_27043`；原始目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043`。

训练107图/168框（broken39、loose129）；val23图/37框（9、28）；test23图/36框（8、28）。沿用原图片级划分。YOLO0/1 → COCO1/2 → 模型0/1；num_classes=2。数据和划分哈希登记于CSV及pretraining_record。

实际设置：100epoch、batch2、workers0、seed42、FP32、SGD lr0.0025/momentum0.9/decay0.0001；warmup500 iter、start_factor0.001；milestones67/92、gamma0.1。Resize边界(1333,800)、keep_ratio=True、flip0.5。实际5400更新，训练含验证/保存/固定训练图预测耗时2193.604秒，峰值allocated约3769.94MiB。

官方完整COCO checkpoint加载：348项中344项一致；ROI fc_cls weight/bias和fc_reg weight/bias因80→2类重新初始化。其余backbone、FPN、RPN和ROI共享层确实加载。初始权重路径、URL、哈希见pretraining_record及CSV。

选模规则：原val native coco/bbox_mAP的三位小数最高值，同值保留首次；epoch66已固定。完整精度复算仅核查，不更换checkpoint。

## 已保存评价

|集合|事件|口径|mAP50|mAP50-95|broken AP50/50-95|loose AP50/50-95|AR100|
|---|---|---|---:|---:|---|---|---:|
|val|selected_epoch_validation|MMDetection-native-CocoMetric-rounded|3.8000%|1.3000%|未保存/未计算/0.8000%|未保存/未计算/1.7000%|未保存/未计算|
|val|selected_validation_offline_exact|pycocotools-exact-from-saved-predictions|3.7739%|1.2734%|1.6973%/0.8487%|5.8506%/1.6982%|4.3849%|
|test|fixed_selected_test|MMDetection-native-CocoMetric-rounded|0.1000%|0.0000%|未保存/未计算/0.0000%|未保存/未计算/0.1000%|未保存/未计算|
|test|fixed_test_offline_exact|pycocotools-exact-from-saved-predictions|0.0825%|0.0495%|0.0000%/0.0000%|0.1650%/0.0990%|1.0714%|

test状态：已完成并核查原始指标、哈希、映射、坐标、保存预测及可视化。

## 异常、可视化与解释

5400批次总损失及RPN/ROI分量有限，首100次总损失均值约0.35665，末100次约0.01357；val AP仍低。原训练已完成，无需为了整理再运行。

checkpoint首次受限读取拒绝HistoryBuffer元数据，STEP08用限定可信类型的临时safe_globals解决，348/348严格CPU加载通过，无缺项和尺寸不匹配，未进行额外测试推理。早期CUDA脚本将正确nvcc12.8.93误写为12.8.90导致假失败，已更正。

已保存训练损失/学习率图、原图GT和管线图，以及固定train8图片epoch0/25/50/75/100预测。正式test可视化阈值0.30，AP预测过滤0.001；显示无框不能证明零预测。

已确认事实：val mAP50约3.7739%、mAP50-95约1.2734%，性能仍低。尚待验证原因：数据量、目标尺寸、缩放/增强及anchor匹配等；不将这些推测当成实验已经证明的结论。

比较限制：框架AP实现、输入、后处理、maxDet、优化器及batch不同；100epoch计算量不同。参考对照，单seed不做显著性结论。train-fit自检不作为泛化指标。

## 完整路径索引

- 训练配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\resolved_val_001.py`
- 代码/版本/哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\pretraining_record.json`
- 依赖：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\requirements_freeze.txt`
- 实际预训练加载：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\actual_training_weight_load.json`
- 损失及学习率：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\loss_curves.png`
- 完整损失：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\losses.csv`
- 选中权重记录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\selected_checkpoint.json`
- 验证预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\validation\epoch_066\predictions.bbox.json`
- val原生指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\validation\epoch_066\metrics.json`
- val完整精度：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\best_validation_metrics.json`
- 训练日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step07_235_27043`
- test结果父目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\test_selected_epoch66`
- 数据转换核查：`D:\小论文\conductor_defect_dataset\03_coco_dataset\audit`
- 管线核查：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\STEP05_debug8_static_check.txt`
- 统一CSV：`D:\小论文\model_projects\baseline_analysis\10_all_experiments_metrics.csv`
- 汇总：`D:\小论文\model_projects\baseline_analysis\11_all_experiments_summary_for_teacher.md`

核查状态：训练完成和val指标、选中权重哈希已核验；test 已完成并核查。

test事实：123个返回框、score>0.30仅1个。cond_000031的loose框score=0.6756，与同类GT最大IoU=0，按现有标注为误检。36个GT中仅1个有同类候选IoU≥0.50；此项是候选覆盖诊断，不作为普通Recall。test精确mAP50=0.082508%、mAP50-95=0.049505%、AR100=1.071429%；broken AP=0。

结果核查：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step08_review\acceptance.json`；画框总览：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step08_review\test_gt_and_visible_predictions.jpg`。

test泛化性能很低。现有检查通过不等于已排除所有可能问题，也不能确定低分的唯一原因。MERGE130沿用训练前固定设置，不依据本次test调参。
