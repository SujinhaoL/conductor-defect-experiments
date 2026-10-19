# FRCNN-R50FPN-MERGE130-001

目的：执行原train107与val23合并为130图的既定实验。该设置在VAL-001 test前已固定。

训练130图/205框（broken48、loose157），原test23图/36框（8、28）保留。没有独立val；原val已经成为训练数据，不能报告其独立验证成绩。YOLO0/1→COCO1/2→模型0/1（broken/loose）。

固定设置：同一官方完整COCO初始checkpoint重新开始，不接着VAL-001私人数据checkpoint、不resume。100epoch、batch2、workers0、FP32、seed42、SGD0.0025/0.9/0.0001、warmup500、milestones67/92、gamma0.1；Resize边界(1333,800)保持比例、flip0.5、官方anchor。每轮65更新，共6500；实际运行状态及核查记录见下方。

checkpoint规则：固定最后epoch100，不按val/test或固定train8拟合分数挑epoch。本训练阶段不做test。原test已有前两次Ultralytics基础实验和FRCNN VAL-001的使用记录，后续分数不得用于调参。

## 运行记录

- 运行编号：`step09_109_20338`
- 状态：COMPLETED_REVIEWED
- 原始目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338`
- 完整日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step09_109_20338\step09_console.txt`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\resolved_merge130_001.py`
- 代码/环境/命令/数据哈希/初始权重来源哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\pretraining_record.json`
- 实际训练：100轮，6500更新；训练含保存/快照耗时2153.535秒；allocated峰值3771.71MiB。
- 完整COCO加载证据：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\artifacts\actual_training_weight_load.json`
- 损失/LR曲线：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\artifacts\loss_curves.png`
- 逐次loss：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\artifacts\losses.csv`
- epoch0/25/50/75/100固定train8预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\artifacts`
- 最后一轮权重/哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\selected_checkpoint.json`

## 评价及核查状态

val：不适用。test：已完成并核查指标、checkpoint/文件哈希、类别映射、原图坐标、保存预测及可视化。

异常及可视化：本次100轮训练已完成并核查，未发现非有限训练损失；已检查损失/LR曲线和固定训练图片快照。

解释与比较限制：训练规模由107变130，更新数由5400变6500；无独立val。不能视为严格同规模单变量对照，也不能保证合并数据提高性能。单seed不做统计显著性结论。

- 阶段命令与文件：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\STEP09_merge130_training.txt`
- 统一CSV：`D:\小论文\model_projects\baseline_analysis\10_all_experiments_metrics.csv`
- 总Excel：`D:\小论文\model_projects\baseline_analysis\outputs\模型实验结果总表.xlsx`
- 汇总：`D:\小论文\model_projects\baseline_analysis\11_all_experiments_summary_for_teacher.md`

## 训练核查

100轮/6500更新的损失及学习率核查通过。首100次总loss均值0.361302，末100次0.012629。344个兼容COCO张量加载与4个类别输出层重新初始化已确认，未接着VAL-001 checkpoint训练。

损失/LR曲线已查看，总损失下降，训练快照部分图片出现高置信框，仍有图片在0.30下无可见框。固定train8的最终拟合mAP50约60.396%、mAP50-95约59.777%，只作过程诊断，不能作为130图整体或泛化成绩。终端ROI acc不能当作缺陷检测Recall。

核查证据：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step09_review\acceptance.json`。训练流程验收通过；固定epoch100 test已完成并核查，未根据test改变训练设置或checkpoint。

## 固定epoch100测试结果

|评价编号|口径|mAP50|mAP50-95|AR100|broken AP50/AP50-95|loose AP50/AP50-95|
|---|---|---:|---:|---:|---|---|
|EV-3a40bf61cdc04443|MMDetection-native-CocoMetric-rounded|0.000000%|0.000000%|未保存/未计算|未保存/未计算/0.000000%|未保存/未计算/0.000000%|
|EV-be4f62d04282ccb5|pycocotools-exact-from-saved-predictions|0.028289%|0.016973%|1.071429%|0.000000%/0.000000%|0.056577%/0.033946%|

指标（EV-3a40bf61cdc04443）：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\native_coco_metrics.json`；预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\predictions.bbox.json`；状态：VERIFIED_RESULT_REVIEW。

指标（EV-be4f62d04282ccb5）：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\test_metrics.json`；预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\predictions.bbox.json`；状态：VERIFIED_RESULT_REVIEW。


## 最终测试核查与解释

原test23图/36框的129个预测已核查。2张在score>0.001后无返回框；score>0.30只有6个显示框，分布在4张图。6个显示框与同类GT均未达到IoU0.50；cond_000026的broken最大同类IoU约0.184，其余5个为0，按现有标注为误检。不能把未画出框解释成完全没有预测。

36个GT中1个有同类候选IoU≥0.50；这只是候选覆盖，不是普通Recall或一对一检出率。普通Precision/Recall本轮未计算，不填0；COCO AR100单列。

精确COCO mAP50=0.028289%、mAP50-95=0.016973%、AR100=1.071429%；broken AP50/AP50-95均为0，loose分别0.056577%/0.033946%。原生mAP输出0.000属于三位小数舍入，保存预测复算精确值非全零；两种口径均保留，不改变权重选择。

与107图训练、val选epoch66的原test精确结果0.082508%/0.049505%相比，本次AP数值更低，AR100相同。本次合并未显示泛化提升；训练规模、更新数、checkpoint规则不同且仅一个seed，不能推断合并数据普遍降低性能。

流程核查通过，最终泛化表现很差，两者应分别表述。数据量、小目标缩放、anchor匹配、类别不平衡等只能列为尚待验证的可能因素，现有证据无法确定唯一原因。没有依据test新增训练、挑epoch或调整参数。

- 完整测试结果目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846`
- 原生指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\native_coco_metrics.json`
- 精确指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\test_metrics.json`
- 预测坐标、类别及置信度：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\predictions_details.json`
- COCO预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\predictions.bbox.json`
- 23张测试可视化：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\predicted_images`
- 测试命令/环境/数据和选择哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\pretest_record.json`
- 完整测试终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step10_13778_21846\step10_console.txt`
- 结果核查：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step10_review\acceptance.json`
- GT与已保存预测总览：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step10_review\test_gt_and_visible_predictions.jpg`
- 离线独立COCO核对：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step10_review\independent_coco_check.txt`

核查状态：COMPLETED_REVIEWED。原计划的两个正式Faster R-CNN实验及小样本诊断已完成；无需为了登记重新训练或推理。跨框架仍按原生口径参考比较，未统一评价的限制继续保留。
