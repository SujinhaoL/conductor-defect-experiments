# FRCNN-R50FPN-SCALE1280-VAL-001

状态：训练、验证及固定第5轮test结果均已复核；STEP15末尾归档错误已通过保存结果恢复，未重跑推理。原始运行文件保持原样，其REVIEW_PENDING是运行时记录，最新状态见本说明及audit/step11_review/acceptance.json。

运行编号：`step11_31112_14115`。原始目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115`。

## 目的与实际设置

检验Faster R-CNN仅提高整图输入分辨率的影响。对照VAL-001，原train107/168框（broken39、loose129）、val23/37框（9、28），test23/36框未参与本轮。类别YOLO0/1→COCO1/2→模型0/1，num_classes=2。

实际训练、验证及固定训练图快照均Resize(2133,1280)、keep_ratio=True；batch2、FP32、workers0、seed42、100epoch、5400更新。SGD lr0.0025/momentum0.9/weight_decay0.0001；warmup500，milestones67/92、gamma0.1；flip0.5。配置还原尺寸后与冻结原基线一致，5400条实际学习率轨迹也一致。

同一官方完整COCO检测器重新初始化：`D:\小论文\model_projects\Faster R-CNN\workspace\weights\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`；来源：https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth；SHA256：`047c8118fc5ca88ba5ae1fab72f2cd6b070501fe3af2f3cba5cfa9a89b44b03e`。344项兼容权重逐值加载；仅ROI分类/回归输出的weight/bias四项因80→2类别重新初始化，不是只加载ImageNet骨干。

耗时（含val、保存和固定图快照）3095.629秒，约51.59分钟；峰值CUDA allocated 8903.871MiB（约8.70GiB），不是设备总占用。原基线对应2193.604秒、3769.939MiB。

## 模型选择与原始结果

按每轮原生val coco/bbox_mAP（三位小数）最大值、同值保留首次，选定第5轮。不得按mAP50或完整精度复算另挑轮次。最佳权重：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\best_coco_bbox_mAP_epoch_5.pth`；SHA256：`c8ff3af83efeec7cf3f126f49833da6e42489d9f022809d6237b94d9efdb08d3`。末轮权重：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\epoch_100.pth`；SHA256：`7031e53e49689327b3d238855ba0a84f9356fb2b405018238b7f85b25ce4d6dd`。

|指标（23张val）|原800基线，第66轮|新1280，第5轮|
|---|---:|---:|
|mAP50|3.773949%|3.189325%|
|mAP50-95|1.273413%|1.487789%|
|COCO AR100|4.384921%|11.607143%|
|broken AP50|1.697313%|0.000000%|
|broken AP50_95|0.848656%|0.000000%|
|loose AP50|5.850585%|6.378650%|
|loose AP50_95|1.698170%|2.975578%|

原生mAP50=0.032、mAP50-95=0.015；上表为相同保存预测的精确COCO复算。普通Precision/Recall未计算，AR100单独报告。补测test的结果与异常记录见末尾STEP15部分。

## 训练过程与可视化发现

5400条损失及分项全部有限。首100次总损失均值0.39741879，末100次0.01741165。固定train8/15GT末轮mAP50-95：原800为58.663366%，1280为80.198020%。本次broken训练AP为100%，loose约60.396%；11/15GT有同类候选IoU≥0.50，另外4个loose仍未覆盖。该覆盖数不等于普通Recall，也不是RPN proposal recall。

第100轮val精确mAP50=0.594059%、mAP50-95=0.162796%，低于选定第5轮。训练拟合增强而验证仍低且后期下降，是泛化问题/过拟合的迹象，不能单凭此确定唯一机制。

选定第5轮保存2300个预测，23张各100个，均未超过显示阈值0.30；AP过滤阈值是0.001。图上无红框不代表零预测。检查了原图坐标范围、类别映射、全100轮文件和固定train8详细预测导出。loss与预测图已目视检查，但没有全量逐层特征/RPN/ROI内部采样检查。

## 结论与下一步边界

本次完成了预定尺寸敏感性对照；mAP50-95变化+0.214376个百分点，mAP50变化-0.584624个百分点。不能说放大尺寸已解决问题、也不能证明尺寸完全无影响。单seed、23张val不足以宣称显著性；本轮同时提高训练和验证尺度，不区分二者贡献。

训练阶段结束时没有运行test；后续RPN→ROI诊断、anchor对照与最后一次1600尺寸对照已分别归档。用户于2026-10-14指定固定已有权重补齐最终test，第5轮补测已核查。没有根据此次test改选epoch或调整预测阈值。旧test已被查看，不再称完全未接触盲测。

## 证据完整路径

- 运行配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\resolved_scale1280_val_001.py`
- 依赖：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\requirements_freeze.txt`
- 初始化/数据/归档哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\pretraining_record.json`
- 实际加载：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\artifacts\actual_training_weight_load.json`
- 损失与学习率：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\artifacts\loss_curves.png`
- 固定训练图末轮：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\artifacts\epoch_100\predictions_contact_sheet.jpg`
- 选定验证预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\artifacts\validation\epoch_005\predictions.bbox.json`
- 精确指标及checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\selected_checkpoint.json`
- 完整日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step11_31112_14115\step11_console.txt`
- 独立复核：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_review\acceptance.json`
- 训练验证对照曲线：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_review\resolution_training_validation_comparison.png`
- 验证GT与显示预测：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_review\selected_val_gt_green_pred_red.jpg`
- 逐GT诊断：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_review\train8_epoch100_per_gt.csv`

## STEP15 固定第5轮最终test

用户于2026-10-14启动原test23图/36框（broken8、loose28）评价。test运行编号step15_23467_7311。保持Resize(2133,1280)、keep_ratio=True、默认anchor8、FP32、score_thr0.001、ROI NMS0.5、max100。选模仍为此前val确定的第5轮，348项权重严格加载核查通过。

|指标|val：37框|test：36框|
|---|---:|---:|
|mAP50|3.189325%|0.819432%|
|mAP50-95|1.487789%|0.286644%|
|COCO AR100|11.607143%|6.250000%|
|broken AP50|0.000000%|0.323806%|
|broken AP50-95|0.000000%|0.032381%|
|loose AP50|6.378650%|1.315058%|
|loose AP50-95|2.975578%|0.540908%|

上表均为已保存预测的精确COCO指标。原生test mAP50=0.008、mAP50-95=0.003，类别AP50-95舍入为broken0.000、loose0.005；舍入零不表示精确零。普通P/R未计算，不填0。COCO AR100不是普通Recall。

保存2300个低阈值预测，每图100个；均不超过显示阈值0.30，所以23张预测图没有可见框。已核对类别、原图坐标、COCO导出一致性、23图完整解码和哈希，独立COCO复算与原保存指标一致。助手已查看保存图总览；无可见框不等于模型无预测，也不宣称全链路无错。

异常：模型完成23图推理和指标计算后，最终哈希函数接收字符串而非Path对象，导致AttributeError。保留原日志/manifest中的FAIL，不改写历史。已修复当前脚本，并仅根据保存预测完成CPU复核、补齐test_summary与证据manifest，没有重新模型推理或训练。原始运行未及时落盘的推理耗时和峰值显存留空，不估算或填0；训练耗时/显存记录完整保留。

结果显示该固定模型的test AP仍很低，不能据此确定唯一原因。与val数值不同是不同集合的观测，不能混成同一评价结果；单seed、小test、旧test已有使用的限制保留。本阶段的目的已达成：补齐固定权重最终评价并留存证据。按原先指定顺序下一项是1280 anchor4第15轮，不根据本次分数改选。

- 完整test结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311`
- 精确指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\test_metrics.json`
- 原生指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\native_coco_metrics.json`
- 全部预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\predictions.bbox.json`
- 坐标/类别/分数：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\predictions_details.json`
- 23张预测图：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\predicted_images`
- 恢复记录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\postprocessing_recovery.json`
- 原完整终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step15_23467_7311\step15_console.txt`
- 核查：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step15_review\acceptance.json`
- 独立复算：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step15_review\independent_coco_check.txt`
- 保存图总览：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step15_review\saved_test_visualizations.jpg`
