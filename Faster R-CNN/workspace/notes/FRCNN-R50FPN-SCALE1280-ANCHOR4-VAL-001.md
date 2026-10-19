# FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001

状态：100轮训练、验证及固定第15轮test结果均已复核；STEP16补测正常完成并保留原始证据。原始运行文件保持不变，REVIEW_PENDING为生成时状态，当前状态见本说明和独立acceptance.json。

运行：`step13_27622_23378`；目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378`。

## 问题、实验定位与边界

检验STEP12提示的小目标anchor匹配限制。它是固定1280输入的anchor尺寸消融，与此前纯输入尺寸对照是两个不同问题；是辅助诊断证据，不是论文必做实验，也不是新网络模块。用户强调实验必须服务论文且不无限复杂化：本次完成后停止自动追加尺寸、anchor或其他训练。

## 实际设置与复现

原train107/168框（broken39、loose129）、val23/37框（9、28）；test23/36框本轮未用。YOLO0/1→COCO1/2→模型0/1，num_classes=2，ROI背景不是第三个缺陷。

直接对照STEP11/step11_31112_14115。唯一科学配置变化为RPN anchor_generator.scales=[8]→[4]；三种ratio0.5/1/2和stride4/8/16/32/64不变，每位置3个anchor，输出张量形状不变。训练/验证/固定快照仍Resize(2133,1280)、keep_ratio=True。

batch2、FP32、workers0、seed42；100epoch、每轮54步、总5400次更新。SGD0.0025/momentum0.9/weight_decay0.0001；warmup500/start_factor0.001，milestones67/92、gamma0.1；flip0.5。5400条实际学习率与对照一致；5400条损失有限。score_thr0.001、ROI NMS0.5、max100、COCO maxDets[1,10,100]、显示0.30。

同一官方完整COCO初始化：`D:\小论文\model_projects\Faster R-CNN\workspace\weights\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`；来源：https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth；SHA256：`047c8118fc5ca88ba5ae1fab72f2cd6b070501fe3af2f3cba5cfa9a89b44b03e`。首次更新前344项逐值一致；仅ROI fc_cls/fc_reg weight/bias因80→2重置。anchor几何变化保持加载兼容，但并不意味着它与COCO原先几何相同。未续训私人权重。

耗时3110.653秒（51.84分钟，含val/保存/固定快照），峰值CUDA allocated 8903.870MiB（8.70GiB）。

## 模型选择与验证结果

按原生val coco/bbox_mAP三位小数最大值、同值保留首次，选第15轮。权重：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\best_coco_bbox_mAP_epoch_15.pth`；SHA256：`080c27214809847dce4e2d1b5e760a3e8b868bdb0d0f2941e5e43f219ffb548a`。完整精度复算不更换已选权重。末轮SHA256：`21ef28787593f368bfab10ee40c7839c0529cb9d8f603dc9abd4a257394b71d5`。

|23张val，保存预测精确COCO复算|1280 anchor8，第5轮|1280 anchor4，第15轮|
|---|---:|---:|
|mAP50|3.189325%|5.450693%|
|mAP50-95|1.487789%|1.159139%|
|COCO AR100|11.607143%|5.000000%|
|broken AP50|0.000000%|0.000000%|
|broken AP50_95|0.000000%|0.000000%|
|loose AP50|6.378650%|10.901386%|
|loose AP50_95|2.975578%|2.318277%|

原生mAP50=0.055、mAP50-95=0.012；上表是同一份保存预测的完整精度。普通P/R未单独登记，AR100不能代替普通Recall。固定第15轮补测test结果见末尾STEP16部分。

## 过程、预测及假设检验

末轮固定train8：mAP50=100.000000%、mAP50-95=96.190594%，对照对应mAP50-95=80.198020%。15/15GT有IoU≥0.50最终同类框，原4个loose漏检GT87/123/124/160的最佳同类IoU分别约0.949/0.990/0.977/0.960。

这支持较小anchor改善了已观察训练目标的拟合。它是用107张训练后，在其中固定8张上的观察，不是全107张完全过拟合的证明，也不是重跑原8张专用过拟合实验。没有采集本轮新RPN/ROI中间数组，不能从最终框反推出每个中间阶段已全部证明正常。

主要val mAP50-95变化-0.328651个百分点；mAP50变化+2.261368个百分点；broken AP仍0。第100轮val mAP50-95=0.189563%，早期较好、后期下降。不能只挑mAP50改善宣布实验成功或模型泛化改善。

选定val有2100个低阈值预测，19张达到100个上限；显示阈值>0.30留下15个，按同类、分数降序、一对一IoU≥0.50规则为TP=4、FP=11、未匹配GT=33。这只描述固定显示阈值，不用于计算AP或调整阈值。

训练拟合和val差距仍大，符合泛化受限/过拟合迹象；数据量、分布、目标外观和标注口径等仍是假设，当前不能归因于某一个因素。没有异常中断；比例/类别/原图坐标、100轮val文件、5次固定快照、哈希与选模均通过。

## 对小论文的用途与停止条件

可作为问题分析或消融：官方800→1280没有明显解决主要val指标；在1280下缩小anchor能恢复部分训练小目标的拟合，却未改善主要val指标。不能声称已找到所有低分原因、实现全链路无错，或获得具有显著性的结论。单seed、23张val且多轮开发的限制应如实写出；旧test已被查看。

该训练阶段已归档。用户于2026-10-14指定五项已有权重的最终test补齐，第15轮补测已完成；未按test改选epoch或阈值。本轮不追加训练，后续新训练仍需先明确论文问题、单一因素、评价标准和停止条件。

## 证据完整路径

- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\resolved_scale1280_anchor4_val_001.py`
- 依赖：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\requirements_freeze.txt`
- 初始化/数据/命令/代码归档哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\pretraining_record.json`
- 实际权重加载：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\artifacts\actual_training_weight_load.json`
- 损失与学习率：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\artifacts\loss_curves.png`
- 训练图末轮预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\artifacts\epoch_100\predictions_contact_sheet.jpg`
- 选中权重及精确val：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\selected_checkpoint.json`
- 选中val预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\artifacts\validation\epoch_015\predictions.bbox.json`
- 完整日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step13_27622_23378\step13_console.txt`
- 独立复算：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step13_review\acceptance.json`
- 训练/验证趋势对照：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step13_review\anchor_training_validation_comparison.png`
- 四个目标前后对照：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step13_review\four_targets_anchor8_vs_anchor4.jpg`

## STEP16 固定第15轮最终test

用户2026-10-14启动，运行step16_27428_3464，正常退出码0，无中断。原test23图/36框（broken8、loose28），沿用选定val的Resize(2133,1280)、keep_ratio=True、anchor4、FP32、score_thr0.001、ROI NMS0.5、max100及COCO maxDets[1,10,100]。348项权重逐值加载核查通过，类别映射和坐标核查通过，test未参与选模。

|精确COCO指标|val：37框|test：36框|
|---|---:|---:|
|mAP50|5.450693%|1.908924%|
|mAP50-95|1.159139%|0.424630%|
|COCO AR100|5.000000%|8.660714%|
|broken AP50|0.000000%|3.410178%|
|broken AP50-95|0.000000%|0.708449%|
|loose AP50|10.901386%|0.407671%|
|loose AP50-95|2.318277%|0.140812%|

原生test mAP50=0.019、mAP50-95=0.004，类别AP50-95 broken=0.007、loose=0.001；上表为同次保存预测独立复算的精确值。普通P/R未计算，保持空值；AR100单列，不能当普通Recall。

2135个低阈值预测全部保存。score>0.30有27框；按固定同类、分数降序、一对一IoU>=0.50的显示阈值诊断，TP0、FP27、未匹配GT36。27个显示框不能解释为检出27个目标，低阈值预测中仍有贡献AP的正确框。该诊断不替代COCO AP/AR，不写入正式普通P/R列，也不据test更改阈值。

评价含原图读取/23图推理和预测图片保存共8.061794秒，峰值CUDA allocated1057.304MiB；此耗时不含完整程序启动/CPU权重加载/末尾指标计算，峰值不是设备总占用，不作为严格效率排名。

已确认结果：固定anchor4模型test AP比固定anchor8第5轮的对应test数值更高，但整体仍低；val主要AP方向不同。单seed、每集合23张、不同val选定epoch和旧test使用限制须说明，不能证明该anchor修改普遍提升泛化或低分机制已经确定。此次按固定清单补齐最终评价的目的已达成；下一项仍是1600默认anchor第12轮，不根据本次分数改变顺序或调参。

- 完整test结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464`
- 原生指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\native_coco_metrics.json`
- 精确COCO指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\test_metrics.json`
- 全部低阈值预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\predictions.bbox.json`
- 坐标、类别、置信度及处理尺寸：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\predictions_details.json`
- 23张预测可视化：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\predicted_images`
- 配置/命令/版本/哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\pretest_record.json`
- 原始终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step16_27428_3464\step16_console.txt`
- 验收：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step16_review\acceptance.json`
- 独立保存预测复算：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step16_review\independent_coco_check.txt`
- GT与已保存预测总览：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step16_review\test_gt_and_visible_predictions.jpg`
- 固定显示阈值诊断：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step16_review\display_threshold_diagnostic.json`
