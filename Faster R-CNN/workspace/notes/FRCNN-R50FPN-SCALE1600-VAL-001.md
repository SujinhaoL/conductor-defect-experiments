# FRCNN-R50FPN-SCALE1600-VAL-001

核查日期：2026-10-14（Asia/Shanghai）。状态：100轮训练、val及固定第12轮test结果均已复核，STEP17正常完成。原运行内REVIEW_PENDING保留生成时状态，当前验收见本说明及独立acceptance.json。

运行编号：`step14_3575_9657`；原始目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657`。

## 科学问题与实际设置

这是用户明确要求的最后一次整图尺寸试验。检验同一Faster R-CNN、数据划分和训练预算下，短边目标1280提高到1600是否改善val。直接对照STEP11默认anchor8，不与STEP13的anchor4混称尺寸单因素实验。

原train107张/168框（broken39、loose129），val23张/37框（broken9、loose28）；test23张/36框本轮未用。沿用已有COCO及原划分。YOLO0/1→COCO1/2→模型0/1，num_classes=2；ROI背景不属于第三种缺陷。

冻结配置逐项还原后与STEP11一致：唯一科学参数变化是训练、val、固定train8快照的Resize(2133,1280)→(2666,1600)，keep_ratio=True。长边限制2666、短边尽量1600，保持比例，后续按32倍数补边，不强制拉伸。

默认anchor scale8、ratio0.5/1/2；batch2、FP32、workers0、seed42、deterministic=True；100epoch、每轮54批、总5400更新。SGD lr0.0025/momentum0.9/weight_decay0.0001；warmup500/start_factor0.001，milestones67/92、gamma0.1；RandomFlip0.5。实际5400条学习率与对照完全相同，所有loss有限。

AP预测score_thr0.001、ROI NMS0.5、max_per_img100、COCO maxDets[1,10,100]；显示阈值0.30。未改batch/AMP、未自动resume或重试。

同一官方完整COCO检测checkpoint重新初始化：`D:\小论文\model_projects\Faster R-CNN\workspace\weights\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`；来源：https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth；SHA256：`047c8118fc5ca88ba5ae1fab72f2cd6b070501fe3af2f3cba5cfa9a89b44b03e`。首次更新前344项逐值加载；仅ROI fc_cls/fc_reg的weight/bias因80→2重新初始化。没有续训私人权重。

实际版本：{'torch': '2.7.1+cu128', 'torchvision': '0.22.1+cu128', 'mmcv': '2.1.0', 'mmengine': '0.10.7', 'mmdet': '3.3.0', 'numpy': '1.26.4', 'pycocotools': '2.0.10'}；MMDetection commit：`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`。运行命令、依赖和冻结输入哈希见下方证据。

## 选模与验证结果

按原生val coco/bbox_mAP（三位小数）最大值、同值保留首次，选第12轮。最佳权重：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\best_coco_bbox_mAP_epoch_12.pth`；SHA256：`f2431a883b6e39ea03e906fd5546f4d7f6823d35bb8f72cb43ae5db019ab8167`。末轮SHA256：`52498439d946ec239688376c1ce2e009538c042563fd7579aa966248e48a72a7`。离线完整精度复算不重新挑选epoch。

|同一23张val，默认anchor8|800，第66轮|1280，第5轮|1600，第12轮|
|---|---:|---:|---:|
|mAP50|3.773949%|3.189325%|5.168345%|
|mAP50-95|1.273413%|1.487789%|1.284096%|
|COCO AR100|4.384921%|11.607143%|14.940476%|
|broken AP50|1.697313%|0.000000%|0.066748%|
|broken AP50_95|0.848656%|0.000000%|0.040049%|
|loose AP50|5.850585%|6.378650%|10.269942%|
|loose AP50_95|1.698170%|2.975578%|2.528143%|

上表为保存预测的精确pycocotools COCO复算。原生mAP50=0.052、mAP50-95=0.013另行保留。broken的原生类别AP50-95显示0.000，但精确结果为0.0004004894871509623（0.040049%），不能写成精确全零。普通P/R未单独计算留空，AR100不能代替普通Recall；补测test见末尾STEP17记录。

## 是否回答问题、支持什么结论

完成了预先确定的尺寸对照。相较1280，主要val mAP50-95变化-0.203693个百分点，mAP50变化+1.979020个百分点。较宽松IoU=0.50下有收益，但跨IoU阈值平均的主要AP没有提高，绝对表现仍低。不能仅挑mAP50或AR100的提高宣布整体泛化改善。

末轮固定train8的mAP50=99.174917%、mAP50-95=87.904290%；对照1280的mAP50-95=80.198020%。15/15GT存在IoU≥0.50同类候选。这是107张训练后其中8张的拟合诊断，不是全107张完全过拟合或独立泛化成绩。

此前1600的静态anchor几何检查仍有4个零正样本GT，而本次末轮最终框能够覆盖15个GT；这再次说明“零正anchor”不是该GT绝对无法被最终检测的证明。未新增本轮RPN/ROI内部推理，不能把最终框进步解释为已定位所有内部原因。

第100轮val mAP50-95=0.368537%，较早期选中轮低。训练拟合提升而val仍低，符合泛化受限/过拟合迹象；数据规模、外观分布和标注口径等仍为待证实假设，不能归因于单一原因或宣称全链路无错。

本轮保持了原问题与单因素设计，没有混入anchor改动。800、1280、1600的默认设置系列没有显示主要val AP随尺寸持续提高，也未解决低泛化问题；这可以成为小论文中的尺寸敏感性结果，包括没有改善的结果。它不证明任意大图策略都无效，也不足以要求以后所有模型统一用1600。

该尺寸训练已结束，不自动准备更大尺寸、anchor或模块。用户于2026-10-14指定固定已有权重补齐test，第12轮已完成。单seed、23张val及反复使用val开发的局限如实披露；旧test已被查看，不能称为完全未接触盲测。跨框架输入及AP实现不同，前四次仍是参考比较。

## 资源、可视化及异常

本次耗时6703.556秒（111.73分钟，含val/保存/快照），对照51.59分钟。峰值CUDA allocated 13694.866MiB（13.37GiB），对照8.70GiB。allocated不等于reserved或整机显存，单次耗时受运行条件影响，不做精确效率定律推断。

选中val有2021个低阈值预测，17张达到100框上限，所有23张都有低阈值预测。显示>0.30留下10个；按同类、分数降序、一对一IoU≥0.50规则为TP=3、FP=7、未匹配GT=34。仅作显示阈值描述，不调阈值或改变AP。

已目视核对末轮训练图、选中val GT/预测总览及训练/val趋势图。训练图较容易见到预测，val图多数在显示阈值下没有框，少量误检仍存在；显示无框不等于低阈值没有预测。

没有OOM或训练中断。100轮val文件、5次固定train8快照、5400行loss/LR、配置、数据/权重/归档哈希、选模及独立COCO复算均通过。首100次总loss均值0.390894，末100次0.019217。预期ROI四项类别输出层尺寸不匹配已逐项确认，其余344项确实加载。

## 证据完整路径

- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\resolved_scale1600_val_001.py`
- 依赖：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\requirements_freeze.txt`
- 命令/代码/数据与初始化来源及哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\pretraining_record.json`
- 实际预训练加载：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\actual_training_weight_load.json`
- 损失及学习率：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\losses.csv`
- 损失曲线：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\loss_curves.png`
- 固定train8末轮预测图：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\epoch_100\predictions_contact_sheet.jpg`
- 选中checkpoint及val精确指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\selected_checkpoint.json`
- 选中val预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\validation\epoch_012\predictions.bbox.json`
- 完整终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step14_3575_9657\step14_console.txt`
- 独立验收：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_review\acceptance.json`
- 选中val标注和显示框：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_review\selected_val_gt_green_pred_red.jpg`
- 1280与1600训练/val趋势：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_review\resolution_training_validation_comparison.png`
- 实际Resize和GT坐标核查：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_resolution\resolution_audit.json`

## STEP17 固定第12轮最终test

用户2026-10-14启动step17_23519_12608，正常退出码0。原test23图36框（broken8、loose28），沿用选定val的Resize(2666,1600)、keep_ratio=True、默认anchor8、FP32、score_thr0.001、ROI NMS0.5、max100。348项完整加载及数据/配置/权重/预测哈希、类别/原图坐标、23张保存图和独立COCO复算均通过。

|精确COCO指标|val：37框|test：36框|
|---|---:|---:|
|mAP50|5.168345%|5.520830%|
|mAP50-95|1.284096%|1.848211%|
|COCO AR100|14.940476%|14.821429%|
|broken AP50|0.066748%|0.959790%|
|broken AP50-95|0.040049%|0.269667%|
|loose AP50|10.269942%|10.081871%|
|loose AP50-95|2.528143%|3.426755%|

原生test mAP50=0.055、mAP50-95=0.018；上表为同次保存预测的精确复算，普通P/R未计算留空，AR100单列。1981个低阈值预测，8个score>0.30显示框。按同类、分数降序、一对一IoU>=0.50的显示阈值诊断为TP2/FP6/未匹配GT34；只作显示阈值描述，不替代AP/AR或普通P/R，不据test改阈值。

含读图、推理与保存图片耗时9.730124秒，峰值CUDA allocated1527.131MiB；不包含完整启动/CPU加载/末尾指标计算，不当设备总显存或严格效率比较。

本模型test数值高于两个1280设置，但val主要AP并未显示随尺寸持续提高。单seed、每集合23张、不同选中epoch及已有test使用的限制保留；不能根据此次test再挑尺寸/anchor，也不能证明扩大尺寸普遍解决问题。已达成固定权重最终评价的目的，按原清单进入YOLO11s1600第54轮，不追加训练。

- 完整test结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608`
- 精确指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\test_metrics.json`
- 原生指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\native_coco_metrics.json`
- 全部低阈值预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\predictions.bbox.json`
- 坐标/类别/分数：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\predictions_details.json`
- 23张可视化：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\predicted_images`
- 配置/环境/命令/哈希：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\pretest_record.json`
- 完整终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step17_23519_12608\step17_console.txt`
- 验收：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step17_review\acceptance.json`
- 独立保存预测复算：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step17_review\independent_coco_check.txt`
- GT与预测总览：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step17_review\test_gt_and_visible_predictions.jpg`
