# 输电导线缺陷检测实验持续汇总

本文件由已保存原始指标生成。统一CSV为数值登记依据，Excel为显示视图。历史09报告与原四实验Excel保留。

## 评价口径

val与test分别报告；107张训练与130张训练分别注明。当前各框架原生AP只作参考对照，输入缩放、增强、预训练、后处理和评价实现并不相同。单seed结果不能证明统计显著性。

Ultralytics原生AP使用101点插值积分；普通P/R在IoU0.50、平滑平均分类F1最大处计算，实际阈值数值未保存。MMDetection使用pycocotools COCO AP，普通P/R未计算；AR100不能代替普通Recall。

YOLO：等比例缩放和LetterBox；RT-DETR：直接方形缩放；Faster R-CNN：保持比例的Resize边界，原基线(1333,800)、尺寸敏感性实验(2133,1280)和(2666,1600)。YOLO NMS IoU0.7、max300；RT-DETR不使用NMS、max300；Faster R-CNN ROI NMS0.5、max100。评价预测过滤0.001与FRCNN展示阈值0.30分别记录。

原生MMDetection输出保留三位小数，类别字段`*_precision`实际为AP50-95。保存预测的完整精度COCO复算另列，不改变已经固定的第66轮选模。历史1600实验选中轮次结果和自动重验证结果是两次评价，分别列示；终端输出不增加虚假小数位。

## 验证集结果（23图片、37框）

|评价编号|模型/输入|训练图片数|epoch|评价事件/口径|mAP50|mAP50-95|P|R|AR100|
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
|EV-6b0ae7bf163fff46|YOLO11s/1280|107|15|formal_independent_evaluation; Ultralytics-native-AP|2.2262%|0.4386%|1.2958%|23.4127%|未计算/未保存|
|EV-000db8ca7a751020|RT-DETR-L/1280|107|71|formal_independent_evaluation; Ultralytics-native-AP|1.8199%|0.5388%|50.8274%|1.7857%|未计算/未保存|
|EV-41ecd5b4af6778c7|YOLO11s/1600|107|54|selected_epoch_validation; Ultralytics-native-AP|3.0020%|1.4830%|17.1320%|5.5560%|未计算/未保存|
|EV-55e77811ab640740|YOLO11s/1600|107|54|automatic_best_checkpoint_revalidation; Ultralytics-native-AP|3.0000%|1.4800%|17.0000%|5.5600%|未计算/未保存|
|EV-be590b45b7acdf6a|RT-DETR-L/1600|107|42|selected_epoch_validation; Ultralytics-native-AP|2.2920%|1.1970%|53.5520%|3.5710%|未计算/未保存|
|EV-a18c7b02d7b1ec63|RT-DETR-L/1600|107|42|automatic_best_checkpoint_revalidation; Ultralytics-native-AP|2.1100%|0.9980%|53.3000%|3.5700%|未计算/未保存|
|EV-40a51201f85da60f|Faster R-CNN R50-FPN/1333x800 bounds|107|66|selected_epoch_validation; MMDetection-native-CocoMetric-rounded|3.8000%|1.3000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-ad6be269f0ee371b|Faster R-CNN R50-FPN/1333x800 bounds|107|66|selected_validation_offline_exact; pycocotools-exact-from-saved-predictions|3.7739%|1.2734%|未计算/未保存|未计算/未保存|4.3849%|
|EV-452954bbe43a2a66|Faster R-CNN R50-FPN/2133x1280 bounds|107|5|selected_epoch_validation; MMDetection-native-CocoMetric-rounded|3.2000%|1.5000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-14781186d6cd01f9|Faster R-CNN R50-FPN/2133x1280 bounds|107|5|selected_validation_offline_exact; pycocotools-exact-from-saved-predictions|3.1893%|1.4878%|未计算/未保存|未计算/未保存|11.6071%|
|EV-259b3e65cb0c4488|Faster R-CNN R50-FPN/2133x1280 bounds|107|15|selected_epoch_validation; MMDetection-native-CocoMetric-rounded|5.5000%|1.2000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-5ed5a5780b192a95|Faster R-CNN R50-FPN/2133x1280 bounds|107|15|selected_validation_offline_exact; pycocotools-exact-from-saved-predictions|5.4507%|1.1591%|未计算/未保存|未计算/未保存|5.0000%|
|EV-ffdb3a6bb25dbcdb|Faster R-CNN R50-FPN/2666x1600 bounds|107|12|selected_epoch_validation; MMDetection-native-CocoMetric-rounded|5.2000%|1.3000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-112e87e3c366da30|Faster R-CNN R50-FPN/2666x1600 bounds|107|12|selected_validation_offline_exact; pycocotools-exact-from-saved-predictions|5.1683%|1.2841%|未计算/未保存|未计算/未保存|14.9405%|

## 测试集结果（23图片、36框）

|评价编号|模型/输入|训练图片数|epoch|评价事件/口径|mAP50|mAP50-95|P|R|AR100|
|---|---|---:|---:|---|---:|---:|---:|---:|---:|
|EV-62e95ae48182ad81|YOLO11s/1280|107|15|formal_independent_evaluation; Ultralytics-native-AP|0.8108%|0.2288%|1.6253%|20.5357%|未计算/未保存|
|EV-c228a7b3cccdebdd|RT-DETR-L/1280|107|71|formal_independent_evaluation; Ultralytics-native-AP|0.0000%|0.0000%|0.0000%|0.0000%|未计算/未保存|
|EV-be51b4493b6c7e80|Faster R-CNN R50-FPN/1333x800 bounds|107|66|fixed_selected_test; MMDetection-native-CocoMetric-rounded|0.1000%|0.0000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-0949e68c6341c66c|Faster R-CNN R50-FPN/1333x800 bounds|107|66|fixed_test_offline_exact; pycocotools-exact-from-saved-predictions|0.0825%|0.0495%|未计算/未保存|未计算/未保存|1.0714%|
|EV-3a40bf61cdc04443|Faster R-CNN R50-FPN/1333x800 bounds|130|100|fixed_last_epoch_test; MMDetection-native-CocoMetric-rounded|0.0000%|0.0000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-be4f62d04282ccb5|Faster R-CNN R50-FPN/1333x800 bounds|130|100|fixed_last_test_offline_exact; pycocotools-exact-from-saved-predictions|0.0283%|0.0170%|未计算/未保存|未计算/未保存|1.0714%|
|EV-130728a8cb98331b|Faster R-CNN R50-FPN/2133x1280 bounds|107|5|fixed_selected_test; MMDetection-native-CocoMetric-rounded|0.8000%|0.3000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-f55e1b0a9e76409c|Faster R-CNN R50-FPN/2133x1280 bounds|107|5|fixed_test_offline_exact; pycocotools-exact-from-saved-predictions|0.8194%|0.2866%|未计算/未保存|未计算/未保存|6.2500%|
|EV-5f94298ae373ff9f|Faster R-CNN R50-FPN/2133x1280 bounds|107|15|fixed_selected_test; MMDetection-native-CocoMetric-rounded|1.9000%|0.4000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-d19b474c5188b02f|Faster R-CNN R50-FPN/2133x1280 bounds|107|15|fixed_test_offline_exact; pycocotools-exact-from-saved-predictions|1.9089%|0.4246%|未计算/未保存|未计算/未保存|8.6607%|
|EV-a748606a96200062|Faster R-CNN R50-FPN/2666x1600 bounds|107|12|fixed_selected_test; MMDetection-native-CocoMetric-rounded|5.5000%|1.8000%|未计算/未保存|未计算/未保存|未计算/未保存|
|EV-15c135fb2e1d0e9e|Faster R-CNN R50-FPN/2666x1600 bounds|107|12|fixed_test_offline_exact; pycocotools-exact-from-saved-predictions|5.5208%|1.8482%|未计算/未保存|未计算/未保存|14.8214%|

## 分类别AP

|评价编号|集合|broken AP50|broken AP50-95|loose AP50|loose AP50-95|
|---|---|---:|---:|---:|---:|
|EV-6b0ae7bf163fff46|val|1.2778%|0.2556%|3.1746%|0.6217%|
|EV-62e95ae48182ad81|test|0.8929%|0.2679%|0.7288%|0.1897%|
|EV-000db8ca7a751020|val|0.0000%|0.0000%|3.6397%|1.0775%|
|EV-c228a7b3cccdebdd|test|0.0000%|0.0000%|0.0000%|0.0000%|
|EV-41ecd5b4af6778c7|val|该来源未保存|该来源未保存|该来源未保存|该来源未保存|
|EV-55e77811ab640740|val|5.7500%|2.8700%|0.2520%|0.0907%|
|EV-be590b45b7acdf6a|val|该来源未保存|该来源未保存|该来源未保存|该来源未保存|
|EV-a18c7b02d7b1ec63|val|0.0000%|0.0000%|4.2200%|2.0000%|
|EV-40a51201f85da60f|val|该来源未保存|0.8000%|该来源未保存|1.7000%|
|EV-ad6be269f0ee371b|val|1.6973%|0.8487%|5.8506%|1.6982%|
|EV-be51b4493b6c7e80|test|该来源未保存|0.0000%|该来源未保存|0.1000%|
|EV-0949e68c6341c66c|test|0.0000%|0.0000%|0.1650%|0.0990%|
|EV-3a40bf61cdc04443|test|该来源未保存|0.0000%|该来源未保存|0.0000%|
|EV-be4f62d04282ccb5|test|0.0000%|0.0000%|0.0566%|0.0339%|
|EV-452954bbe43a2a66|val|该来源未保存|0.0000%|该来源未保存|3.0000%|
|EV-14781186d6cd01f9|val|0.0000%|0.0000%|6.3786%|2.9756%|
|EV-130728a8cb98331b|test|该来源未保存|0.0000%|该来源未保存|0.5000%|
|EV-f55e1b0a9e76409c|test|0.3238%|0.0324%|1.3151%|0.5409%|
|EV-259b3e65cb0c4488|val|该来源未保存|0.0000%|该来源未保存|2.3000%|
|EV-5ed5a5780b192a95|val|0.0000%|0.0000%|10.9014%|2.3183%|
|EV-5f94298ae373ff9f|test|该来源未保存|0.7000%|该来源未保存|0.1000%|
|EV-d19b474c5188b02f|test|3.4102%|0.7084%|0.4077%|0.1408%|
|EV-ffdb3a6bb25dbcdb|val|该来源未保存|0.0000%|该来源未保存|2.5000%|
|EV-112e87e3c366da30|val|0.0667%|0.0400%|10.2699%|2.5281%|
|EV-a748606a96200062|test|该来源未保存|0.3000%|该来源未保存|3.4000%|
|EV-15c135fb2e1d0e9e|test|0.9598%|0.2697%|10.0819%|3.4268%|

RT-DETR部分类别在原生P/R曲线插值中记录P=1、R=0。这是原生数值约定，不能解释为该类检测可靠；分类AP仍为0。


## 训练与评价状态

|实验|运行|train/val/test|迭代数/更新数|test状态|
|---|---|---|---|---|
|Y11-BASE-001-R1|Y11-BASE-001-R1_20260917_203044_883451|107/23/23|2700/未保存或未运行|COMPLETED_ONCE|
|RTD-BASE-001-R1|RTD-BASE-001-R1_20260918_172120|107/23/23|2700/未保存或未运行|COMPLETED_ONCE|
|Y11-SENS-IMGSZ1600-001|Y11-SENS-IMGSZ1600-001|107/23/23|2700/未保存或未运行|NOT_EVALUATED_BY_DESIGN|
|RTD-SENS-IMGSZ1600-001|RTD-SENS-IMGSZ1600-001|107/23/23|2700/未保存或未运行|NOT_EVALUATED_BY_DESIGN|
|FRCNN-R50FPN-VAL-001|step07_235_27043|107/23/23|5400/5400|COMPLETED_REVIEWED|
|FRCNN-R50FPN-MERGE130-001|step09_109_20338|130/不适用/23|6500/6500|COMPLETED_REVIEWED|
|FRCNN-R50FPN-SCALE1280-VAL-001|step11_31112_14115|107/23/23|5400/5400|COMPLETED_REVIEWED|
|FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001|step13_27622_23378|107/23/23|5400/5400|COMPLETED_REVIEWED|
|FRCNN-R50FPN-SCALE1600-VAL-001|step14_3575_9657|107/23/23|5400/5400|COMPLETED_REVIEWED|

四次Ultralytics实验为2700训练batch。梯度累积和AMP使它们不等于2700优化器更新；实际更新数历史未保存。Faster R-CNN VAL-001为5400更新，MERGE130-001为6500更新。同样100epoch不代表相同计算量。训练时间和显存采集口径见原始说明，不做精确效率排名。

YOLO1280和RTDETR1280的test各已使用一次；1600实验未做test。FRCNN VAL实验选模只依据val，后续测试分数不得用于挑epoch或调参。MERGE130从同一官方COCO初始权重重新训练，100epoch选最后轮；原val已进入train后不能作为独立val。MERGE130的固定epoch100 test已保存，核查状态见上表。

DEBUG8为诊断实验。最终train-fit mAP50约63.93%、mAP50-95约53.51%，未完全过拟合；不能放入正式泛化排名。固定官方anchor下有部分小框无正RPN anchor，这是已观察到的流程限制，不能直接认定为正式AP低的唯一原因。

## 已确认事实与解释限制

当前Faster R-CNN选中epoch66，val精确COCO mAP50=3.7739%、mAP50-95=1.2734%，仍较低。损失下降不能直接证明泛化良好。不能把没有显示框解释成完全没有预测，须同时查看低阈值保存预测。数据量、小目标缩放、类别不平衡等是可能因素，现有结果不能确定单一原因。

## 输入分辨率敏感性实验

FRCNN-R50FPN-SCALE1280-VAL-001保持107/23、COCO初始化及训练设置，改为Resize(2133,1280)。100轮5400更新，val选第5轮。精确val mAP50=3.1893%、mAP50-95=1.4878%、AR100=11.6071%。较原基线mAP50变化-0.5846个百分点，mAP50-95变化+0.2144个百分点，broken AP仍为0。

末轮固定train8的mAP50-95由58.6634%升至80.1980%，这是训练拟合诊断，不能作为val或泛化成绩。验证最优出现在早期，之后下降；仅放大整图没有解决低泛化问题，不能据此确定唯一原因或宣布尺寸无影响。

尺寸训练开发与选模依据val。用户随后指定固定第5轮补测；当前test完成状态见运行表与下方补测记录。旧test已被查看，不能称其为未接触盲测集。固定train8的RPN与ROI诊断见单独说明。

Faster R-CNN两次固定test均已完成并核查。107图训练、val选epoch66：精确COCO mAP50=0.082508%、mAP50-95=0.049505%、AR100=1.071429%；130图训练、固定epoch100：mAP50=0.028289%、mAP50-95=0.016973%、AR100=1.071429%。两次broken AP50/AP50-95均为0。MERGE130本次结果未显示泛化提升；训练规模、更新数及checkpoint选择规则不同，单seed不能证明合并训练必然降低效果。

MERGE130保存129个预测，score>0.30有6个；按现有标注，这6个与同类GT均未达到IoU0.50。36个GT中只有1个有低阈值同类候选IoU≥0.50，这是候选覆盖诊断，不是普通Recall。原生三位小数指标保留为0.000，同时单列保存预测的精确COCO复算结果，不将舍入的0解释为精确全零。

最终结果核查：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step10_review\acceptance.json`。原计划的DEBUG8、VAL-001和MERGE130-001已完成；没有据test成绩新增训练、挑epoch或调参。跨框架评价实现仍未统一，因此与前四次属于参考对照。

## 固定1280输入的anchor尺寸对照

FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001仅把RPN anchor scale8改为4，其余实际配置及5400条学习率轨迹与SCALE1280对照一致，同一官方COCO重新初始化。100轮完成，val按原规则选第15轮。精确val mAP50=5.4507%（对照3.1893%）、mAP50-95=1.1591%（对照1.4878%）、AR100=5.0000%；broken AP仍0。

固定train8末轮mAP50=100.0000%、mAP50-95=96.1906%，15/15GT存在IoU≥0.50同类框。此前4个loose训练漏检都获得准确的最终框，但这是107张训练后的8图拟合诊断，不能称全107张已完全过拟合，也不能声称恢复了原8图专用过拟合实验。

本次支持anchor尺寸匹配曾限制部分小训练目标拟合；主要val mAP50-95未提升、mAP50上升，不能宣布泛化问题已解决，不能只取有利指标。扩大整图和缩小anchor是两个不同因素，不能把本次当作纯输入尺寸实验或论文创新模块。它提供消融/诊断证据，不是论文必做项目。

当前停止自动追加尺寸、anchor或其他训练。用户随后授权固定第15轮补测test，完成状态见运行表。训练拟合与验证表现的差距仍需结合数据和标注标准解释；不把数据问题当成已经证实的结论。

说明及原始证据：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md`。

## 最后一次整图尺寸对照：1600

FRCNN-R50FPN-SCALE1600-VAL-001/step14_3575_9657完成100轮5400更新，原107/23划分，同一官方COCO重新初始化。相对STEP11默认anchor8的1280对照，仅训练/val/快照Resize提高为(2666,1600)，其余配置和5400条学习率一致。按原val规则选第12轮；不与STEP13的anchor4混称尺寸单因素比较。

精确val mAP50=5.1683%（1280为3.1893%），主要mAP50-95=1.2841%（1280为1.4878%），AR100=14.9405%。宽松IoU=0.50下有收益，主要AP未改善，不能仅挑有利指标。broken AP50=0.066748%、AP50-95=0.040049%，原生类别AP显示0.000只是舍入，精确值并非零。

末轮固定train8 mAP50-95=87.9043%（1280为80.1980%），15/15GT有IoU≥0.50同类候选；仅为训练子集拟合，不是全107图完全过拟合或泛化成绩。早期val较好、后期降低，训练/val差距仍大。

耗时111.73分钟，峰值CUDA allocated 13.37GiB，无OOM或中断。选定val保存2021个低阈值框、17张达到100框上限；显示>0.30有10框，一对一同类IoU≥0.50匹配TP3/FP7/未匹配GT34，仅作显示阈值描述，不改变AP或选模。

800→1280→1600默认anchor系列未显示主要val AP随尺寸持续提升，未解决低泛化问题；不能据此断言尺寸无影响或低分只有一个原因。本轮尺寸训练已结束，不追加训练。用户随后授权固定第12轮补测test。单seed、23张val且多轮开发、旧test已被查看的限制仍保留。

本次说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1600-VAL-001.md`。test状态见运行表，未完成项留空。

## 固定已有权重的最终test补齐

按用户指定顺序补测，每项保持已固定的val选模权重与评价配置。每次先核查并登记，再进行下一项；不根据test改选epoch或阈值。原test已有历史使用，不能重新宣称盲测。

FRCNN-R50FPN-SCALE1280-VAL-001，第5轮，原23图test：mAP50=0.819432%、mAP50-95=0.286644%、COCO AR100=6.250000%。保存2300个低阈值预测，score>0.30显示0个；没有显示框不能解释为零预测。

核查状态：VERIFIED_RESULT_REVIEW；完整结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311`。

原测试在23图预测和指标已保存后，因哈希函数收到字符串路径而在末尾归档报错。已仅用保存预测完成复核和补齐归档，没有再次模型推理；原FAIL日志保留。该次未落盘的推理耗时/显存留空，不估算或填0。

FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001，第15轮，原23图test：mAP50=1.908924%、mAP50-95=0.424630%、COCO AR100=8.660714%。保存2135个低阈值预测，score>0.30显示27个；没有显示框不能解释为零预测。

核查状态：VERIFIED_RESULT_REVIEW；完整结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464`。

FRCNN-R50FPN-SCALE1600-VAL-001，第12轮，原23图test：mAP50=5.520830%、mAP50-95=1.848211%、COCO AR100=14.821429%。保存1981个低阈值预测，score>0.30显示8个；没有显示框不能解释为零预测。

核查状态：VERIFIED_RESULT_REVIEW；完整结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608`。

## 固定训练图RPN与ROI诊断（不列入泛化成绩）

FRCNN-DIAG-RPNROI-TRAIN8/step12_24867_6686已完成并复核。两份epoch100权重各在固定原train8/15GT上诊断，16份保存数组独立复算、权重哈希、FPN形状和原末轮快照一致性检查通过。没有训练更新、val/test推理，也未改变原val选中的epoch66/5。

按IoU≥0.50的候选覆盖诊断，800的RPN/过滤前同类ROI/最终同类框分别覆盖8/8/8个GT；1280分别为11/11/11个。它们是逐GT是否存在候选的覆盖计数，不是普通Recall或独立泛化指标。

1280剩余GT87/123/124/160均为loose；RPN交给ROI的候选及ROI同类回归在过滤前最大IoU均低于0.50，不能只通过降低显示阈值解决。它们恰好对应默认anchor下零正样本目标，支持anchor几何匹配假设，但不是唯一原因的证明。RPN保存的是NMS/top1000后的输出，未恢复历史训练随机采样或全量特征图。

完整诊断说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-DIAG-RPNROI-TRAIN8.md`；复核：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step12_review\acceptance.json`。本诊断不向正式CSV/Excel新增AP记录；后续准备状态与实际成绩分开记录。

## 原始证据索引

- 数值CSV：`D:\小论文\model_projects\baseline_analysis\10_all_experiments_metrics.csv`
- 总Excel：`D:\小论文\model_projects\baseline_analysis\outputs\模型实验结果总表.xlsx`
- 登记验收：`D:\小论文\model_projects\baseline_analysis\audit\registration_acceptance.json`

### EV-6b0ae7bf163fff46

- 指标：`D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\val_metrics.json`
- 指标SHA256：`98bed9a3a93cbcce2f39ad6a82f6461b5da69f1ebf9b8e2a744bd303dad793cf`
- checkpoint：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\weights\best.pt`
- checkpoint SHA256：`01f6246e45a0b42fe93762120c8784d8a885dcba21567ab5906d94af2edb1cc9`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\args.yaml`
- 日志：`D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\Y11-BASE-001-R1_20260917_203044_883451_console.log`
- 说明：`D:\小论文\model_projects\yolo11\workspace\notes\formal_baseline_single_run_report.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-62e95ae48182ad81

- 指标：`D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\test_metrics.json`
- 指标SHA256：`1bc959ad5f1ed06379b2fb7fe2acee799343600fdade3738f662aeba62f08281`
- checkpoint：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\weights\best.pt`
- checkpoint SHA256：`01f6246e45a0b42fe93762120c8784d8a885dcba21567ab5906d94af2edb1cc9`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\args.yaml`
- 日志：`D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\Y11-BASE-001-R1_20260917_203044_883451_console.log`
- 说明：`D:\小论文\model_projects\yolo11\workspace\notes\formal_baseline_single_run_report.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-000db8ca7a751020

- 指标：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_val_metrics.json`
- 指标SHA256：`655813e93a82be0d6afb1c86a7b555bc9098125628c3d8dc3fcc1683f9efaf7a`
- checkpoint：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\weights\best.pt`
- checkpoint SHA256：`15ba1e62e1170d21e0628753be3eef0a710e06300643b3f9dd421f2c02d53ccd`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\args.yaml`
- 日志：`D:\小论文\model_projects\RT-DETR\workspace\logs\RTD-BASE-001-R1_20260918_172120.log`
- 说明：`D:\小论文\model_projects\RT-DETR\workspace\notes\rtdetr_single_run_report.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-c228a7b3cccdebdd

- 指标：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_test_metrics.json`
- 指标SHA256：`3074d18a4be53628d377c1188c5710f7d812dd560ebe37625ce828c0fdda3e86`
- checkpoint：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\weights\best.pt`
- checkpoint SHA256：`15ba1e62e1170d21e0628753be3eef0a710e06300643b3f9dd421f2c02d53ccd`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\args.yaml`
- 日志：`D:\小论文\model_projects\RT-DETR\workspace\logs\RTD-BASE-001-R1_20260918_172120.log`
- 说明：`D:\小论文\model_projects\RT-DETR\workspace\notes\rtdetr_single_run_report.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-41ecd5b4af6778c7

- 指标：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001\results.csv`
- 指标SHA256：`fe1622d4a461202930eb55e1fbe5d19019c543f06bbbdd2f504001e35c5b1178`
- checkpoint：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001\weights\best.pt`
- checkpoint SHA256：`2cc998a09da3eb36d717df79dfa28adf2a41cebb1121ecd43d984ff32ea4c92e`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001\args.yaml`
- 日志：`D:\小论文\model_projects\baseline_analysis\evidence\legacy_console\Y11-SENS-IMGSZ1600-001.txt`
- 说明：`D:\小论文\model_projects\yolo11\workspace\notes\Y11-SENS-IMGSZ1600-001.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-55e77811ab640740

- 指标：`D:\小论文\model_projects\baseline_analysis\evidence\legacy_console\Y11-SENS-IMGSZ1600-001.txt`
- 指标SHA256：`01231c54bc28632815080e0a9aaee9bd14257da9f95a32903a722c2607c27e63`
- checkpoint：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001\weights\best.pt`
- checkpoint SHA256：`2cc998a09da3eb36d717df79dfa28adf2a41cebb1121ecd43d984ff32ea4c92e`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001\args.yaml`
- 日志：`D:\小论文\model_projects\baseline_analysis\evidence\legacy_console\Y11-SENS-IMGSZ1600-001.txt`
- 说明：`D:\小论文\model_projects\yolo11\workspace\notes\Y11-SENS-IMGSZ1600-001.md`
- 核查状态：VERIFIED_PRINTED_PRECISION

### EV-be590b45b7acdf6a

- 指标：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001\results.csv`
- 指标SHA256：`6e1abf48b95ff2877db18ba2afcd54ad3d3a572e842c9738c885f193e543f321`
- checkpoint：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001\weights\best.pt`
- checkpoint SHA256：`29d23c8bb93bd650800ea439c9f43755b81f1763d0550ce4e14119dee4121a13`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001\args.yaml`
- 日志：`D:\小论文\model_projects\baseline_analysis\evidence\legacy_console\RTD-SENS-IMGSZ1600-001.txt`
- 说明：`D:\小论文\model_projects\RT-DETR\workspace\notes\RTD-SENS-IMGSZ1600-001.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-a18c7b02d7b1ec63

- 指标：`D:\小论文\model_projects\baseline_analysis\evidence\legacy_console\RTD-SENS-IMGSZ1600-001.txt`
- 指标SHA256：`61b1a2886ffd164695f09f9ecdc28e6ac9e27d05f23ffd6ae6392d5062c16774`
- checkpoint：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001\weights\best.pt`
- checkpoint SHA256：`29d23c8bb93bd650800ea439c9f43755b81f1763d0550ce4e14119dee4121a13`
- 机器可读预测未保存。引用已有图片，不为填表重跑推理。
- 配置：`D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001\args.yaml`
- 日志：`D:\小论文\model_projects\baseline_analysis\evidence\legacy_console\RTD-SENS-IMGSZ1600-001.txt`
- 说明：`D:\小论文\model_projects\RT-DETR\workspace\notes\RTD-SENS-IMGSZ1600-001.md`
- 核查状态：VERIFIED_PRINTED_PRECISION

### EV-40a51201f85da60f

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\validation\epoch_066\metrics.json`
- 指标SHA256：`0e9c3172156f3f4fe7cc5116d3d23a80da0ad611717055979b6c67ce5793995f`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\best_coco_bbox_mAP_epoch_66.pth`
- checkpoint SHA256：`7d80317d4f291fd8f7954423123af8e8154ad2dd4dad366353b05123ba326b39`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\validation\epoch_066\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\resolved_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step07_235_27043\train_console.log`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-VAL-001.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-ad6be269f0ee371b

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\best_validation_metrics.json`
- 指标SHA256：`dc9a3166b913b73d55c57d219ea49f0bb57d9cfc8537b6d220d6dae4a24ee9f3`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\best_coco_bbox_mAP_epoch_66.pth`
- checkpoint SHA256：`7d80317d4f291fd8f7954423123af8e8154ad2dd4dad366353b05123ba326b39`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\validation\epoch_066\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\resolved_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step07_235_27043\train_console.log`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-VAL-001.md`
- 核查状态：VERIFIED_SAVED_EVIDENCE

### EV-be51b4493b6c7e80

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\test_selected_epoch66\step08_31489_5707\native_coco_metrics.json`
- 指标SHA256：`ba0d610a46d062ad7f425aa346df4074734b69d33bcfe19146b9c0fcaa8dcf49`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\best_coco_bbox_mAP_epoch_66.pth`
- checkpoint SHA256：`7d80317d4f291fd8f7954423123af8e8154ad2dd4dad366353b05123ba326b39`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\test_selected_epoch66\step08_31489_5707\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\test_selected_epoch66\step08_31489_5707\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step08_31489_5707\step08_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-0949e68c6341c66c

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\test_selected_epoch66\step08_31489_5707\test_metrics.json`
- 指标SHA256：`5dac98074b0ef47c04a8876df2ae757403b6096e07fc4a33fdaf4e2da8b430f2`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\best_coco_bbox_mAP_epoch_66.pth`
- checkpoint SHA256：`7d80317d4f291fd8f7954423123af8e8154ad2dd4dad366353b05123ba326b39`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\test_selected_epoch66\step08_31489_5707\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\test_selected_epoch66\step08_31489_5707\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step08_31489_5707\step08_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-3a40bf61cdc04443

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\native_coco_metrics.json`
- 指标SHA256：`8ec3265041b4bd47d5e67feaf6980fe6c3b1e98541dc765da025b9f80f58213f`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\epoch_100.pth`
- checkpoint SHA256：`fccdc599d1e2bf2603b319d7f990f6e795dc8a24eef726e9a415723a409fd15a`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step10_13778_21846\step10_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-MERGE130-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-be4f62d04282ccb5

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\test_metrics.json`
- 指标SHA256：`79792faa8468ee44346eab5f4671ee4d3601277d4e473b29804a0a32c95f26e7`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\epoch_100.pth`
- checkpoint SHA256：`fccdc599d1e2bf2603b319d7f990f6e795dc8a24eef726e9a415723a409fd15a`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\test_selected_epoch100\step10_13778_21846\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step10_13778_21846\step10_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-MERGE130-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-452954bbe43a2a66

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\artifacts\validation\epoch_005\metrics.json`
- 指标SHA256：`7d3eaf7e9c769a95c265f553b4ad47461d8d638b2b0d1d60df5ef149e12fa0ee`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\best_coco_bbox_mAP_epoch_5.pth`
- checkpoint SHA256：`c8ff3af83efeec7cf3f126f49833da6e42489d9f022809d6237b94d9efdb08d3`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\artifacts\validation\epoch_005\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\resolved_scale1280_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step11_31112_14115\step11_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-14781186d6cd01f9

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\selected_checkpoint.json`
- 指标SHA256：`90c00307029c42b7cf2c12d28a15e6727c64606dbf289a7ef868adb6324e1613`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\best_coco_bbox_mAP_epoch_5.pth`
- checkpoint SHA256：`c8ff3af83efeec7cf3f126f49833da6e42489d9f022809d6237b94d9efdb08d3`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\artifacts\validation\epoch_005\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\resolved_scale1280_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step11_31112_14115\step11_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-130728a8cb98331b

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\native_coco_metrics.json`
- 指标SHA256：`c319820d6f8c407cc751951a34f861ed379274aa2744476dd2726b781da68e1b`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\best_coco_bbox_mAP_epoch_5.pth`
- checkpoint SHA256：`c8ff3af83efeec7cf3f126f49833da6e42489d9f022809d6237b94d9efdb08d3`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step15_23467_7311\step15_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-f55e1b0a9e76409c

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\test_metrics.json`
- 指标SHA256：`47bb9e7323706f72d1d6d16e9da5a17672c501f6de71ed14ae8520bc7ce34a06`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\best_coco_bbox_mAP_epoch_5.pth`
- checkpoint SHA256：`c8ff3af83efeec7cf3f126f49833da6e42489d9f022809d6237b94d9efdb08d3`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\test_selected_epoch5\step15_23467_7311\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step15_23467_7311\step15_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-259b3e65cb0c4488

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\artifacts\validation\epoch_015\metrics.json`
- 指标SHA256：`60bfd54daaf6feeffc5ba2d6e07c4626eefd9cd10f148be4972b22e0d47ff2b5`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\best_coco_bbox_mAP_epoch_15.pth`
- checkpoint SHA256：`080c27214809847dce4e2d1b5e760a3e8b868bdb0d0f2941e5e43f219ffb548a`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\artifacts\validation\epoch_015\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\resolved_scale1280_anchor4_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step13_27622_23378\step13_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-5ed5a5780b192a95

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\selected_checkpoint.json`
- 指标SHA256：`cacc1dbe7e74d9e2ed19aa9b4b88ec32e9f68e9dcb37a32cd0c0be72bd1f9841`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\best_coco_bbox_mAP_epoch_15.pth`
- checkpoint SHA256：`080c27214809847dce4e2d1b5e760a3e8b868bdb0d0f2941e5e43f219ffb548a`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\artifacts\validation\epoch_015\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\resolved_scale1280_anchor4_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step13_27622_23378\step13_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-5f94298ae373ff9f

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\native_coco_metrics.json`
- 指标SHA256：`e7e546aa5b426f1fee8132e1853118d9fc221ca3e6c59fc903ed14611550ac91`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\best_coco_bbox_mAP_epoch_15.pth`
- checkpoint SHA256：`080c27214809847dce4e2d1b5e760a3e8b868bdb0d0f2941e5e43f219ffb548a`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step16_27428_3464\step16_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-d19b474c5188b02f

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\test_metrics.json`
- 指标SHA256：`bbaa97e361351feb84c291e8acd0b89530ea944c8f35a474387f7123fff0b5a4`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\best_coco_bbox_mAP_epoch_15.pth`
- checkpoint SHA256：`080c27214809847dce4e2d1b5e760a3e8b868bdb0d0f2941e5e43f219ffb548a`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378\test_selected_epoch15\step16_27428_3464\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step16_27428_3464\step16_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-ffdb3a6bb25dbcdb

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\validation\epoch_012\metrics.json`
- 指标SHA256：`674dd19ae5292745f8583883cce5d1165865a95236ed845a6d54f62273299e66`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\best_coco_bbox_mAP_epoch_12.pth`
- checkpoint SHA256：`f2431a883b6e39ea03e906fd5546f4d7f6823d35bb8f72cb43ae5db019ab8167`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\validation\epoch_012\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\resolved_scale1600_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step14_3575_9657\step14_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1600-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-112e87e3c366da30

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\selected_checkpoint.json`
- 指标SHA256：`b5c34f171685c88cc2039e4ce334ded8475bf64c0677b2cfcbc0f7ef2fd3df4c`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\best_coco_bbox_mAP_epoch_12.pth`
- checkpoint SHA256：`f2431a883b6e39ea03e906fd5546f4d7f6823d35bb8f72cb43ae5db019ab8167`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\artifacts\validation\epoch_012\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\resolved_scale1600_val_001.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step14_3575_9657\step14_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1600-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-a748606a96200062

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\native_coco_metrics.json`
- 指标SHA256：`c28abce914f744596db9aa75b1ce11258192e357cbe7c4dcf9eaf66feb38c5d3`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\best_coco_bbox_mAP_epoch_12.pth`
- checkpoint SHA256：`f2431a883b6e39ea03e906fd5546f4d7f6823d35bb8f72cb43ae5db019ab8167`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step17_23519_12608\step17_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1600-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

### EV-15c135fb2e1d0e9e

- 指标：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\test_metrics.json`
- 指标SHA256：`7b458f42874b7f43a558fc9456788d15540f86d7ae3bd81fde2ae510f7518b04`
- checkpoint：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\best_coco_bbox_mAP_epoch_12.pth`
- checkpoint SHA256：`f2431a883b6e39ea03e906fd5546f4d7f6823d35bb8f72cb43ae5db019ab8167`
- 预测：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\predictions.bbox.json`
- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657\test_selected_epoch12\step17_23519_12608\resolved_test_config.py`
- 日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step17_23519_12608\step17_console.txt`
- 说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1600-VAL-001.md`
- 核查状态：VERIFIED_RESULT_REVIEW

