# 输电导线缺陷检测项目约定

本文件只服务于 `D:\小论文\model_projects` 内的本项目，不是用户全局配置。首次整理日期：2026-10-13。内容根据实际文件核对；实验状态可能变化，继续前应查最新日志和记录，不把本文件中的阶段状态视为永久事实。

## 工作方式

- 使用中文解释，面向初学者，科学、客观，不迎合；事实、假设和待核查项分别表述。
- 用户自己在 VS Code 的 Windows **cmd** 终端启动训练及测试。助手不得自行启动训练，包括小样本过拟合、带更新的显存试跑；也不得为补表自行启动新的模型推理。
- 助手可读取本地文件，准备官方代码/权重、配置和脚本，做无模型前向的 CPU 数据/坐标/anchor 检查，复算已保存预测，整理结果。
- 一次推进一个阶段。准备后给出完整路径、可直接复制的 cmd 命令、正常标志、需要反馈的输出；收到输出先验收再进入下一阶段。已完成工作不要仅为重新整理而重跑。
- 每次实验后说明：目的是否达成、当前证据支持什么、是否偏离原问题、下一步的必要性与停止条件。不要因分数低无限追加实验。
- 优先 cmd。需要 PowerShell 时说明原因。只用对应项目环境，不混用 YOLO11、RT-DETR、Faster R-CNN 的 Python。
- 官方源码保持原样，实验修改放 workspace。独立运行使用唯一目录，失败/中断也保留；禁止覆盖已有结果、自动重试训练或静默修改 batch/AMP/学习率/尺寸。

## 路径与数据规则

- 总项目：`D:\小论文\model_projects`
- Faster R-CNN：`D:\小论文\model_projects\Faster R-CNN`
- 官方源码：`D:\小论文\model_projects\Faster R-CNN\mmdetection`
- 自定义文件：`D:\小论文\model_projects\Faster R-CNN\workspace`
- Faster 专用解释器：`D:\小论文\model_projects\Faster R-CNN\.venv\Scripts\python.exe`
- 原 YOLO 数据：`D:\小论文\conductor_defect_dataset\02_yolo11_dataset`
- 派生 COCO 数据：`D:\小论文\conductor_defect_dataset\03_coco_dataset`
- 原 YOLO 配置：`D:\小论文\model_projects\yolo11\workspace\configs\conductor_bl_v1.yaml`

原图/标签及原图片级划分不得覆盖、移动或重新随机划分。COCO 已转换并核对，不要重复建数据集。

|集合|图片|框|broken|loose|
|---|---:|---:|---:|---:|
|train|107|168|39|129|
|val|23|37|9|28|
|test|23|36|8|28|
|trainval|130|205|48|157|

共153张、241框。COCO annotations 下有 instances_train/val/test/trainval.json。图片在 images/train、images/val、images/test；trainval JSON 直接引用 train、val，不额外合并复制图片。

类别：YOLO `0=broken,1=loose` → COCO `1=broken,2=loose` → 模型内部 `0=broken,1=loose`。num_classes=2；ROI分类器的第3项是内部背景，不是第3个缺陷类别。

split_group_id 共138个（127个单图组、11个多图组共26张），不是138个已确认地点。已登记组未跨集合，scene_id 空；用户此前做过相似图人工分组。不能宣称完全排除所有同源泄漏，也不要要求用户补造未知 scene_id。发现新证据先解释，不擅自重划。

## 已核实环境与初始化

RTX 5070 Ti 16GB；Python3.10.20、torch2.7.1+cu128、torchvision0.22.1+cu128、mmcv2.1.0、mmengine0.10.7、mmdet3.3.0、numpy1.26.4、pycocotools2.0.10。
官方 MMDetection commit：`44ebd17b145c2372c4b700bfb9cb20dbd28ab64a`。MMCV已针对sm_120构建并通过算子/反向检查，官方COCO示例和实际训练已运行，不要重复安装。

初始权重：`D:\小论文\model_projects\Faster R-CNN\workspace\weights\faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth`
SHA256：`047c8118fc5ca88ba5ae1fab72f2cd6b070501fe3af2f3cba5cfa9a89b44b03e`
以上哈希在实际使用前必须与 step03_preparation.json 的 checkpoint.sha256 核对；不得仅依赖手工抄录。
来源：https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth

这是完整COCO检测器预训练，不是只有ImageNet骨干。348个state_dict项中344项兼容并逐值加载；仅ROI fc_cls/fc_reg的weight/bias因80→2目标类重新初始化。新的正式实验从同一官方COCO权重开始，不能接着私人数据权重训练来冒充独立对照。

## 已完成实验与证据入口

前四次：YOLO11s 1280、RT-DETR-L 1280、YOLO11s 1600、RT-DETR-L 1600。均100epoch，1280基础实验有test，1600实验没有test；精确设置以实际args.yaml、日志和结果为准。不要把optimizer=auto的lr0误当实际学习率，也不要把2700训练batch当2700次优化器更新。

Faster工作目录下 workspace/runs 的已完成运行：

- `FRCNN-DEBUG-OVERFIT-8\step06_24506_3904`：仅原train固定8图，100轮400更新；未完全过拟合，诊断用，不作为泛化成绩。原scale下15个GT中8个无RPN正anchor是已核实限制，不能认定为全部低分的唯一原因。
- `FRCNN-R50FPN-VAL-001\step07_235_27043`：107训练、23验证，100轮5400更新；按原生val mAP50-95选第66轮；test位于 `test_selected_epoch66\step08_31489_5707`，已核查。
- `FRCNN-R50FPN-MERGE130-001\step09_109_20338`：train+val130，100轮6500更新，无独立val；从COCO重新开始，固定最后第100轮；test位于 `test_selected_epoch100\step10_13778_21846`，已核查。
- `FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115`：107/23，Resize(2133,1280)，100轮5400更新；按同一原生val规则选第5轮。已完成训练及val复核，test未运行；核查见workspace/audit/step11_review/acceptance.json。mAP50=0.03189324821274552、mAP50-95=0.014877891782027844、AR100=0.11607142857142856（均为0..1底值），broken AP仍0。
- `FRCNN-DIAG-RPNROI-TRAIN8\step12_24867_6686`：已完成且复核通过；原800和新1280的epoch100各诊断同一原train8一次，共16图次。仅拟合诊断，不列入正式泛化成绩。独立复算见workspace/audit/step12_review/acceptance.json，说明见workspace/notes/FRCNN-DIAG-RPNROI-TRAIN8.md。

MERGE130的原val不能作为该运行的独立验证集。另起107/23对照且从官方COCO重新训练时，可以继续使用原val进行模型开发；不能使用MERGE130私人权重后再声称对原val独立验证。

两次正式Faster共同设置：Resize(1333,800)、keep_ratio=True，batch2、FP32、seed42、SGD lr0.0025/momentum0.9/weight_decay0.0001、warmup500、milestones67/92、gamma0.1、flip0.5、官方anchor。score_thr0.001、ROI NMS0.5、max_per_img100，显示阈值0.30。COCO maxDets=[1,10,100]。

优先读以下现有说明，按实际文件核验状态：

- `D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-VAL-001.md`
- `D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-MERGE130-001.md`
- `D:\小论文\model_projects\Faster R-CNN\workspace\notes\STEP06_result_review.txt`
- `D:\小论文\model_projects\Faster R-CNN\workspace\notes\EXPERIMENT_RECORDING_RULES.md`
- `D:\小论文\model_projects\baseline_analysis\audit\run_status_index.json`

原图GT：`D:\小论文\conductor_defect_dataset\03_coco_dataset\audit\ground_truth`
实际Resize/归一化/补边图：`D:\小论文\conductor_defect_dataset\03_coco_dataset\audit\debug8\step05_20493_32385`
正式训练运行的 artifacts 下保存losses.csv、loss_curves.png、epoch_000/025/050/075/100固定train8预测；VAL运行另有每轮validation预测。
完整日志在 Faster工作目录的 workspace/logs。已有检查覆盖数据转换、处理管线、权重加载、loss、anchor分配及保存预测评价。STEP12另保存16次诊断的RPN交接候选框、ROI类别回归/概率数组、逐GT局部框图和FPN形状；没有全面证明所有内部环节无错，未恢复历史训练随机ROI采样，也未保存完整特征图及RPN过滤前全量候选。

## 2026-10-13 的当前状态与下一阶段

尺寸对照STEP11已完成并核查，不能再当成未启动或重跑。以VAL-001为对照，保持107/23和训练设置，仅训练/验证/快照Resize改为(2133,1280)，从同一COCO权重重新训练。实际耗时3095.629秒、峰值CUDA allocated 8903.871MiB；配置差异、5400条学习率、归档哈希、100轮val文件、保存预测COCO复算均核对。
本次val mAP50-95小幅提高而mAP50下降，未解决低泛化问题。末轮固定train8诊断mAP50-95约80.198%，不作为泛化成绩；11/15GT有同类候选IoU≥0.50，4个loose仍未覆盖。第5轮val的2300个低阈值预测均不超过0.30，显示无框不等于无预测。
说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-VAL-001.md`。历史准备说明STEP11_resolution1280_plan.md和运行内冻结code保留原状态；实时核查状态以acceptance.json、当前说明及总表为准。

STEP12已完成，不应为整理再跑：800的RPN/ROI过滤前/最终同类框均覆盖8/15GT，1280均11/15（覆盖指存在IoU≥0.50候选，不是普通Recall）。1280未覆盖GT87/123/124/160均loose，RPN候选及ROI同类回归在过滤前最大IoU都低于0.50；四者恰好对应默认anchor几何检查中零正样本目标。支持anchor匹配假设，但不证明全部val误差由其导致。800的GT88零正anchor仍被检出，不能声称绝对不可检测。ROI训练还会加入GT候选，不能声称模型完全未学该目标。
16份保存数组独立复算、原图坐标/类别/概率/FPN形状、checkpoint哈希、末轮快照一致性均通过。RPN是自身NMS/top1000之后的交接候选，不是原始全量候选；没有定位RPN排序/NMS各自贡献。四个漏检局部图已目视核查，不凭外观断言标注错误。原val最佳epoch66/5不变，epoch100仅用于拟合诊断。

下一阶段已准备、尚未运行：`FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001`，命令入口workspace/scripts/run_step13_train_anchor4.cmd，方案workspace/notes/STEP13_anchor4_plan.md。唯一科学参数变化是RPN anchor_generator.scales=[8]→[4]；继承STEP11冻结配置，仍Resize(2133,1280)、107/23、batch2、FP32、seed42、SGD0.0025、100epoch5400更新、同调度/flip/过滤/val选模规则，从同一官方COCO重新开始，不续训私人权重。
原train107/168框的CPU anchor核查（workspace/audit/step13_anchor_geometry/anchor_geometry.json）显示：固定1280，scale8→4使零正anchor目标33→3，未翻转/水平翻转均如此；30个获得正anchor、原有正anchor者无新增遗漏，诊断4个目标均获正anchor，仍剩GT102/103/104。是采样前几何结果，不是召回率/AP或实际训练采样记录。每位置仍3个anchor，RPN张量形状不变；预训练张量原来对应默认几何，兼容加载不保证收益。
助手已执行CPU几何核查和STEP13 --check-preparation，未启动训练或推理。用户完成这一次训练后先验收并扩展正式登记器；不能直接假定新实验自动入总表。停止条件：完成固定100轮后评价假设；失败保留日志，不自动重试或改batch/AMP/尺度/阈值，不机械连续搜索更多anchor或尺寸。
按val进行模型开发，不反复用test调参，不自动递增到1600；静态1600数据只属于几何分析。诊断实验不放入正式泛化表。

## 指标、结果登记与比较边界

- 原始结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\<实验编号>\<运行编号>`；日志在workspace/logs；单次说明在workspace/notes/<实验编号>.md。
- 统一CSV：`D:\小论文\model_projects\baseline_analysis\10_all_experiments_metrics.csv`
- 总Excel：`D:\小论文\model_projects\baseline_analysis\outputs\模型实验结果总表.xlsx`
- 总报告：`D:\小论文\model_projects\baseline_analysis\11_all_experiments_summary_for_teacher.md`
- 历史四次报告 `D:\小论文\model_projects\baseline_analysis\09_four_experiments_summary_for_teacher.md` 和旧Excel保留。
- 已有登记入口：`D:\小论文\model_projects\baseline_analysis\scripts\refresh_experiment_records.py`。已扩展至七次正式实验，含STEP11的原生/精确val两行，共16个唯一评价记录；919个CSV/Excel相关单元格匹配。STEP11没有test行。其他新实验仍需先扩展读取逻辑并核对真实源文件，不能假定自动登记。
- 每行代表运行+checkpoint+评价集合+口径/事件，唯一评价ID防重复；从原始指标生成CSV和Excel，不手抄多份冲突数字。记录配置/代码版本/环境/命令/数据/初始及选中权重的来源与哈希。
- 比例底值0..1，Excel显示百分数。未算/不适用/未保存留空并记录原因，真实0才填0。
- MMDetection class *_precision 是类别AP50-95，不是普通Precision；AR100不能替代普通Recall。候选IoU覆盖数也不是普通Recall。
- 原生三位小数与保存预测的精确COCO复算分开登记；复算不得追溯改变已固定的checkpoint。只凭0.000不能断言精确全零。
- val/test分表；107/130训练规模注明。跨框架输入、AP计算、预训练、后处理不同，未统一前只能参考比较；同epoch不代表同更新数或同算力。单seed不宣称统计显著性。
- 用户已查看旧test结果。后续尺寸/模型开发用val，不能把旧test称为从未接触的盲测集；新泛化结论需要诚实披露这一限制。
- 每轮更新核对CSV/Excel数字、百分数、空值、集合、类映射、checkpoint/epoch/run及来源路径，最后明确新增文件和仍待核查项目。不为补表重跑模型。
