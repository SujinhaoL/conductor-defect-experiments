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
完整日志在 Faster工作目录的 workspace/logs。已有检查覆盖数据转换、处理管线、权重加载、loss、anchor分配及保存预测评价；没有全面证明所有内部环节无错，也未保存完整特征图/RPN proposals/ROI采样可视化。

## 2026-10-13 的下一阶段

用户希望验证提高输入分辨率是否有帮助。先准备 `FRCNN-R50FPN-SCALE1280-VAL-001`：以VAL-001为对照，保持107/23划分和训练设置，只把训练/验证/固定快照的Resize改为 `(2133,1280)`、keep_ratio=True；从同一COCO权重重新训练。尚未运行的计划不能登记成结果。
准备与解释见 `D:\小论文\model_projects\Faster R-CNN\workspace\notes\STEP11_resolution1280_plan.md`；静态尺寸核查在 workspace/audit/step11_resolution。
这是短边目标1280，不等同于YOLO imgsz1280。需要同时核对实际模型输入像素数、GT缩放后尺寸、batch、更新数、显存和耗时。
按val比较尺寸，不反复用test调参。先做一个尺寸，收到用户输出后验收；不得自动递增尺寸跑一串训练。静态检查中的1600只是候选几何分析，不是已启动训练。

## 指标、结果登记与比较边界

- 原始结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\<实验编号>\<运行编号>`；日志在workspace/logs；单次说明在workspace/notes/<实验编号>.md。
- 统一CSV：`D:\小论文\model_projects\baseline_analysis\10_all_experiments_metrics.csv`
- 总Excel：`D:\小论文\model_projects\baseline_analysis\outputs\模型实验结果总表.xlsx`
- 总报告：`D:\小论文\model_projects\baseline_analysis\11_all_experiments_summary_for_teacher.md`
- 历史四次报告 `D:\小论文\model_projects\baseline_analysis\09_four_experiments_summary_for_teacher.md` 和旧Excel保留。
- 已有登记入口：`D:\小论文\model_projects\baseline_analysis\scripts\refresh_experiment_records.py`。它目前只识别前六次正式实验；新增实验必须先扩展读取逻辑并核对真实源文件，不能假定新运行会自动进入总表。
- 每行代表运行+checkpoint+评价集合+口径/事件，唯一评价ID防重复；从原始指标生成CSV和Excel，不手抄多份冲突数字。记录配置/代码版本/环境/命令/数据/初始及选中权重的来源与哈希。
- 比例底值0..1，Excel显示百分数。未算/不适用/未保存留空并记录原因，真实0才填0。
- MMDetection class *_precision 是类别AP50-95，不是普通Precision；AR100不能替代普通Recall。候选IoU覆盖数也不是普通Recall。
- 原生三位小数与保存预测的精确COCO复算分开登记；复算不得追溯改变已固定的checkpoint。只凭0.000不能断言精确全零。
- val/test分表；107/130训练规模注明。跨框架输入、AP计算、预训练、后处理不同，未统一前只能参考比较；同epoch不代表同更新数或同算力。单seed不宣称统计显著性。
- 用户已查看旧test结果。后续尺寸/模型开发用val，不能把旧test称为从未接触的盲测集；新泛化结论需要诚实披露这一限制。
- 每轮更新核对CSV/Excel数字、百分数、空值、集合、类映射、checkpoint/epoch/run及来源路径，最后明确新增文件和仍待核查项目。不为补表重跑模型。
