# 固定1280输入的anchor尺寸对照

实验编号：FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001。
准备日期：2026-10-13。当前状态：配置及CPU检查已准备，训练尚未启动。

## 目的与依据

STEP12已在两份epoch100权重、同一固定原train8上完成RPN→ROI诊断。1280下4个loose目标在RPN传给ROI的候选框中均无IoU≥0.50框，类别相关ROI回归在过滤前仍未达到0.50。这4个目标也对应默认anchor下无正样本的目标。它是可检验的线索，不是所有验证误差的已证实原因。

补充CPU几何核查只读取原train107/168框，使用当前官方AnchorGenerator和MaxIoUAssigner，固定Resize(2133,1280)，分别检查不翻转和水平翻转；没有模型前向、训练、val/test评价。默认scale8的零正anchor目标为33/168（broken2、loose31），预先选择的单个候选scale4为3/168（全loose）。两种翻转状态结论相同，30个目标新增正anchor，原来有正anchor的目标没有变成零正anchor。诊断4个目标都获得正anchor；仍剩GT102/103/104，不继续搜索更多尺寸。

几何检查位于：
- `D:\小论文\model_projects\Faster R-CNN\workspace\audit\step13_anchor_geometry\anchor_geometry.json`
- `D:\小论文\model_projects\Faster R-CNN\workspace\audit\step13_anchor_geometry\train107_anchor_comparison.csv`

这些是随机采样前的分配结果，不代表每次训练必定采到每个目标，不是RPN召回率或AP，也不保证泛化改善。检查脚本已按CocoDataset先将xywh转换为xyxy、再转换float32，与既有debug8真实Resize检查逐项一致；开发时发现并修正了相反运算顺序引起的浮点一致性断言失败，没有改动原始数据或既有实验结果。

## 唯一实验因素

以已完成的FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115为直接对照。

- 只将 `model.rpn_head.anchor_generator.scales` 从 `[8]` 改成 `[4]`。
- 最小层的方形anchor边长从32变为16输入像素；五层方形边长从32/64/128/256/512变为16/32/64/128/256。三种长宽比仍为0.5/1/2，每个位置仍3个anchor，RPN输出层张量形状不变。
- 输入仍为Resize(2133,1280)、keep_ratio=True；train/val及固定训练图快照一致。
- 原train107/168框、val23/37框；数据、标签、划分不变。
- batch2、FP32、workers0、seed42、100epoch、每轮54步、总5400次优化更新。
- SGD lr0.0025、momentum0.9、weight_decay0.0001；warmup500、factor0.001；milestones67/92、gamma0.1；flip0.5。
- RPN/ROI结构、匹配阈值、采样、损失、score_thr0.001、ROI NMS0.5、max_per_img100均不改；显示阈值0.30。
- 同一官方COCO完整检测checkpoint重新开始，resume=False；绝不续训STEP11私人权重。COCO来源和哈希从step03_preparation.json核对。训练首次更新前仍检查344项精确加载、4项ROI输出层因80→2重置。

anchor几何变化不增加参数数量，预训练RPN张量形状仍兼容，但这些张量原先在默认anchor几何上训练；兼容加载不等于新的几何先验没有影响。这正是本次对照要检验的因素，不能把它冒充新的网络创新模块。

配置继承STEP11实际运行的冻结配置。启动前将scale4还原为8，去除experiment_id/work_dir记录字段后，完整配置必须与冻结对照完全相等。

## 评价与停止条件

每轮仅评价原val。按相同原生三位小数coco/bbox_mAP（mAP50-95）最大值选checkpoint，同值保留首次；完整精度保存预测复算用于核查，不另挑epoch。
主比较为选定checkpoint的val mAP50-95，同时报告mAP50、各类AP和AR100。固定train8的拟合与4个目标的变化只作机制诊断，不混入泛化成绩。普通Precision/Recall未算时留空。

本轮不运行test；也不自动增大分辨率、更改anchor比例、降低匹配阈值或追加另一轮训练。完成100轮并核查后结束本次对照；若几何改善没有带来可解释的训练/val收益，不能继续把低分归因于默认anchor，应先总结证据再决定研究方向。单seed、反复使用的23张val不支持统计显著性或所有模型的普适结论。已有test被查看的历史如实保留。
OOM、非有限损失、加载或数据断言异常时停止，保留失败目录，不自动重试、resume或改batch/AMP。此前同1280运行约51.59分钟、峰值allocated8.70GiB，只供时间和资源参考，不保证本次完全相同。

## 用户cmd命令

```cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
call "D:\小论文\model_projects\Faster R-CNN\workspace\scripts\run_step13_train_anchor4.cmd"
```

该命令将启动一次训练。助手只执行过 `--check-preparation` 分支，没有构建检测器或启动训练。
正常开始：ANCHOR_ONLY_CONFIG_DIFF_PASS、STEP13_PREPARATION_PASS、FORMAL_OFFICIAL_COCO_WEIGHT_LOAD_PASS（matched=344 expected_new_outputs=4）。
正常结束：FORMAL_100_EPOCHS_5400_UPDATES_COMPLETED、BEST_VAL_EPOCH=...、STEP13_VAL_TRAIN_COMPLETED、TEST_NOT_RUN、STEP13_EXIT_CODE=0。
反馈最后约60行、RUN_DIR及LOG_DIR；如报错，反馈完整Traceback，先不要重跑。

## 文件与记录

- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\configs\frcnn_r50fpn_scale1280_anchor4_val_001.py`
- 启动脚本：`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\step13_run_anchor4.py`
- 静态验收：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step13_preflight\preparation_check.json`
- 原始结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_<唯一编号>`
- 完整日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step13_<同一编号>`，包括step13_console.txt和train_console.log。

沿用已验证的记录hook：冻结配置/脚本/依赖/数据与初始化哈希、首次实际加载证据、总损失与RPN/ROI损失及学习率、耗时/显存、epoch0/25/50/75/100固定train8完整预测、每轮val预测与原生指标、最佳/末轮checkpoint和哈希。不会复制大型初始权重和原图。
本轮未增加历史ROI随机采样或完整特征图导出；已有STEP12完成有目标的中间框诊断，不需为开本实验重跑该诊断。

训练完成后由助手验收，扩展 `D:\小论文\model_projects\baseline_analysis\scripts\refresh_experiment_records.py` 读取本轮真实结果，再更新统一CSV/Excel/总报告及本轮说明。当前登记器识别七次已完成正式实验，本轮尚未登记为成绩。禁止为补表自行推理或将未评价字段填0。
