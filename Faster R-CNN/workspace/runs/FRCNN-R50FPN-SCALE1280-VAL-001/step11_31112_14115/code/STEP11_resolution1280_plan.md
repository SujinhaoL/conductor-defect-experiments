# 输入分辨率敏感性实验与现有流程证据

准备日期：2026-10-13（Asia/Shanghai）。实验编号：FRCNN-R50FPN-SCALE1280-VAL-001。
状态：准备完成，静态检查通过；尚未启动训练。CPU数据处理/anchor检查不等于训练或检测器推理。

## 问题与假设

已有六次正式实验的泛化表现很低。缩小高清图片会使缺陷只占少量像素，这可能贡献了误差；现有结果不能证明它是唯一原因。前四次中的YOLO11和RT-DETR已经做过1280→1600敏感性实验，部分val AP提高但绝对值仍低，不能说尺寸完全没试过或完全有效。

本次检验：在Faster R-CNN R50-FPN、原train107/val23、初始化/训练设置不变时，提高整图输入分辨率是否改善验证表现。它是一个单seed探索性对照，不能只凭一次正/负结果解释所有六次实验，更不能保证以后各模型都必须用这个尺寸。

Faster R-CNN是经典的两阶段参考模型。它提供RPN候选区域和ROI分类/回归两个可诊断环节，可与YOLO11、RT-DETR形成不同检测范式的参考对照。它不是确定标注真假的“金标准”，也不保证在本数据上优于其他模型。导师的具体动机需以导师本人表述为准。

## 两个尺寸数字的含义

当前mmcv.image.geometric.rescale_size源码采用：

`s = min(长边上限/max(W,H), 短边目标/min(W,H))`

新宽高约为(W*s,H*s)，再按实际取整。原配置(1333,800)表示保持比例，短边尽量达到800、同时长边不超过1333；极宽长图会先触及长边上限，因此短边可能小于800。不是把每张图强行拉成1333×800。后续batch还有32倍数的补边。

例如本数据存在6000×4000原图：原配置得到1200×800；新配置(2133,1280)得到1920×1280。5472×3078的16:9图在原配置下约1333×750，新配置约2133×1200。像素面积约增加2.56倍，但显存/耗时不会严格线性等比例，图片总量少也不能保证单batch放得下。

Faster的短边1280不等于YOLO的imgsz1280；YOLO按其LetterBox规则，RT-DETR按其固定正方形缩放。跨模型应记录实际输入/补边/目标像素，不能只比较数字名称。

## 本轮实际做过的静态检查

来源：原train107/168框与val23/37框；不利用test图片或分数选尺寸。使用当前安装版本的mmcv尺寸函数、真实MMDetection LoadImage/LoadAnnotations/Resize以及原AnchorGenerator/MaxIoUAssigner，未建立模型或优化器，未做前向/反向。

|尺度（长边上限,短边目标）|train框短边中位数|train短边<4像素|train短边<8像素|固定train8中零正anchor目标|
|---|---:|---:|---:|---:|
|(1333,800)|12.506像素|7/168|44/168|8/15|
|(2133,1280)|20.009像素|2/168|12/168|4/15|
|(2666,1600)|25.011像素|0/168|7/168|4/15|

小于4/8像素仅为描述性统计，不是通用可检测阈值。anchor结果只对应固定8图、固定未翻转batch、采样前分配，不等于全部训练集或RPN proposal recall；不能从几何变化直接推出AP变化。
cond_000087中一个loose由约19.49×3.11变成31.19×4.98像素；1280仍未消除所有细小目标。1600仅做了CPU几何分析，没有准备或启动第二个尺寸训练。

- 明细/哈希/统计：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_resolution\resolution_audit.json`
- 训练/验证逐框尺寸：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_resolution\box_sizes_train_val.csv`
- 固定8图anchor明细：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_resolution\debug8_anchor_sizes.csv`
- 1:1像素对照图：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_resolution\train_debug8_resize_800_vs_1280.png`
- 训练框短边分布：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_resolution\train_box_short_edge_cdf.png`

## 固定实验设计

对照为FRCNN-R50FPN-VAL-001/step07_235_27043。新配置继承该运行的冻结resolved_val_001.py；静态检查将新尺寸还原后逐项比较整个配置，除实验标签/work_dir之外所有设置必须相同。

- 原train107/168框训练，val23/37框选模，test23不参与本轮。
- 同一个官方完整COCO checkpoint重新初始化；不接着epoch66/100私人权重。
- Resize仅从(1333,800)变为(2133,1280)，训练、验证、固定train8快照同时修改，keep_ratio=True。
- batch2、workers0、FP32、seed42、deterministic=True、100epoch、5400次更新。
- SGD lr0.0025/momentum0.9/weight_decay0.0001；warmup500/start_factor0.001，milestones67/92、gamma0.1。
- RandomFlip0.5，其余处理和官方anchor、RPN/ROI结构均沿用。
- score_thr0.001、ROI NMS0.5、max_per_img100；显示阈值0.30。
- 每轮val。与原基线一致：按原生三位小数coco/bbox_mAP选最佳，同值保留首次；保存每轮预测供完整精度复算，复算不事后换权重。
- 主比较指标：相同val上的选定checkpoint精确COCO mAP50-95，同时看mAP50、两类AP、AR100、每轮轨迹及固定train图变化。普通P/R未计算时留空。
- 不根据test尝试不同尺寸/epoch，不自动追加1600训练。OOM/非有限损失/异常时保留日志并停止，不静默降batch、开AMP或resume。

现有RTX5070Ti约16GB，原batch2训练记录的allocated峰值约3.77GiB；1280可能可行但尚未做GPU训练显存实测，不能保证不会OOM。保留batch与FP32是为了控制变量，真有资源问题再单独决定并记录，不拿不等价实验冒充只改尺寸。

## 由用户在cmd启动一次

```cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
call "D:\小论文\model_projects\Faster R-CNN\workspace\scripts\run_step11_train_scale1280.cmd"
```

不需要新建环境、重新转数据或手动改原图。以上命令会开始本次训练；助手只运行过--check-preparation静态分支。

正常开始：SCALE_ONLY_CONFIG_DIFF_PASS、STEP11_PREPARATION_PASS、FORMAL_OFFICIAL_COCO_WEIGHT_LOAD_PASS: matched=344 expected_new_outputs=4，然后Epoch(train)[1][.../54]。预期80→2类输出层四项shape mismatch可以解释，其他加载异常应停止。
正常结束：FORMAL_100_EPOCHS_5400_UPDATES_COMPLETED、BEST_VAL_EPOCH=...、STEP11_VAL_TRAIN_COMPLETED、STEP11_EXIT_CODE=0。
把最后约60行及RUN_DIR、LOG_DIR反馈给助手；报错时给完整Traceback，先不要重试。训练结束核查与登记后再决定下一阶段。

原始证据随训练保存：

- `D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_<唯一编号>`
- `D:\小论文\model_projects\Faster R-CNN\workspace\logs\step11_<同一编号>`，包含step11_console.txt和train_console.log。
- run内有冻结配置/脚本/依赖、数据及初始权重哈希、实际加载证据、losses.csv、loss_curves.png、epoch0/25/50/75/100固定train图、每轮val预测及指标、选定checkpoint与哈希。

记录辅助代码只改了快照读取尺寸参数、显示实验名及输出尺寸信息；旧运行code内冻结版本与已有结果保留。若只改主配置不改原来快照里固定800的尺寸，会造成新训练与可视化尺度不一致，本次已纠正并核对。
当前总表脚本只识别前六次正式实验；本轮完成并反馈后应扩展读取器、核对val结果并新增唯一评价行。不能为补表跑test，也不能把“准备完成”录成实际成绩。

## 是否要把以后所有实验统一用大尺寸

如果新尺寸在val有稳定且有意义的收益，并且资源代价可接受，可在后续一组基线/创新模块对照开始前固定该预处理策略。单次单seed提升只支持当前模型/设置下的探索性结论；正式主张尽量用事先确定的多seed或其他独立证据检验，不根据看到的test挑设置。
创新模块对照中，基线与改进模型应使用同样的数据、输入策略、训练预算/优化设置和评价口径。跨模型可采用统一输入政策或相同资源预算，但要说明几何变换差异，不能机械认为相同imgsz数字就是等价。
若提升很小、两类失衡仍严重，或显存/时间代价过高，应停止盲目放大，结合已有证据考虑切片/导线ROI/anchor等明确的新问题；不能同时改多个因素来证明“尺寸是原因”。切片是新的处理策略，应单列实验，按原图划分再切片，检测映射回原图后评价。
原test已被多次查看，后续不能称完全未接触的盲测；最终研究结论应披露这一点，条件允许时增加新独立测试数据。

## 现有可视化与封装链路检查入口

- 原图GT：`D:\小论文\conductor_defect_dataset\03_coco_dataset\audit\ground_truth`
- 真正的原图→Resize→归一化/补边反变换：`D:\小论文\conductor_defect_dataset\03_coco_dataset\audit\debug8\step05_20493_32385\pipeline_contact_sheet.jpg`
- 管线逐图报告：同目录pipeline_report.json、step05_summary.json（8图15框、坐标检查、344权重项加载通过）。
- VAL-001损失/LR曲线：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\loss_curves.png`
- 仅修正总标题的VAL-001曲线副本：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step11_resolution\baseline107_loss_curves_title_corrected.png`。原画图函数的循环变量覆盖了总标题参数，导致原图顶端显示Learning rate used；本次使用原5400条loss数值重新绘图，没有修改曲线数据、权重或成绩。新图表函数已修正这一显示问题，旧原图保留。
- MERGE130损失/LR曲线：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-MERGE130-001\step09_109_20338\artifacts\loss_curves.png`
- 两个运行的artifacts/epoch_000、025、050、075、100下有固定train8预测总览和逐图预测。
- VAL-001每轮验证：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-VAL-001\step07_235_27043\artifacts\validation`
- 原尺寸anchor诊断：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step06_review\per_box_review.csv`
- 数据转换验收：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step04_independent_acceptance.json`
- 最终预测/评价核查：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step08_review\acceptance.json`与`step10_review\acceptance.json`。

已验证图像与标签匹配、类别/原图坐标、实际Resize/颜色归一化/补边、预训练加载、有限损失、固定train拟合、默认anchor正样本分配、预测导出/COCO评价一致性。DEBUG8未完全过拟合，故不能写“全链路已充分证明无误”。
尚未完整保存每层特征图、RPN全部proposal、ROI采样/回归中间状态；没有完整的RPN proposal recall诊断。不要把损失曲线等同于所有内部层已可视化。若导师特指这些环节，应另设有问题导向的诊断；仅画热力图不能证明实现正确。

## 可借鉴的原始论文

- [Faster R-CNN，2015](https://arxiv.org/abs/1506.01497)：RPN生成候选区域，后接检测分类/回归，是两阶段参考与诊断分解的依据；并不意味着本任务精度必然高。
- [SNIP，CVPR 2018](https://arxiv.org/abs/1711.08189)：分析目标尺度变化，使用图像金字塔和按尺度选择目标训练；提示应关注输入中的目标尺寸与训练/评价尺度匹配，而不是盲用默认尺寸。
- [SNIPER，NeurIPS 2018](https://arxiv.org/abs/1805.09300)：在多尺度图像中取带上下文的局部chips训练，兼顾目标尺度和计算成本；不要求始终把整张高清图放大。
- [SAHI，ICIP 2022](https://arxiv.org/abs/2202.06934)：切片辅助推理与微调，用于小目标；其公开实验不能直接保证本导线数据改善。切片边界、重叠合并、完整原图坐标评价与额外计算预算要单独记录。

这些工作给出研究路线，不给本数据通用最优尺寸。本次先做可解释的800→1280对照，再依据val和资源记录决定。
