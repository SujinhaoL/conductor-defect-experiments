# 最后一次整图输入尺寸对照：FRCNN-R50FPN-SCALE1600-VAL-001

日期：2026-10-13。用户明确要求最后再做一次增大输入尺寸的实验。状态：文件已准备，训练尚未启动；实际静态检查结果见workspace/audit/step14_preflight/preparation_check.json和workspace/audit/step14_resolution/resolution_audit.json。不得把本计划当作实验成绩。

## 为什么做、与哪次比

本次只回答：在现有Faster R-CNN、原train107/val23和固定训练预算下，把Resize从(2133,1280)提高到(2666,1600)，是否改善验证表现？它补全默认anchor设置下的800、1280、1600输入尺寸敏感性对照，可作为小论文中分析整图缩放影响的证据。

直接对照：FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115，使用默认anchor scales=[8]。新配置继承该运行冻结的resolved_scale1280_val_001.py，逐项还原Resize后要求整个科学配置一致。基线800可作系列背景。

STEP13是1280+anchor4的独立匹配诊断，不是本次尺寸单因素的直接对照。本次使用与STEP11相同的默认anchor8；不能把本次与STEP13的差异全部归因于尺寸。这样是在继续最初的尺寸系列，不是把两个改动合在一起。

## 为什么选1600，以及不能提前断言什么

`scale=(2666,1600), keep_ratio=True`表示短边尽量达到1600，长边不超过2666，保持比例；不是强行拉伸为2666×1600。具体缩放因子为min(2666/max(W,H),1600/min(W,H))，宽高按当前MMCV规则取整，batch还按32倍数补边。

6000×4000原图从1280设置下的1920×1280变成2400×1600；5472×3078原图从2133×1200变成2666×1500。线性尺寸约提高25%，像素面积约提高56%，这是一个有限的下一档，不能等同于YOLO imgsz1600。

已有CPU几何统计的标注哈希将再次核对，旧文件保留原样：

|原train168个GT框|1280|1600|
|---|---:|---:|
|缩放后框短边中位数|20.009像素|25.011像素|
|短边小于4像素|2|0|
|短边小于8像素|12|7|
|固定train8的零正anchor目标（15个GT）|4|4|

目标像素变多，有检验细节保留效果的理由；但默认anchor的这4个几何匹配问题没有因此消失，所以不能保证提高AP，也不能预先认定尺寸就是低分主因。小于4/8像素是描述性统计，不是统一的可检测界限。

这是单seed、反复使用23张val的探索性比较。提高只支持当前设置下的关联证据，不能据此证明所有模型必须使用大图；不提高只表示这一设置下没有观察到改善，不能证明任意更大尺寸、切片或其他策略都无效。本次不扩展到这些策略。

## 开始前固定的设置与判断规则

- 训练：原train107张/168框（broken39、loose129）；验证：原val23张/37框（broken9、loose28）。不重分、不合并，不使用test选尺寸或epoch。
- 官方Faster R-CNN R50-FPN，num_classes=2，默认anchor scales=[8]及ratio=[0.5,1,2]。模型结构、匹配规则和损失不变。
- 从同一官方完整COCO检测checkpoint重新开始，resume=False；不续训任何私人数据权重。启动前核对step03_preparation.json中的来源与SHA256，训练开始前逐值确认344项加载，4项ROI类别输出层重新初始化。
- 仅训练、验证和固定train8快照的Resize改为(2666,1600)；插值、归一化、补边、RandomFlip0.5保持原设置。
- batch2、FP32、workers0、seed42、deterministic=True、100epoch，每轮54次更新，总5400次更新。
- SGD lr0.0025/momentum0.9/weight_decay0.0001；warmup500/start_factor0.001；milestones67/92、gamma0.1。不自动缩放学习率。
- 每轮val；按原生coco/bbox_mAP（三位小数的mAP50-95）选最佳，同值保留首次。保存预测供精确复算，复算不改变已经固定的checkpoint。
- 主要判断：选定checkpoint的val精确COCO mAP50-95；同时完整报告mAP50、broken/loose的AP50及AP50-95、AR100、曲线、耗时和显存。不能只挑改善的指标，单seed不能宣称统计显著。
- AP预测score_thr=0.001，ROI NMS0.5，max_per_img100，COCO maxDets=[1,10,100]；显示阈值0.30。显示无框不等于无预测。
- 这次完成100轮和结果核验后结束本轮尺寸尝试，不因分数低自动追加更大尺寸、anchor、训练轮数或模块。若OOM/报错则保留失败日志并停止，不自动重试、resume、减batch或开AMP。

## 显存与成本

STEP11实测峰值CUDA allocated=8903.8706MiB（约8.70GiB），含验证/快照等的耗时3095.629秒。按面积粗略放大allocated约13.59GiB，仅作风险提示，不是实测、不是总显存，更不包含所有reserved、系统和其他进程开销；耗时也不按此比例保证。

本机GPU总显存16303MiB。准备时nvidia-smi显示已有约3118MiB占用，且会变化。1600+batch2+FP32比较吃紧，启动前应关闭不需要的其他GPU任务（保留本实验终端），不要同时训练其他模型。不能承诺不会OOM。保留batch2和FP32是为了比较清楚；若放不下，本阶段记录资源限制，后续改变batch/AMP须重新明确设计，不能冒称只改尺寸。

新预检通过nvidia-smi读取实时显存并打印GPU_MEMORY_SNAPSHOT（total/used的单位为MiB）；不进行带更新或无更新的GPU模型试跑。等待训练完成的启动器不创建额外CUDA上下文，由训练子进程使用CUDA，以免无必要地占用显存。这不改变训练配置。CPU检查只验证配置、图片缩放坐标与依赖/数据/权重哈希，不证明训练显存一定够。

## 文件、保存内容与启动

- 配置：`D:\小论文\model_projects\Faster R-CNN\workspace\configs\frcnn_r50fpn_scale1600_val_001.py`
- 入口：`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\run_step14_train_scale1600.cmd`
- 执行与预检：`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\step14_run_scale1600.py`
- CPU核查：`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\step14_resolution_audit.py`
- 尺寸及坐标报告：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_resolution\resolution_audit.json`
- 逐图尺寸：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_resolution\image_dimensions_train_val.csv`
- 固定train8实际Resize的1:1局部GT对照：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_resolution\train8_resize_1280_vs_1600.png`
- 静态配置预检：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step14_preflight\preparation_check.json`

旧STEP11几何报告中“1600未授权训练”是当时状态；旧文件和旧运行内冻结副本保留。本次新授权、尺寸选择和资源说明写在本文件及当前AGENTS.md，不回改历史证据。

由用户在Windows cmd中执行一次：

```cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
call "D:\小论文\model_projects\Faster R-CNN\workspace\scripts\run_step14_train_scale1600.cmd"
```

脚本创建唯一step14运行编号并打印RUN_DIR、LOG_DIR；助手不执行上述训练分支。

- 原始结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1600-VAL-001\step14_<唯一编号>`
- 完整日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step14_<同一编号>`，含step14_console.txt、train_console.log、step14_manifest.json。
- run内保留resolved配置、输入代码及哈希、官方初始化来源/哈希、实际加载证据、依赖版本、命令、数据标注哈希；不重复复制大型初始权重或原图。
- artifacts保留5400次loss/LR及曲线；第0/25/50/75/100轮固定train8预测/可视化及处理后尺寸；100轮val预测及原生指标；选中val的精确COCO复算、每类AP/AR及checkpoint哈希。沿用现有checkpoint保留规则，保留最佳和末轮等，不承诺保存全部100份权重。
- 没有独立test推理，也不新增RPN/ROI专项诊断。主要比较已有保存证据，原test的既往使用在报告中如实披露。

正常开始：SCALE_ONLY_CONFIG_DIFF_PASS、STEP14_PREPARATION_PASS、FORMAL_OFFICIAL_COCO_WEIGHT_LOAD_PASS: matched=344 expected_new_outputs=4、Epoch(train)[1][.../54]。
正常结束：FORMAL_100_EPOCHS_5400_UPDATES_COMPLETED、BEST_VAL_EPOCH=...、STEP14_VAL_TRAIN_COMPLETED、STEP14_EXIT_CODE=0。

完成后反馈最后约60行、RUN_DIR、LOG_DIR；失败时反馈完整报错及LOG_DIR，先不重试。收到后检查是否达成问题、结果与成本，再更新单次实验说明、统一CSV、Excel和文字总报告。现有汇总读取器只识别八次正式实验，需要依据新运行真实文件扩展后核验；本次准备不改动已有18行指标，不登记尚未获得的成绩，不为补表启动新推理。
