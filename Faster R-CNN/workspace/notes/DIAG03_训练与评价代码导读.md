# DIAG03：1280默认anchor的训练与评价代码导读

日期：2026-10-20。状态：助手已完成限定范围的静态代码与保存日志核查；用户阅读是可选学习，不是诊断验收的前置条件。没有启动训练、模型推理或test，也没有修改训练配置。

主运行：D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115

## 这一步回答什么

第二步确认了冻结数据管线中图像、标注和预处理数值对应。第三步接着检查：我们的启动与记录代码是否接入了正确的官方训练流程？加载的是否是COCO检测权重？模型输出是否按正确类别和原图坐标评价？

这一步不改变尺寸、anchor（锚框）、学习率或权重，不增加一个新训练实验。没有必要为了读代码再跑一次训练。

## 如果想学习代码，可以这样操作（可选）

1. VS Code“文件→打开文件夹”：D:\小论文\model_projects。
2. 如需Python解释器，Ctrl+Shift+P → Python: Select Interpreter（选择解释器）→ D:\小论文\model_projects\Faster R-CNN\.venv\Scripts\python.exe。本次只阅读文件，不需要执行Python。
3. 终端右上角下拉菜单选择Command Prompt（命令提示符/cmd），不要点击旧训练脚本的运行按钮。
4. 打开本文，Ctrl+Shift+V看中文预览。下面的四个位置按顺序阅读，Ctrl+G输入行号。

可以在VS Code的cmd终端执行以下命令打开本文；它只打开编辑器：

```cmd
cd /d "D:\小论文\model_projects"
code -r "D:\小论文\model_projects\Faster R-CNN\workspace\notes\DIAG03_训练与评价代码导读.md"
```

如果code命令不在PATH里，直接在VS Code资源管理器找到本文即可，不需要为此安装环境。

## ① 到底是谁训练模型

打开：D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\code\step11_run_scale1280.py，第129行training_child函数。

```python
official = ROOT / 'mmdetection/tools/train.py'
sys.argv = [str(official), str(run_dir / 'resolved_scale1280_val_001.py'), '--work-dir', str(run_dir), '--launcher', 'none']
runpy.run_path(str(official), run_name='__main__')
```

读法：指定官方train.py，给它冻结配置和本次结果目录，再运行官方入口。这个项目启动器负责准备、核验和留档，没有另写一套Faster R-CNN的前向、损失和反向传播。

继续追踪时，官方D:\小论文\model_projects\Faster R-CNN\mmdetection\tools\train.py第110行创建Runner（训练流程管理器），第117行runner.train()启动训练。这里只阅读，不能把这两行单独复制到终端运行。

本次重新确认官方源码commit为44ebd17b145c2372c4b700bfb9cb20dbd28ab64a，已跟踪文件无修改。

## ② 实际配置和初始权重是什么

打开：D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\resolved_scale1280_val_001.py。

这是继承关系展开后的配置，代表当时实际运行设置；阅读它比只看一个继承配置更直接。

|行号|实际值|含义|
|---|---|---|
|34|load_from=官方faster_rcnn_r50_fpn_1x_coco…pth|从完整COCO检测器初始化|
|115|num_classes=2|目标类别为broken、loose|
|137|scales=[8]|默认anchor尺寸设置|
|168|score_thr=0.001，NMS IoU=0.5，max_per_img=100|预测后处理，不是绿色GT显示|
|218|SGD lr=0.0025，accumulative_counts=1|普通FP32优化器，每个训练batch执行一次更新|
|237|resume=False|不是接着旧私人数据实验续训|
|262|100 epoch；每epoch验证一次|固定训练轮数和验证频率|
|265、292|batch_size=2；Resize(2133,1280)|每批2图；保持长宽比的缩放边界|

第57行frozen_stages=1、第60行norm_eval=True是保留的官方骨干设置：冻结早期部分骨干，训练时固定BN（批归一化）的运行统计。不能把“不是所有骨干参数都更新”直接当程序错误；也不能说本实验更新了所有参数。

打开同一运行目录的artifacts\actual_training_weight_load.json。它是在首次优化器更新前保存的实测记录：344项兼容state_dict参数/缓冲量与COCO文件逐值相同；4项ROI输出层尺寸不兼容，保留新模型初始化。

|输出层|COCO模型|两类模型|原因|
|---|---|---|---|
|fc_cls.weight / bias（分类）|[81,1024] / [81]|[3,1024] / [3]|80目标类+背景 → 2目标类+背景|
|fc_reg.weight / bias（框回归）|[320,1024] / [320]|[8,1024] / [8]|每目标类4个回归量：80×4 → 2×4|

ROI（Region of Interest，候选区域）指对候选框覆盖的特征进行进一步分类、定位的部分。分类器内部背景不等于数据集多了第三种缺陷。

证据代码：同一运行code\frcnn_formal_hooks.py第32—43行。它不是仅检查“文件能打开”，而是在当时模型加载后用torch.equal检查344项完全相同。本次只复核保存记录及其实现，没有新建模型重新加载。

本次重新计算了官方初始权重哈希，并与step03_preparation.json及pretraining_record.json一致核对；也复核第5轮和第100轮权重哈希。没有混用anchor4或1600权重。

## ③ 训练有没有只记loss却不更新

已有证据：artifacts\losses.jsonl中5400条迭代连续，每轮54个batch，共100轮；五项loss（损失）和学习率均为有限值，学习率逐条与原800基线一致。前100次平均总损失0.3974187850，后100次0.0174116464。

107张训练图、batch=2且不丢最后一批，所以每轮54次。配置没有梯度累积、没有AMP跳步机制；对应官方流程每批执行一次优化器更新。

代码路径：

- D:\小论文\model_projects\Faster R-CNN\.venv\Lib\site-packages\mmengine\runner\loops.py，第110—139行：每轮进入train模式，再调用model.train_step，随后记录after_train_iter。
- D:\小论文\model_projects\Faster R-CNN\.venv\Lib\site-packages\mmengine\model\base_model\base_model.py，第112—117行：预处理 → 算loss → optim_wrapper.update_params。
- D:\小论文\model_projects\Faster R-CNN\.venv\Lib\site-packages\mmengine\optim\optimizer\optimizer_wrapper.py，第195—220行：反向传播算梯度 → step更新 → zero_grad清理梯度。

项目code\frcnn_debug_hooks.py第184行只是读取已经返回的loss和当前学习率，再写日志，不替换优化器步骤。

同一文件第210行snapshot（阶段性预测快照）临时切到eval（评价模式）；第264行finally恢复每个模块原先的模式及Python、NumPy、PyTorch随机状态。官方验证循环也在eval和no_grad（不计算梯度）下运行，下一轮官方训练重新进入train模式。

核查边界：这些代码和连续日志支持训练更新流程实际走完；没有保存每一步每一层的梯度数组，因此不能声称所有参数在每次更新时都有非零梯度。损失下降也不等于验证集泛化良好。

## ④ 0.30是否把用于AP评价的框删掉了

打开：同一运行code\frcnn_debug_hooks.py，第241—253行。

```python
for box, label, score in zip(boxes, labels, scores):
    x1, y1, x2, y2 = box
    coco_predictions.append(dict(image_id=int(info["img_id"]), category_id=label + 1,
                                 bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
# ...
draw_predictions(rgb, boxes, labels, scores, self.visual_threshold, path)
# ...
metrics = evaluate_train_fit(self.dataset.ann_file, coco_predictions, ids, destination / 'train_fit')
```

这里删节了无关记录行、统一了引号，逻辑与冻结源码一致。先把返回的框全部加入coco_predictions，再把同一份返回结果交给画图函数。0.30只在draw_predictions第85—87行决定画不画，未用来过滤coco_predictions。

正式val更直接：code\frcnn_formal_hooks.py第55—63行只给官方CocoMetric设置保存路径；预测转换和AP计算仍由官方实现负责。

因此需要分清：模型后处理已经按score>0.001、NMS（重叠框去重）和每图最多100框筛选；保存的结果不等于原始全部候选。显示阈值0.30进一步影响人能看到哪些框，不会把已有AP计算改成0.30口径。

坐标和类别核查：

- 模型内部0/1 → COCO 1/2。快照用label+1；官方CocoMetric用self.cat_ids[label]，本数据cat_ids为[1,2]。
- 官方two_stage.py第196行predict默认rescale=True；bbox_head.py第545—548行除以缩放系数，把预测还原至原图坐标。COCO结果用原图[x,y,width,height]。
- 官方coco_metric.py第238—244行逐框写JSON；第474—477行设置类别、图片ID、maxDets和IoU阈值。
- 冻结配置中COCO maxDets=[1,10,100]，第三项100表示每图每类别在COCO评价中最多考虑100个最高分检测；模型另有跨类别每图总共最多100框的限制。两者是不同层次的上限。

本次只检查train/val保存文件。100轮val预测的图片ID均属于原val，类别和坐标格式有效；没有任何一轮触发“空预测补零”适配分支。该分支存在于formal hook第67—81行，但没有参与本次100轮结果。

已有STEP11保存预测的精确COCO复算证据及文件哈希已重新核对，本次没有再次运行模型或另改正式AP。

## 最佳第5轮为什么不改

官方CocoMetric总体AP保存为三位小数，官方CheckpointHook按严格大于才替换，因此同值保留首次。重新读100轮validation_history.jsonl，按当时规则仍选第5轮。

第5轮精确val mAP50=0.03189324821274552，mAP50-95=0.014877891782027844。原生数值分别为0.032、0.015，是舍入差异。本次不重新选轮次，也不按test挑参数。

字段coco/broken_precision和coco/loose_precision在这一CocoMetric实现中表示分类别AP，不能因名字有precision就当某个固定置信度的普通精确率。

## 本次结论、文件改动和停止条件

已核查的配置、初始化记录、更新调用、模式切换、预测导出与评价坐标未发现需要重训修复的具体冲突。这不等于整个框架、所有标注语义和每一层梯度已证明无错。

新建本文（以前没有），保存只读核查记录：D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-CODE1280-001\diag03_20261020_review\review.json。它记录来源哈希和核查边界。

本轮没有修改任何旧训练/测试.py、模型配置或官方源码。DIAG02的.py是上一阶段新增、这次用户实际运行的同一版本；本次未再改它。新增了第二步验收JSON，并更新DIAG02教学说明、排查路线和AGENTS当前状态；前后变化用Git与验收目录的修改前副本保留。

如想学习，优先看上述①②④三处关键代码，不必通读框架源码。有不明白的行可以提出，但无需反馈“是否读完”或证明整个程序正确。技术核查由助手负责；本步没有需要用户运行的新实验，也没有必须提交的学习反馈。

本步到代码与已有证据核查为止。下一步先解释旧STEP12的RPN/ROI（候选生成/候选区域分类定位）证据，再判断是否有具体缺口；不自动安排新的推理、过拟合训练、扩数据或恢复STEP18补测。总进度与给老师看的结论见 [诊断全流程总览](<D:/小论文/model_projects/Faster R-CNN/workspace/notes/FRCNN1280_诊断全流程总览_讨论版.md>)。原核查JSON的USER_READING_PENDING是当时记录，予以保留，当前不作为阻塞条件。
