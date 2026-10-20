# DIAG04：第5轮验证集三图候选过程

2026-10-21准备。状态：无模型预检通过，等待用户启动推理；还没有真实三图诊断结果。

## 先核对总体路线

已读取 `D:\小论文\model_projects\AGENTS.md` 及 `D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN1280_排查路线.md`。

当前仍在第④步“只在存在具体缺口时补候选框过程”。前三步已在限定范围内验收，GT172/194/171的用户观察也已完成。这次不重做这些验收，不恢复STEP18，不改尺寸/anchor，不扩充数据，不训练。

具体缺口：第5轮最终保存框能说明类别/位置表现，不能区分RPN交接候选不足、ROI分类/定位变化、分数/NMS/最多100框过滤。旧STEP12固定为第100轮train8，不是这批第5轮val证据，不能直接用旧命令代替。

范围固定如下，依据来自val观察，不依据test选图：

|顺序|原验证图片|主要观察|待回答的问题|
|---|---|---|---|
|1|val/cond_000027.jpg|GT172，断股|重合较好的松股框来自哪个候选？同一候选的断股框和类别分数是什么？|
|2|val/cond_000125.jpg|GT194，松股|IoU约0.492的定位范围，在候选/回归/筛选各阶段怎样变化？|
|3|val/cond_000015.jpg|GT171，断股|最终无重合，是交接候选就不足，还是后续没有保留合格框？|

第二张图片另有GT195，所以三张图共4个人工标注；会附带记录GT195，但不加图片，不要求重复判断类别。这些是有目的的案例诊断，不是全验证集原因占比或新泛化成绩。

停止条件：获得这三图在指定阶段的证据、复核与历史输出的一致性，然后解释支持到哪一步；若交接RPN已不足，只能先定位到“ROI之前或交接时”的范围，不能声称区分了RPN原始生成/排序/去重。无需为了齐全自动追加所有层的特征、全部图片或新训练。

## 现在你需要做什么

1. VS Code选择“文件 → 打开文件夹”：`D:\小论文\model_projects`。这是Git根目录，方便看差异。
2. `Ctrl+Shift+P` → `Python: Select Interpreter` → 选择/输入：
   `D:\小论文\model_projects\Faster R-CNN\.venv\Scripts\python.exe`。
   使用已有Faster专用环境，不新建环境、不安装依赖。
3. “终端 → 新建终端”，在终端的下拉菜单选择 `Command Prompt`（命令提示符/cmd）。若默认打开PowerShell，另选cmd。
4. 输入以下命令。即使提示符没有显示 `(.venv)`，显式指定解释器仍会使用正确环境：

```cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
.venv\Scripts\python.exe -I -X utf8 workspace\scripts\diag04_roi_trace.py --run
```

`-I`只隔离本次Python进程对用户级包和环境变量路径的干扰，不是另建虚拟环境。`--run`才执行模型；助手只运行过`--check-only`和无模型的CPU核查。不要运行旧STEP12，也不要运行训练命令。

## 正常输出和必须反馈的内容

正常会依次看到：

```text
DIAG04_USER_VAL3_INFERENCE
RESULT_DIR=...唯一结果目录...
LOG_DIR=...唯一日志目录...
DIAG04_PREPARATION_PASS: epoch5; 3 val images; no model built by checks
DIAG04_CHECKPOINT_LOAD_PASS: epoch5 matched=348; eval; gradients disabled
DIAG04_IMAGE_PASS 1/3 val/cond_000027.jpg saved_match=True
DIAG04_IMAGE_PASS 2/3 val/cond_000125.jpg saved_match=True
DIAG04_IMAGE_PASS 3/3 val/cond_000015.jpg saved_match=True
DIAG04_TRACE_PASS: 3 val images; historical outputs matched; review pending
REPORT=...index.html
DIAG04_EXIT_CODE=0
```

这里只列关键标志，框架可能夹杂普通加载信息。`PASS`表示工具完成及一致性检查通过，具体原因解释仍待助手核查。

**你必须反馈：复制从DIAG04开始到结束的完整终端输出，尤其RESULT_DIR、LOG_DIR和退出码。** 不需要证明代码正确、不需要再确认三例缺陷类别、不需要自行解释数组。助手收到后先检查真实输出/保存数组，再解释图中现象和本阶段结论。

如果出现FAIL、Traceback或非零退出码，发送完整错误及两个目录；不要自行改阈值、换权重或重复运行。已输出的文件会保留，可以先判断是否能用已保存数组修复整理阶段，避免重做模型推理。

成功后可选打开网页；这不是提交终端输出的前置条件：

```cmd
start "" "D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-TRACE1280-001\latest.html"
```

网页内会有中文阶段表、同一局部窗口的框对照、同一候选的类别概率及位置变化。绿色为人工标注，蓝色为RPN交接候选，橙色为同类ROI框。独立最大IoU表不是同一框轨迹；页面另列同一候选的记录。背景是内部分类选项，不是第三个缺陷类别。

## 文件隔离和保存位置

诊断编号：`FRCNN-DIAG-TRACE1280-001`。沿用原test已被使用的历史，本次只读原val图，不读取test预测、不产生新的AP、不更新正式成绩表。

使用已固定的第5轮私人数据权重进行**诊断推理**，不是续训、不是独立训练对照；此处不应改用官方COCO初始权重：

`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\best_coco_bbox_mAP_epoch_5.pth`

SHA256：`c8ff3af83efeec7cf3f126f49833da6e42489d9f022809d6237b94d9efdb08d3`，实际启动时再次与选模记录核对。

- 原模型配置直接读取：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115\resolved_scale1280_val_001.py`。
- 新结果：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-TRACE1280-001\diag04_<日期时间>\`。
- 完整终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\diag04_<日期时间>\console.txt`。
- 每次使用唯一目录，不覆盖历史；报告中间数组、框图留在本机，不上传大型结果或原图。

结果保存：`code`中的工具/清单/冻结模型配置副本，`preparation.json`来源信息，`weight_load.json`加载核查，三份`*_intermediates.npz`数组，`per_gt.json`逐目标记录，`images.json`历史输出差异，`summary.json`状态，`manifest.json`来源及输出哈希，`index.html`和局部框图。不重复复制大权重或全图数据。

## 本次代码新增了什么，为什么

旧训练/测试.py、旧STEP12、模型配置和官方源码均未改。以下文件原来不存在，全部是新增，不虚构“修改前的代码”：

|文件|作用|
|---|---|
|`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\diag04_roi_trace.py`|一次性调用三图官方推理，捕获ROI输入/输出，复用官方解码与NMS，记录结果。|
|`D:\小论文\model_projects\Faster R-CNN\workspace\configs\diag04_trace1280_val3.json`|固定第5轮运行、三图编号、主要GT、依赖版本及15份来源哈希。它是诊断清单，不是修改后的模型配置。|
|本操作说明|解释目的、代码、用户操作和反馈要求。|

为什么需要新增一个工具：现有STEP12把两个模型、第100轮、train8和相应预测格式写死；直接改它容易破坏历史复现。新工具复用其“捕获官方ROI输入/输出”的方法，同时增加独立清单和官方返回的候选索引，拆开score、NMS与数量限制。没有复制训练流程。后续同类工作优先评估复用此工具，不默认再复制一份。

可在VS Code中按`Ctrl+G`跳到以下位置；学习可选，不影响运行：

- `diag04_roi_trace.py:50`，`check()`：核对环境、来源哈希和配置。没有创建模型。
- `diag04_roi_trace.py:115`，`load_checkpoint()`：安全读取第5轮权重，要求epoch5/270次更新。
- `diag04_roi_trace.py:156`，`target_rows()`：按同一候选编号追踪被哪一步移除，独立记录各阶段最大IoU。候选数不是普通召回率。
- `diag04_roi_trace.py:219`，`trace()`：使用冻结验证数据集、官方模型和真实预处理器；`eval()`、关闭梯度，不创建优化器。
- `diag04_roi_trace.py:256`：两个hook（钩子，即只读观察点）捕获RPN交给ROI的框以及ROI分类/回归输出，不替换模块。
- `diag04_roi_trace.py:265`：`result = model.test_step(batch)[0]`。这里的`test_step`是框架通用推理方法名，输入仍是原val，不意味着用了test集合。
- `diag04_roi_trace.py:275`：对同一次前向保存的张量调用官方`multiclass_nms()`，取得候选身份。不再运行神经网络。
- `diag04_roi_trace.py:291`：与历史预测比较类别、框位置和分数；不一致就停止，不能把新运行差异偷偷当成旧实验结论。
- `diag04_roi_trace.py:335`，`report()`：生成中文表和局部画框图，属于显示副本，不给模型输入画好框的图。

关键新增代码示例：

```python
# 与历史验证一致的数据处理和模型推理；不训练
with torch.inference_mode():
    result = model.test_step(batch)[0]
```

```python
# 只对捕获的输出做辅助后处理，观察“限100框”之前的结果
dets, labels, candidate_ids = multiclass_nms(
    pre.bboxes, pre.scores, filt.score_thr, filt.nms,
    max_num=-1, return_inds=True)
```

`max_num=-1`只用于辅助张量分析；真正模型输出仍按原配置每图最多100框。随后会检查辅助结果的前100框与真实模型输出一致。没有降低模型阈值，也没有用扩展候选重新算AP。

科学边界：RPN交接框已经过RPN自身筛选，不能声称导出原始所有候选；框的类别、位置与排序可能共同影响结果，不能仅凭一个最大值认定某模块“代码坏了”。

## 如何看修改前后

VS Code打开Git根目录后，“源代码管理”里可查看本阶段改动；新增文件整份显示为新增，原来没有旧版本。`AGENTS.md`和排查路线/总览只更新当前状态与操作规则。提交后可在cmd查看：

```cmd
cd /d "D:\小论文\model_projects"
git show --stat HEAD
git show HEAD
```

准备核查：15份来源哈希、冻结配置、专用环境通过；CPU读取权重验证元信息；合成框验证分数过滤、NMS、数量截断的候选身份，空候选处理及报告图片解码通过。助手没有创建检测器或运行真实图像前向，不能将准备通过写成真实诊断已经成功。
