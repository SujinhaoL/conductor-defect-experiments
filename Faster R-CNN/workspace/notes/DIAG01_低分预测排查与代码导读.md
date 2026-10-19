# DIAG01：用已有预测学习第一步排查

这一步不训练模型，也不重新让模型预测。它把已保存的数字转换成更容易理解的表和图。导师提出的第一问是“没有画出框，究竟没有预测，还是被显示规则隐藏了？”

## 先在VS Code打开正确项目和环境

1. VS Code → 文件 → 打开文件夹：`D:\小论文\model_projects\Faster R-CNN`。左边`mmdetection`是官方源码，`.venv`是独立Python环境，`workspace`是本实验自己的脚本/配置/记录。
2. `Ctrl+Shift+P` → `Python: Select Interpreter`（Python：选择解释器）→ 选择或输入 `D:\小论文\model_projects\Faster R-CNN\.venv\Scripts\python.exe`。
3. 终端右上角下拉菜单 → 选择默认配置文件 → `Command Prompt`（命令提示符）→ 新建终端。提示符应像 `D:\小论文\model_projects\Faster R-CNN>`；不是`PS ...>`。
4. 后续命令仍明确写`.venv\Scripts\python.exe`，防止终端的默认python指向其他模型环境。无需再安装环境；`-I`是Python的隔离启动选项，避免用户级包/环境变量干扰，不会创建第二个环境；`-X utf8`用于中文文件和终端编码。

## 本次改了哪些文件

|文件完整路径|原来 / 现在|为什么|
|---|---|---|
|`D:\小论文\model_projects\AGENTS.md`|原来没有Oct19教学与单模型计划；现在在开头补充，保留历史内容。|以后交接必须解释代码、环境、当前步骤，不重复旧实验。|
|`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\diag01_saved_predictions.py`|新增文件，原来没有。|读取历史预测，核对类别/坐标/数量/哈希，按分数排序，计算IoU和画局部图。|
|`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\diag01_viewer.html`|新增文件，原来没有。|提供中文页面，可改显示阈值、前N框、查看单框和局部图。|
|`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN1280_排查路线.md`|新增计划。|记录科学问题、阶段、验收和停止条件。|
|`D:\小论文\model_projects\Faster R-CNN\workspace\notes\DIAG01_低分预测排查与代码导读.md`|新增本文。|解释术语、代码和操作。|

训练配置、官方源码、原始数据、checkpoint（权重文件）、原始预测、历史总表均未因本次诊断改动。**不是每次实验/排查都要修改模型代码**；这次新增的是解释结果的代码。

AGENTS修改前副本：`D:\小论文\model_projects\baseline_analysis\audit\AGENTS.before_diag01_20261019.txt`。在VS Code cmd可运行：

```cmd
code --diff "D:\小论文\model_projects\baseline_analysis\audit\AGENTS.before_diag01_20261019.txt" "D:\小论文\model_projects\AGENTS.md"
```

左边是修改前，右边是修改后。若cmd找不到code，可在VS Code先打开两个文件，对左边文件右键“选择以进行比较”，再对另一文件右键“与已选项目进行比较”。新增脚本不存在旧版本，不伪造逐行“修改前”。

## 读四处代码就够了

在新Python脚本用`Ctrl+F`搜索以下函数/文字；对话交付同时给具体行号。

当前行号（VS Code按`Ctrl+G`输入行号）：Python第59行`xyxy`、第65行`iou`、第74行`best_overlap`、第95行`crop_panel`、第124行`verify_existing`、第177行`cases`；HTML第44行筛选显示框。无需一次读完300行，先读下面四处。

### 1. `cases = [`：知道读了什么数据

三行分别读取train8第100轮、val23第100轮、val23第5轮。前两个是同一权重的训练/验证对照；第三个是历史选定的最佳验证权重。这里没有test，也没有重新调用模型。

### 2. `def xyxy`：知道框如何表示

```python
def xyxy(box):
    x, y, width, height = box
    return [x, y, x + width, y + height]
```

COCO保存的是`[左上x, 左上y, 宽, 高]`；画矩形需要`[左上x, 左上y, 右下x, 右下y]`。把宽高误当右下角会画错位置。本步只转换表达方式，不改变预测内容。

### 3. `def iou` 和 `def best_overlap`：知道“框到了哪里”

IoU = 两个框相交面积 / 两个框并集面积。完全重合为1，完全不重合为0。

```python
best = max(candidates, key=lambda p: (iou(gt["bbox"], p["bbox"]), p["score"]))
```

先找和人工框位置最重合的预测；若IoU相同，再选高分者。这不是最高置信度预测，也不是评价AP的完整匹配方法。同一个预测可以成为两个GT各自的最大重合候选，所以统计数不能直接叫正确检出数或普通Recall。

### 4. 网页 `allowed=current.predictions.filter`：知道为什么过去看不见

旧绘图行为是只显示`score > 0.30`。新网页默认显示阈值0，可以切回0.30。这是**显示方法对照**，不是对历史绘图代码的覆盖修改：

旧代码依据：`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\step11_review_results.py`第192行按`p['score']>.3`取可见框；`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\frcnn_debug_hooks.py`第80行的`draw_predictions`在第86行跳过`score <= threshold`。这些历史文件没有被本次改写。

```javascript
// 旧显示规则的等价写法：
predictions.filter(p => p.score > 0.30)
// 新网页：threshold由界面设置，初始0；原始预测列表仍保留。
current.predictions.filter(p => p.score > threshold)
```

默认画前10框防止遮满图，可选全部100；表格仍显示该图全部保存框。点某一行可以单框聚焦，聚焦模式暂时忽略显示阈值，网页会明确提示。

保存结果在历史模型输出时已经过`score>0.001`、NMS阈值0.5及每图最多100框。显示阈值改0不能恢复当时未保存的候选，也没有修改AP的过滤设置。若以后确有必要检查更早的候选，需要另行设计用户启动的推理诊断，不在这一步悄悄运行。

## 你问到的术语和旧图

|术语|中文与作用|为什么检查 / 能说明什么|
|---|---|---|
|GT / Ground Truth|人工标注的真实目标框及类别|判断预测位置/类别需要参照；人工标注本身也可能需核查，不能仅凭模型差就说标签错。|
|score / confidence|模型给预测类别的置信度分数|决定排序和显示过滤；不是位置重合率，不是可靠性经过校准的概率保证。|
|IoU / Intersection over Union|交并比，两个框的位置重合程度|区分“附近有框”与“框得够准”；IoU0.5和置信度0.5不是同一阈值。|
|anchor|预设的参考框，具有不同尺寸/长宽比|帮助RPN生成候选与分配训练目标；当前保持原默认anchor，不改它。|
|RPN / Region Proposal Network|区域候选网络，先找可能存在目标的位置|看目标在较早阶段是否有重合候选；不是最终两类缺陷分类结果。|
|ROI / Region of Interest|感兴趣区域，即拿去进一步判断的一块候选区域|ROI head（区域分类/回归头）判断断股、松股或背景，并调整框的位置。|
|RPN/ROI中间数组|阶段输出的数字表：框坐标、各类别分数等|用来追踪候选存在与否、类别和定位；不是必须全部看懂的神秘文件，也不是额外训练数据。|
|NMS / Non-Maximum Suppression|非极大值抑制，按分数减少高度重叠的重复框|保存预测可能已经被去重；不能把最终没有某框等同于网络从来没产生过它。|
|Resize|按规则缩放图像，同时缩放框坐标|检查图与标签有没有同步变换；(2133,1280)在keep_ratio=True下是长/短边约束，实际图像尺寸由原比例决定。|
|归一化 / Normalize|按通道减均值、除标准差，改变像素数值范围|使数值符合模型预期；它不移动框。可逆变换后看图像是否合理。|
|补边 / Padding|为批处理或特征尺度对齐补齐边缘|核对真实内容区域、补边位置、尺寸和框，不能把补边区当原图。|
|Tensor / 张量|模型使用的多维数字数组，例如[批次,通道,高,宽]|检查模型输入前的实际数据，而非仅查看未经处理的原图。|

你提到的旧图：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step12_review\four_uncovered_targets_comparison.jpg`。

`GT123 loose`指`train/cond_000108.jpg`内编号123的人工标注，类别松股；123不是照片序号，loose也不是lo0se。

- `RPN IoU 0.353`：旧保存的RPN交接候选中，和这个GT最重合的框，IoU约0.353。保存的RPN候选已经过其自身NMS和top1000，不能代表所有原始候选。
- `NO_RPN_IOU50`：这些候选中没有一个达到IoU≥0.50。不是“RPN没输出框”。
- `final IoU 0.342`：最终保存的同类预测中，最大IoU约0.342。也不是置信度0.342。本次复算该框实际置信度约0.061482。
- 两个最大IoU可能来自不同候选，不能说同一个框从0.353回归后变差到0.342。
- 旧图这类标签用来定位检查方向，不自动证明错误发生于哪个算法。原ROI还存在背景分数等信息，必要时下一阶段再讲。

新中文局部图：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-SAVED1280-001\diag01_20261019_010031_868401\crops\train8_epoch100_gt123.jpg`。绿色人工框、橙色同类最大重合预测。本次放大局部仅用于看清楚，不是将模型输入尺寸调大。

低分但位置已有一定重合的例子：第5轮val `cond_000060.jpg`的GT177，最大同类IoU约0.615529，而该框分数只有0.051652。局部图为结果目录下`crops\val23_epoch005_gt177.jpg`。它在0.30显示规则下看不见，但不能说模型完全没找到附近位置；也不能凭这一框认定整套检测结果已经好。

## 我做了哪些核验

89个GT的最大IoU用另一个numpy向量实现独立核算通过；train8第100轮及val第5轮的逐GT结果与旧诊断一致；对应历史配置、输入标注、选中预测的已登记哈希一致；89张局部图均完整解码。人工查看了6种代表情况的拼图，并单独放大GT123和GT177；没有声称逐一人工审阅全部89图。

复核脚本：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-SAVED1280-001\review_diag01.py`（新增，只服务于这次验收）；结果目录`review\acceptance.json`保存数值/代码静态检查结论，`review\visual_contact_sheet.jpg`是核查拼图，`review\AGENTS_teaching_change.diff`保存AGENTS修改前后差异。无需用户再运行复核脚本。

HTML嵌入数据与JSON一致、JavaScript语法检查通过。应用内浏览器的安全规则不允许打开本地`file:`页面，所以没有完成浏览器交互实测；不绕过这一限制。请按下面命令用你本机默认浏览器打开，确认切换图片和阈值能正常工作。

## 本阶段结果和解释

查看当前结果目录的`index.html`即可，不必先打开所有JSON/CSV。

- 第5轮val保存2300框，最高分仅0.170571，旧显示阈值0.30下自然看不到框。16/37GT在保存结果中存在同类IoU≥0.50候选；21/37没有。不能把低AP完全解释为画图阈值。
- 同一个第100轮：训练8图的15个GT有11个存在同类IoU≥0.50候选，验证37个GT只有4个。训练8图表现不是全部训练集表现，也不能当验证成绩。
- 第100轮val虽有8个score>0.30框，但没有为任何GT提供同类IoU≥0.50候选，说明高置信度也不保证框得对。
- 所有覆盖数都只是几何诊断。正式AP沿用原评价，不用这些数字替换Recall/AP，不因诊断改选最佳轮。

## 现在你执行什么

文件已生成，下面只验证文件及来源哈希，学习使用环境和脚本；不会重复生成，更不会训练/推理：

```cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
.venv\Scripts\python.exe -I -X utf8 workspace\scripts\diag01_saved_predictions.py --verify-existing workspace\audit\FRCNN-DIAG-SAVED1280-001\diag01_20261019_010031_868401
start "" "D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-SAVED1280-001\diag01_20261019_010031_868401\index.html"
```

正常会打印三行统计，以及`DIAG01_VERIFY_PASS: saved predictions only; no training/inference/test`，浏览器打开本地页面。也可在资源管理器直接双击index.html；无需部署服务器或安装插件。

建议阅读顺序：先看第100轮训练8图，然后同轮验证23图；最后第5轮验证23图切换显示阈值0与0.30。对任一目标，分别观察分数、IoU、局部框。先看GT123的中文图，再找一张第5轮“仅低分同类候选达到IoU0.50”的图。

请反馈：完整验证输出；页面能否正常出现表格和图片；一两个你仍看不懂的图片名/GT编号。收到反馈后验收本步，再开始真实输入数据管线检查。
