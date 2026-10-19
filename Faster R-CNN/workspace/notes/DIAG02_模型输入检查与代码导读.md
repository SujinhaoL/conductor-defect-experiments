# DIAG02：模型输入前的图像与标注检查

## 2026-10-20 全量结果已验收：无需重跑

用户实际结果：D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-INPUT1280-001\diag02_20261020_010559_436827。完整终端以DIAG02_INPUT_PIPELINE_PASS、DIAG02_EXIT_CODE=0结束。用户看图反馈：翻转同步、最终输入正常、局部绿框包住缺陷。

复核通过：130张不同原图；训练不翻转107图168框、翻转107图168框、验证23图37框，共237条记录；15份来源/63份输出/130图哈希一致；独立坐标复算通过，最大误差0.000194073像素；57张PNG完整解码，61个报告文件链接存在。

验收文件：上述运行目录\review\acceptance.json。原summary中的Visual review pending是生成报告当时的状态，保留原文件；当前状态以新增验收记录为准。下文命令是准备时的历史操作说明，不是现在要求再执行。

“画图”是为了人检查，在用于展示的照片上叠加绿色矩形和文字；网络实际输入没有画上这些矩形。框的数值是单独的标注数据。

“副本”指另取一份显示用数据。比如原图上的框为[1000,600,1400,800]，图片缩小一半后，显示框用[500,300,700,400]；这是举例，不是GT177的实际坐标。评价仍可使用原图坐标。程序第49—50行array函数的.copy()复制框数组，第224—229行只缩放显示副本；没有覆盖原标注，也没有另建数据集。

本次验收没有修改该.py，运行脚本与当前脚本哈希一致。新建第三步中文代码导读DIAG03_训练与评价代码导读.md，接下来阅读已有训练与评价代码，不需要重训。此前准备记录保留如下。

准备日期：2026-10-19。诊断编号：FRCNN-DIAG-INPUT1280-001。状态：工具CPU预检通过，原train107/val23全量检查等待用户执行。本次没有训练、模型推理或test评价。

## 本阶段到底验证什么

第一步看到的是保存预测；这一步看模型收到的输入。目标是核对：原人工标注经过图片加载、Resize（缩放）、RandomFlip（水平翻转）、PackDetInputs（打包成张量）、DetDataPreprocessor（真实数据预处理器）后，坐标、类别、颜色、像素数值和补边是否符合冻结配置。

只使用既有运行 D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115 的 resolved_scale1280_val_001.py。原配置、原标注、官方代码和权重均不修改；不需要加载checkpoint（模型权重）。

老STEP05检查的是800尺寸固定train8；STEP11还做过1280几何/anchor核查。这些证据保留，本次补足当前1280真实预处理器、训练翻转两条分支和验证坐标约定的系统检查。没有复制训练器，也不重跑检测模型。

## VS Code打开哪里、选哪个环境

1. 打开文件夹：D:\小论文\model_projects。这样能同时看Git差异及各模型目录。
2. Ctrl+Shift+P → Python: Select Interpreter（选择解释器）→ 输入解释器路径：D:\小论文\model_projects\Faster R-CNN\.venv\Scripts\python.exe。
3. 终端菜单 → 新建终端；终端右上角下拉菜单选择Command Prompt（cmd）。
4. 先打开新文件 D:\小论文\model_projects\Faster R-CNN\workspace\scripts\diag02_input_pipeline.py。下面命令使用专用Python的明确路径，即使终端没有显示(.venv)也可以；不用另建环境。

## 这次哪些代码变了

|文件|修改前|修改后/目的|
|---|---|---|
|Faster R-CNN/workspace/scripts/diag02_input_pipeline.py|原来没有|新增一个CPU诊断工具，直接调用冻结配置对应的官方组件；不创建检测器|
|Faster R-CNN/workspace/notes/DIAG02_模型输入检查与代码导读.md|原来没有|本说明，解释操作和证据边界|
|AGENTS.md|第二步尚未准备|记录第二步已准备、CPU预检通过、等待用户全量运行|
|Faster R-CNN/workspace/notes/FRCNN1280_排查路线.md|第二步待开始|第二步工具已准备、全量结果仍待验收|

没有改任何旧训练/测试Python文件或模型配置。旧版本在Git提交 e0aabcf 中；目前保留未提交更改供你查看红绿差异。VS Code左侧“源代码管理”里，新文件标记U（尚未跟踪），已有说明标记M（已修改）。点击即可查看；新文件没有虚构的旧代码。全量检查及反馈验收后，再登记提交。

## 先只读四处代码

文件：D:\小论文\model_projects\Faster R-CNN\workspace\scripts\diag02_input_pipeline.py。Ctrl+G可跳到行号。

- 第59行 prepare：确认是正确的Faster环境、冻结配置、官方commit和原train/val标注哈希。第111行只创建数据预处理器，并放在CPU上。
- 第129行 expected_geometry：不依赖框变换的输出，直接从COCO原标注用算式计算预期坐标；第142行 prepare_item 将它与官方管线输出对比。
- 第189行 check_batch：调用真实预处理器，检查RGB通道、归一化数值、批次补边、处理前后GT是否保持正确。
- 第314行 run：遍历两种训练翻转分支和验证集，收集记录和对照图。

关键代码一：原来没有这个诊断调用；现在新增的是数据处理组件，不是完整检测模型。

~~~python
processor = MODELS.build(copy.deepcopy(cfg.model.data_preprocessor)).cpu()
...
out = processor(pseudo_collate(copy.deepcopy(packed)), training=training)
~~~

training=True在这里表示“按训练阶段规则预处理数据”，不表示启动训练。本配置没有批次增强；程序中没有检测器、损失、反向传播或优化器。

关键代码二：水平翻转检查的数学依据。xyxy表示左上角(x1,y1)和右下角(x2,y2)；宽为W时，翻转后的左右坐标是W-x2、W-x1。

~~~python
scaled = raw * np.array([new_w / width, new_h / height] * 2)
if flip:
    scaled[:, [0, 2]] = new_w - scaled[:, [2, 0]]
~~~

我们仅在诊断配置的内存副本中把RandomFlip概率设为0和1，分别检查两条分支。原训练配置仍是prob=0.5；不修改或重训原实验。

## 一个容易误判的官方约定

训练流程：先加载标注，再Resize，因此训练框跟着图像缩放；随后图像与框一起翻转。验证流程：先Resize图像，再加载原标注；验证GT保留原图坐标，因为预测在评价前会还原到原图。本运行CocoMetric还显式使用原COCO标注JSON。

所以验证GT数值没有缩放，不能直接判断为标签错位。本页在缩小图上画验证框时，仅缩放一个用于显示的副本；保存在records.json的actual_sample_gt_xyxy仍是实际原图坐标。不要为“修复”这个正常约定修改验证管线顺序。

本地依据：
- D:\小论文\model_projects\Faster R-CNN\mmdetection\mmdet\models\detectors\two_stage.py 中 predict(rescale=True)。
- D:\小论文\model_projects\Faster R-CNN\mmdetection\mmdet\models\roi_heads\bbox_heads\bbox_head.py 中 rescale分支用scale_factor还原框。
- D:\小论文\model_projects\Faster R-CNN\mmdetection\mmdet\evaluation\metrics\coco_metric.py 中 process/compute_metrics及配置的ann_file。

## 现在在cmd中执行

第一条进入Faster目录，第二条使用本模型专用环境运行CPU检查。-I是Python隔离启动选项，用来减少其他Python搜索路径的干扰，不是创建第二个环境；-X utf8让中文输出统一为UTF-8。

~~~cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
.venv\Scripts\python.exe -I -X utf8 workspace\scripts\diag02_input_pipeline.py
~~~

正常情况下会依次看到：

~~~text
DIAG02_PREPARATION_PASS: frozen1280; CPU data preprocessor only; no checkpoint
DIAG02_PROGRESS branch=train_no_flip images=107
DIAG02_PROGRESS branch=train_flip images=107
DIAG02_PROGRESS branch=val images=23
DIAG02_INPUT_PIPELINE_PASS
REPORT=...\index.html
DIAG02_EXIT_CODE=0
~~~

每20张会打印一次进度。两遍训练图各168框，验证23图37框；总计130张不同图片、237个“图片×处理分支”检查记录，不是新增237张数据。数据原划分不变。出现FAIL请保留完整输出；不要自行改batch、尺寸或反复运行。

看到PASS之后再打开最新报告：

~~~cmd
start "" "D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-INPUT1280-001\latest.html"
~~~

latest.html只在完整检查成功后更新。每次执行创建唯一目录，失败也保留。预检报告不会冒充全量报告。

## 结果在哪里

完整结果：D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-INPUT1280-001\diag02_<日期时间>\

完整日志：D:\小论文\model_projects\Faster R-CNN\workspace\logs\diag02_<同一日期时间>\console.txt

- index.html：中文对照页。固定train8各显示不翻转/翻转两种，共16组；val选cond_000012（含broken）和cond_000060（含GT177），共2组。选图不是根据test表现。
- views：原图标注、处理后图像和最终输入画框图；GT123/177另有输入局部图。所有框均是人工标注，没有预测框。
- records.json / per_image.csv：逐图尺寸、类别、坐标、误差与补边；坐标顺序xyxy，尺寸字段为高×宽。
- summary.json：全量数量、最大误差和运行边界。
- effective_input_config.json：实际读取的预处理配置及诊断分支说明。
- manifest.json：来源/输出哈希、图像哈希、Git版本/改动状态、执行命令。
- code/diag02_input_pipeline.py：本次运行程序的副本，便于以后追溯。

图像等派生输出留在本机，仍由Git规则排除，不上传到GitHub。此诊断不生成mAP，不更新正式成绩总表。

## 看图时只检查这三件事

1. 不翻转与翻转两组：缺陷换到另一侧时，绿色框是否也跟着移动，类别文字是否仍对应。先看cond_000006、cond_000008，再看cond_000108/GT123。
2. 第二列与第三列：除补边外，画面内容、颜色和框位置是否一致。第三列是从真实预处理器输出张量逆归一化后显示；橙色边界标出有效图片，边界外的灰褐色可能是补边。
3. 验证cond_000060的GT177局部：缩放到模型输入后目标还剩多少细节，框是否仍包住同一个目标。局部放大3倍只为了看清像素，不是在给模型增大输入尺寸，也不是预测结果。

不必逐一看完237条数值。先看上面的代表图；如发现框明显错位、翻转不一致或颜色异常，发具体文件名/截图给助手。程序会对全量130张图做数值检查。

## 当前已做的准备检查及局限

助手在4张图片上执行CPU预检：train6/8分别走不翻转及翻转两条分支，val108/115按原验证流程。最终预检目录：D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-INPUT1280-001\preflight_20261019_215606_703453。

结果PREFLIGHT_ONLY_PASS，框坐标最大误差0.000194073像素；归一化与独立NumPy计算最大误差0；逆归一化最大误差0.0000076294像素。已目视检查翻转后cond_000008输入图和GT177局部；全量与用户看图仍待完成。

准备阶段首次试运行因PyTorch SequentialSampler不提供MMDetection所需dataset属性而停止；已在新诊断工具中改用官方DefaultSampler(shuffle=False)。这是本工具的准备错误，不是旧实验训练错误；原失败记录保留。随后通过的检查只做CPU数据处理，没有模型推理。

检查通过可支持：当前冻结管线在检查范围内遵守图像/框/类别/归一化/补边约定。它不能恢复过去100轮未保存的随机批次，不能证明标注语义全部正确，不能证明RPN/ROI或泛化能力无问题。若通过，本阶段到此为止；再进入既定第三步配置、权重与训练/评价适配代码核查，不自动追加训练。

请反馈：完整终端输出（包含RESULT_DIR、LOG_DIR、REPORT、退出码），以及上述代表图里“框是否跟随、颜色和补边是否正常、GT177局部是否仍对应目标”的观察。有问题发具体截图；没有问题也直接说明即可。
