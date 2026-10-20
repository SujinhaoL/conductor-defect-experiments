# 输电导线缺陷检测项目约定

本文件只服务于 `D:\小论文\model_projects` 内的本项目，不是用户全局配置。首次整理日期：2026-10-13。内容根据实际文件核对；实验状态可能变化，继续前应查最新日志和记录，不把本文件中的阶段状态视为永久事实。

## 优先读取：2026-10-21 DIAG04已运行并验收，本轮基础排查阶段性收束

用户已运行DIAG04并回传PASS/退出码0。结果：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-TRACE1280-001\diag04_20261021_020858_182370`；完整日志为 `workspace/logs/diag04_20261021_020858_182370/console.txt`。助手仅离线复核：15份来源及36份输出哈希；三图300个最终预测的类别、顺序、框、分数与历史epoch5结果完全一致；独立NumPy复算softmax、框解码（最大误差0.000765323原图像素）、按类NMS的候选身份和顺序、4GT的20项阶段统计及去向；24张PNG完整解码、27个报告链接存在。12格主要案例拼图已查看，不宣称浏览器交互或全标注语义审查。验收为该运行 `review/acceptance.json`，细节为 `review/independent_check.json`；原始summary/manifest等保留运行时状态，不改写。

主要发现：GT172在ROI同类过滤前有7个IoU≥0.50候选，5个被NMS去重，余2个去重后跨类别分数排名143/168，未进每图总共前100；最终仍有错类松股框。GT194有2个合格松股ROI框（最高IoU约0.687），被同一个更高分但GT IoU约0.492的松股框抑制；独立NMS已定位抑制者。GT171有2个合格断股ROI框，去重后1个IoU约0.619、分数0.002438的框排第450，被前100限制移除。三主要例均有合格RPN交接候选，不能再说它们一开始就没找到位置。附带GT195的RPN最大IoU约0.00685，属于不同情况。

这些合格框均通过了模型score>0.001门槛；旧显示阈值0.30不是它们在最终保存框中消失的全部原因。NMS的0.5比较预测框之间的重叠，不是拿GT选最准的框；不能把官方按分数筛选的正常执行直接称为代码bug。当前证据支持低分/排序与后处理共同影响这些案例，不能宣称仅改NMS或max_per_img就能解决整体AP/泛化，也不将三例推广为全val原因比例。正式AP、原模型配置、权重选择、标注及test均不动。

总体路线第④步的限定停止条件已满足，前三步加本轮材料可做阶段性讨论。没有发现需要重训修复的具体实现冲突；第⑤步受控拟合是条件步骤，目前不自动启动。当前必需用户反馈已齐，不再要求重跑DIAG04、重复确认三例类别或学习代码后才能继续。`start "" ".../FRCNN-DIAG-TRACE1280-001/latest.html"`仅打开已生成报告，可选阅读GT171/194的阶段差异；无新增推理。当前产物为总览修订6及DIAG04验收说明，下一研究方案需要依据这些证据另行确定，不自动追加全层特征、全部图片、阈值搜索、训练、test或STEP18。

## 2026-10-21 DIAG04三图候选过程准备记录（用户已运行，见上方验收）

已重新对照本文件和 `workspace/notes/FRCNN1280_排查路线.md`：仍是第④步有具体缺口才补内部证据。前三步、三例人工反馈不重做。下一步已准备为 `FRCNN-DIAG-TRACE1280-001`：固定STEP11最佳第5轮权重和val/cond_000027.jpg、cond_000125.jpg、cond_000015.jpg三图，对应主要GT172/194/171；GT194同图另有GT195，所以总共4GT。只诊断候选流程，不训练、不调参、不读test、不恢复STEP18。不得把三图原因推广为全val原因比例。

新增工具 `D:\小论文\model_projects\Faster R-CNN\workspace\scripts\diag04_roi_trace.py`，清单 `workspace/configs/diag04_trace1280_val3.json`，操作/代码说明 `workspace/notes/DIAG04_三图候选过程操作与代码导读.md`。旧训练/测试脚本、STEP12和官方代码未改。工具采用冻结val管线与官方model.test_step，hook只捕获ROI输入/输出；官方解码与NMS返回候选身份，分别记录分数筛选、去重与限100框影响。辅助NMS未截断结果只用于分析，真实模型配置仍score>0.001、NMS0.5、最多100；不算新AP。RPN只到已筛选交接候选，不承诺RPN过滤前全量输出。

助手已完成无模型预检、CPU权重元信息检查、合成框筛选身份/空候选/报告图片核查；没有建检测器或前向。真实三图输出尚未产生，必须由用户执行 `.venv\Scripts\python.exe -I -X utf8 workspace\scripts\diag04_roi_trace.py --run`。先核对348项严格加载；实际输出与历史epoch5保存预测的类别/数量/顺序须相同，框坐标容差0.1原图像素、分数容差1e-4；不同就停止，不自动重试或放宽门槛。

用户本阶段必须反馈完整终端输出（尤其RESULT_DIR、LOG_DIR、DIAG04_EXIT_CODE）；无需重复判断类别或学习代码才能推进。新结果为 `workspace/audit/FRCNN-DIAG-TRACE1280-001/diag04_<日期时间>`，日志为 `workspace/logs/diag04_<日期时间>/console.txt`。成功后可选打开该诊断目录latest.html；助手收到输出先复核真实数组，再解释结论与是否达到停止条件。不要再次以“下一步可以准备”结束而不给已准备的操作。

## 2026-10-21 第5轮验证集三例人工反馈验收记录

用户已明确反馈：GT172非常肯定为断股；GT171放大后可辨认断股；GT194绿色GT基本覆盖整体可见缺陷，橙框覆盖散股较明显的中心、未完整包含两侧较轻区域，边界有讨论空间。本阶段人工观察已完成，不再要求用户重复提交这三项反馈。精确原话、数值来源及解释边界见 `D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-SAVED1280-001\diag01_20261019_010031_868401\review\val_epoch005_20261020\user_observation_20261021.json`。本次核对14份来源/既有输出哈希，原analysis/manifest及预测/标注未改写。

GT172保存结果的错类loose框最大IoU=0.7155926083，同类broken最高IoU=0.2326237602；这是最终输出层面的类别不符，不足以单独归因某个ROI分类模块。GT194同类最高IoU=0.4918699039，记录为缺陷边界定义与预测范围差异，不能直接认定人工标错、为迁就预测调整GT或降低评价门槛；一个边界案例不解释全体验证低AP。GT171全部保存框与GT无面积重合，原图缺陷可辨不能证明RPN无候选，也不证明Resize后细节无损。

诊断总览更新到修订4；前三例只补充语义和范围反馈，不等于全37个GT语义全部验收、全框架无错或根因已确定。此次仅更新记录，没有修改任何训练/测试.py、模型配置、正式指标或官方源码；没有运行模型。

下一项明确的证据缺口是：第5轮这三张val图的RPN交接候选、ROI分类/定位输出及最终过滤过程。旧STEP12是第100轮train8，不能代替该证据。若继续，先准备固定第5轮权重、这三张val图、原配置不变的少量内部诊断，由用户cmd启动；优先复用已核查方法。目的为区分候选覆盖不足、ROI输出变化及后处理过滤，不追加训练或调参，不读test，不恢复STEP18。停止条件是记录能支持的环节判断及剩余边界；不要自动扩为全层特征、全数据推理或重训。本轮反馈记录阶段尚未准备或运行这项新推理；不能把计划写成已完成。

## 2026-10-20 诊断总览与三例待反馈记录（已由上方2026-10-21状态更新）

当前交接（2026-10-20）：用户追问“现在需要做什么”，助手已完成第5轮完整val23图37GT保存预测的离线分类，不能继续只说“下一步再分析”。具体入口：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-SAVED1280-001\diag01_20261019_010031_868401\review\val_epoch005_20261020\第5轮验证集逐目标诊断_讨论版.html`。用户现在只需打开网页看三个新例并描述观察：GT172的缺陷类别、GT194的绿色标注范围、GT171能否辨认缺陷；允许“看不清/不确定”，不要求用户证明代码正确。GT177为可选参照，不必重复反馈。此页引用本机原图，通过SVG视窗画框；没有复制或改写数据。打开HTML不使用Python环境。

本次分类及独立复核：第5轮val23图37GT、2300保存框全部核对原始预测与COCO标注；16个同类IoU≥0.50均为loose，另外21个中1个只有错类达到0.50（broken的GT172），20个所有类别均未达到0.50（8 broken、12 loose），其中9个与全部保存框零重合。逐GT独立找最大IoU，不是普通召回率、一对一TP或新增AP；不能证明RPN/ROI/NMS具体根因。三例选择规则：唯一错类覆盖例172、未覆盖中同类IoU最高且最接近0.50的194、全部零重合中编号最小的171。统计覆盖全部37GT，三例只用于人工理解与语义反馈。

本次用Faster专用环境做无模型的独立数值复核，37个结果一致，4张示例原图完整解码及视窗坐标通过；来源、逐目标结果、核查记录在同目录analysis.json、independent_check.json、manifest.json。没有新增常驻实验脚本、没有改旧.py/模型配置/官方代码、没有模型前向/训练/test。诊断总览已更新；先检查用户的三例观察，必要时确认标注定义，再判断最小内部证据缺口，不自动重训或新推理。

2026-10-20补充：用户要求本次及后续交付文件后缀使用“讨论版”，不加“教师”二字。本次总览MD和HTML已按此改名，相关当前链接同步；历史原始证据不批量改名。用户阅读总览无需提交学习反馈才可继续；已提交的DIAG02三项视觉反馈有效。

本次澄清已核实：DIAG01只在浏览器筛选保存框，不是以更低模型输出阈值运行新推理。GT123同类保存框最大IoU=0.3415882722890863，因此仅改变显示门槛不会修复这批框的位置；不外推到未保存的原始候选。原train8最初由STEP05按seed42分层选2张同时含两类、3张仅broken、3张仅loose，并排除重复已登记split_group_id，共5个broken/10个loose框。用于流程/拟合自检，后续固定做训练快照及STEP12诊断；不是网页临时选图，也不是全train107代表性成绩。

网页范围：DIAG01含8+23+23=54个图片×权重记录、31张不同图片，验证两组都是完整23图；下拉框切换图片，默认只画前10框。DIAG02数值核查130张不同图片、237条处理记录，但只展示固定train8×2种翻转分支+val2=18个案例、10张不同图片；不能说全部130张都已人工看图。三组预测的用途分别是末轮训练子集拟合、同末轮完整验证对照、已固定最佳第5轮验证表现，不是必须只诊断三组的通用规范。正式STEP11使用全部train107训练。

第5轮完整val23图37GT保存结果分类现已完成，数值与当前反馈要求见本节最前；旧train8继续作为局部证据，不能取代全量验证或全train拟合结论。优先只读现有结果，不因用户询问抽样就自动追加全train推理或重训。

用户希望明确下一步，并用一份完整过程记录与老师讨论。已新增持续维护的主记录：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN1280_诊断全流程总览_讨论版.md`。本次阶段性快照在 `D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-OVERVIEW1280-001\discussion_20261020_v1\诊断全流程总览_讨论版.html`；两张代表图内嵌，可单独发送HTML，其他原始证据仍只在本机。每个后续阶段完成后更新主记录并保留版本，不把计划写成已完成，不覆盖旧证据。

前三步在各自限定范围内已核查；用户已反馈第一、二步的视觉观察。第三步技术核查由助手负责，代码导读是可选学习，不要求用户证明程序正确，也不再把“是否读完代码”设为进入下一步的条件。每次交付明确区分：助手已完成什么、用户必须操作/反馈什么、哪些只是可选学习；没有必要操作时直接说明。

下一步是第四步候选框过程的解释与缺口判断，先复用现有STEP12，不启动新实验。本次引用的28份STEP12保存文件哈希均与历史验收一致；历史数组独立复算仍引用原验收，没有宣称本次重新运行全部数组复算。GT177是第5轮val低置信度例；GT123是第100轮train定位不足例，二者只能说明不同现象，不能当同条件比较。STEP12只覆盖第100轮固定train8，不能据此认定第5轮val低分的全部原因。

第5轮val的21个未覆盖GT已按DIAG01保存结果细分为1个仅错类覆盖和20个任何类均不覆盖。只有具体关键问题无法从保存文件回答时，才提议少量train/val新推理并交用户cmd启动。当前没有新的训练/推理命令，不恢复STEP18、不扩充数据、不改尺寸或anchor。给老师的主材料是诊断总览；如需各正式实验val/test成绩，再附已有成绩简表。总览不取代原始证据，也不宣称全部框架无错。

## 2026-10-20 DIAG02验收与DIAG03限定范围核查记录

用户运行DIAG02全量检查并回传DIAG02_INPUT_PIPELINE_PASS、退出码0及三张网页截图；明确反馈“翻转同步、最终输入也正常、局部的绿框也有包住缺陷”。本机报告可读取，latest.html指向diag02_20261020_010559_436827/index.html。

本次助手复核15份来源、63份输出、130张原train/val图片哈希全部一致；独立按原COCO复算237条记录的Resize/水平翻转、类别及框对应、补边尺寸，57张保存PNG完整解码，61个网页文件链接存在。train_no_flip与train_flip各107图168框，val23图37框，130张不同图片。最大框误差0.00019407255058467854像素，归一化误差0。用户看图加数值检查支持本阶段通过，不宣称全部标注语义和模型内部无错，也不声称恢复了历史100轮随机batch。

原始报告/summary/manifest不改写；运行时summary中的Visual review pending是历史状态，新增验收：D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-INPUT1280-001\diag02_20261020_010559_436827\review\acceptance.json。代表图及用户反馈范围写在其中。无需重跑DIAG02。

解释“画图副本”：用于人看报告的图像与矩形显示数据，不是给模型输入画好绿框的照片，更不是复制一套训练数据。验证GT保留原图坐标；diag02_input_pipeline.py第49—50行复制框数组，第224—229行仅对显示副本乘scale_factor。训练GT正常随Resize/翻转；不能擅改官方验证管线。运行过的DIAG02脚本哈希304285bde31fdd1ddacd70151936bcb5a1ec9e09b4b914e58264e8e625c13b36，本次未修改。

第三步配置/权重加载/训练更新/评价适配代码核查：助手已只读冻结代码及官方本地实现，核对12份冻结归档、4份历史验收源哈希、官方初始及第5/100轮权重哈希、5400条连续有限loss及一致LR、100轮val选模记录；未发现这些检查范围内需要重训修复的具体冲突。正式val的空预测补零分支100轮均未触发。初始化344项逐值一致是历史首次更新前记录，本次未新建检测器复测；也没有保存历史逐层梯度，不能宣称所有参数每步均更新。

第三步证据：D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-CODE1280-001\diag03_20261020_review\review.json；教学说明：D:\小论文\model_projects\Faster R-CNN\workspace\notes\DIAG03_训练与评价代码导读.md。原review中的USER_READING_PENDING保留为当时状态；现已明确代码阅读可选，不是待验收实验。没有新增常驻训练脚本或修改任何旧训练/测试.py、模型配置、官方源码；不改正式指标表。

第四步优先解读已有STEP12材料并判断具体证据缺口，不自动新推理/训练、不恢复STEP18，不把低分本身当追加实验理由。第二步代码与验收说明已提交；本次总览和状态说明按已授权Git规则管理，派生图像/大型输出继续留本机。以前的准备状态是历史记录，最新安排以文首为准。

## 优先读取：2026-10-19 Git版本管理与第一步用户验收

用户已明确要求创建GitHub仓库管理当前实验。项目Git根目录为 `D:\小论文\model_projects`，远程仓库为 https://github.com/SujinhaoL/conductor-defect-experiments ，保持私有。首次提交只代表2026-10-19现存文件快照，不能伪造之前STEP的提交历史，也不能说本次Git登记重新验证了所有历史实验。

- 代码变动前先读本文件、查看 `git status` 和相关实际文件，尊重用户未提交修改。解释改动的文件、函数/行号、前后diff及原因；本阶段验收后可以提交本阶段已核查的项目代码/说明，并报告commit。提交/推送范围限于已授权项目内容，不能夹带其他未确认修改。
- 默认跟踪自定义脚本、配置、说明、汇总指标及明确允许的小型冻结代码/配置/证据索引；原始图片和标签、环境、官方源码克隆、权重、安装器、大型运行结果和Excel派生文件继续本地保留。新增目录先检查.gitignore，不用git add -f绕过排除规则。GitHub不是完整数据/权重备份；从仓库克隆也不能直接运行旧命令，需要原数据、环境、官方代码和权重。
- 官方MMDetection和Ultralytics源码继续保持各自独立Git仓库，根仓库仅记录来源commit，不改成嵌套子模块、不复制官方源码。保持现有文件路径、原始换行及历史冻结文件，不为建仓库重构实验。
- 日常查看diff可在VS Code打开整个 `D:\小论文\model_projects`，进入“源代码管理”；已提交的版本可用git show或指定两个commit比较。本阶段没有修改训练/评价Python逻辑或任何模型参数。

用户已回传DIAG01_VERIFY_PASS和本机网页截图，确认页面能打开并显示低分预测及GT177/178局部框。这是第一步用户检查通过，不代表用户逐一审查了所有89幅局部图，也不表示所有网页交互均完整测试。

解释边界：GT177同类最大IoU约0.615529、对应分数约0.051652；该预测已经保留并参与原AP评价，只是在旧显示阈值0.30下不画出。不能写成“模型把它过滤掉所以AP低”。第5轮val仍有21/37个GT没有保存的同类IoU>=0.50候选；不能归结为只有显示阈值问题。

Git建仓及用户核验已完成，第二步“真实模型输入前的数据管线与标注对应”已于2026-10-20验收，最新状态见文首；不自动运行新模型、不重训、不读test预测、不恢复STEP18。第一步用户确认记录见诊断目录review/user_acceptance_20261019.json。

## 优先读取：2026-10-19 教学方式与当前单模型排查

用户希望学习代码，不能再只给两三句cmd命令。每次交付阶段必须说明：VS Code打开的完整文件夹；Python解释器完整路径；如何选择cmd终端；哪些文件新增/修改、具体行号或函数、修改前后代码、每个修改的目的和科学理由；执行命令、正常标志、反馈内容。对已有文件保留可比较的修改前副本或补丁；新增文件明确“原来没有”，不能虚构旧代码。关键术语首次出现附中文解释，图上的缩写、编号、阈值与坐标格式要解释，并区分事实、假设和待核查项。不是每一步都必须修改模型代码；纯诊断应明确不改变训练配置。

2026-10-19补充：用户担心每个STEP新增多份脚本造成复杂化。后续同类实验优先复用官方训练入口及少量已核查的公共工具，通过独立小配置/参数和唯一结果目录区分实验，不默认复制整份训练或测试启动脚本。只有现有工具不能回答具体问题时才新增工具，并解释新增必要性。保留历史脚本及冻结证据，不能为整理而批量移动、删除、重写或立即大规模重构。每次先说明本步的目的、输入、是否调用模型、验证什么、什么结果能支持何种判断；明确哪些离线整理由助手已完成，哪些等待用户启动，避免把生成报告说成用户已经运行了新实验。

代码学习可继续在桌面端对话并用VS Code查看文件；插件不是正确复现的必要条件。修改前后优先展示真实diff（差异），保留历史版本，新增文件明确全部为新增。此前项目根目录和Faster workspace未建立Git仓库，曾使用VS Code的两个文件比较；2026-10-19用户要求建立根目录Git仓库，当前规则见文首。官方mmdetection仍是独立Git仓库。不得将备份文件比较冒充Git提交历史，或未经说明将环境/权重/数据/全部结果加入Git。

当前只围绕 `FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115`（Resize(2133,1280)、默认anchor8）重新组织排查。用户说“从0开始”是从基础讲明白、按顺序核查，复用可靠历史证据，不自动重跑已完成工作。当前不扩充数据、不改尺寸/anchor、不恢复STEP18及RT1600补测。完整路线见 `D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN1280_排查路线.md`。

顺序：①已有预测的低分框、局部定位、同轮训练/验证对照；②真实模型输入前的数据管线及框（含训练翻转、Resize、归一化、补边）；③配置、权重加载、训练更新及预测/评价适配代码；④只在证据缺口确有必要时补RPN/ROI与过滤链诊断；⑤只有具体疑点仍未解决时，再提出最小受控拟合训练。每步先验收再前进；新推理/训练由用户cmd启动。不能承诺有限检查能证明所有框架内部无错，也不把全特征图、全部随机ROI采样当作论文必做项目。

第一步诊断编号 `FRCNN-DIAG-SAVED1280-001`，已离线生成至 `D:\小论文\model_projects\Faster R-CNN\workspace\audit\FRCNN-DIAG-SAVED1280-001\diag01_20261019_010031_868401`。入口index.html，证据manifest.json，统计summary.json。脚本workspace/scripts/diag01_saved_predictions.py与diag01_viewer.html，教学说明workspace/notes/DIAG01_低分预测排查与代码导读.md。只读原train8 epoch100、val23 epoch100、val23 epoch5保存预测，不读取test预测、不运行模型。第100轮同一权重用于训练/验证对照；第5轮仅单独解释已固定最佳验证结果，不能用第100轮训练与第5轮验证冒充同轮对照。

本次复算：train8 epoch100为33框、11框score>0.30、11/15GT存在同类IoU>=0.50候选；val23 epoch100为78框、8框score>0.30、4/37GT有同类IoU>=0.50候选（这4个均仅有低分合格候选）；val23 epoch5为2300框、最高score0.1705705822、0框score>0.30、16/37GT有同类IoU>=0.50低分候选。候选覆盖不是普通Recall/AP，不能当一对一TP数。AP引用既有验收，不改变选模、正式总表或历史结果。

显示阈值0仅意味着不再过滤保存结果；保存文件已经过score>0.001、ROI NMS0.5、每图最多100。不能声称恢复了模型原始全量框。RPN旧诊断保存的是自身NMS/top1000后的交接候选。GT123是cond_000108.jpg的人工标注编号，不是图片编号；旧RPN最大IoU0.352908、最终同类最大IoU0.341588是不同集合的最大值，不能说同一个框回归后变坏。

第一步用户已于2026-10-19回传核验通过输出和网页截图，随后Git版本管理已完成；第二步现已验收，第三步限定核查已完成，代码导读为可选学习。后续先查看文首状态、诊断总览及证据，仍一次只推进一阶段。Oct17 test暂停与自动隔离规则仍有效。

第一步验收：89个GT独立IoU复算及历史逐GT比对通过，89幅局部图完整解码，6个代表案例拼图及GT123/177放大图已人工查看；见结果目录review/acceptance.json与review/visual_review.md。网页数据和JS语法已核查；应用内浏览器不允许file协议，用户随后已在本机浏览器打开并反馈截图，尚不声称所有交互均完整测试。AGENTS修改前备份baseline_analysis/audit/AGENTS.before_diag01_20261019.txt，差异见结果目录review/AGENTS_teaching_change.diff；Oct17及之后历史正文保留原样。

## 优先读取：2026-10-17 当前决定与自动隔离规则

用户已决定暂不继续剩下两项补测，下次另行讨论研究方案。STEP18（YOLO11s 1600）仅完成准备，未运行；RT-DETR-L 1600补测也未运行。保留准备脚本、原权重、日志、冻结队列和已有结果，不自动恢复补测、不继续准备下一项，不把未测试填成0或声称最终test对照表已全部完成。下方2026-10-14的补测安排属于历史授权，本节取代其“当前下一项/继续补齐”待办；只有用户后续明确要求继续时才恢复。

本次核对实际指标文件、运行目录和补测日志：九次正式实验中七次已有test，两次尚未test。均沿用原test的23图36框，具体来源及文件哈希见 `D:\小论文\model_projects\baseline_analysis\audit\test_usage_review_20261017\verified_status.json`；补测进度见 `D:\小论文\model_projects\baseline_analysis\audit\test_completion_review_20261014\test_completion_progress.json`。这是2026-10-17快照；以后开始实验前必须再读最新记录。

|正式实验|选定轮次|test状态|
|---|---:|---|
|YOLO11s 1280 / Y11-BASE-001-R1|15|已完成|
|RT-DETR-L 1280 / RTD-BASE-001-R1|71|已完成|
|YOLO11s 1600 / Y11-SENS-IMGSZ1600-001|54|未运行；STEP18准备保留，用户暂不继续|
|RT-DETR-L 1600 / RTD-SENS-IMGSZ1600-001|42|未运行；用户暂不继续|
|Faster R-CNN (1333,800)，train107 / FRCNN-R50FPN-VAL-001|66|已完成，STEP08|
|Faster R-CNN (1333,800)，trainval130 / FRCNN-R50FPN-MERGE130-001|100|已完成，STEP10；无独立val|
|Faster R-CNN (2133,1280)，默认anchor / FRCNN-R50FPN-SCALE1280-VAL-001|5|已完成，STEP15；归档错误已从保存结果恢复|
|Faster R-CNN (2133,1280)，anchor4 / FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001|15|已完成，STEP16|
|Faster R-CNN (2666,1600)，默认anchor / FRCNN-R50FPN-SCALE1600-VAL-001|12|已完成，STEP17|

另有DEBUG-OVERFIT-8和DIAG-RPNROI-TRAIN8两项流程/拟合诊断，仅使用原train固定8图，不计入上述九次正式实验，也不需要补test。YOLO/RT-DETR 1600已有训练和val结果，当前只是缺少对应test成绩。

### 后续每次实验由助手主动执行，用户无需重复隔离提示词

- 每次准备训练、测试或续训前，先读本AGENTS.md、最新实验说明、运行/补测状态及相关实际日志。主动核查历史test使用情况；不能因为用户未提醒或换了模型就当作全新盲测。原test已在本项目多次评价且结果已被查看，这段历史适用于所有继续使用该test的模型，不仅是同一个模型。
- 每个新的独立训练建立新实验编号和唯一运行目录，明确对应模型的环境、配置、数据划分、初始权重完整路径及哈希；独立对照默认从预先确定的官方预训练权重开始，关闭误resume，不继承旧实验的私人数据权重。保留全部旧权重和日志；通常复用已核实的同模型环境，不因“隔离”重复安装环境或复制数据。
- 独立测试明确唯一的训练运行、已固定checkpoint及其哈希，在该运行下创建独立测试结果目录，核对实际评价集合；不得混用其他尺寸/模型的权重，不覆盖旧结果。已有保存预测足以完成的整理/复算，使用独立衍生文件并记录来源，不启动新的模型推理。
- 用户明确要求续训、迁移或其他依赖旧权重的实验时，按实际研究目的记录父运行、加载权重和resume行为，不强行改为从头训练，也不得把这种实验冒充独立初始化对照。范围或必要条件确实不清楚时才询问，不为常规隔离反复索要确认。
- 在用户执行命令前，用简短中文说明本次已做的隔离：实验编号、结果目录、使用哪份权重、是否续训、允许使用哪些集合及已有test历史。无需再次要求用户粘贴一整段提示词；隔离不授权助手代为启动训练/测试。
- 改模型/参数和选择checkpoint主要依据train/val；测试前固定模型、配置、权重、评价口径及必要的阈值选择规则，不依据test再挑epoch、尺寸、anchor或阈值。用户想法若可能来自test，先说明并如实记录其依据，必要时询问；不能编造“未受test影响”的理由。
- 文件隔离能避免用错权重、误续训和覆盖记录，不能消除已经看过test的信息影响，也不能保证结果无偏。即使尚未测试某个1600模型，同一test也不是本项目从未接触的数据。新方案若参考了test结果，标注探索性质；正式报告披露历史test使用，不称为严格盲测。若研究需要新的独立盲测结论，再讨论另有未用于开发的数据，不擅自重划现有数据，也不要求用户为了继续工作立即收集新数据。

## 工作方式

- 使用中文解释，面向初学者，科学、客观，不迎合；事实、假设和待核查项分别表述。
- 每次准备告诉用户“下一步”前，先实际读取本AGENTS.md、总体排查路线及相关最新验收，说明本步对应总计划哪一步、具体证据缺口、最小范围及停止条件。不能只凭长对话记忆推演，不把计划或工具准备写成已运行结果。交付时明确用户现在的必要操作、正常标志和必须反馈的内容；可选代码学习/看图另行标明。若确实无需用户操作，说明当前阶段结论，不反复留下只有口头计划的交接。
- 用户自己在 VS Code 的 Windows **cmd** 终端启动训练及测试。助手不得自行启动训练，包括小样本过拟合、带更新的显存试跑；也不得为补表自行启动新的模型推理。
- 助手可读取本地文件，准备官方代码/权重、配置和脚本，做无模型前向的 CPU 数据/坐标/anchor 检查，复算已保存预测，整理结果。
- 一次推进一个阶段。准备后给出完整路径、可直接复制的 cmd 命令、正常标志、需要反馈的输出；收到输出先验收再进入下一阶段。已完成工作不要仅为重新整理而重跑。
- 每次实验后说明：目的是否达成、当前证据支持什么、是否偏离原问题、下一步的必要性与停止条件。不要因分数低无限追加实验。
- 新实验前用初学者能理解的语言说明：它回答小论文的什么问题、已有证据、改哪个因素、预期如何判断以及何时停止。输入尺寸对照和anchor诊断是不同问题，范围改变必须明确解释；不把辅助诊断说成论文必做项目。用户再次强调避免复杂化，不自动追加训练；STEP13归档后用户明确要求最后再做一次增大输入尺寸的实验，范围见下方STEP14。
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
- `FRCNN-R50FPN-SCALE1280-VAL-001\step11_31112_14115`：107/23，Resize(2133,1280)，100轮5400更新；按同一原生val规则选第5轮。训练/val核查见workspace/audit/step11_review/acceptance.json。val mAP50=0.03189324821274552、mAP50-95=0.014877891782027844、AR100=0.11607142857142856（均为0..1底值），val broken AP为0。固定第5轮test已完成并核查，结果在test_selected_epoch5/step15_23467_7311；末尾归档错误仅从保存预测恢复，原FAIL日志保留，未重跑模型。test精确mAP50=0.008194318011112834、mAP50-95=0.002866443023255048、AR100=0.0625；验收见workspace/audit/step15_review/acceptance.json。
- `FRCNN-DIAG-RPNROI-TRAIN8\step12_24867_6686`：已完成且复核通过；原800和新1280的epoch100各诊断同一原train8一次，共16图次。仅拟合诊断，不列入正式泛化成绩。独立复算见workspace/audit/step12_review/acceptance.json，说明见workspace/notes/FRCNN-DIAG-RPNROI-TRAIN8.md。
- `FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001\step13_27622_23378`：已完成100轮5400更新，按原val规则选第15轮。唯一科学变化为固定1280下anchor scale8→4。精确val mAP50=0.054506931335776214、mAP50-95=0.011591385071287288、AR100=0.05，val broken AP为0；训练/val复核见workspace/audit/step13_review/acceptance.json。固定第15轮test已完成并登记，结果在test_selected_epoch15/step16_27428_3464；精确test mAP50=0.019089244127215983、mAP50-95=0.004246304645777617、AR100=0.08660714285714285；验收见workspace/audit/step16_review/acceptance.json。
- `FRCNN-R50FPN-SCALE1600-VAL-001\step14_3575_9657`：已完成并核查100轮5400更新，按原val规则选第12轮。相对STEP11默认anchor8，仅Resize(2133,1280)→(2666,1600)。精确val mAP50=0.05168344875198208、mAP50-95=0.012840958397665272、AR100=0.1494047619047619；训练/val复核见workspace/audit/step14_review/acceptance.json。固定第12轮test已完成并登记，结果test_selected_epoch12/step17_23519_12608；精确test mAP50=0.05520830195125764、mAP50-95=0.018482112673997353、AR100=0.14821428571428572；验收workspace/audit/step17_review/acceptance.json。

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

## 2026-10-14 训练与诊断阶段记录（最新安排见文首）

尺寸对照STEP11已完成并核查，不能再当成未启动或重跑。以VAL-001为对照，保持107/23和训练设置，仅训练/验证/快照Resize改为(2133,1280)，从同一COCO权重重新训练。实际耗时3095.629秒、峰值CUDA allocated 8903.871MiB；配置差异、5400条学习率、归档哈希、100轮val文件、保存预测COCO复算均核对。
本次val mAP50-95小幅提高而mAP50下降，未解决低泛化问题。末轮固定train8诊断mAP50-95约80.198%，不作为泛化成绩；11/15GT有同类候选IoU≥0.50，4个loose仍未覆盖。第5轮val的2300个低阈值预测均不超过0.30，显示无框不等于无预测。
说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1280-VAL-001.md`。历史准备说明STEP11_resolution1280_plan.md和运行内冻结code保留原状态；实时核查状态以acceptance.json、当前说明及总表为准。

STEP12已完成，不应为整理再跑：800的RPN/ROI过滤前/最终同类框均覆盖8/15GT，1280均11/15（覆盖指存在IoU≥0.50候选，不是普通Recall）。1280未覆盖GT87/123/124/160均loose，RPN候选及ROI同类回归在过滤前最大IoU都低于0.50；四者恰好对应默认anchor几何检查中零正样本目标。支持anchor匹配假设，但不证明全部val误差由其导致。800的GT88零正anchor仍被检出，不能声称绝对不可检测。ROI训练还会加入GT候选，不能声称模型完全未学该目标。
16份保存数组独立复算、原图坐标/类别/概率/FPN形状、checkpoint哈希、末轮快照一致性均通过。RPN是自身NMS/top1000之后的交接候选，不是原始全量候选；没有定位RPN排序/NMS各自贡献。四个漏检局部图已目视核查，不凭外观断言标注错误。原val最佳epoch66/5不变，epoch100仅用于拟合诊断。

STEP13已完成并验收，不要重跑：`FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001`。唯一科学参数变化是RPN anchor_generator.scales=[8]→[4]；继承STEP11冻结配置，仍Resize(2133,1280)、107/23、batch2、FP32、seed42、SGD0.0025、100epoch5400更新、同调度/flip/过滤/val选模规则，从同一官方COCO重新开始，不续训私人权重。历史准备方案STEP13_anchor4_plan.md和运行code保留，最新结果见workspace/notes/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md。
原train107/168框的CPU anchor核查（workspace/audit/step13_anchor_geometry/anchor_geometry.json）显示：固定1280，scale8→4使零正anchor目标33→3，未翻转/水平翻转均如此；30个获得正anchor、原有正anchor者无新增遗漏，诊断4个目标均获正anchor，仍剩GT102/103/104。是采样前几何结果，不是召回率/AP或实际训练采样记录。每位置仍3个anchor，RPN张量形状不变；预训练张量原来对应默认几何，兼容加载不保证收益。
STEP13保存预测独立COCO复算、100轮val文件、5400条有限loss及与对照相同的学习率、冻结输入/权重哈希/344项COCO加载、类别坐标和可视化均核查通过；助手没有启动新的模型前向或训练。固定train8末轮mAP50=1.0、mAP50-95=0.9619059405940594，15/15GT有IoU≥0.50同类框；原4个loose漏检均恢复（IoU约0.949/0.990/0.977/0.960）。这是107张训练后的8图拟合诊断，不是全107张完全过拟合，也不是重跑原8图专用过拟合。
STEP13主要val mAP50-95由STEP11的1.4878%降至1.1591%，mAP50由3.1893%升至5.4507%，不能只挑有利指标宣称泛化改善。耗时3110.653秒，峰值allocated8903.870MiB；选中val2100个低阈值框，显示>0.30有15个；固定同类分数降序一对一IoU≥0.50检查TP4/FP11/未匹配GT33，仅作显示阈值诊断，不改变AP/选模。最佳epoch15 SHA256=080c27214809847dce4e2d1b5e760a3e8b868bdb0d0f2941e5e43f219ffb548a，末轮SHA256=21ef28787593f368bfab10ee40c7839c0529cb9d8f603dc9abd4a257394b71d5，以selected_checkpoint.json核验。
STEP13的停止条件已满足：完成固定100轮并总结假设。本次支持部分训练小目标的anchor匹配限制，不证明所有val低分的原因或全链路无错。诊断实验不放入正式泛化表；不为填表跑test。

STEP14已完成并验收，不要重跑：用户明确要求的最后一次增大输入尺寸实验 `FRCNN-R50FPN-SCALE1600-VAL-001/step14_3575_9657`。当前说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\FRCNN-R50FPN-SCALE1600-VAL-001.md`。历史STEP14_resolution1600_plan.md、step14_preflight及运行code保留准备时状态，不当作当前未启动证据；一次性step14_verify_handoff.py不得在已运行后重跑。
直接对照STEP11默认anchor8+1280，冻结配置仅训练/验证/固定train8快照Resize改为(2666,1600)、keep_ratio=True；其余107/23、100epoch5400更新、batch2、FP32、seed42、SGD/调度/flip/过滤/val选模规则相同，从同一官方COCO权重重新开始。不是与STEP13 anchor4做尺寸单因素比较。100轮val、5次train8快照、5400条有限loss和相同LR、归档/数据/权重哈希、344项加载、预测原图坐标/类别、COCO精确复算和可视化均通过；助手未运行新的模型推理或训练。
按原生val coco/bbox_mAP三位小数最大值、同值保留首次，选第12轮。选中SHA256=f2431a883b6e39ea03e906fd5546f4d7f6823d35bb8f72cb43ae5db019ab8167；末轮SHA256=52498439d946ec239688376c1ce2e009538c042563fd7579aa966248e48a72a7，以实际selected_checkpoint.json核验。精确val mAP50=5.168345%、mAP50-95=1.284096%、AR100=14.940476%；相对1280主AP下降而AP50上升，不能只挑有利指标。broken AP50=0.066748%、AP50-95=0.040049%，原生类别AP=0.000是舍入，不是精确全零；loose AP50=10.269942%、AP50-95=2.528143%。训练验收当时尚未test，随后已在STEP17完成并登记，详见下文。
末轮固定train8 mAP50=99.174917%、mAP50-95=87.904290%，15/15GT有IoU≥0.50同类候选；是107训练中的8图拟合诊断，不是全107图完全过拟合或独立泛化。1600静态anchor仍有4个零正GT却最终能覆盖，不能把零正anchor说成绝对不可检测。未为本轮新增RPN/ROI内部推理。
本次实测耗时6703.556秒（111.73分钟，含val/保存/快照）、峰值CUDA allocated13694.866MiB（13.37GiB），无OOM/中断；不把allocated当整机显存，不把单次耗时差异全归因于尺寸。选中val2021个低阈值框，17张达100框上限；显示>0.30有10框，同类分数降序一对一IoU≥0.50匹配TP3/FP7/未匹配GT34，只作显示阈值诊断。
停止条件已满足：本轮尺寸尝试结束，当前没有准备下一次训练，不自动搜索更大尺寸/anchor/模块，也不为补表跑test。默认anchor下800、1280、1600未显示主要val AP持续提高，未解决泛化低的问题；不能证明尺寸无影响或低分只有一个原因。单seed、小val多轮开发及旧test已被查看的限制仍须披露。原18行记录保留，新增1600原生/精确val两行。

## 2026-10-14 的 test 补齐安排与执行记录（余下两项于2026-10-17暂不继续）

用户在2026-10-14要求按以下顺序补齐五项test，前三项已完成。2026-10-17用户决定暂不继续余下两项，最新安排见文首；以下保留历史过程，不作为自动恢复任务的指令。仍禁止助手自行启动模型推理或训练。补测不是追加训练、重新挑选checkpoint或按test调参。

1. Faster R-CNN 1280默认anchor：STEP11第5轮。
2. Faster R-CNN 1280 anchor4：STEP13第15轮。
3. Faster R-CNN 1600默认anchor：STEP14第12轮。
4. YOLO11s 1600：第54轮对应best.pt。
5. RT-DETR-L 1600：第42轮对应best.pt。

固定清单及五份权重完整路径/哈希：`D:\小论文\model_projects\baseline_analysis\audit\test_completion_review_20261014\fixed_test_queue.json`。保留冻结时的顺序、权重及状态，不用其中TEST_PENDING字段推断当前待办。若用户以后明确恢复，每次只推进一项，先核对实际文件，由用户cmd启动，助手验收并登记后再准备下一项；不以分数低为由跳过届时已授权评价。

第一项STEP15已完成并登记，结果在`Faster R-CNN/workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115/test_selected_epoch5/step15_23467_7311`，日志在workspace/logs/step15_23467_7311。348项严格加载、23图完整推理、原生/精确COCO指标和2300个预测均已保存；score>0.30为0个。最后sha收到str而非Path导致归档FAIL；当前脚本已修复，仅通过`step15_recover_saved_results.py`复核保存预测并补齐summary/manifest，无新的模型推理或训练。原FAIL日志/manifest和冻结code保持原样；恢复文件`postprocessing_recovery.json`明确记录原因。未落盘的原推理耗时/显存留空。不要再运行STEP15。验收`workspace/audit/step15_review/acceptance.json`，单次说明和总表已更新。

第二项STEP16已完成、验收并登记：原STEP13训练目录下`test_selected_epoch15/step16_27428_3464`，日志workspace/logs/step16_27428_3464；退出码0，无中断。348项加载、数据/权重/配置/预测哈希、原图坐标和类别、23张图、2135个低阈值预测、精确COCO复算与原生舍入均通过。27个score>0.30框分布在11张图，固定同类一对一IoU>=0.50诊断TP0/FP27/未匹配GT36；不作为普通P/R或AP替代，不据此调阈值。含推理/读图/画图8.061794秒、峰值allocated1057.304MiB，不当设备总占用或严格效率比较。证据workspace/audit/step16_review/acceptance.json；说明workspace/notes/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md。不要重新运行STEP16。

第三项STEP17已完成并验收登记：原STEP14训练目录下test_selected_epoch12/step17_23519_12608，日志workspace/logs/step17_23519_12608；退出码0，无中断。348项加载、数据/配置/权重/预测哈希、类别/原图坐标、23张可视化及独立COCO复算均通过。1981个低阈值预测，8个score>0.30框分布在7图，显示阈值诊断TP2/FP6/未匹配GT34；不替代AP/AR或普通P/R。含读图/推理/画图9.730124秒、峰值allocated1527.131MiB，不当总设备占用或严格效率排名。不要重新运行STEP17。本次三个Faster补测均已完成；训练/val/test分别解释，不依据较高test分数再选尺寸或anchor。

原计划下一项STEP18（2026-10-17用户暂不继续）：YOLO11s1600，第54轮对应best.pt。入口`D:\小论文\model_projects\yolo11\workspace\scripts\run_step18_test_yolo1600.cmd`，程序step18_test_yolo1600.py、项目侧保存预测适配step18_yolo1600_evidence.py、配置workspace/configs/step18_yolo1600_test.yaml，说明workspace/notes/STEP18_yolo1600_final_test.txt。已通过CPU准备及合成坐标/原生指标接口核查，证据YOLO workspace/audit/step18_preflight；2026-10-17复核仍未启动test，无本项模型推理结果或运行日志。准备文件保留，不再作为当前待执行命令；恢复时重新核对环境和路径。
必须使用YOLO自己的`.venv/Scripts/python.exe`（Python3.12.13、torch2.8.0+cu128、torchvision0.23.0+cu128、ultralytics8.4.152、numpy2.5.2、pillow12.3.0、pyyaml6.0.3），官方YOLO源码commit9aa41d679d121afb8b4cc512d7a162f4f36886c3，tracked源码无改动。固定weight哈希以冻结队列核对。原YOLO test23图36框，imgsz1600、batch4、workers0、rect=True、quantize=16（FP16）、conf0.001、NMS0.7、max300。batch/workers/FP16匹配原1600训练结束验证器；旧1280独立测试为默认FP32，必须披露此差异。
保持save_json=False：该版本在自定义数据集开JSON时会触发COCO评价并替换原生AP，不能为保存文件静默改变口径。项目侧验证器只在官方update_metrics后导出克隆/缩放的完整预测，在原get_stats前保存原生匹配数组，不改NMS或AP计算；画图复用同次输出。原生P/R按IoU0.50及平滑均值F1最大处报告，保存其阈值，不把test统计阈值用于开发。复用readonly_dataset只读保护及workspace标签缓存适配，数据和官方源码不写。结果原YOLO1600训练目录下test_selected_epoch54/step18_<唯一编号>，日志YOLO workspace/logs同编号。已有test目录时停止，不覆盖或自动重试；只有用户以后明确恢复并运行后，才验收并扩展Ultralytics补测登记逻辑、更新表和报告，再根据届时授权决定是否准备RT-DETR-L1600第42轮。

`refresh_experiment_records.py`已支持三项尺寸/anchor实验的已验收補测读取，必须有该测试的acceptance、visual_review及证据哈希才能登记。当前26个唯一评价记录，STEP15/16/17各原生/精确test两行；更新前24行指标及源文件字段保持不变，1448个CSV/Excel相关单元格一致。登记证据baseline_analysis/audit/step17_registration_verification.json。Excel结果页标注anchor4以区分同1280尺寸；原生/精确复算为同次预测的不同精度记录，不是两个实验。旧四次Excel保留历史版本，val/test分别展示。每完成一项就登记；补测进度baseline_analysis/audit/test_completion_review_20261014/test_completion_progress.json，固定权重清单保留冻结状态。

用户2026-10-14要求更简约、英文附中文解释的教师表，已新增`D:\小论文\model_projects\baseline_analysis\outputs\模型实验简表_教师讨论版.xlsx`；原详细总表作为证据索引保留，旧四次Excel不改写。给老师主文件改用简表，建议配11_all_experiments_summary_for_teacher.md。仅两页val（验证集）/test（测试集），各九次正式实验，一次实验只展示一行，主要指标在上、断股/松股AP在下；未测/未算留空，合并130训练val不适用。Faster取精确COCO保存预测复算；Ultralytics取固定权重的原生独立评价，1600的val统一采用训练结束best.pt重验证（与训练内CSV选中轮次可能略有不同，不能混用同一行总体与类别来源）。单次来源映射在audit/teacher_simple_plan.json。342个保存XLSX单元格、百分比/空值/类别核查通过，证据audit/teacher_simple_acceptance.json；无新增模型推理。独立脚本prepare_teacher_simple.py/build_teacher_simple.mjs/verify_teacher_simple.py生成，已接入refresh_experiment_records.py登记后自动刷新，数字从同一CSV派生，不手工维护第二套数值。原详细总表的英文机器标识继续保留；教师简表提供中文状态/指标解释。

后续新实验统一执行文首“后续每次实验由助手主动执行”的隔离规则，无需用户再次粘贴隔离提示词。

## 指标、结果登记与比较边界

- 原始结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\<实验编号>\<运行编号>`；日志在workspace/logs；单次说明在workspace/notes/<实验编号>.md。
- 统一CSV：`D:\小论文\model_projects\baseline_analysis\10_all_experiments_metrics.csv`
- 总Excel：`D:\小论文\model_projects\baseline_analysis\outputs\模型实验结果总表.xlsx`
- 总报告：`D:\小论文\model_projects\baseline_analysis\11_all_experiments_summary_for_teacher.md`
- 历史四次报告 `D:\小论文\model_projects\baseline_analysis\09_four_experiments_summary_for_teacher.md` 和旧Excel保留。
- 已有登记入口：`D:\小论文\model_projects\baseline_analysis\scripts\refresh_experiment_records.py`。九次正式实验，含STEP11/13/14各原生/精确val，以及STEP15/16/17各原生/精确test，共26个唯一评价记录。YOLO/RT-DETR1600未test，用户于2026-10-17决定暂不继续；若日后恢复并完成，仍需扩展各自真实来源读取，不能假定自动登记。CSV/Excel一致性见baseline_analysis/audit/registration_acceptance.json；教师简表由同一次CSV自动刷新。旧表中这两项“待测试”表示缺少成绩，不代表当前安排继续执行；当前安排以文首和补测进度记录为准。
- 每行代表运行+checkpoint+评价集合+口径/事件，唯一评价ID防重复；从原始指标生成CSV和Excel，不手抄多份冲突数字。记录配置/代码版本/环境/命令/数据/初始及选中权重的来源与哈希。
- 比例底值0..1，Excel显示百分数。未算/不适用/未保存留空并记录原因，真实0才填0。
- MMDetection class *_precision 是类别AP50-95，不是普通Precision；AR100不能替代普通Recall。候选IoU覆盖数也不是普通Recall。
- 原生三位小数与保存预测的精确COCO复算分开登记；复算不得追溯改变已固定的checkpoint。只凭0.000不能断言精确全零。
- val/test分表；107/130训练规模注明。跨框架输入、AP计算、预训练、后处理不同，未统一前只能参考比较；同epoch不代表同更新数或同算力。单seed不宣称统计显著性。
- 用户已查看旧test结果。后续尺寸/模型开发用val，不能把旧test称为从未接触的盲测集；新泛化结论需要诚实披露这一限制。
- 每轮更新核对CSV/Excel数字、百分数、空值、集合、类映射、checkpoint/epoch/run及来源路径，最后明确新增文件和仍待核查项目。不为补表重跑模型。
