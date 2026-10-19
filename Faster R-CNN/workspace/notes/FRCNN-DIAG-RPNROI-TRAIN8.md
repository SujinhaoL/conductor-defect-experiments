# FRCNN-DIAG-RPNROI-TRAIN8

状态：STEP12用户运行成功；保存结果独立复核PASS。属于训练集流程诊断，不是正式泛化实验。原始summary.json中的REVIEW_PENDING保留为运行时历史状态，当前以本说明和独立acceptance.json为准。

运行编号：`step12_24867_6686`。原始目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-DIAG-RPNROI-TRAIN8\step12_24867_6686`。完整终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step12_24867_6686\step12_console.txt`。

## 目的和范围

定位800→1280后固定训练图仍有4个目标没有足够重合的最终同类框，查看RPN输出、ROI类别回归及过滤环节。采用原107张训练运行的epoch100权重（VAL-001和SCALE1280-VAL-001），同一原train8/15GT，各推理8张。不是8图重新训练，不做优化器更新，不使用val/test。原val最佳epoch66/5未改变。

类别COCO1=broken/2=loose，模型0/1，ROI第3个概率为背景。显示阈值0.30与评价过滤阈值0.001分开；局部诊断图的红框是最终返回框中与该GT同类IoU最高的框，不另用0.30筛选。

## 已核验的事实

两份checkpoint各348项严格加载，哈希/配置对应。16份NPZ保存数组均独立复算：RPN坐标N×4，ROI类别相关回归N×2×4，分类概率N×3；类别、有限值、原图坐标边界、FPN层尺寸通过。诊断最终预测与既有epoch100快照逐项比较通过（原图框绝对容差0.1像素，score绝对容差1e-4）。没有为复核再次运行模型。

|模型尺寸|GT总数|RPN候选覆盖|ROI同类回归、过滤前覆盖|最终同类框覆盖|
|---|---:|---:|---:|---:|
|baseline800|15|8|8|8|
|scale1280|15|11|11|11|

覆盖定义为每个GT是否存在IoU≥0.50候选；不做一对一检测匹配，不考虑完整精确率曲线。因此不是普通Recall或COCO AR100，也不是验证/测试AP。800→1280恢复了GT6/18/122，其余原8个覆盖目标保留。

|1280未覆盖GT|图片|输入中的框宽×高（像素）|默认anchor最大IoU|RPN最大IoU|ROI同类过滤前最大IoU|
|---|---|---:|---:|---:|---:|
|87（loose）|train/cond_000087.jpg|31.19×4.98|0.1517|0.2748|0.3493|
|123（loose）|train/cond_000108.jpg|11.71×9.91|0.1133|0.3529|0.3661|
|124（loose）|train/cond_000108.jpg|14.55×11.09|0.1576|0.2904|0.3976|
|160（loose）|train/cond_000143.jpg|19.95×7.07|0.1379|0.2632|0.3320|

4个目标在RPN传给ROI的候选中均无IoU≥0.50框，ROI同类回归在过滤前最大IoU仍小于0.50。因此不能只靠降低显示阈值或更改最后NMS把这些当前预测变成IoU≥0.50命中。4个目标恰好对应默认anchor几何检查中无正样本的目标，支持进一步检验anchor尺寸匹配。

## 尚未证实的解释与可视化边界

这不是证实MMDetection实现有bug，也不是所有val/test低分由anchor造成的证明。800下GT88没有正anchor却仍被检测到，说明零正anchor不是数学上的绝对不可检测。ROI训练配置会加入GT作为候选，不能说RPN零正anchor意味着整个模型从未学习该目标。

保存的RPN候选已经过RPN自身NMS及top1000筛选，尚未区分原始候选生成、评分、排序与RPN NMS的各自贡献。已保存RPN/ROI中间框数组、每GT局部对照图和FPN形状；没有恢复历史训练的随机ROI采样或保存所有层的完整特征图。

已目视检查4个1280漏检局部图及800/1280并排图。绿色GT通常比候选更紧、更小；这说明定位重合不足，不代表标注一定错误。缺陷语义及标注边界标准需领域判断，不凭本次图像改标。

## 本阶段结论与下一阶段

目的达成：诊断将这4个训练目标的问题定位到RPN交接候选覆盖不足、ROI现有回归没有修复，排除了仅由显示阈值造成的解释。它直接服务于小目标低分问题，没有因低分盲目加尺寸。诊断范围有限，不宣称全链路全部无错。

随后仅用原train107/168框做CPU匹配核查：固定1280，anchor scale8→4使无正anchor目标33→3，翻转前后结论一致，原正样本目标无新增遗漏。下一次只准备固定1280、anchor scale4的单因素对照，从同一COCO权重重训107/23，100epoch5400更新，val选模，不做test。具体设置和停止条件见STEP13_anchor4_plan.md；尚未启动，不能记成成绩。

## 证据完整路径

- 原始诊断摘要：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-DIAG-RPNROI-TRAIN8\step12_24867_6686\summary.json`
- 原800逐GT：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-DIAG-RPNROI-TRAIN8\step12_24867_6686\baseline800\per_gt.csv`
- 新1280逐GT：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-DIAG-RPNROI-TRAIN8\step12_24867_6686\scale1280\per_gt.csv`
- 独立复算验收：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step12_review\acceptance.json`
- anchor与RPN/ROI联合表：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step12_review\anchor_proposal_roi_comparison.csv`
- 四个目标800/1280对照图：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step12_review\four_uncovered_targets_comparison.jpg`
- 复算脚本：`D:\小论文\model_projects\Faster R-CNN\workspace\scripts\step12_review_results.py`
- 后续CPU几何核查：`D:\小论文\model_projects\Faster R-CNN\workspace\audit\step13_anchor_geometry\anchor_geometry.json`
- 下一阶段方案：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\STEP13_anchor4_plan.md`
