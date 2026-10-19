# 实验保存、登记与核查规则

本规则来自用户2026-10-12的补充要求。后续每阶段完成后执行，不能等所有实验结束才整理。

## 保存

- 原始结果：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\<实验编号>\<唯一运行编号>`。
- 终端日志：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\<阶段运行编号>`。
- 每个正式实验说明：`D:\小论文\model_projects\Faster R-CNN\workspace\notes\<实验编号>.md`。
- 本次VAL运行沿用`step07_235_27043`。固定epoch66的test作为其子目录`test_selected_epoch66\step08_<随机运行编号>`保存。它是同一次训练的评价，不是新训练运行。
- 每次独立训练使用唯一编号。保留失败/中断记录和原始日志。已有大权重和图片直接引用，不重复复制。
- 保存实际配置、代码commit、依赖、命令、数据及划分哈希、COCO初始checkpoint来源/哈希/加载记录、损失/LR、选定权重、原生指标、完整预测和可视化。未来阶段沿用此结构。

## 登记

- 统一CSV：`D:\小论文\model_projects\baseline_analysis\10_all_experiments_metrics.csv`。
- 总Excel：`D:\小论文\model_projects\baseline_analysis\outputs\模型实验结果总表.xlsx`。
- 文字汇总：`D:\小论文\model_projects\baseline_analysis\11_all_experiments_summary_for_teacher.md`。
- CSV从保存的原始指标读取，Excel和报告从CSV派生。不要手动维护多份数字。
- 一个评价编号对应同一个具体训练运行、checkpoint、集合、评价口径及评价事件。原选中轮次val与自动重验证、原生与精确离线COCO结果分别登记，不能择高覆盖。
- 未评价不创建虚假的评价行。待评价/不适用状态写在运行状态表、Excel登记说明和单实验说明中。
- CSV比例为0~1；Excel保留原数值并显示百分数。真实0保留0；未保存、未计算、不适用留空并记录原因。
- Precision/Recall必须记录IoU、阈值选择和计算规则。COCO AR100单独列，不能替代普通Recall。MMDetection类别`*_precision`为AP50-95，不能填入普通Precision。

## 比较与核查

- val/test分开；107与130训练规模分开。MERGE130无独立val，原val进入train后不能当独立验证。
- 保留原生选模依据。离线复算不事后改变已经固定的checkpoint。优先使用保存的预测，不自行启动推理补表。
- 各框架AP、输入缩放、后处理和maxDet不同，未统一时只作参考对照。相同epoch不代表相同更新数和算力。
- 原test使用记录如实保存；后续不根据test调参或挑epoch。单seed不宣称显著性。
- DEBUG8诊断实验单独记录，不混入正式泛化表。
- 每次登记核查CSV/Excel数值、比例单位、空值状态、val/test、类别、checkpoint哈希、epoch、运行编号、来源路径及唯一评价编号。保留历史09报告和四实验Excel。
- 测试脚本自动保存并登记后仍等待用户反馈，再检查实际结果与可视化，报告事实、推测及未完成项目。

## 仅整理现有结果的恢复命令

如果Excel被打开导致写入失败，关闭Excel后只运行以下整理命令。该命令不加载模型，不训练、不推理，不能通过重跑test来修复登记失败。

```cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
"D:\小论文\model_projects\Faster R-CNN\.venv\Scripts\python.exe" -I -X utf8 "D:\小论文\model_projects\baseline_analysis\scripts\refresh_experiment_records.py"
```
