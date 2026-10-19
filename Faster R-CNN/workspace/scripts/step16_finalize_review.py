"""Finalize the reviewed STEP16 evidence, update notes, back up canonical records. No model imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent/'baseline_analysis'
RUN = ROOT/'workspace/runs/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001/step13_27622_23378'
DEST = RUN/'test_selected_epoch15/step16_27428_3464'
AUDIT = ROOT/'workspace/audit/step16_review'
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
save = lambda path,value: path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


review = read(AUDIT/'acceptance.json')
assert review['status']=='PASS' and review['display_threshold_TP']==0 and review['display_threshold_FP']==27
review.update(visual_review_pending=False,visual_review_completed=True,
    visual_review_note='Assistant inspected the all23-image contact sheet of existing predictions with original GT added in green.27 score>0.30 boxes occur on11 images. Same-class one-to-one IoU>=0.50 matching finds0 TP and27 FP, checked separately from AP; no new inference.')
save(AUDIT/'acceptance.json',review)
note_path = ROOT/'workspace/notes/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md'
note = note_path.read_text(encoding='utf-8')
note = note.replace('状态：100轮训练及验证结果已复核；test未运行。', '状态：100轮训练、验证及固定第15轮test结果均已复核；STEP16补测正常完成并保留原始证据。')
note = note.replace('普通P/R未单独登记，AR100不能代替普通Recall；test未运行不填0。', '普通P/R未单独登记，AR100不能代替普通Recall。固定第15轮补测test结果见末尾STEP16部分。')
note = note.replace('当前只归档并总结，不继续自动加大尺寸/改anchor/加入模块，不为填表推理或test。后续如需新实验，应先明确论文问题、单一因素、预算、评价标准和停止条件，再决定是否必要。', '该训练阶段已归档。用户于2026-10-14指定五项已有权重的最终test补齐，第15轮补测已完成；未按test改选epoch或阈值。本轮不追加训练，后续新训练仍需先明确论文问题、单一因素、评价标准和停止条件。')
if '## STEP16 固定第15轮最终test' not in note:
    note += f'''
## STEP16 固定第15轮最终test

用户2026-10-14启动，运行step16_27428_3464，正常退出码0，无中断。原test23图/36框（broken8、loose28），沿用选定val的Resize(2133,1280)、keep_ratio=True、anchor4、FP32、score_thr0.001、ROI NMS0.5、max100及COCO maxDets[1,10,100]。348项权重逐值加载核查通过，类别映射和坐标核查通过，test未参与选模。

|精确COCO指标|val：37框|test：36框|
|---|---:|---:|
|mAP50|5.450693%|1.908924%|
|mAP50-95|1.159139%|0.424630%|
|COCO AR100|5.000000%|8.660714%|
|broken AP50|0.000000%|3.410178%|
|broken AP50-95|0.000000%|0.708449%|
|loose AP50|10.901386%|0.407671%|
|loose AP50-95|2.318277%|0.140812%|

原生test mAP50=0.019、mAP50-95=0.004，类别AP50-95 broken=0.007、loose=0.001；上表为同次保存预测独立复算的精确值。普通P/R未计算，保持空值；AR100单列，不能当普通Recall。

2135个低阈值预测全部保存。score>0.30有27框；按固定同类、分数降序、一对一IoU>=0.50的显示阈值诊断，TP0、FP27、未匹配GT36。27个显示框不能解释为检出27个目标，低阈值预测中仍有贡献AP的正确框。该诊断不替代COCO AP/AR，不写入正式普通P/R列，也不据test更改阈值。

评价含原图读取/23图推理和预测图片保存共8.061794秒，峰值CUDA allocated1057.304MiB；此耗时不含完整程序启动/CPU权重加载/末尾指标计算，峰值不是设备总占用，不作为严格效率排名。

已确认结果：固定anchor4模型test AP比固定anchor8第5轮的对应test数值更高，但整体仍低；val主要AP方向不同。单seed、每集合23张、不同val选定epoch和旧test使用限制须说明，不能证明该anchor修改普遍提升泛化或低分机制已经确定。此次按固定清单补齐最终评价的目的已达成；下一项仍是1600默认anchor第12轮，不根据本次分数改变顺序或调参。

- 完整test结果：`{DEST}`
- 原生指标：`{DEST/'native_coco_metrics.json'}`
- 精确COCO指标：`{DEST/'test_metrics.json'}`
- 全部低阈值预测：`{DEST/'predictions.bbox.json'}`
- 坐标、类别、置信度及处理尺寸：`{DEST/'predictions_details.json'}`
- 23张预测可视化：`{DEST/'predicted_images'}`
- 配置/命令/版本/哈希：`{DEST/'pretest_record.json'}`
- 原始终端日志：`{ROOT/'workspace/logs/step16_27428_3464/step16_console.txt'}`
- 验收：`{AUDIT/'acceptance.json'}`
- 独立保存预测复算：`{AUDIT/'independent_coco_check.txt'}`
- GT与已保存预测总览：`{AUDIT/'test_gt_and_visible_predictions.jpg'}`
- 固定显示阈值诊断：`{AUDIT/'display_threshold_diagnostic.json'}`
'''
note_path.write_text(note,encoding='utf-8')
backup = BASE/'audit/step16_registration_before'
backup.mkdir(exist_ok=True)
manifest = {}
for source in [BASE/'10_all_experiments_metrics.csv',BASE/'outputs/模型实验结果总表.xlsx',BASE/'11_all_experiments_summary_for_teacher.md',BASE/'audit/workbook_plan.json']:
    target = backup/source.name
    if not target.exists():
        shutil.copy2(source,target)
    manifest[str(source)] = dict(backup=str(target),sha256=sha(target))
save(backup/'backup_manifest.json',manifest)
print('STEP16_REVIEW_FINALIZED; original results unchanged; registration backup saved; no inference/training')
