"""Complete saved STEP17 review and notes; preserve source evidence. No model imports."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent/'baseline_analysis'
RUN = ROOT/'workspace/runs/FRCNN-R50FPN-SCALE1600-VAL-001/step14_3575_9657'
DEST = RUN/'test_selected_epoch12/step17_23519_12608'
AUDIT = ROOT/'workspace/audit/step17_review'
read = lambda path:json.loads(path.read_text(encoding='utf-8'))
save = lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
review = read(AUDIT/'acceptance.json')
assert review['status']=='PASS' and review['display_threshold_TP']==2 and review['display_threshold_FP']==6
review.update(visual_review_pending=False,visual_review_completed=True,
    visual_review_note='Assistant inspected all23 existing prediction images with original GT in green.8 score>0.30 boxes on7 images; same-class one-to-one IoU>=0.50 diagnostic TP2/FP6. No new inference.')
save(AUDIT/'acceptance.json',review)
summary = read(DEST/'test_summary.json')
selected = read(RUN/'selected_checkpoint.json')
note_path = ROOT/'workspace/notes/FRCNN-R50FPN-SCALE1600-VAL-001.md'
note = note_path.read_text(encoding='utf-8')
note = note.replace('状态：100轮训练与val结果已复核，test未运行。', '状态：100轮训练、val及固定第12轮test结果均已复核，STEP17正常完成。')
note = note.replace('AR100不能代替普通Recall，test未运行不填0。', 'AR100不能代替普通Recall；补测test见末尾STEP17记录。')
note = note.replace('按用户要求结束这一轮增大尺寸试验，不自动准备更大尺寸、anchor、模块或test。', '该尺寸训练已结束，不自动准备更大尺寸、anchor或模块。用户于2026-10-14指定固定已有权重补齐test，第12轮已完成。')
if '## STEP17 固定第12轮最终test' not in note:
    table = ['|精确COCO指标|val：37框|test：36框|','|---|---:|---:|']
    val,test = selected['validation_metrics'],summary['metrics']
    for label,key in [('mAP50','mAP50'),('mAP50-95','mAP50_95'),('COCO AR100','AR100')]:
        table.append(f'|{label}|{val[key]*100:.6f}%|{test[key]*100:.6f}%|')
    for cls in ('broken','loose'):
        for label,key in [('AP50','AP50'),('AP50-95','AP50_95')]:
            table.append(f'|{cls} {label}|{val["per_class"][cls][key]*100:.6f}%|{test["per_class"][cls][key]*100:.6f}%|')
    note += '\n## STEP17 固定第12轮最终test\n\n'
    note += '用户2026-10-14启动step17_23519_12608，正常退出码0。原test23图36框（broken8、loose28），沿用选定val的Resize(2666,1600)、keep_ratio=True、默认anchor8、FP32、score_thr0.001、ROI NMS0.5、max100。348项完整加载及数据/配置/权重/预测哈希、类别/原图坐标、23张保存图和独立COCO复算均通过。\n\n'
    note += '\n'.join(table)+'\n\n'
    note += '原生test mAP50=0.055、mAP50-95=0.018；上表为同次保存预测的精确复算，普通P/R未计算留空，AR100单列。1981个低阈值预测，8个score>0.30显示框。按同类、分数降序、一对一IoU>=0.50的显示阈值诊断为TP2/FP6/未匹配GT34；只作显示阈值描述，不替代AP/AR或普通P/R，不据test改阈值。\n\n'
    note += f'含读图、推理与保存图片耗时{summary["inference_and_image_saving_seconds"]:.6f}秒，峰值CUDA allocated{summary["peak_cuda_allocated_mib"]:.3f}MiB；不包含完整启动/CPU加载/末尾指标计算，不当设备总显存或严格效率比较。\n\n'
    note += '本模型test数值高于两个1280设置，但val主要AP并未显示随尺寸持续提高。单seed、每集合23张、不同选中epoch及已有test使用的限制保留；不能根据此次test再挑尺寸/anchor，也不能证明扩大尺寸普遍解决问题。已达成固定权重最终评价的目的，按原清单进入YOLO11s1600第54轮，不追加训练。\n\n'
    for label,path in [('完整test结果',DEST),('精确指标',DEST/'test_metrics.json'),('原生指标',DEST/'native_coco_metrics.json'),
        ('全部低阈值预测',DEST/'predictions.bbox.json'),('坐标/类别/分数',DEST/'predictions_details.json'),
        ('23张可视化',DEST/'predicted_images'),('配置/环境/命令/哈希',DEST/'pretest_record.json'),
        ('完整终端日志',ROOT/'workspace/logs/step17_23519_12608/step17_console.txt'),('验收',AUDIT/'acceptance.json'),
        ('独立保存预测复算',AUDIT/'independent_coco_check.txt'),('GT与预测总览',AUDIT/'test_gt_and_visible_predictions.jpg')]:
        note += f'- {label}：`{path}`\n'
note_path.write_text(note,encoding='utf-8')
backup = BASE/'audit/step17_registration_before'
backup.mkdir(exist_ok=True)
for source in [BASE/'10_all_experiments_metrics.csv',BASE/'outputs/模型实验结果总表.xlsx',BASE/'11_all_experiments_summary_for_teacher.md',BASE/'audit/workbook_plan.json']:
    target = backup/source.name
    if not target.exists():shutil.copy2(source,target)
print('STEP17_REVIEW_FINALIZED; originals unchanged; registration backup saved; no inference/training')
