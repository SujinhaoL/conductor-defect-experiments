"""Record reviewed STEP13 evidence and render saved boxes; no model operations."""
from datetime import datetime, timezone
import csv
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'workspace/scripts'))
from step13_review_results import read, save, sha, xyxy, iou
from PIL import Image, ImageDraw, ImageFont, ImageOps


def main():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    audit = ROOT / 'workspace/audit/step13_review'
    a = read(audit / 'acceptance.json'); run = Path(a['run_dir'])
    assert a['status'] == 'PASS' and a['selected_epoch'] == 15
    base = ROOT.parent / 'baseline_analysis'
    backup = audit / 'records_before_update'; backup.mkdir(exist_ok=True)
    sources = [base / '10_all_experiments_metrics.csv', base / 'outputs/模型实验结果总表.xlsx',
        base / '11_all_experiments_summary_for_teacher.md', base / 'audit/registration_acceptance.json',
        base / 'audit/run_status_index.json', ROOT.parent / 'AGENTS.md']
    receipt = read(base / 'audit/registration_acceptance.json')
    assert receipt['csv_sha256'] == sha(sources[0]) and receipt['excel_sha256'] == sha(sources[1])
    for p in sources:
        if not (backup / p.name).exists():
            shutil.copy2(p, backup / p.name)
    data = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
    fixed = read(data / 'annotations/instances_debug8.json')
    imgs = {v['id']: v for v in fixed['images']}
    oldrun = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'
    preds = [read(p / 'artifacts/epoch_100/predictions_coco.json') for p in (oldrun, run)]
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 18)
    canvas = Image.new('RGB', (1160, 1480), 'white')
    d = ImageDraw.Draw(canvas)
    d.text((10, 5), 'Same 1280 input | epoch100 train-only | green: GT, red: nearest same-class final box', font=font, fill='black')
    details = []
    for row, aid in enumerate((87, 123, 124, 160)):
        ann = next(x for x in fixed['annotations'] if x['id'] == aid)
        im = imgs[ann['image_id']]; g = xyxy(ann['bbox'])
        gw, gh = ann['bbox'][2:]; cx, cy = (g[0] + g[2]) / 2, (g[1] + g[3]) / 2
        cw, ch = max(160, gw * 3), max(100, gh * 4)
        window = (max(0, int(cx-cw/2)), max(0, int(cy-ch/2)), min(im['width'], math.ceil(cx+cw/2)), min(im['height'], math.ceil(cy+ch/2)))
        for col, prediction in enumerate(preds):
            cand = [p for p in prediction if p['image_id'] == ann['image_id'] and p['category_id'] == ann['category_id']]
            p = max(cand, key=lambda v: iou(g, xyxy(v['bbox'])))
            overlap = iou(g, xyxy(p['bbox']))
            with Image.open(data / 'images' / im['file_name']) as src:
                crop = src.convert('RGB').crop(window)
            crop = ImageOps.contain(crop, (560, 280))
            sx, sy = crop.width / (window[2]-window[0]), crop.height / (window[3]-window[1])
            pen = ImageDraw.Draw(crop)
            for b, color in [(g, 'lime'), (xyxy(p['bbox']), 'red')]:
                pen.rectangle([(b[0]-window[0])*sx, (b[1]-window[1])*sy, (b[2]-window[0])*sx, (b[3]-window[1])*sy], outline=color, width=3)
            x, y = col * 580 + 10, row * 360 + 40
            canvas.paste(crop, (x, y+65))
            d.text((x, y), f'GT{aid} {Path(im["file_name"]).stem} anchor{8 if col == 0 else 4}', font=font, fill='black')
            d.text((x, y+25), f'IoU {overlap:.3f}; score {p["score"]:.4f}', font=font, fill='black')
            details.append(dict(annotation_id=aid, anchor_scale=8 if col == 0 else 4, max_same_class_iou=overlap, score=p['score']))
    canvas.save(audit / 'four_targets_anchor8_vs_anchor4.jpg', quality=95)
    # Fixed display-threshold error counts; separate from AP and not a tuning step.
    selection = read(run / 'selected_checkpoint.json')
    truth = read(data / 'annotations/instances_val.json')
    visible = sorted([p for p in read(selection['validation_predictions']) if p['score'] > .3], key=lambda p: -p['score'])
    matched = set(); comparisons = []
    for p in visible:
        targets = [g for g in truth['annotations'] if g['image_id'] == p['image_id'] and g['category_id'] == p['category_id'] and g['id'] not in matched]
        best = max(targets, key=lambda g: iou(xyxy(g['bbox']), xyxy(p['bbox'])), default=None)
        overlap = iou(xyxy(best['bbox']), xyxy(p['bbox'])) if best else 0
        if overlap >= .5:
            matched.add(best['id'])
        comparisons.append(dict(**p, match_iou=overlap, matched_annotation=best['id'] if overlap >= .5 else None))
    errors = dict(score_rule='strict score > 0.30; class-specific, descending-score, one-to-one IoU >= 0.50 matching; original val23',
        predictions=len(visible), true_positives=len(matched), false_positives=len(visible)-len(matched),
        unmatched_gt=len(truth['annotations'])-len(matched), matches=comparisons, changes_to_predictions_or_ap=False)
    save(audit / 'selected_val_display_error_counts.json', errors)
    save(audit / 'four_targets_comparison.json', details)
    a.update(visual_review_pending=False, visual_review_completed=True,
        visual_review_scope='Assistant inspected the actual epoch100 train contact sheet, selected-val GT/visible-prediction overlay and train/val anchor comparison curves.',
        four_prior_misses_recovered_in_final_predictions=True, four_target_details=details,
        selected_val_display_error_counts={k:v for k,v in errors.items() if k != 'matches'})
    assert all(r['max_same_class_iou'] > .94 for r in details if r['anchor_scale'] == 4)
    save(audit / 'acceptance.json', a)
    m, b = a['selected_metrics'], a['baseline_selected_metrics']
    pre = read(run / 'pretraining_record.json')
    lines = ['# FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001', '',
        '状态：100轮训练及验证结果已复核；test未运行。原始运行文件保持不变，REVIEW_PENDING为生成时状态，当前状态见本说明和独立acceptance.json。', '',
        f'运行：`{run.name}`；目录：`{run}`。', '',
        '## 问题、实验定位与边界', '',
        '检验STEP12提示的小目标anchor匹配限制。它是固定1280输入的anchor尺寸消融，与此前纯输入尺寸对照是两个不同问题；是辅助诊断证据，不是论文必做实验，也不是新网络模块。用户强调实验必须服务论文且不无限复杂化：本次完成后停止自动追加尺寸、anchor或其他训练。', '',
        '## 实际设置与复现', '',
        '原train107/168框（broken39、loose129）、val23/37框（9、28）；test23/36框本轮未用。YOLO0/1→COCO1/2→模型0/1，num_classes=2，ROI背景不是第三个缺陷。', '',
        '直接对照STEP11/step11_31112_14115。唯一科学配置变化为RPN anchor_generator.scales=[8]→[4]；三种ratio0.5/1/2和stride4/8/16/32/64不变，每位置3个anchor，输出张量形状不变。训练/验证/固定快照仍Resize(2133,1280)、keep_ratio=True。', '',
        'batch2、FP32、workers0、seed42；100epoch、每轮54步、总5400次更新。SGD0.0025/momentum0.9/weight_decay0.0001；warmup500/start_factor0.001，milestones67/92、gamma0.1；flip0.5。5400条实际学习率与对照一致；5400条损失有限。score_thr0.001、ROI NMS0.5、max100、COCO maxDets[1,10,100]、显示0.30。', '',
        f'同一官方完整COCO初始化：`{pre["checkpoint"]["path"]}`；来源：{pre["checkpoint"]["url"]}；SHA256：`{pre["checkpoint"]["sha256"]}`。首次更新前344项逐值一致；仅ROI fc_cls/fc_reg weight/bias因80→2重置。anchor几何变化保持加载兼容，但并不意味着它与COCO原先几何相同。未续训私人权重。', '',
        f'耗时{a["wall_seconds"]:.3f}秒（{a["wall_seconds"]/60:.2f}分钟，含val/保存/固定快照），峰值CUDA allocated {a["peak_cuda_allocated_mib"]:.3f}MiB（{a["peak_cuda_allocated_mib"]/1024:.2f}GiB）。', '',
        '## 模型选择与验证结果', '',
        f'按原生val coco/bbox_mAP三位小数最大值、同值保留首次，选第15轮。权重：`{selection["best_checkpoint"]}`；SHA256：`{selection["best_checkpoint_sha256"]}`。完整精度复算不更换已选权重。末轮SHA256：`{selection["last_checkpoint_sha256"]}`。', '',
        '|23张val，保存预测精确COCO复算|1280 anchor8，第5轮|1280 anchor4，第15轮|', '|---|---:|---:|']
    for key, name in [('mAP50', 'mAP50'), ('mAP50_95', 'mAP50-95'), ('AR100', 'COCO AR100')]:
        lines.append(f'|{name}|{b[key]*100:.6f}%|{m[key]*100:.6f}%|')
    for cls in ('broken', 'loose'):
        for key in ('AP50', 'AP50_95'):
            lines.append(f'|{cls} {key}|{b["per_class"][cls][key]*100:.6f}%|{m["per_class"][cls][key]*100:.6f}%|')
    lines += ['', '原生mAP50=0.055、mAP50-95=0.012；上表是同一份保存预测的完整精度。普通P/R未单独登记，AR100不能代替普通Recall；test未运行不填0。', '',
        '## 过程、预测及假设检验', '',
        f'末轮固定train8：mAP50={a["train8_final"]["mAP50"]*100:.6f}%、mAP50-95={a["train8_final"]["mAP50_95"]*100:.6f}%，对照对应mAP50-95={a["baseline_train8_final"]["mAP50_95"]*100:.6f}%。15/15GT有IoU≥0.50最终同类框，原4个loose漏检GT87/123/124/160的最佳同类IoU分别约0.949/0.990/0.977/0.960。', '',
        '这支持较小anchor改善了已观察训练目标的拟合。它是用107张训练后，在其中固定8张上的观察，不是全107张完全过拟合的证明，也不是重跑原8张专用过拟合实验。没有采集本轮新RPN/ROI中间数组，不能从最终框反推出每个中间阶段已全部证明正常。', '',
        f'主要val mAP50-95变化{(m["mAP50_95"]-b["mAP50_95"])*100:+.6f}个百分点；mAP50变化{(m["mAP50"]-b["mAP50"])*100:+.6f}个百分点；broken AP仍0。第100轮val mAP50-95={a["final_val_metrics"]["mAP50_95"]*100:.6f}%，早期较好、后期下降。不能只挑mAP50改善宣布实验成功或模型泛化改善。', '',
        f'选定val有2100个低阈值预测，19张达到100个上限；显示阈值>0.30留下{errors["predictions"]}个，按同类、分数降序、一对一IoU≥0.50规则为TP={errors["true_positives"]}、FP={errors["false_positives"]}、未匹配GT={errors["unmatched_gt"]}。这只描述固定显示阈值，不用于计算AP或调整阈值。', '',
        '训练拟合和val差距仍大，符合泛化受限/过拟合迹象；数据量、分布、目标外观和标注口径等仍是假设，当前不能归因于某一个因素。没有异常中断；比例/类别/原图坐标、100轮val文件、5次固定快照、哈希与选模均通过。', '',
        '## 对小论文的用途与停止条件', '',
        '可作为问题分析或消融：官方800→1280没有明显解决主要val指标；在1280下缩小anchor能恢复部分训练小目标的拟合，却未改善主要val指标。不能声称已找到所有低分原因、实现全链路无错，或获得具有显著性的结论。单seed、23张val且多轮开发的限制应如实写出；旧test已被查看。', '',
        '当前只归档并总结，不继续自动加大尺寸/改anchor/加入模块，不为填表推理或test。后续如需新实验，应先明确论文问题、单一因素、预算、评价标准和停止条件，再决定是否必要。', '',
        '## 证据完整路径', '']
    for name, p in [('配置', run / 'resolved_scale1280_anchor4_val_001.py'), ('依赖', run / 'requirements_freeze.txt'),
        ('初始化/数据/命令/代码归档哈希', run / 'pretraining_record.json'), ('实际权重加载', run / 'artifacts/actual_training_weight_load.json'),
        ('损失与学习率', run / 'artifacts/loss_curves.png'), ('训练图末轮预测', run / 'artifacts/epoch_100/predictions_contact_sheet.jpg'),
        ('选中权重及精确val', run / 'selected_checkpoint.json'), ('选中val预测', Path(selection['validation_predictions'])),
        ('完整日志', ROOT / 'workspace/logs' / run.name / 'step13_console.txt'), ('独立复算', audit / 'acceptance.json'),
        ('训练/验证趋势对照', audit / 'anchor_training_validation_comparison.png'), ('四个目标前后对照', audit / 'four_targets_anchor8_vs_anchor4.jpg')]:
        assert p.exists(), p
        lines.append(f'- {name}：`{p}`')
    note = ROOT / 'workspace/notes/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md'
    note.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (ROOT / 'workspace/notes/STEP13_result_review.txt').write_text(
        f'STEP13_RESULT_REVIEW_PASS\nRUN_DIR={run}\nBEST_VAL_EPOCH=15\nNOTE={note}\nNo model inference or training during review; test not run.\n', encoding='utf-8')
    print(json.dumps(dict(note=str(note), fixed_train8_metrics=a['train8_final'], visible_val_errors={k:v for k,v in errors.items() if k != 'matches'}), ensure_ascii=False))
    print('STEP13_NOTES_RECORDED; saved predictions only; no model training/inference')


if __name__ == '__main__':
    main()
