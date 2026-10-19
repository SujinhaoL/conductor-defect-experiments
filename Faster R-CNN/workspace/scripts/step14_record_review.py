"""Archive and document the completed 1600 run using saved evidence only."""
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'workspace/scripts'))
from step14_review_results import read, save, sha, xyxy, iou


def main():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    audit = ROOT / 'workspace/audit/step14_review'
    a = read(audit / 'acceptance.json'); run = Path(a['run_dir'])
    assert a['status'] == 'PASS' and a['selected_epoch'] == 12
    base = ROOT.parent / 'baseline_analysis'
    backup = audit / 'records_before_update'; backup.mkdir(exist_ok=True)
    sources = [base / '10_all_experiments_metrics.csv', base / 'outputs/模型实验结果总表.xlsx',
        base / '11_all_experiments_summary_for_teacher.md', base / 'audit/registration_acceptance.json',
        base / 'audit/run_status_index.json', base / 'audit/workbook_plan.json',
        base / 'scripts/refresh_experiment_records.py', ROOT.parent / 'AGENTS.md']
    receipt = read(base / 'audit/registration_acceptance.json')
    assert receipt['csv_sha256'] == sha(sources[0]) and receipt['excel_sha256'] == sha(sources[1])
    for p in sources:
        if not (backup / p.name).exists():
            shutil.copy2(p, backup / p.name)
    data = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
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
    a.update(visual_review_pending=False, visual_review_completed=True,
        visual_review_scope='Assistant inspected epoch100 train8 contact sheet, selected-val GT/prediction overlay and 1280/1600 train-validation curves.',
        selected_val_display_error_counts={k: v for k, v in errors.items() if k != 'matches'})
    save(audit / 'acceptance.json', a)
    m, b = a['selected_metrics'], a['baseline_selected_metrics']
    pre = read(run / 'pretraining_record.json')
    original = read(ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/selected_checkpoint.json')['validation_metrics']
    control = Path(pre['direct_control_run'])
    control_summary = read(control / 'artifacts/formal_training_summary.json')
    lines = ['# FRCNN-R50FPN-SCALE1600-VAL-001', '',
        '核查日期：2026-10-14（Asia/Shanghai）。状态：100轮训练与val结果已复核，test未运行。原运行内REVIEW_PENDING保留生成时状态，当前验收见本说明及独立acceptance.json。', '',
        f'运行编号：`{run.name}`；原始目录：`{run}`。', '',
        '## 科学问题与实际设置', '',
        '这是用户明确要求的最后一次整图尺寸试验。检验同一Faster R-CNN、数据划分和训练预算下，短边目标1280提高到1600是否改善val。直接对照STEP11默认anchor8，不与STEP13的anchor4混称尺寸单因素实验。', '',
        '原train107张/168框（broken39、loose129），val23张/37框（broken9、loose28）；test23张/36框本轮未用。沿用已有COCO及原划分。YOLO0/1→COCO1/2→模型0/1，num_classes=2；ROI背景不属于第三种缺陷。', '',
        '冻结配置逐项还原后与STEP11一致：唯一科学参数变化是训练、val、固定train8快照的Resize(2133,1280)→(2666,1600)，keep_ratio=True。长边限制2666、短边尽量1600，保持比例，后续按32倍数补边，不强制拉伸。', '',
        '默认anchor scale8、ratio0.5/1/2；batch2、FP32、workers0、seed42、deterministic=True；100epoch、每轮54批、总5400更新。SGD lr0.0025/momentum0.9/weight_decay0.0001；warmup500/start_factor0.001，milestones67/92、gamma0.1；RandomFlip0.5。实际5400条学习率与对照完全相同，所有loss有限。', '',
        'AP预测score_thr0.001、ROI NMS0.5、max_per_img100、COCO maxDets[1,10,100]；显示阈值0.30。未改batch/AMP、未自动resume或重试。', '',
        f'同一官方完整COCO检测checkpoint重新初始化：`{pre["checkpoint"]["path"]}`；来源：{pre["checkpoint"]["url"]}；SHA256：`{pre["checkpoint"]["sha256"]}`。首次更新前344项逐值加载；仅ROI fc_cls/fc_reg的weight/bias因80→2重新初始化。没有续训私人权重。', '',
        f'实际版本：{pre["versions"]}；MMDetection commit：`{pre["mmdet_commit"]}`。运行命令、依赖和冻结输入哈希见下方证据。', '',
        '## 选模与验证结果', '',
        f'按原生val coco/bbox_mAP（三位小数）最大值、同值保留首次，选第{a["selected_epoch"]}轮。最佳权重：`{selection["best_checkpoint"]}`；SHA256：`{selection["best_checkpoint_sha256"]}`。末轮SHA256：`{selection["last_checkpoint_sha256"]}`。离线完整精度复算不重新挑选epoch。', '',
        '|同一23张val，默认anchor8|800，第66轮|1280，第5轮|1600，第12轮|', '|---|---:|---:|---:|']
    for key, name in [('mAP50', 'mAP50'), ('mAP50_95', 'mAP50-95'), ('AR100', 'COCO AR100')]:
        lines.append(f'|{name}|{original[key]*100:.6f}%|{b[key]*100:.6f}%|{m[key]*100:.6f}%|')
    for cls in ('broken', 'loose'):
        for key in ('AP50', 'AP50_95'):
            lines.append(f'|{cls} {key}|{original["per_class"][cls][key]*100:.6f}%|{b["per_class"][cls][key]*100:.6f}%|{m["per_class"][cls][key]*100:.6f}%|')
    inv = a['selected_prediction_inventory']
    lines += ['', '上表为保存预测的精确pycocotools COCO复算。原生mAP50=0.052、mAP50-95=0.013另行保留。broken的原生类别AP50-95显示0.000，但精确结果为0.0004004894871509623（0.040049%），不能写成精确全零。普通P/R未单独计算留空，AR100不能代替普通Recall，test未运行不填0。', '',
        '## 是否回答问题、支持什么结论', '',
        f'完成了预先确定的尺寸对照。相较1280，主要val mAP50-95变化{(m["mAP50_95"]-b["mAP50_95"])*100:+.6f}个百分点，mAP50变化{(m["mAP50"]-b["mAP50"])*100:+.6f}个百分点。较宽松IoU=0.50下有收益，但跨IoU阈值平均的主要AP没有提高，绝对表现仍低。不能仅挑mAP50或AR100的提高宣布整体泛化改善。', '',
        f'末轮固定train8的mAP50={a["train8_final"]["mAP50"]*100:.6f}%、mAP50-95={a["train8_final"]["mAP50_95"]*100:.6f}%；对照1280的mAP50-95={a["baseline_train8_final"]["mAP50_95"]*100:.6f}%。15/15GT存在IoU≥0.50同类候选。这是107张训练后其中8张的拟合诊断，不是全107张完全过拟合或独立泛化成绩。', '',
        '此前1600的静态anchor几何检查仍有4个零正样本GT，而本次末轮最终框能够覆盖15个GT；这再次说明“零正anchor”不是该GT绝对无法被最终检测的证明。未新增本轮RPN/ROI内部推理，不能把最终框进步解释为已定位所有内部原因。', '',
        f'第100轮val mAP50-95={a["final_val_metrics"]["mAP50_95"]*100:.6f}%，较早期选中轮低。训练拟合提升而val仍低，符合泛化受限/过拟合迹象；数据规模、外观分布和标注口径等仍为待证实假设，不能归因于单一原因或宣称全链路无错。', '',
        '本轮保持了原问题与单因素设计，没有混入anchor改动。800、1280、1600的默认设置系列没有显示主要val AP随尺寸持续提高，也未解决低泛化问题；这可以成为小论文中的尺寸敏感性结果，包括没有改善的结果。它不证明任意大图策略都无效，也不足以要求以后所有模型统一用1600。', '',
        '按用户要求结束这一轮增大尺寸试验，不自动准备更大尺寸、anchor、模块或test。单seed、23张val及反复使用val开发的局限如实披露；旧test已被查看，不能称为完全未接触盲测。跨框架输入及AP实现不同，前四次仍是参考比较。', '',
        '## 资源、可视化及异常', '',
        f'本次耗时{a["wall_seconds"]:.3f}秒（{a["wall_seconds"]/60:.2f}分钟，含val/保存/快照），对照{control_summary["wall_seconds_including_validation_checkpoints_snapshots"]/60:.2f}分钟。峰值CUDA allocated {a["peak_cuda_allocated_mib"]:.3f}MiB（{a["peak_cuda_allocated_mib"]/1024:.2f}GiB），对照{control_summary["observed_peak_cuda_allocated_mib"]/1024:.2f}GiB。allocated不等于reserved或整机显存，单次耗时受运行条件影响，不做精确效率定律推断。', '',
        f'选中val有{inv["count"]}个低阈值预测，{inv["images_at_limit"]}张达到100框上限，所有23张都有低阈值预测。显示>0.30留下{errors["predictions"]}个；按同类、分数降序、一对一IoU≥0.50规则为TP={errors["true_positives"]}、FP={errors["false_positives"]}、未匹配GT={errors["unmatched_gt"]}。仅作显示阈值描述，不调阈值或改变AP。', '',
        '已目视核对末轮训练图、选中val GT/预测总览及训练/val趋势图。训练图较容易见到预测，val图多数在显示阈值下没有框，少量误检仍存在；显示无框不等于低阈值没有预测。', '',
        f'没有OOM或训练中断。100轮val文件、5次固定train8快照、5400行loss/LR、配置、数据/权重/归档哈希、选模及独立COCO复算均通过。首100次总loss均值{a["loss_first100"]:.6f}，末100次{a["loss_last100"]:.6f}。预期ROI四项类别输出层尺寸不匹配已逐项确认，其余344项确实加载。', '',
        '## 证据完整路径', '']
    for name, p in [('配置', run / 'resolved_scale1600_val_001.py'), ('依赖', run / 'requirements_freeze.txt'),
        ('命令/代码/数据与初始化来源及哈希', run / 'pretraining_record.json'), ('实际预训练加载', run / 'artifacts/actual_training_weight_load.json'),
        ('损失及学习率', run / 'artifacts/losses.csv'), ('损失曲线', run / 'artifacts/loss_curves.png'),
        ('固定train8末轮预测图', run / 'artifacts/epoch_100/predictions_contact_sheet.jpg'),
        ('选中checkpoint及val精确指标', run / 'selected_checkpoint.json'), ('选中val预测', Path(selection['validation_predictions'])),
        ('完整终端日志', ROOT / 'workspace/logs' / run.name / 'step14_console.txt'),
        ('独立验收', audit / 'acceptance.json'), ('选中val标注和显示框', audit / 'selected_val_gt_green_pred_red.jpg'),
        ('1280与1600训练/val趋势', audit / 'resolution_training_validation_comparison.png'),
        ('实际Resize和GT坐标核查', ROOT / 'workspace/audit/step14_resolution/resolution_audit.json')]:
        assert p.exists(), p
        lines.append(f'- {name}：`{p}`')
    note = ROOT / 'workspace/notes/FRCNN-R50FPN-SCALE1600-VAL-001.md'
    note.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (ROOT / 'workspace/notes/STEP14_result_review.txt').write_text(
        f'STEP14_RESULT_REVIEW_PASS\nRUN_DIR={run}\nBEST_VAL_EPOCH=12\nNOTE={note}\nTest not run; no assistant model inference/training; size experiment series stops here.\n', encoding='utf-8')
    print(f'STEP14_NOTES_RECORDED: display TP={errors["true_positives"]} FP={errors["false_positives"]}; no model inference/training')


if __name__ == '__main__':
    main()
