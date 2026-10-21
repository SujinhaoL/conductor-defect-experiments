"""Prepare a concise teacher workbook from canonical saved metrics. No inference/training."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parents[1]
CSV=BASE/'10_all_experiments_metrics.csv'
with CSV.open(encoding='utf-8-sig',newline='') as handle:records=list(csv.DictReader(handle))
status=json.loads((BASE/'audit/run_status_index.json').read_text(encoding='utf-8'))['runs']
experiments=[
    ('Y11-BASE-001-R1','YOLO11s（小型模型）','1280，等比例缩放＋补边'),
    ('RTD-BASE-001-R1','RT-DETR-L（大型模型）','1280×1280，直接缩放'),
    ('Y11-SENS-IMGSZ1600-001','YOLO11s（小型模型）','1600，等比例缩放＋补边'),
    ('RTD-SENS-IMGSZ1600-001','RT-DETR-L（大型模型）','1600×1600，直接缩放'),
    ('FRCNN-R50FPN-VAL-001','Faster R-CNN（两阶段检测）','(1333,800)，默认锚框'),
    ('FRCNN-R50FPN-MERGE130-001','Faster R-CNN（两阶段检测）','(1333,800)，合并训练'),
    ('FRCNN-R50FPN-SCALE1280-VAL-001','Faster R-CNN（两阶段检测）','(2133,1280)，默认锚框'),
    ('FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001','Faster R-CNN（两阶段检测）','(2133,1280)，anchor4（锚框尺度4）'),
    ('FRCNN-R50FPN-SCALE1600-VAL-001','Faster R-CNN（两阶段检测）','(2666,1600)，默认锚框'),
]


def number(row,key):
    value=row.get(key,'') if row else ''
    return float(value) if value!='' else None


def choose(experiment,split):
    rows=[r for r in records if r['experiment_id']==experiment and r['evaluation_split']==split]
    priority={'pycocotools-exact-from-saved-predictions':100,'formal_independent_evaluation':90,
              'automatic_best_checkpoint_revalidation':80,'selected_epoch_validation':60}
    if not rows:return None
    return max(rows,key=lambda row:priority.get(row['evaluation_protocol'],priority.get(row['evaluation_event'],0)))


plan=dict(output_path=str(BASE/'outputs/模型实验简表_讨论版.xlsx'),
    preview_dir=str(BASE/'audit/teacher_simple_previews'),source_csv=str(CSV),
    source_csv_sha256=hashlib.sha256(CSV.read_bytes()).hexdigest(),sheets=[])
for split,name,counts in [('val','val（验证集）','23张图片，37个框：broken（断股）9个、loose（松股）28个'),
                          ('test','test（测试集）','23张图片，36个框：broken（断股）8个、loose（松股）28个')]:
    rows,class_rows,sources=[],[],[]
    for index,(experiment,model,size) in enumerate(experiments,1):
        source=choose(experiment,split)
        run=next(r for r in status if r['experiment_id']==experiment and r.get('run_id'))
        state='已评价' if source else '不适用（已并入训练）' if split=='val' and experiment=='FRCNN-R50FPN-MERGE130-001' else '待测试'
        epoch=int(source['selected_epoch']) if source else run.get('selected_epoch')
        rows.append([index,model,size,int(run['train_images']),epoch]+[number(source,k) for k in
            ('map50','map50_95','precision','recall','coco_ar100')]+[state])
        class_rows.append([index,model,size]+[number(source,k) for k in
            ('broken_ap50','broken_ap50_95','loose_ap50','loose_ap50_95')]+[state])
        sources.append(dict(experiment_id=experiment,evaluation_id=source['evaluation_id'] if source else None,
            state=state,metric_source=source['metric_source'] if source else None))
    plan['sheets'].append(dict(name=name,split=split,counts=counts,rows=rows,class_rows=class_rows,sources=sources,
        note='空白表示未评价、不适用或该框架未计算；真实计算为0才显示0.00%。AR100是COCO平均召回率，不等同于Recall（召回率）。',
        source_note='Faster R-CNN采用保存预测的精确COCO复算；YOLO/RT-DETR采用原生权重验证，1600的val采用训练结束best.pt重验证。同一实验只展示一个评价事件。'))
for sheet in plan['sheets']:
    assert len(sheet['rows'])==len(sheet['class_rows'])==9
    if sheet['split']=='test':
        for index in (2,3):
            if sheet['sources'][index]['evaluation_id'] is None:assert sheet['rows'][index][5:10]==[None]*5
    else:assert sheet['rows'][5][5:10]==[None]*5
path=BASE/'audit/teacher_simple_plan.json'
path.write_text(json.dumps(plan,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print('TEACHER_SIMPLE_PLAN_READY;2 sheets;9 experiments per sheet;test pending cells empty;no inference/training')
