"""Audit existing STEP11 files and saved predictions. Never construct/run a model."""
from collections import Counter
from contextlib import redirect_stdout
import copy
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'
BASE = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
OUT = ROOT / 'workspace/audit/step11_review'
DATA = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'workspace/cache/matplotlib'))
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from mmengine.config import Config
import matplotlib.pyplot as plt


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def save(p, v):
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def history(run):
    return [json.loads(x) for x in (run/'artifacts/validation_history.jsonl').read_text().splitlines()]


def evaluate(ann, pred, label):
    with (OUT/(label+'_coco.txt')).open('w', encoding='utf-8') as f, redirect_stdout(f):
        gt = COCO(str(ann))
        dt = gt.loadRes(pred) if pred else COCO()
        if not pred:
            dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'], annotations=[])
            dt.createIndex()
        ev = COCOeval(gt, dt, 'bbox')
        ev.params.imgIds = sorted(gt.getImgIds()); ev.params.catIds = [1, 2]; ev.params.maxDets = [1, 10, 100]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    m = dict(mAP50_95=float(ev.stats[0]), mAP50=float(ev.stats[1]), AR100=float(ev.stats[8]), per_class={})
    for k, name in enumerate(('broken', 'loose')):
        p, r = ev.eval['precision'], ev.eval['recall']
        m['per_class'][name] = {key: float(x[x > -1].mean()) for key, x in
            [('AP50_95', p[:, :, k, 0, 2]), ('AP50', p[0, :, k, 0, 2]), ('AR100', r[:, k, 0, 2])]}
    return m


def compare(actual, expected):
    for k in ('mAP50', 'mAP50_95', 'AR100'):
        assert abs(actual[k]-expected[k]) < 1e-12, k
    for c in ('broken', 'loose'):
        for k in ('AP50', 'AP50_95', 'AR100'):
            assert abs(actual['per_class'][c][k]-expected['per_class'][c][k]) < 1e-12, (c, k)


def xyxy(b):
    return [b[0], b[1], b[0]+b[2], b[1]+b[3]]


def iou(a, b):
    inter = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0


def check_predictions(truth, preds):
    imgs = {im['id']: im for im in truth['images']}
    counts = Counter()
    for p in preds:
        assert p['image_id'] in imgs and p['category_id'] in (1, 2)
        assert 0.001 < p['score'] <= 1 and math.isfinite(p['score'])
        x, y, w, h = p['bbox']; im = imgs[p['image_id']]
        assert all(math.isfinite(t) for t in (x, y, w, h)) and x >= -0.001 and y >= -0.001 and w > 0 and h > 0
        assert x+w <= im['width']+0.001 and y+h <= im['height']+0.001
        counts[p['image_id']] += 1
    assert max(counts.values(), default=0) <= 100
    return dict(count=len(preds), images_at_limit=sum(n == 100 for n in counts.values()),
                images_empty=len(imgs)-len(counts), above_display_threshold=sum(p['score'] > .3 for p in preds),
                category_counts=dict(Counter(p['category_id'] for p in preds)))


def coverage(truth, preds, label):
    rows = []
    for a in truth['annotations']:
        cand = [p for p in preds if p['image_id'] == a['image_id'] and p['category_id'] == a['category_id']]
        best = max(cand, key=lambda p: iou(xyxy(a['bbox']), xyxy(p['bbox'])), default=None)
        overlap = iou(xyxy(a['bbox']), xyxy(best['bbox'])) if best else 0
        rows.append(dict(annotation_id=a['id'], image_id=a['image_id'], category_id=a['category_id'],
                         best_same_class_iou=overlap, score=best['score'] if best else None,
                         candidate_iou_ge_0_5=overlap >= .5))
    with (OUT/(label+'_per_gt.csv')).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def main():
    assert Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    pre, sel, summary = (read(RUN/p) for p in ('pretraining_record.json', 'selected_checkpoint.json', 'artifacts/formal_training_summary.json'))
    logs = ROOT/'workspace/logs'/RUN.name
    assert read(logs/'step11_manifest.json')['status'] == 'TRAINING_COMPLETED_VAL_REVIEW_PENDING'
    for name, digest in pre['saved_inputs_sha256'].items():
        assert sha(RUN/'code'/Path(name).name) == digest, name
    for split, digest in pre['annotation_hashes'].items():
        assert sha(DATA/f'annotations/instances_{split}.json') == digest
    assert sha(pre['checkpoint']['path']) == pre['checkpoint']['sha256']
    assert pre['checkpoint']['sha256'] == read(BASE/'pretraining_record.json')['checkpoint']['sha256']
    assert sha(sel['best_checkpoint']) == sel['best_checkpoint_sha256']
    assert sha(sel['last_checkpoint']) == sel['last_checkpoint_sha256']
    cfg = Config.fromfile(str(RUN/'resolved_scale1280_val_001.py'), import_custom_modules=False).to_dict()
    ref = Config.fromfile(str(BASE/'resolved_val_001.py'), import_custom_modules=False).to_dict()
    for k in ('experiment_id', 'snapshot_scale', 'work_dir'):
        cfg.pop(k, None); ref.pop(k, None)
    for p in [cfg['train_pipeline'], cfg['test_pipeline'], cfg['train_dataloader']['dataset']['pipeline'], cfg['val_dataloader']['dataset']['pipeline']]:
        resize = [t for t in p if t['type'] == 'Resize']; assert len(resize) == 1
        assert tuple(resize[0]['scale']) == (2133,1280) and resize[0]['keep_ratio']
        resize[0]['scale'] = (1333,800)
    assert cfg == ref and cfg['test_dataloader'] is None and not cfg['resume']
    load = read(RUN/'artifacts/actual_training_weight_load.json')
    assert load['status'] == 'PASS' and load['loaded_tensors'] == 344 and len(load['expected_shape_changes']) == 4 and load['loaded_before_first_optimizer_update']
    with (RUN/'artifacts/losses.csv').open() as f:
        losses = list(csv.DictReader(f))
    assert len(losses) == summary['optimizer_updates'] == 5400 and summary['epochs'] == 100
    for i, r in enumerate(losses):
        assert int(r['iteration']) == i+1 and int(r['epoch']) == i//54+1 and int(r['batch_index']) == i%54
        assert all(math.isfinite(float(x)) for x in r.values())
    with (BASE/'artifacts/losses.csv').open() as f:
        baseline_losses = list(csv.DictReader(f))
    assert [r['learning_rate'] for r in losses] == [r['learning_rate'] for r in baseline_losses]
    h = history(RUN); bh = history(BASE)
    assert [r['epoch'] for r in h] == list(range(1,101))
    best = max(h, key=lambda r:r['metrics']['coco/bbox_mAP'])
    assert best['epoch'] == sel['selected_epoch'] == 5 and best == summary['best_validation']
    ann = DATA/'annotations/instances_val.json'; truth = read(ann)
    assert len(truth['images']) == 23 and Counter(a['category_id'] for a in truth['annotations']) == {1:9, 2:28}
    inventory = []
    for r in h:
        d = RUN/f'artifacts/validation/epoch_{r["epoch"]:03d}'
        assert read(d/'metrics.json') == r
        inventory.append(dict(epoch=r['epoch'], **check_predictions(truth, read(d/'predictions.bbox.json'))))
    selected = read(sel['validation_predictions'])
    exact = evaluate(ann, selected, 'selected_val'); compare(exact, sel['validation_metrics'])
    for nk, ek in [('coco/bbox_mAP','mAP50_95'), ('coco/bbox_mAP_50','mAP50')]:
        assert abs(best['metrics'][nk]-exact[ek]) <= .00051
    final = evaluate(ann, read(RUN/'artifacts/validation/epoch_100/predictions.bbox.json'), 'final_val')
    fixedann = DATA/'annotations/instances_debug8.json'; fixed = read(fixedann)
    progress = read(RUN/'artifacts/train_fit_progress.json'); baseline_progress = read(BASE/'artifacts/train_fit_progress.json')
    assert [r['epoch'] for r in progress] == [0,25,50,75,100]
    for r in progress:
        assert r['resize_scale'] == [2133,1280]
        d = RUN/f'artifacts/epoch_{r["epoch"]:03d}'
        pred = read(d/'predictions_coco.json'); check_predictions(fixed, pred)
        details = read(d/'predictions_details.json'); exported = []
        assert [v['image_id'] for v in details] == r['image_ids']
        for v in details:
            assert Path(v['image_path']).parent.name == 'train'
            assert (d/Path(v['image_path']).name).is_file()
            for b, label, score in zip(v['boxes_xyxy'],v['model_labels'],v['scores']):
                exported.append(dict(image_id=v['image_id'], category_id=label+1,bbox=[b[0],b[1],b[2]-b[0],b[3]-b[1]],score=score))
        assert exported == pred
        compare(evaluate(fixedann, pred, f'train8_{r["epoch"]:03d}'), r)
    traincov = coverage(fixed, read(RUN/'artifacts/epoch_100/predictions_coco.json'), 'train8_epoch100')
    valcov = coverage(truth, selected, 'selected_val')
    # Re-render already saved boxes, not new detector output.
    canvas = Image.new('RGB', (1600,2080), 'white'); font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',16)
    for i, im in enumerate(truth['images']):
        with Image.open(DATA/'images'/im['file_name']) as src:
            thumb = ImageOps.contain(src.convert('RGB'),(520,215))
        draw = ImageDraw.Draw(thumb); sx,sy=thumb.width/im['width'],thumb.height/im['height']
        for a in truth['annotations']:
            if a['image_id'] == im['id']:
                b=xyxy(a['bbox']);draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],outline='lime',width=2)
        visible=[p for p in selected if p['image_id']==im['id'] and p['score']>.3]
        for p in visible:
            b=xyxy(p['bbox']);draw.rectangle([b[0]*sx,b[1]*sy,b[2]*sx,b[3]*sy],outline='red',width=2)
        x,y=(i%3)*533,(i//3)*260
        canvas.paste(thumb,(x+(533-thumb.width)//2,y+35))
        ImageDraw.Draw(canvas).text((x+4,y+8),f'{Path(im["file_name"]).stem} visible>{0.3}: {len(visible)}',font=font,fill='black')
    canvas.save(OUT/'selected_val_gt_green_pred_red.jpg', quality=95)
    fig, axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for hist,prog,label in [(bh,baseline_progress,'800 baseline'),(h,progress,'1280 input')]:
        for ax,key,title in zip(axes[:2],['coco/bbox_mAP','coco/bbox_mAP_50'],['Validation mAP50-95','Validation mAP50']):
            ax.plot([x['epoch'] for x in hist],[100*x['metrics'][key] for x in hist],label=label);ax.set_title(title+' (native rounded)')
        axes[2].plot([x['epoch'] for x in prog],[100*x['mAP50_95'] for x in prog],marker='o',label=label)
    axes[2].set_title('Fixed train8 mAP50-95 (diagnostic)')
    for ax in axes: ax.set_xlabel('Epoch');ax.set_ylabel('%');ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'resolution_training_validation_comparison.png',dpi=160);plt.close(fig)
    baseline=read(BASE/'selected_checkpoint.json')['validation_metrics']
    finding=dict(status='PASS',reviewed_utc=datetime.now(timezone.utc).isoformat(),run_dir=str(RUN),
        selected_epoch=5,selected_checkpoint_sha256=sel['best_checkpoint_sha256'],
        epochs=100,optimizer_updates=5400,only_resize_scientific_diff=True,resize_scale=[2133,1280],
        archived_input_hashes_verified=len(pre['saved_inputs_sha256']),validation_epochs_checked=100,
        finite_loss_rows=5400,lr_matches_baseline=True,loss_first100=float(np.mean([float(r['loss']) for r in losses[:100]])),
        loss_last100=float(np.mean([float(r['loss']) for r in losses[-100:]])),
        selected_metrics=exact,baseline_selected_metrics={k:baseline[k] for k in exact},final_val_metrics=final,
        selected_prediction_inventory=inventory[4],final_prediction_inventory=inventory[-1],
        train8_final=progress[-1],baseline_train8_final=baseline_progress[-1],
        train8_gt_with_same_class_candidate_iou_ge_0_5=sum(r['candidate_iou_ge_0_5'] for r in traincov),
        selected_val_gt_with_same_class_candidate_iou_ge_0_5=sum(r['candidate_iou_ge_0_5'] for r in valcov),
        coverage_note='Candidate coverage, not ordinary Recall or RPN proposal recall.',
        wall_seconds=summary['wall_seconds_including_validation_checkpoints_snapshots'],peak_cuda_allocated_mib=summary['observed_peak_cuda_allocated_mib'],
        source_files_sha256={str(p):sha(p) for p in [RUN/'selected_checkpoint.json',RUN/'resolved_scale1280_val_001.py',RUN/'artifacts/losses.csv',Path(sel['validation_predictions'])]},
        test_evaluated=False,training_started=False,detector_forward_performed=False,visual_review_pending=True)
    save(OUT/'acceptance.json',finding);save(OUT/'validation_prediction_inventory.json',inventory)
    print('STEP11_RESULT_REVIEW_PASS: 100 epochs / 5400 finite updates; configuration, hashes, selection, saved COCO predictions verified')
    print(json.dumps({k:finding[k] for k in ['selected_metrics','final_val_metrics','selected_prediction_inventory','train8_gt_with_same_class_candidate_iou_ge_0_5','loss_first100','loss_last100']},ensure_ascii=False))


if __name__ == '__main__':
    main()
