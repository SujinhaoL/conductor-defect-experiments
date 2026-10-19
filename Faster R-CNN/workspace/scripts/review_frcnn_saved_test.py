"""Review saved, user-run Faster R-CNN test evidence on CPU. No model import or inference."""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('D:/小论文/conductor_defect_dataset/03_coco_dataset')


def require(value, message):
    if not value:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = max(0., a[2]-a[0])*max(0., a[3]-a[1]) + max(0., b[2]-b[0])*max(0., b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True, type=Path)
    args = parser.parse_args()
    require(Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve(), 'Use Faster environment')
    dest = args.result_dir.resolve()
    require(dest.is_relative_to(ROOT/'workspace/runs'), 'Result outside Faster runs')
    stage = dest.name.split('_')[0]
    require(stage in ('step16', 'step17'), 'This reviewer is for the next two fixed Faster tests')
    audit = ROOT/'workspace/audit'/f'{stage}_review'
    audit.mkdir(parents=True, exist_ok=True)
    summary = read(dest/'test_summary.json')
    pre = read(dest/'pretest_record.json')
    epoch = summary['selected_epoch']
    expected_epoch, expected_anchor, expected_scale = (15, [4], [2133,1280]) if stage == 'step16' else (12, [8], [2666,1600])
    require(epoch == expected_epoch and summary['anchor_scales'] == expected_anchor and summary['resize_scale'] == expected_scale,
            'Wrong experiment/epoch/anchor/resize')
    log = ROOT/'workspace/logs'/dest.name
    original = read(log/f'{stage}_manifest.json')
    require(original['status'] == summary['status'] == pre['status'] == 'PASS' and original['test_inference_started'], 'Original test not complete')
    require(not summary['training_started'] and not summary['test_used_for_epoch_selection'], 'Unexpected training or test selection')
    require(summary['images'] == 23 and summary['boxes'] == 36 and summary['broken'] == 8 and summary['loose'] == 28, 'Test counts differ')
    files = read(dest/'test_evidence_manifest.json')['files_sha256']
    for source, digest in files.items():
        require(sha(source) == digest, 'Saved result changed: '+source)
    for source, digest in pre['saved_inputs_sha256'].items():
        require(sha(dest/'code'/Path(source).name) == digest, 'Frozen launch input changed: '+source)
    for source, digest in pre['test_image_sha256'].items():
        require(sha(source) == digest, 'Original test image changed: '+source)
    checkpoint = Path(summary['selected_checkpoint'])
    selection = read(dest/'code/selected_checkpoint.json')
    require(sha(checkpoint) == summary['selected_checkpoint_sha256'] == pre['selected_checkpoint_sha256'] == selection['best_checkpoint_sha256'], 'Selected weight changed')
    require(selection['selected_epoch'] == epoch, 'Selection snapshot differs')
    load = read(dest/'selected_checkpoint_load_audit.json')
    require(load['status'] == 'PASS' and load['loaded_tensors'] == 348 and not load['missing_keys'] and
            not load['unexpected_keys'] and not load['shape_mismatches'] and load['sha256'] == sha(checkpoint), 'Checkpoint full-load check failed')
    annotation = DATA/'annotations/instances_test.json'
    require(sha(annotation) == pre['test_annotation_sha256'], 'Annotation changed')
    truth = read(annotation)
    images = {info['id']:info for info in truth['images']}
    require(len(images) == 23 and len(truth['annotations']) == 36, 'Original test inventory differs')
    require([c['name'] for c in truth['categories']] == ['broken','loose'], 'Original category order differs')
    details = read(dest/'predictions_details.json')
    jsonl = [json.loads(line) for line in (dest/'predictions_details.jsonl').read_text(encoding='utf-8').splitlines()]
    require(details == jsonl and [row['image_id'] for row in details] == list(images), 'Per-image records/order incomplete')
    expected = []
    empty, capped, visible_images, degenerate = 0, 0, 0, 0
    visible_by_class = {'broken':0, 'loose':0}
    all_scores = []
    for row in details:
        info = images[row['image_id']]
        boxes, labels, scores = row['boxes_xyxy'], row['model_labels'], row['scores']
        require(len(boxes) == len(labels) == len(scores) == row['returned_count'] <= 100, 'Array sizes/cap differ')
        require(row['original_shape'][:2] == [info['height'], info['width']], 'Original image dimensions differ')
        require(Path(row['image_path']).name == Path(info['file_name']).name, 'Image path differs')
        new_h, new_w = row['resized_shape'][:2]
        require(min(new_h,new_w) <= expected_scale[1] and max(new_h,new_w) <= expected_scale[0], 'Resize bounds differ')
        require(np.allclose(row['scale_factor'], [new_w/info['width'], new_h/info['height']], atol=1e-6, rtol=0), 'Resize coordinate factors differ')
        visible = sum(score > .30 for score in scores)
        require(visible == row['visible_count'], 'Displayed count differs')
        empty += len(scores) == 0
        capped += len(scores) == 100
        visible_images += visible > 0
        for box,label,score,cls in zip(boxes,labels,scores,row['classes']):
            require(label in (0,1) and cls == ('broken','loose')[label], 'Class mapping differs')
            require(all(math.isfinite(v) for v in box) and math.isfinite(score) and .001 <= score <= 1, 'Prediction not finite/score invalid')
            x1,y1,x2,y2 = box
            require(0 <= x1 <= x2 <= info['width']+1e-3 and 0 <= y1 <= y2 <= info['height']+1e-3, 'Not original image coordinates')
            degenerate += x1 == x2 or y1 == y2
            expected.append(dict(image_id=row['image_id'],category_id=label+1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
            all_scores.append(score)
            if score > .30:
                visible_by_class[cls] += 1
        with Image.open(dest/'predicted_images'/(Path(row['image_path']).stem+'.jpg')) as image:
            image.load()
            require(image.size == (info['width'],info['height']), 'Saved image dimensions differ')
    predictions = read(dest/'predictions.bbox.json')
    require(len(predictions) == len(expected) == summary['returned_predictions'], 'COCO prediction count differs')
    require(sum(visible_by_class.values()) == summary['visible_predictions_above_0_3'], 'Visible count differs')
    for category in (1,2):
        a = [row for row in predictions if row['category_id'] == category]
        b = [row for row in expected if row['category_id'] == category]
        require(len(a) == len(b), 'Category export count differs')
        for first,second in zip(a,b):
            require(first['image_id'] == second['image_id'] and abs(first['score']-second['score']) < 1e-8 and
                    np.allclose(first['bbox'],second['bbox'],atol=1e-3,rtol=0), 'COCO vs detailed boxes/labels/scores differ')
    with (audit/'independent_coco_check.txt').open('w',encoding='utf-8') as handle:
        with contextlib.redirect_stdout(handle):
            gt = COCO(str(annotation))
            evaluator = COCOeval(gt,gt.loadRes(predictions),'bbox')
            evaluator.params.imgIds = list(images)
            evaluator.params.catIds = [1,2]
            evaluator.params.maxDets = [1,10,100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    exact = read(dest/'test_metrics.json')
    require(exact == summary['metrics'], 'Summary exact metrics differ')
    for key,index in [('mAP50_95',0),('mAP50',1),('AR100',8)]:
        require(abs(exact[key]-float(evaluator.stats[index])) < 1e-12, 'Saved metric differs: '+key)
    precision,recall = evaluator.eval['precision'],evaluator.eval['recall']
    mean_valid = lambda values: float(values[values>=0].mean())
    for index,name in enumerate(('broken','loose')):
        calculated = dict(AP50_95=mean_valid(precision[:,:,index,0,-1]), AP50=mean_valid(precision[0,:,index,0,-1]), AR100=mean_valid(recall[:,index,0,-1]))
        require(all(abs(value-exact['per_class'][name][key]) < 1e-12 for key,value in calculated.items()), 'Class metric differs')
    native = read(dest/'native_coco_metrics.json')
    require(native == summary['native_coco_metrics'], 'Native summary differs')
    for key,index in [('coco/bbox_mAP',0),('coco/bbox_mAP_50',1),('coco/bbox_mAP_75',2),('coco/bbox_mAP_s',3),('coco/bbox_mAP_m',4),('coco/bbox_mAP_l',5)]:
        require(native[key] == round(float(evaluator.stats[index]),3), 'Native rounded metric differs')
    for name in ('broken','loose'):
        require(native['coco/'+name+'_precision'] == round(exact['per_class'][name]['AP50_95'],3), 'Native category AP differs')
    require(len(list((dest/'predicted_images').glob('*.jpg'))) == 23, 'Saved image count differs')
    console = (log/f'{stage}_console.txt').read_text(encoding='utf-8')
    require(console.count('TEST_IMAGE=') == 23 and f'{stage.upper()}_TEST_EVAL_PASS' in console, 'Original full console incomplete')
    # This threshold-specific diagnostic does not enter formal P/R columns or change AP.
    tp,fp = 0,0
    matched_ids = set()
    per_visible = []
    for row in details:
        targets = [a for a in truth['annotations'] if a['image_id'] == row['image_id']]
        order = sorted((i for i,s in enumerate(row['scores']) if s>.30),key=lambda i:row['scores'][i],reverse=True)
        for index in order:
            box,label,score = row['boxes_xyxy'][index],row['model_labels'][index],row['scores'][index]
            candidates = [a for a in targets if a['category_id']==label+1 and a['id'] not in matched_ids]
            overlaps = [(iou(box,[a['bbox'][0],a['bbox'][1],a['bbox'][0]+a['bbox'][2],a['bbox'][1]+a['bbox'][3]]),a['id']) for a in candidates]
            best,ann_id = max(overlaps,default=(0.,None))
            hit = best >= .50
            if hit:
                matched_ids.add(ann_id); tp += 1
            else:
                fp += 1
            per_visible.append(dict(image_id=row['image_id'],label=label,score=score,box_xyxy=box,best_available_same_class_iou=best,matched_gt=ann_id if hit else None))
    save(audit/'display_threshold_diagnostic.json',dict(score_threshold=.30,iou_threshold=.50,
        rule='same-class score descending one-to-one; diagnostic only, not COCO AP/AR or reported ordinary P/R',
        TP=tp,FP=fp,unmatched_GT=36-tp,visible_by_class=visible_by_class,predictions=per_visible))
    contact = Image.new('RGB',(1600,6*300),'white')
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    for index,row in enumerate(details):
        x,y = (index%4)*400,(index//4)*300
        with Image.open(dest/'predicted_images'/(Path(row['image_path']).stem+'.jpg')) as saved:
            image = saved.convert('RGB')
            draw = ImageDraw.Draw(image)
            for ann in truth['annotations']:
                if ann['image_id']==row['image_id']:
                    bx,by,bw,bh = ann['bbox']
                    draw.rectangle([bx,by,bx+bw,by+bh],outline=(40,255,40),width=max(3,image.width//500))
            thumb = ImageOps.contain(image,(390,270))
            contact.paste(thumb,(x+(400-thumb.width)//2,y+28))
        ImageDraw.Draw(contact).text((x+5,y+5),f'{Path(row["image_path"]).stem} / visible: {row["visible_count"]} / GT: green',font=font,fill='black')
    contact.save(audit/'test_gt_and_visible_predictions.jpg',quality=94)
    acceptance = dict(status='PASS',reviewed_utc=datetime.now(timezone.utc).isoformat(),result_dir=str(dest),
        selected_epoch=epoch,selected_checkpoint_sha256=sha(checkpoint),metrics=exact,images=23,boxes=36,
        returned_predictions=len(predictions),visible_predictions_above_0_3=sum(visible_by_class.values()),
        images_at_limit=capped,images_empty=empty,images_with_visible_predictions=visible_images,degenerate_model_boxes_retained=degenerate,
        score_min=min(all_scores) if all_scores else None,score_max=max(all_scores) if all_scores else None,
        display_threshold_TP=tp,display_threshold_FP=fp,display_threshold_unmatched_GT=36-tp,
        original_files_unchanged=True, frozen_launch_inputs_hashes_verified=True,full_checkpoint_load_verified=True,
        labels_and_original_coordinates_verified=True,saved_predictions_independent_coco_match=True,native_rounding_matches=True,
        visual_review_pending=True,visual_review_completed=False,model_created=False,detector_forward_performed=False,training_started=False,
        source_files_sha256={str(dest/name):sha(dest/name) for name in ('test_summary.json','test_metrics.json','native_coco_metrics.json','predictions.bbox.json','test_evidence_manifest.json')})
    save(audit/'acceptance.json',acceptance)
    print(stage.upper()+'_SAVED_TEST_REVIEW_PASS; no model inference/training; visual review pending')
    print(f'TEST_AP50={exact["mAP50"]:.12f} TEST_AP50_95={exact["mAP50_95"]:.12f} COCO_AR100={exact["AR100"]:.12f}')
    print(f'DISPLAY_DIAGNOSTIC TP={tp} FP={fp} unmatched_GT={36-tp}; not formal P/R')


if __name__=='__main__':
    main()
