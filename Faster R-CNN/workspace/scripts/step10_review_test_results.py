"""Review saved STEP10 predictions and evidence; never load or run a detector."""
from collections import Counter
from contextlib import redirect_stdout
import csv
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
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-MERGE130-001/step09_109_20338'
DEST = RUN / 'test_selected_epoch100/step10_13778_21846'
AUDIT = ROOT / 'workspace/audit/step10_review'
ANN = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset\annotations\instances_test.json')
CLASSES = ['broken', 'loose']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0


def main():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    AUDIT.mkdir(parents=True, exist_ok=True)
    truth = read(ANN)
    summary = read(DEST / 'test_summary.json')
    details = read(DEST / 'predictions_details.json')
    preds = read(DEST / 'predictions.bbox.json')
    pre = read(DEST / 'pretest_record.json')
    manifest = read(DEST / 'test_evidence_manifest.json')
    for name, digest in manifest['files_sha256'].items():
        assert sha(name) == digest, name
    for name, digest in pre['saved_inputs_sha256'].items():
        archived = DEST / 'code' / Path(name).name
        assert sha(archived) == digest, archived
    assert sha(ANN) == pre['test_annotation_sha256'] == '42081faac0bb2fd36adec5e41669239e9e8fb39226b40bdf85a1822fe2ac9baa'
    selection = read(RUN / 'selected_checkpoint.json')
    assert sha(selection['checkpoint']) == selection['checkpoint_sha256'] == summary['selected_checkpoint_sha256'] == pre['selected_checkpoint_sha256']
    assert sha(RUN / 'selected_checkpoint.json') == pre['selection_sha256']
    assert summary['status'] == 'PASS' and summary['selected_epoch'] == selection['selected_epoch'] == 100
    assert summary['training_images'] == 130 and summary['independent_validation'] is False
    assert selection['fresh_official_coco_initialization'] and selection['independent_validation'] is False
    assert not summary['test_used_for_epoch_selection'] and not summary['training_started']
    load = read(DEST / 'selected_checkpoint_load_audit.json')
    assert load['status'] == 'PASS' and load['loaded_tensors'] == 348
    assert not load['missing_keys'] and not load['unexpected_keys'] and not load['shape_mismatches']
    assert load['class_names'] == CLASSES and load['sha256'] == selection['checkpoint_sha256']
    assert summary['model_score_threshold'] == pre['score_threshold'] == 0.001
    assert summary['visualization_score_threshold'] == pre['visualization_score_threshold'] == 0.30
    assert summary['max_per_image'] == pre['max_per_image'] == 100
    assert summary['rcnn_nms_iou'] == pre['rcnn_nms_iou'] == 0.5
    ids = [img['id'] for img in truth['images']]
    assert ids == [row['image_id'] for row in details] == summary['metrics']['image_ids'] == list(range(131, 154))
    assert len(ids) == len(set(ids)) == 23 and len(truth['annotations']) == 36
    assert Counter(a['category_id'] for a in truth['annotations']) == {1: 8, 2: 28}
    assert [(c['id'], c['name']) for c in truth['categories']] == [(1, 'broken'), (2, 'loose')]
    assert len(preds) == 129 == summary['returned_predictions'] == summary['metrics']['prediction_count']
    exported, visible = [], []
    for row, img in zip(details, truth['images']):
        assert Path(row['image_path']).name == Path(img['file_name']).name
        assert '/images/test/' in row['image_path'].replace('\\', '/')
        assert len(row['scores']) == len(row['model_labels']) == len(row['boxes_xyxy']) == row['returned_count'] <= 100
        assert row['visible_count'] == sum(s > 0.30 for s in row['scores'])
        with Image.open(row['image_path']) as image:
            image.load()
            assert image.size == (img['width'], img['height'])
        assert (DEST / 'predicted_images' / Path(row['image_path']).name).is_file()
        gt_for_image = [a for a in truth['annotations'] if a['image_id'] == img['id']]
        for box, label, score in zip(row['boxes_xyxy'], row['model_labels'], row['scores']):
            assert label in (0, 1) and 0.001 < score <= 1 and math.isfinite(score)
            assert all(math.isfinite(x) for x in box) and 0 <= box[0] < box[2] <= img['width'] and 0 <= box[1] < box[3] <= img['height']
            exported.append(dict(image_id=img['id'], category_id=label + 1, bbox=[box[0], box[1], box[2]-box[0], box[3]-box[1]], score=score))
            if score > 0.30:
                visible.append(dict(image_id=img['id'], file_name=img['file_name'], class_name=CLASSES[label], score=score,
                    best_same_class_iou=max((iou(box, xyxy(a['bbox'])) for a in gt_for_image if a['category_id'] == label + 1), default=0),
                    best_any_class_iou=max((iou(box, xyxy(a['bbox'])) for a in gt_for_image), default=0)))
    assert exported == preds, 'COCO export and original-coordinate detailed predictions disagree'
    assert len(visible) == 6 == summary['visible_predictions_above_0_3']
    per_box = []
    for a in truth['annotations']:
        candidates = [p for p in preds if p['image_id'] == a['image_id'] and p['category_id'] == a['category_id']]
        best = max(candidates, key=lambda p: iou(xyxy(a['bbox']), xyxy(p['bbox'])), default=None)
        overlap = iou(xyxy(a['bbox']), xyxy(best['bbox'])) if best else 0
        per_box.append(dict(annotation_id=a['id'], image_id=a['image_id'], class_name=CLASSES[a['category_id']-1],
            best_same_class_iou=overlap, best_candidate_score=best['score'] if best else None,
            has_candidate_iou_at_least_0_5=overlap >= 0.5))
    with (AUDIT / 'per_box_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(per_box[0])); writer.writeheader(); writer.writerows(per_box)
    # COCOeval consumes saved coordinates only; no model, torch or detector import.
    with (AUDIT / 'independent_coco_check.txt').open('w', encoding='utf-8') as f, redirect_stdout(f):
        gt = COCO(str(ANN)); dt = gt.loadRes(preds)
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = ids; evaluator.params.catIds = [1, 2]; evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    for index, key in [(0, 'mAP50_95'), (1, 'mAP50'), (8, 'AR100')]:
        assert abs(evaluator.stats[index] - summary['metrics'][key]) < 1e-12, key
    precision, recall = evaluator.eval['precision'], evaluator.eval['recall']
    for k, name in enumerate(CLASSES):
        for key, values in [('AP50_95', precision[:, :, k, 0, 2]), ('AP50', precision[0, :, k, 0, 2]), ('AR100', recall[:, k, 0, 2])]:
            actual = float(np.mean(values[values > -1]))
            assert abs(actual - summary['metrics']['per_class'][name][key]) < 1e-12, (name, key)
    # Reuse already saved display predictions and add GT; no new inference.
    sheet = Image.new('RGB', (1600, 8 * 260), 'white')
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 16)
    for index, (row, img) in enumerate(zip(details, truth['images'])):
        with Image.open(DEST / 'predicted_images' / Path(img['file_name']).name) as image:
            thumbnail = ImageOps.contain(image.convert('RGB'), (520, 220))
        sx, sy = thumbnail.width / img['width'], thumbnail.height / img['height']
        draw = ImageDraw.Draw(thumbnail)
        for a in truth['annotations']:
            if a['image_id'] == img['id']:
                b = xyxy(a['bbox'])
                draw.rectangle([b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy], outline=(0, 240, 0), width=2)
        x, y = (index % 3)*533, (index // 3)*260
        sheet.paste(thumbnail, (x+(533-thumbnail.width)//2, y+35))
        ImageDraw.Draw(sheet).text((x+5, y+7), f'{Path(img["file_name"]).stem}: returned {row["returned_count"]}, visible {row["visible_count"]}', font=font, fill='black')
    sheet.save(AUDIT / 'test_gt_and_visible_predictions.jpg', quality=95)
    findings = dict(status='PASS', reviewed_utc=datetime.now(timezone.utc).isoformat(), result_dir=str(DEST),
        selected_checkpoint_sha256=summary['selected_checkpoint_sha256'], selected_epoch=100,
        training_images=130, independent_validation=False, evaluated_images=23, gt_boxes=36, broken=8, loose=28, predictions=129,
        images_with_zero_returned_predictions=sum(not row['scores'] for row in details),
        visualization_threshold=0.30, visible_predictions=visible,
        visible_predictions_with_same_class_iou_at_least_0_5=sum(p['best_same_class_iou'] >= 0.5 for p in visible),
        gt_with_same_class_candidate_iou_at_least_0_5=sum(row['has_candidate_iou_at_least_0_5'] for row in per_box),
        coverage_note='GT candidate coverage only, not ordinary Recall or a one-to-one detection metric',
        metrics=summary['metrics'], evidence_hashes_verified=len(manifest['files_sha256']),
        archived_input_hashes_verified=len(pre['saved_inputs_sha256']),
        coordinates_and_category_export_match=True, saved_prediction_coco_recompute_match=True, per_class_metrics_reproduced=True,
        training_started=False, detector_forward_performed=False, test_results_used_to_change_training_settings=False)
    (AUDIT / 'acceptance.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('STEP10_RESULT_REVIEW_PASS: 23 images / 36 GT / 129 saved predictions; all-class and per-class exact COCO metrics reproduced')
    print(json.dumps(visible, ensure_ascii=False))
    print('GT_CANDIDATE_COVERAGE_AT_IOU0.5=' + str(findings['gt_with_same_class_candidate_iou_at_least_0_5']))


if __name__ == '__main__':
    main()
