"""Verify existing STEP08 evidence and make review figures. No model inference."""
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
DEST = RUN / 'test_selected_epoch66/step08_31489_5707'
AUDIT = ROOT / 'workspace/audit/step08_review'
ANN = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset\annotations\instances_test.json')
CLASSES = ['broken', 'loose']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union > 0 else 0


def xyxy(b):
    return [b[0], b[1], b[0] + b[2], b[1] + b[3]]


def main():
    assert Path(sys.executable).resolve() == ROOT / '.venv/Scripts/python.exe'
    AUDIT.mkdir(parents=True, exist_ok=True)
    truth, summary, details, preds = read(ANN), read(DEST / 'test_summary.json'), read(DEST / 'predictions_details.json'), read(DEST / 'predictions.bbox.json')
    pre = read(DEST / 'pretest_record.json')
    manifest = read(DEST / 'test_evidence_manifest.json')
    for name, digest in manifest['files_sha256'].items():
        assert sha(name) == digest, name
    assert sha(ANN) == pre['test_annotation_sha256']
    selection = read(RUN / 'selected_checkpoint.json')
    assert sha(selection['best_checkpoint']) == selection['best_checkpoint_sha256'] == summary['selected_checkpoint_sha256']
    assert sha(RUN / 'selected_checkpoint.json') == pre['selection_sha256']
    assert summary['status'] == 'PASS' and summary['selected_epoch'] == 66
    assert not summary['test_used_for_epoch_selection'] and not summary['training_started']
    ids = [img['id'] for img in truth['images']]
    assert ids == [row['image_id'] for row in details] == summary['metrics']['image_ids']
    assert len(ids) == len(set(ids)) == 23 and len(truth['annotations']) == 36
    assert Counter(a['category_id'] for a in truth['annotations']) == {1: 8, 2: 28}
    assert len(preds) == 123 == summary['returned_predictions']
    exported = []
    visible = []
    for row, img in zip(details, truth['images']):
        assert Path(row['image_path']).name == Path(img['file_name']).name
        assert '/images/test/' in row['image_path'].replace('\\', '/')
        assert len(row['scores']) == len(row['model_labels']) == len(row['boxes_xyxy']) == row['returned_count'] <= 100
        with Image.open(row['image_path']) as image:
            assert image.size == (img['width'], img['height'])
        assert (DEST / 'predicted_images' / Path(row['image_path']).name).is_file()
        for box, label, score in zip(row['boxes_xyxy'], row['model_labels'], row['scores']):
            assert label in (0, 1) and 0.001 < score <= 1 and math.isfinite(score)
            assert all(math.isfinite(x) for x in box) and 0 <= box[0] < box[2] <= img['width'] and 0 <= box[1] < box[3] <= img['height']
            exported.append(dict(image_id=img['id'], category_id=label + 1, bbox=[box[0], box[1], box[2]-box[0], box[3]-box[1]], score=score))
            if score > 0.30:
                relevant = [a for a in truth['annotations'] if a['image_id'] == img['id'] and a['category_id'] == label + 1]
                visible.append(dict(image_id=img['id'], file_name=img['file_name'], class_name=CLASSES[label], score=score,
                                    best_same_class_iou=max((iou(box, xyxy(a['bbox'])) for a in relevant), default=0)))
    assert exported == preds, 'COCO export and detailed original coordinates disagree'
    assert len(visible) == 1 == summary['visible_predictions_above_0_3']
    per_box = []
    for a in truth['annotations']:
        candidates = [p for p in preds if p['image_id'] == a['image_id'] and p['category_id'] == a['category_id']]
        best = max(candidates, key=lambda p: iou(xyxy(a['bbox']), xyxy(p['bbox'])), default=None)
        overlap = iou(xyxy(a['bbox']), xyxy(best['bbox'])) if best else 0
        per_box.append(dict(annotation_id=a['id'], image_id=a['image_id'], class_name=CLASSES[a['category_id']-1],
                            best_same_class_iou=overlap, best_candidate_score=best['score'] if best else None,
                            has_candidate_iou_at_least_0_5=overlap >= 0.5))
    with (AUDIT / 'per_box_review.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_box[0])); w.writeheader(); w.writerows(per_box)
    # Independent saved-prediction recomputation. Does not call a detector.
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from contextlib import redirect_stdout
    with (AUDIT / 'independent_coco_check.txt').open('w', encoding='utf-8') as f, redirect_stdout(f):
        gt = COCO(str(ANN)); dt = gt.loadRes(preds)
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = ids; evaluator.params.catIds = [1, 2]; evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate(); evaluator.accumulate(); evaluator.summarize()
    for index, key in [(0, 'mAP50_95'), (1, 'mAP50'), (8, 'AR100')]:
        assert abs(evaluator.stats[index] - summary['metrics'][key]) < 1e-12
    # Contact sheet: existing display predictions in cyan/red; GT in green.
    sheet = Image.new('RGB', (1600, 8 * 260), 'white')
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 16)
    for index, (row, img) in enumerate(zip(details, truth['images'])):
        with Image.open(DEST / 'predicted_images' / Path(img['file_name']).name) as image:
            thumbnail = ImageOps.contain(image.convert('RGB'), (520, 220))
        ratio = thumbnail.width / img['width']
        d = ImageDraw.Draw(thumbnail)
        for a in truth['annotations']:
            if a['image_id'] == img['id']:
                d.rectangle([v * ratio for v in xyxy(a['bbox'])], outline=(0, 240, 0), width=2)
        x, y = (index % 3)*533, (index // 3)*260
        sheet.paste(thumbnail, (x+(533-thumbnail.width)//2, y+35))
        ImageDraw.Draw(sheet).text((x+5, y+7), f'{Path(img["file_name"]).stem}: returned {row["returned_count"]}, visible {row["visible_count"]}', font=font, fill='black')
    sheet.save(AUDIT / 'test_gt_and_visible_predictions.jpg', quality=95)
    findings = dict(status='PASS', reviewed_utc=datetime.now(timezone.utc).isoformat(), result_dir=str(DEST),
                    selected_checkpoint_sha256=summary['selected_checkpoint_sha256'], selected_epoch=66,
                    evaluated_images=23, gt_boxes=36, broken=8, loose=28, predictions=123,
                    images_with_zero_returned_predictions=sum(not row['scores'] for row in details),
                    visualization_threshold=0.30, visible_predictions=visible,
                    gt_with_same_class_candidate_iou_at_least_0_5=sum(row['has_candidate_iou_at_least_0_5'] for row in per_box),
                    coverage_note='GT candidate coverage only, not ordinary Recall or one-to-one detection metric',
                    metrics=summary['metrics'], evidence_hashes_verified=len(manifest['files_sha256']),
                    coordinates_and_category_export_match=True, saved_prediction_coco_recompute_match=True,
                    training_started=False, detector_forward_performed=False,
                    test_results_used_to_change_training_settings=False)
    (AUDIT / 'acceptance.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('STEP08_RESULT_REVIEW_PASS: 23 images / 36 GT / 123 saved predictions; COCO exact metrics reproduced')
    print(json.dumps(visible, ensure_ascii=False))
    print('GT_CANDIDATE_COVERAGE_AT_IOU0.5=' + str(findings['gt_with_same_class_candidate_iou_at_least_0_5']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
