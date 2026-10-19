"""Repair STEP15 final bookkeeping from existing predictions only. No detector or training imports."""
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'
DEST = RUN / 'test_selected_epoch5/step15_23467_7311'
LOG = ROOT / 'workspace/logs/step15_23467_7311'
AUDIT = ROOT / 'workspace/audit/step15_review'
ANNOTATION = Path('D:/小论文/conductor_defect_dataset/03_coco_dataset/annotations/instances_test.json')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    require(Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve(), 'Use Faster R-CNN environment')
    started = time.perf_counter()
    AUDIT.mkdir(parents=True, exist_ok=True)
    original = read(LOG / 'step15_manifest.json')
    require(original['status'] == 'FAIL' and original['test_inference_started'] and not original['training_started'], 'Unexpected original attempt state')
    require("'str' object has no attribute 'open'" in original['traceback'] and 'Checkpoint file changed during test' in original['traceback'], 'Unexpected failure; do not silently recover')
    preserved_paths = [LOG / 'step15_manifest.json', LOG / 'step15_console.txt',
        DEST / 'predictions.bbox.json', DEST / 'predictions_details.json', DEST / 'predictions_details.jsonl',
        DEST / 'native_coco_metrics.json', DEST / 'test_metrics.json', DEST / 'test_coco_eval.txt',
        DEST / 'pretest_record.json', DEST / 'resolved_test_config.py', DEST / 'selected_checkpoint_load_audit.json']
    before = {str(path): sha(path) for path in preserved_paths}
    pre = read(DEST / 'pretest_record.json')
    selection = read(DEST / 'code/selected_checkpoint.json')
    queue = read(DEST / 'code/fixed_test_queue.json')['items'][0]
    require(pre['selected_epoch'] == selection['selected_epoch'] == queue['epoch'] == 5, 'Wrong epoch')
    checkpoint = Path(selection['best_checkpoint'])
    require(sha(checkpoint) == selection['best_checkpoint_sha256'] == pre['selected_checkpoint_sha256'] == queue['checkpoint_sha256'], 'Checkpoint changed')
    for source, digest in pre['saved_inputs_sha256'].items():
        require(sha(DEST / 'code' / Path(source).name) == digest, 'A frozen launch input changed: ' + source)
    for source, digest in pre['test_image_sha256'].items():
        require(sha(source) == digest, 'A test image changed: ' + source)
    require(sha(ANNOTATION) == pre['test_annotation_sha256'], 'Test annotation changed')
    require(pre['resize'] == [2133, 1280] and pre['keep_ratio'] and pre['score_threshold'] == .001
            and pre['rcnn_nms_iou'] == .5 and pre['max_per_image'] == 100, 'Evaluation settings differ')
    load = read(DEST / 'selected_checkpoint_load_audit.json')
    require(load['status'] == 'PASS' and load['loaded_tensors'] == 348 and not load['missing_keys']
            and not load['unexpected_keys'] and load['sha256'] == sha(checkpoint), 'Full checkpoint loading not verified')
    truth = read(ANNOTATION)
    images = {entry['id']: entry for entry in truth['images']}
    require(len(images) == 23 and len(truth['annotations']) == 36, 'Original test counts differ')
    details = read(DEST / 'predictions_details.json')
    jsonl = [json.loads(line) for line in (DEST / 'predictions_details.jsonl').read_text(encoding='utf-8').splitlines()]
    require(details == jsonl and len(details) == 23, 'Per-image records incomplete')
    require([row['image_id'] for row in details] == list(images), 'IDs/order differ')
    predictions = read(DEST / 'predictions.bbox.json')
    require(len(predictions) == sum(row['returned_count'] for row in details) == 2300, 'Prediction records incomplete')
    expected_rows = []
    score_min, score_max = 1., 0.
    for row in details:
        info = images[row['image_id']]
        boxes, labels, scores = row['boxes_xyxy'], row['model_labels'], row['scores']
        require(len(boxes) == len(labels) == len(scores) == row['returned_count'] <= 100, 'Prediction array lengths differ')
        require(len(scores) == 100, 'Expected 100 retained predictions per image')
        require(row['original_shape'][:2] == [info['height'], info['width']], 'Original coordinate metadata differs')
        require(Path(row['image_path']).name == Path(info['file_name']).name, 'Image filename mismatch')
        visible = sum(score > .30 for score in scores)
        require(visible == row['visible_count'] == 0, 'Displayed prediction count differs')
        new_h, new_w = row['resized_shape'][:2]
        require(min(new_h, new_w) <= 1280 and max(new_h, new_w) <= 2133, 'Resize bounds differ')
        for box, label, score, cls in zip(boxes, labels, scores, row['classes']):
            require(label in (0, 1) and cls == ('broken', 'loose')[label], 'Class mapping differs')
            require(all(math.isfinite(v) for v in box) and math.isfinite(score) and .001 <= score <= 1., 'Non-finite/invalid predictions')
            x1, y1, x2, y2 = box
            require(0 <= x1 < x2 <= info['width'] + 1e-3 and 0 <= y1 < y2 <= info['height'] + 1e-3, 'Prediction coordinates not in original image bounds')
            score_min, score_max = min(score_min, score), max(score_max, score)
            expected_rows.append(dict(image_id=row['image_id'], category_id=label + 1,
                bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
        output = DEST / 'predicted_images' / (Path(row['image_path']).stem + '.jpg')
        with Image.open(output) as image:
            image.load()
            require(image.size == (info['width'], info['height']), 'Saved visualization dimensions differ')
    # CocoMetric exports predictions class by class; compare within each class in stable image order.
    for category in (1, 2):
        actual = [row for row in predictions if row['category_id'] == category]
        expected = [row for row in expected_rows if row['category_id'] == category]
        require(len(actual) == len(expected), 'COCO category count differs')
        for a, e in zip(actual, expected):
            require(a['image_id'] == e['image_id'] and abs(a['score'] - e['score']) < 1e-8,
                    'COCO prediction IDs/scores differ')
            require(np.allclose(a['bbox'], e['bbox'], atol=1e-3, rtol=0), 'COCO xywh vs saved xyxy mismatch')
    with (AUDIT / 'independent_coco_check.txt').open('w', encoding='utf-8') as handle:
        with contextlib.redirect_stdout(handle):
            gt = COCO(str(ANNOTATION))
            evaluator = COCOeval(gt, gt.loadRes(predictions), 'bbox')
            evaluator.params.imgIds = list(images)
            evaluator.params.catIds = [1, 2]
            evaluator.params.maxDets = [1, 10, 100]
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    exact = read(DEST / 'test_metrics.json')
    for key, index in [('mAP50_95', 0), ('mAP50', 1), ('AR100', 8)]:
        require(abs(exact[key] - float(evaluator.stats[index])) < 1e-12, 'Saved exact metric differs: ' + key)
    precision, recall = evaluator.eval['precision'], evaluator.eval['recall']
    mean_valid = lambda values: float(values[values >= 0].mean())
    for c, name in enumerate(('broken', 'loose')):
        values = dict(AP50_95=mean_valid(precision[:, :, c, 0, -1]),
            AP50=mean_valid(precision[0, :, c, 0, -1]), AR100=mean_valid(recall[:, c, 0, -1]))
        require(all(abs(value - exact['per_class'][name][key]) < 1e-12 for key, value in values.items()), 'Class AP/AR mismatch')
    native = read(DEST / 'native_coco_metrics.json')
    for key, index in [('coco/bbox_mAP', 0), ('coco/bbox_mAP_50', 1), ('coco/bbox_mAP_75', 2),
                       ('coco/bbox_mAP_s', 3), ('coco/bbox_mAP_m', 4), ('coco/bbox_mAP_l', 5)]:
        require(native[key] == round(float(evaluator.stats[index]), 3), 'Native rounded AP mismatch: ' + key)
    for name in ('broken', 'loose'):
        require(native['coco/' + name + '_precision'] == round(exact['per_class'][name]['AP50_95'], 3), 'Native class AP mismatch')
    require(len(list((DEST / 'predicted_images').glob('*.jpg'))) == 23, 'Missing visualization')
    console = (LOG / 'step15_console.txt').read_text(encoding='utf-8')
    require(console.count('TEST_IMAGE=') == 23 and 'TEST_IMAGE=23/23' in console, 'Log not complete for all images')
    summary = dict(status='PASS', completion_mode='RECOVERED_SAVED_RESULTS_ONLY',
        experiment='FRCNN-R50FPN-SCALE1280-VAL-001', split='original test', selected_epoch=5,
        images=23, boxes=36, broken=8, loose=28, metrics=exact, native_coco_metrics=native,
        returned_predictions=2300, visible_predictions_above_0_3=0,
        resize_scale=[2133, 1280], anchor_scales=[8], model_score_threshold=.001,
        visualization_score_threshold=.30, max_per_image=100, rcnn_nms_iou=.5,
        selected_checkpoint=str(checkpoint), selected_checkpoint_sha256=sha(checkpoint),
        inference_and_image_saving_seconds=None, peak_cuda_allocated_mib=None,
        missing_runtime_stats_reason='Original finalization failed before transient runtime statistics were persisted; not estimated or filled with zero.',
        training_started=False, test_used_for_epoch_selection=False, prior_test_results_already_seen=True,
        recovery_detector_forward_performed=False, recovery_model_created=False,
        original_failure_manifest=str(LOG / 'step15_manifest.json'), recovery_record=str(DEST / 'postprocessing_recovery.json'))
    if (DEST / 'test_summary.json').exists():
        require(read(DEST / 'test_summary.json') == summary, 'Existing recovered summary differs')
    else:
        save(DEST / 'test_summary.json', summary)
    after = {str(path): sha(path) for path in preserved_paths}
    require(before == after, 'Original evidence changed during recovery')
    record = dict(status='PASS', recovered_utc=datetime.now(timezone.utc).isoformat(),
        result_dir=str(DEST), original_attempt_status='FAIL', completion_mode='RECOVERED_SAVED_RESULTS_ONLY',
        reason='Final sha helper received str instead of Path after all23 predictions/metrics were saved.',
        preserved_files_sha256=before, checkpoint_sha256=sha(checkpoint),
        model_created=False, model_inference_performed=False, training_started=False,
        cpu_recovery_seconds=time.perf_counter() - started)
    save(DEST / 'postprocessing_recovery.json', record)
    evidence = preserved_paths + [DEST / 'test_summary.json', DEST / 'postprocessing_recovery.json']
    evidence += sorted((DEST / 'predicted_images').glob('*.jpg'))
    save(DEST / 'test_evidence_manifest.json', dict(status='PASS', files_sha256={str(p):sha(p) for p in evidence},
        checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint), no_training=True,
        selected_before_test=True, completion_mode='RECOVERED_SAVED_RESULTS_ONLY'))
    acceptance = dict(status='PASS', reviewed_utc=datetime.now(timezone.utc).isoformat(), result_dir=str(DEST),
        selected_epoch=5, selected_checkpoint_sha256=sha(checkpoint), metrics=exact, images=23, boxes=36,
        returned_predictions=2300, visible_predictions_above_0_3=0,
        score_min=score_min, score_max=score_max, original_files_unchanged=True,
        frozen_launch_inputs_hashes_verified=True, full_checkpoint_load_verified=True,
        labels_and_original_coordinates_verified=True, saved_predictions_independent_coco_match=True,
        native_rounding_matches=True, original_attempt_status='FAIL', results_recovered=True,
        visual_review_pending=True, visual_review_completed=False,
        detector_forward_performed=False, model_created=False, training_started=False,
        source_files_sha256={str(p):sha(p) for p in [DEST/'test_summary.json', DEST/'test_metrics.json',
            DEST/'native_coco_metrics.json', DEST/'predictions.bbox.json', DEST/'test_evidence_manifest.json']})
    save(AUDIT / 'acceptance.json', acceptance)
    # A compact view of existing saved images for assistant visual review. No detector invocation.
    contact = Image.new('RGB', (1200, 6 * 220), 'white')
    draw = ImageDraw.Draw(contact)
    for index, row in enumerate(details):
        x, y = (index % 4) * 300, (index // 4) * 220
        with Image.open(DEST/'predicted_images'/(Path(row['image_path']).stem+'.jpg')) as image:
            thumb = ImageOps.contain(image, (290, 185))
            contact.paste(thumb, (x + (300-thumb.width)//2, y+25))
        draw.text((x+5,y+5), Path(row['image_path']).stem + ' / score>0.30: 0', fill='black')
    contact.save(AUDIT/'saved_test_visualizations.jpg', quality=92)
    print('STEP15_SAVED_RESULTS_RECOVERED_PASS; 23 images, 2300 predictions; no new model inference or training')
    print(f'TEST_mAP50={exact["mAP50"]:.12f} TEST_mAP50_95={exact["mAP50_95"]:.12f} TEST_AR100={exact["AR100"]:.12f}')
    print('RESULT_DIR=' + str(DEST))


if __name__ == '__main__':
    main()
