"""Refresh records from saved evidence only. No training, model load or inference.

Run with the dedicated Faster R-CNN interpreter. Excel is authored by the bundled
artifact-tool Node runtime; this file reads exported XML for independent checks.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile

import yaml

BASE = Path(__file__).resolve().parents[1]
PROJECTS = BASE.parent
FRCNN = PROJECTS / 'Faster R-CNN'
FRUN = FRCNN / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
OUT = BASE / 'outputs'
AUDIT = BASE / 'audit'
CSV = BASE / '10_all_experiments_metrics.csv'
BOOK = OUT / '模型实验结果总表.xlsx'
REPORT = BASE / '11_all_experiments_summary_for_teacher.md'
RUNTIME = Path(r'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\node')
NODE = RUNTIME / 'bin/node.exe'
ATTACHMENTS = Path(r'C:\Users\Administrator\.codex\attachments')
DATA = Path(r'D:\小论文\conductor_defect_dataset')
RATIO = ['map50', 'map50_95', 'precision', 'recall', 'coco_ar100'] + [
    f'{c}_{m}' for c in ('broken', 'loose') for m in ('ap50', 'ap50_95', 'precision', 'recall', 'ar100')]
FIELDS = '''evaluation_id experiment_id run_id model framework framework_version source_commit python_version torch_version
training_status evaluation_status evaluation_split evaluation_event evaluation_protocol comparison_scope
train_images val_images test_images train_boxes train_broken train_loose eval_images eval_boxes eval_broken eval_loose
label_type class_mapping data_version split_manifest_path split_manifest_sha256 annotation_path annotation_sha256
label_files_manifest_sha256 pretrained_source pretrained_path pretrained_sha256 pretrained_load_evidence
resize_mode input_size epochs batch workers seed deterministic amp optimizer learning_rate gradient_accumulation
training_iterations optimizer_updates iteration_count_status training_seconds training_time_scope
checkpoint_path checkpoint_sha256 selected_epoch selection_rule map50 map50_95 precision recall coco_ar100
broken_ap50 broken_ap50_95 broken_precision broken_recall broken_ar100 loose_ap50 loose_ap50_95 loose_precision loose_recall loose_ar100
precision_recall_rule precision_recall_threshold evaluator_version prediction_score_threshold nms_rule nms_iou max_detections coco_max_dets
visualization_threshold metric_precision missing_reasons metric_source metric_source_sha256 predictions_source predictions_sha256
config_source config_sha256 log_source original_log_source training_command_source evaluation_command experiment_note visualization_source evidence_bundle source_verification'''.split()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _hash_stream(f)


def _hash_stream(f):
    digest = hashlib.sha256()
    for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return [{k.strip(): v.strip() for k, v in row.items()} for row in csv.DictReader(f)]


def label_manifest(split):
    entries = [(p.name, sha(p)) for p in sorted((DATA / '02_yolo11_dataset/labels' / split).glob('*.txt'))]
    return hashlib.sha256(json.dumps(entries, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def base_row(experiment, run, model, split):
    manifest = DATA / '02_yolo11_dataset/split_manifest.csv'
    boxes, broken, loose = (37, 9, 28) if split == 'val' else (36, 8, 28)
    return dict(experiment_id=experiment, run_id=run.name, model=model,
                training_status='COMPLETED', evaluation_status='VERIFIED_SAVED_EVIDENCE',
                evaluation_split=split, train_images=107, val_images=23, test_images=23,
                train_boxes=168, train_broken=39, train_loose=129,
                eval_images=23, eval_boxes=boxes, eval_broken=broken, eval_loose=loose,
                data_version='original fixed split 153 images / 241 boxes; STEP04 conversion verified',
                split_manifest_path=str(manifest), split_manifest_sha256=sha(manifest),
                label_files_manifest_sha256=label_manifest(split),
                epochs=100, batch=4, workers=0, seed=42, amp=True,
                comparison_scope='native framework reference; no unified cross-framework reevaluation',
                source_verification='source files and checkpoint hash checked; original observations preserved')


def attach_metric(row, metrics, source, event, precision_note, predictions=None):
    row.update(evaluation_event=event, metric_source=str(source), metric_source_sha256=sha(source),
               metric_precision=precision_note)
    overall = metrics.get('overall', metrics)
    for key in ('map50', 'map50_95', 'precision', 'recall'):
        if key in overall:
            row[key] = overall[key]
    for cls, values in metrics.get('per_class', metrics.get('classes', {})).items():
        for old, new in [('map50', 'ap50'), ('map50_95', 'ap50_95'), ('precision', 'precision'), ('recall', 'recall')]:
            if old in values:
                row[f'{cls}_{new}'] = values[old]
    if predictions:
        row.update(predictions_source=str(predictions), predictions_sha256=sha(predictions))
    return row


def terminal_best(path):
    """Parse exactly the last automatic best.pt validation block, at printed precision."""
    text = path.read_text(encoding='utf-8-sig')
    last = text.rsplit('Validating ', 1)[1]
    matches = re.findall(r'^\s*(all|broken|loose)\s+(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)', last, re.M)
    assert len(matches) == 3, f'Missing best.pt validation rows: {path}'
    result = {'overall': {}, 'per_class': {}}
    for cls, images, boxes, *values in matches:
        assert int(boxes) == {'all': 37, 'broken': 9, 'loose': 28}[cls]
        item = dict(zip(('precision', 'recall', 'map50', 'map50_95'), map(float, values)))
        if cls == 'all':
            assert int(images) == 23
            result['overall'] = item
        else:
            result['per_class'][cls] = item
    return result


def legacy_rows():
    rows, runs = [], []
    specifications = [
        ('yolo11', 'Y11-BASE-001-R1', 'Y11-BASE-001-R1_20260917_203044_883451', 'YOLO11s', 15,
         '01f6246e45a0b42fe93762120c8784d8a885dcba21567ab5906d94af2edb1cc9', None),
        ('RT-DETR', 'RTD-BASE-001-R1', 'RTD-BASE-001-R1_20260918_172120', 'RT-DETR-L', 71,
         '15ba1e62e1170d21e0628753be3eef0a710e06300643b3f9dd421f2c02d53ccd', None),
        ('yolo11', 'Y11-SENS-IMGSZ1600-001', 'Y11-SENS-IMGSZ1600-001', 'YOLO11s', 54,
         '2cc998a09da3eb36d717df79dfa28adf2a41cebb1121ecd43d984ff32ea4c92e', '18aab365-1d88-401b-b0a2-390ae5921449'),
        ('RT-DETR', 'RTD-SENS-IMGSZ1600-001', 'RTD-SENS-IMGSZ1600-001', 'RT-DETR-L', 42,
         '29d23c8bb93bd650800ea439c9f43755b81f1763d0550ce4e14119dee4121a13', 'a08bd723-3391-4616-ac0d-1cc9177e13ec'),
    ]
    for project, exp, run_id, model, epoch, expected, attachment in specifications:
        home = PROJECTS / project / 'workspace'
        run = home / 'runs' / run_id
        args_path = run / 'args.yaml'
        args = yaml.safe_load(args_path.read_text(encoding='utf-8'))
        trace = read_csv(run / 'results.csv')
        assert len(trace) == 100 and [int(float(r['epoch'])) for r in trace] == list(range(1, 101))
        checkpoint = run / 'weights/best.pt'
        assert sha(checkpoint) == expected, f'Historical checkpoint changed: {checkpoint}'
        selected = trace[epoch - 1]
        assert float(selected['metrics/mAP50-95(B)']) == max(float(r['metrics/mAP50-95(B)']) for r in trace)
        rtdetr = model == 'RT-DETR-L'
        initial = home / 'weights' / ('rtdetr-l.pt' if rtdetr else 'yolo11s.pt')
        expected_initial = '6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f' if rtdetr else '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'
        assert sha(initial) == expected_initial
        if attachment:
            original_log = ATTACHMENTS / attachment / '已粘贴的文本.txt'
            raw = BASE / 'evidence/legacy_console' / f'{exp}.txt'
            raw.parent.mkdir(parents=True, exist_ok=True)
            if raw.exists():
                if original_log.exists():
                    assert sha(raw) == sha(original_log), 'Archived terminal evidence changed'
            else:
                shutil.copy2(original_log, raw)
            note = home / 'notes' / f'{exp}.md'
            seconds = float(trace[-1]['time'])
        elif rtdetr:
            raw = home / 'logs' / f'{run_id}.log'
            original_log = raw
            note = home / 'notes/rtdetr_single_run_report.md'
            seconds = read_json(home / 'analysis/rtdetr_single_run_metrics.json')['runtime']['training_wall_seconds']
        else:
            raw = home / 'logs' / run_id / f'{run_id}_console.log'
            original_log = raw
            note = home / 'notes/formal_baseline_single_run_report.md'
            seconds = read_json(home / 'analysis/formal_baseline_single_run_metrics.json')['runtime_seconds']
        text = raw.read_text(encoding='utf-8-sig', errors='replace')
        optim = re.search(r'optimizer:.*?AdamW\(lr=([0-9.eE+-]+), momentum=([0-9.eE+-]+)\)', text)
        assert optim, f'No actual optimizer evidence: {raw}'
        common = dict(framework='Ultralytics', framework_version='8.4.152',
            source_commit='9aa41d679d121afb8b4cc512d7a162f4f36886c3', python_version='3.12.13', torch_version='2.8.0+cu128',
            label_type='YOLO detection xywh normalized', class_mapping='YOLO/model:0=broken,1=loose',
            annotation_path=str(DATA / '02_yolo11_dataset/labels'),
            pretrained_source='official COCO detection checkpoint; verified historical loading',
            pretrained_path=str(initial), pretrained_sha256=expected_initial, pretrained_load_evidence=str(raw),
            resize_mode='stretch to square (RTDETRDataset.load_image rect_mode=False)' if rtdetr else 'aspect-preserving resize + LetterBox / rectangular validation batches',
            input_size=str(args['imgsz']), deterministic=args['deterministic'],
            optimizer='AdamW (resolved optimizer=auto)', learning_rate=float(optim.group(1)),
            gradient_accumulation='warmup varies; afterwards nbs64/batch4=16',
            training_iterations=2700, iteration_count_status='27 batches per epoch x100; optimizer update count not saved (AMP/accumulation)',
            training_seconds=seconds, training_time_scope='results.csv elapsed seconds' if attachment else 'historical wrapper training wall seconds',
            checkpoint_path=str(checkpoint), checkpoint_sha256=expected, selected_epoch=epoch,
            selection_rule='native Ultralytics detection fitness = mAP50-95; selected epoch matched historical record and results.csv',
            evaluation_protocol='Ultralytics-native-AP', evaluator_version='Ultralytics 8.4.152 native AP, 101-point trapezoidal integration',
            precision_recall_rule='IoU0.50, at shared threshold maximizing smoothed mean class F1; macro class mean; native zero-prediction conventions retained',
            precision_recall_threshold='', prediction_score_threshold=0.001,
            nms_rule='none; decoder top-k then conf filter and max_det' if rtdetr else 'class-aware NMS',
            nms_iou='' if rtdetr else 0.7, max_detections=300,
            visualization_threshold='', config_source=str(args_path), config_sha256=sha(args_path),
            log_source=str(raw), original_log_source=str(original_log),
            training_command_source=str(raw if attachment else home / 'scripts' / ('train_rtdetr_formal_baseline.py' if rtdetr else 'run_formal_baseline.py')),
            evaluation_command='saved native training-end/independent validation; exact outer command not retained' if not attachment else 'automatic best.pt validation inside CLI training',
            experiment_note=str(note), evidence_bundle=str(run))
        runs.append(dict(experiment_id=exp, run_id=run_id, model=model, status='COMPLETED',
                         train_images=107, val_images=23, test_images=23, epochs=100, batch=4,
                         training_iterations=2700, optimizer_updates=None, learning_rate=common['learning_rate'],
                         input_size=common['input_size'], resize_mode=common['resize_mode'], selected_epoch=epoch,
                         test_status='NOT_EVALUATED_BY_DESIGN' if attachment else 'COMPLETED_ONCE',
                         source=str(run), note=str(note)))
        for split in (['val'] if attachment else ['val', 'test']):
            row = base_row(exp, run, model, split)
            row.update(common)
            if attachment:
                m = {'overall': {key: float(selected[col]) for key, col in [
                    ('precision', 'metrics/precision(B)'), ('recall', 'metrics/recall(B)'),
                    ('map50', 'metrics/mAP50(B)'), ('map50_95', 'metrics/mAP50-95(B)')]}}
                r = attach_metric(row.copy(), m, run / 'results.csv', 'selected_epoch_validation', 'CSV original saved precision; per-class values not stored here')
                r['visualization_source'] = str(run)
                rows.append(r)
                r = attach_metric(row.copy(), terminal_best(raw), raw, 'automatic_best_checkpoint_revalidation', 'terminal printed decimals only; do not add digits')
                r.update(evaluation_status='VERIFIED_PRINTED_PRECISION', visualization_source=str(run))
            else:
                source = run / f'formal_{split}_metrics.json' if rtdetr else home / 'logs' / run_id / f'{split}_metrics.json'
                r = attach_metric(row, read_json(source), source, 'formal_independent_evaluation', 'original JSON numeric values; historical class decimals retained')
                r['visualization_source'] = str(run / (f'formal_{split}' if rtdetr else f'final_{split}'))
            rows.append(r)
    return rows, runs


def coco_metric(row, native, exact, source, predictions, event):
    row.update(metric_source=str(source), metric_source_sha256=sha(source), predictions_source=str(predictions),
               predictions_sha256=sha(predictions), evaluation_event=event)
    if native is not None:
        row.update(map50=native.get('coco/bbox_mAP_50'), map50_95=native.get('coco/bbox_mAP'),
                   broken_ap50_95=native.get('coco/broken_precision'), loose_ap50_95=native.get('coco/loose_precision'),
                   evaluation_protocol='MMDetection-native-CocoMetric-rounded',
                   metric_precision='native AP rounded to 3 decimal places; class *_precision means AP50-95')
    else:
        row.update(map50=exact['mAP50'], map50_95=exact['mAP50_95'], coco_ar100=exact['AR100'],
                   evaluation_protocol='pycocotools-exact-from-saved-predictions',
                   metric_precision='full numeric JSON; same COCO predictions; no new model inference')
        for c, v in exact['per_class'].items():
            for old, new in [('AP50', 'ap50'), ('AP50_95', 'ap50_95'), ('AR100', 'ar100')]:
                row[f'{c}_{new}'] = v[old]
    return row


def faster_rows():
    pre = read_json(FRUN / 'pretraining_record.json')
    selection = read_json(FRUN / 'selected_checkpoint.json')
    summary = read_json(FRUN / 'artifacts/formal_training_summary.json')
    assert summary['epochs'] == 100 and summary['optimizer_updates'] == 5400
    assert selection['selected_epoch'] == 66 and sha(selection['best_checkpoint']) == selection['best_checkpoint_sha256']
    assert sha(pre['checkpoint']['path']) == pre['checkpoint']['sha256']
    note = FRCNN / 'workspace/notes/FRCNN-R50FPN-VAL-001.md'
    log = FRCNN / 'workspace/logs' / FRUN.name / 'train_console.log'
    assert log.exists()
    common = dict(framework='MMDetection', framework_version=pre['versions']['mmdet'],
        source_commit=pre['mmdet_commit'], python_version='3.10.20', torch_version=pre['versions']['torch'],
        label_type='COCO bbox xywh pixels', class_mapping='YOLO0/1 -> COCO1/2 -> model0/1 (broken/loose); num_classes=2',
        pretrained_source=pre['checkpoint']['url'], pretrained_path=pre['checkpoint']['path'], pretrained_sha256=pre['checkpoint']['sha256'],
        pretrained_load_evidence=str(FRUN / 'artifacts/actual_training_weight_load.json'),
        resize_mode='Resize scale=(1333,800), keep_ratio=True; pad to divisor32', input_size='1333x800 bounds',
        batch=2, amp=False, deterministic=True, optimizer='SGD', learning_rate=0.0025, gradient_accumulation='1',
        training_iterations=5400, optimizer_updates=5400, iteration_count_status='5400 finite batch records verified',
        training_seconds=summary['wall_seconds_including_validation_checkpoints_snapshots'],
        training_time_scope='training + val + checkpoint + fixed-train snapshots',
        checkpoint_path=selection['best_checkpoint'], checkpoint_sha256=selection['best_checkpoint_sha256'], selected_epoch=66,
        selection_rule='max original val coco/bbox_mAP rounded to 3 decimals; first best on tie; frozen before test',
        precision_recall_rule='ordinary Precision/Recall not computed; COCO AR100 is a separate metric',
        evaluator_version='MMDetection 3.3.0 / MMEngine 0.10.7 / pycocotools 2.0.10',
        prediction_score_threshold=0.001, nms_rule='class-aware ROI NMS; RPN pre_nms=1000,nms_iou=0.7,max=1000',
        nms_iou=0.5, max_detections=100, coco_max_dets='1;10;100', visualization_threshold=0.30,
        config_source=str(FRUN / 'resolved_val_001.py'), config_sha256=sha(FRUN / 'resolved_val_001.py'),
        log_source=str(log), original_log_source=str(log),
        training_command_source=str(FRUN / 'code/run_step07_train_val_baseline.cmd'),
        evaluation_command='native validation each epoch inside saved STEP07 training command',
        experiment_note=str(note), evidence_bundle=str(FRUN))
    rows = []
    row = base_row('FRCNN-R50FPN-VAL-001', FRUN, 'Faster R-CNN R50-FPN', 'val')
    row.update(common, annotation_path=str(DATA / '03_coco_dataset/annotations/instances_val.json'),
               annotation_sha256=pre['annotation_hashes']['val'], visualization_source=str(FRUN / 'artifacts'))
    assert sha(row['annotation_path']) == row['annotation_sha256']
    pred = Path(selection['validation_predictions'])
    native_source = FRUN / 'artifacts/validation/epoch_066/metrics.json'
    rows.append(coco_metric(row.copy(), read_json(native_source)['metrics'], None, native_source, pred, 'selected_epoch_validation'))
    exact_source = FRUN / 'artifacts/best_validation_metrics.json'
    exact = read_json(exact_source)
    assert exact == selection['validation_metrics']
    rows.append(coco_metric(row.copy(), None, exact, exact_source, pred, 'selected_validation_offline_exact'))
    tests = sorted((FRUN / 'test_selected_epoch66').glob('*/test_summary.json'))
    assert len(tests) <= 1, 'Multiple completed test evaluations need manual review'
    review_path = FRCNN / 'workspace/audit/step08_review/acceptance.json'
    reviewed = False
    for path in tests:
        s = read_json(path)
        assert s['status'] == 'PASS' and s['selected_epoch'] == 66 and s['images'] == 23 and s['boxes'] == 36
        assert s['selected_checkpoint_sha256'] == common['checkpoint_sha256']
        dest = path.parent
        if review_path.exists():
            review = read_json(review_path)
            reviewed = (review['status'] == 'PASS' and Path(review['result_dir']) == dest
                        and review['selected_checkpoint_sha256'] == s['selected_checkpoint_sha256']
                        and review['metrics'] == s['metrics'])
        row = base_row('FRCNN-R50FPN-VAL-001', FRUN, 'Faster R-CNN R50-FPN', 'test')
        row.update(common, evaluation_status='VERIFIED_RESULT_REVIEW' if reviewed else 'COMPUTATION_PASS_AWAITING_RESULT_REVIEW',
                   annotation_path=str(DATA / '03_coco_dataset/annotations/instances_test.json'), annotation_sha256=pre['annotation_hashes']['test'],
                   visualization_source=str(dest / 'predicted_images'), log_source=str(FRCNN / 'workspace/logs' / dest.name / 'step08_console.txt'),
                   config_source=str(dest / 'resolved_test_config.py'), config_sha256=sha(dest / 'resolved_test_config.py'), evidence_bundle=str(dest))
        row['evaluation_command'] = read_json(dest / 'pretest_record.json').get('command_line', 'see saved STEP08 command and log')
        assert sha(row['annotation_path']) == row['annotation_sha256']
        predictions = dest / 'predictions.bbox.json'
        assert len(read_json(dest / 'predictions_details.json')) == 23
        assert len(list((dest / 'predicted_images').glob('*.jpg'))) == 23
        if s['native_coco_metrics']:
            rows.append(coco_metric(row.copy(), s['native_coco_metrics'], None, dest / 'native_coco_metrics.json', predictions, 'fixed_selected_test'))
        rows.append(coco_metric(row.copy(), None, s['metrics'], dest / 'test_metrics.json', predictions, 'fixed_test_offline_exact'))
    run = dict(experiment_id='FRCNN-R50FPN-VAL-001', run_id=FRUN.name, model='Faster R-CNN R50-FPN',
               status='COMPLETED_REVIEWED' if reviewed else 'TRAINING_COMPLETED', train_images=107, val_images=23, test_images=23, epochs=100, batch=2,
               training_iterations=5400, optimizer_updates=5400, learning_rate=0.0025, input_size='1333x800 bounds',
               resize_mode=common['resize_mode'], selected_epoch=66, test_status=('COMPLETED_REVIEWED' if reviewed else 'COMPLETED_AWAITING_REVIEW') if tests else 'PENDING',
               source=str(FRUN), note=str(note))
    return rows, run


def merged_runs():
    """Index merged training attempts without creating a fake evaluation row."""
    folder = FRCNN / 'workspace/runs/FRCNN-R50FPN-MERGE130-001'
    notes = FRCNN / 'workspace/notes/FRCNN-R50FPN-MERGE130-001.md'
    entries = []
    for run in sorted(folder.glob('step09_*')):
        if not run.is_dir():
            continue
        entry = dict(experiment_id='FRCNN-R50FPN-MERGE130-001', run_id=run.name, model='Faster R-CNN R50-FPN',
            status='INCOMPLETE_NEEDS_REVIEW', train_images=130, val_images=None, test_images=23,
            epochs=None, batch=2, training_iterations=None, optimizer_updates=None, learning_rate=0.0025,
            input_size='1333x800 bounds', selected_epoch=None, test_status='PENDING', source=str(run), note=str(notes))
        log_manifest = FRCNN / 'workspace/logs' / run.name / 'step09_manifest.json'
        if log_manifest.exists():
            entry['status'] = read_json(log_manifest)['status']
        selected = run / 'selected_checkpoint.json'
        summary = run / 'artifacts/formal_training_summary.json'
        if selected.exists() and summary.exists():
            s, stats = read_json(selected), read_json(summary)
            assert s['selected_epoch'] == 100 and stats['epochs'] == 100 and stats['optimizer_updates'] == 6500
            assert not s['independent_validation'] and not stats['continued_from_val001_checkpoint']
            assert sha(s['checkpoint']) == s['checkpoint_sha256']
            entry.update(status='TRAINING_COMPLETED_TEST_PENDING', epochs=100, training_iterations=6500,
                         optimizer_updates=6500, selected_epoch=100)
            review_path = FRCNN / 'workspace/audit/step09_review/acceptance.json'
            if review_path.exists():
                review = read_json(review_path)
                if review['status'] == 'PASS' and Path(review['run_dir']) == run and review['selected_checkpoint_sha256'] == s['checkpoint_sha256']:
                    entry['status'] = 'TRAINING_REVIEWED_TEST_PENDING'
        entries.append(entry)
    if not entries:
        entries.append(dict(experiment_id='FRCNN-R50FPN-MERGE130-001', run_id=None, model='Faster R-CNN R50-FPN',
            status='PREPARED_NOT_STARTED', train_images=130, val_images=None, test_images=23, epochs=100, batch=2,
            training_iterations=None, optimizer_updates=None, learning_rate=0.0025, input_size='1333x800 bounds',
            selected_epoch=None, test_status='NOT_STARTED', source=str(folder), note=str(notes)))
    return entries


def merged_test_rows(runs, template):
    """Read MERGE130 epoch100 test if saved; no new detector evaluation."""
    rows = []
    shared = '''framework framework_version source_commit python_version torch_version label_type class_mapping
resize_mode input_size deterministic amp optimizer gradient_accumulation precision_recall_rule evaluator_version
prediction_score_threshold nms_rule nms_iou max_detections coco_max_dets visualization_threshold comparison_scope'''.split()
    for entry in runs:
        if entry['experiment_id'] != 'FRCNN-R50FPN-MERGE130-001' or not entry.get('run_id'):
            continue
        run = Path(entry['source'])
        tests = sorted((run / 'test_selected_epoch100').glob('*/test_summary.json'))
        assert len(tests) <= 1, 'Multiple completed MERGE130 tests require review'
        for path in tests:
            s = read_json(path)
            selection = read_json(run / 'selected_checkpoint.json')
            pre = read_json(run / 'pretraining_record.json')
            stats = read_json(run / 'artifacts/formal_training_summary.json')
            assert s['status'] == 'PASS' and s['selected_epoch'] == 100 and s['training_images'] == 130
            assert s['images'] == 23 and s['boxes'] == 36 and not s['independent_validation']
            assert s['selected_checkpoint_sha256'] == selection['checkpoint_sha256']
            assert sha(s['selected_checkpoint']) == selection['checkpoint_sha256']
            assert not s['test_used_for_epoch_selection'] and not s['training_started']
            dest = path.parent
            assert len(read_json(dest / 'predictions_details.json')) == 23
            assert len(list((dest / 'predicted_images').glob('*.jpg'))) == 23
            row = base_row(entry['experiment_id'], run, entry['model'], 'test')
            row.update({key: template[key] for key in shared})
            review_path = FRCNN / 'workspace/audit/step10_review/acceptance.json'
            reviewed = False
            if review_path.exists():
                reviewed_data = read_json(review_path)
                reviewed = (reviewed_data['status'] == 'PASS' and Path(reviewed_data['result_dir']) == dest
                            and reviewed_data['selected_checkpoint_sha256'] == selection['checkpoint_sha256']
                            and reviewed_data['metrics'] == s['metrics'])
            row.update(train_images=130, val_images='', train_boxes=205, train_broken=48, train_loose=157,
                evaluation_status='VERIFIED_RESULT_REVIEW' if reviewed else 'COMPUTATION_PASS_AWAITING_RESULT_REVIEW',
                data_version='original train+val130 union; no independent val; original test23 reserved; STEP04 COCO verified',
                pretrained_source=pre['checkpoint']['url'], pretrained_path=pre['checkpoint']['path'], pretrained_sha256=pre['checkpoint']['sha256'],
                pretrained_load_evidence=str(run / 'artifacts/actual_training_weight_load.json'),
                learning_rate=pre['lr'], training_iterations=6500, optimizer_updates=6500,
                iteration_count_status='6500 finite batch records verified',
                training_seconds=stats['wall_seconds_including_checkpoints_snapshots'], training_time_scope='training + checkpoint + fixed-train snapshots; no val',
                checkpoint_path=selection['checkpoint'], checkpoint_sha256=selection['checkpoint_sha256'], selected_epoch=100,
                selection_rule=selection['selection_rule'],
                annotation_path=str(DATA / '03_coco_dataset/annotations/instances_test.json'), annotation_sha256=pre['annotation_hashes']['test'],
                config_source=str(dest / 'resolved_test_config.py'), config_sha256=sha(dest / 'resolved_test_config.py'),
                log_source=str(FRCNN / 'workspace/logs' / dest.name / 'step10_console.txt'),
                original_log_source=str(FRCNN / 'workspace/logs' / dest.name / 'step10_console.txt'),
                training_command_source=str(run / 'code/run_step09_train_merge130.cmd'),
                evaluation_command=read_json(dest / 'pretest_record.json')['command_line'],
                experiment_note=entry['note'], visualization_source=str(dest / 'predicted_images'), evidence_bundle=str(dest))
            assert sha(row['annotation_path']) == row['annotation_sha256']
            prediction_file = dest / 'predictions.bbox.json'
            if s['native_coco_metrics']:
                rows.append(coco_metric(row.copy(), s['native_coco_metrics'], None, dest / 'native_coco_metrics.json', prediction_file, 'fixed_last_epoch_test'))
            rows.append(coco_metric(row.copy(), None, s['metrics'], dest / 'test_metrics.json', prediction_file, 'fixed_last_test_offline_exact'))
            entry.update(status='COMPLETED_REVIEWED' if reviewed else 'TEST_COMPLETED_AWAITING_REVIEW',
                         test_status='COMPLETED_REVIEWED' if reviewed else 'COMPLETED_AWAITING_REVIEW')
    return rows


def finish_row(row):
    # Evaluation identity includes occurrence: automatic revalidation and selection
    # epoch validation are different observations of the same checkpoint.
    identity = [row[k] for k in ('experiment_id', 'run_id', 'checkpoint_sha256', 'evaluation_split', 'evaluation_protocol', 'evaluation_event')]
    row['evaluation_id'] = 'EV-' + hashlib.sha256('|'.join(identity).encode()).hexdigest()[:16]
    reasons = {}
    for key in RATIO:
        value = row.get(key)
        if value is None or value == '':
            row[key] = ''
            reasons[key] = ('not computed; AR100 is not ordinary Recall' if key.endswith('precision') or key.endswith('recall') else
                            'not computed by native Ultralytics' if 'ar100' in key and row['framework'] == 'Ultralytics' else
                            'not saved in this metric source')
        else:
            assert isinstance(value, (float, int)) and math.isfinite(value) and 0 <= value <= 1, (key, value)
    if not row.get('predictions_source'):
        reasons['predictions_source'] = 'no machine-readable raw prediction file saved in legacy run; existing plots retained; no rerun'
    if not row.get('precision_recall_threshold'):
        reasons['precision_recall_threshold'] = 'not saved; best-mean-F1 rule known' if row['framework'] == 'Ultralytics' else 'ordinary P/R not computed'
    if not row.get('optimizer_updates'):
        reasons['optimizer_updates'] = 'actual optimizer update count not logged; 2700 batches must not be called 2700 updates'
    if not row.get('visualization_threshold'):
        reasons['visualization_threshold'] = 'legacy plotting rules differ; no claim of common threshold'
    if not row.get('annotation_sha256'):
        reasons['annotation_sha256'] = 'YOLO labels are multiple files; use label_files_manifest_sha256 instead'
    if not row.get('coco_max_dets'):
        reasons['coco_max_dets'] = 'Ultralytics native AP; COCO maxDets not applicable'
    if not row.get('nms_iou'):
        reasons['nms_iou'] = 'not applicable; RT-DETR validator does not use NMS'
    if row.get('val_images') == '':
        reasons['val_images'] = 'not applicable; original val23 merged into train130; no independent validation'
    row['missing_reasons'] = json.dumps(reasons, ensure_ascii=False, separators=(',', ':'))
    return {key: row.get(key, '') for key in FIELDS}


def scale1280_rows(template, anchor4=False, scale1600=False):
    """Read reviewed validation and any separately reviewed fixed-checkpoint test."""
    assert not (anchor4 and scale1600), 'Separate single-factor experiments'
    experiment = 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001' if anchor4 else 'FRCNN-R50FPN-SCALE1280-VAL-001'
    stage = 'step13' if anchor4 else 'step11'
    resolved = 'resolved_scale1280_anchor4_val_001.py' if anchor4 else 'resolved_scale1280_val_001.py'
    command = 'run_step13_train_anchor4.cmd' if anchor4 else 'run_step11_train_scale1280.cmd'
    if scale1600:
        experiment, stage = 'FRCNN-R50FPN-SCALE1600-VAL-001', 'step14'
        resolved, command = 'resolved_scale1600_val_001.py', 'run_step14_train_scale1600.cmd'
    scale = [2666, 1600] if scale1600 else [2133, 1280]
    bounds = f'{scale[0]}x{scale[1]} bounds'
    comparison = ('single-seed anchor scale [8] to [4] at unchanged Resize(2133,1280) vs SCALE1280-VAL-001; fresh same COCO; original train107/val23; no test evaluation' if anchor4 else 'single-seed input-resolution sensitivity vs VAL-001; same original train107/val23; no test evaluation')
    if scale1600:
        comparison = 'single-seed final input-resolution sensitivity: Resize(2133,1280) to (2666,1600) vs STEP11 SCALE1280-VAL-001; default anchor8 unchanged; fresh same COCO; original train107/val23; no test evaluation'
    folder = FRCNN / 'workspace/runs' / experiment
    rows, runs = [], []
    shared = '''framework framework_version source_commit python_version torch_version label_type class_mapping
deterministic amp optimizer gradient_accumulation precision_recall_rule evaluator_version
prediction_score_threshold nms_rule nms_iou max_detections coco_max_dets visualization_threshold'''.split()
    for run in sorted(folder.glob(stage + '_*')):
        if not run.is_dir():
            continue
        note = FRCNN / 'workspace/notes' / (experiment + '.md')
        entry = dict(experiment_id=experiment, run_id=run.name, model='Faster R-CNN R50-FPN',
            status='INCOMPLETE_NEEDS_REVIEW', train_images=107, val_images=23, test_images=23,
            epochs=None, batch=2, training_iterations=None, optimizer_updates=None, learning_rate=.0025,
            input_size=bounds, selected_epoch=None, test_status='NOT_RUN_DEVELOPMENT_VAL_ONLY', source=str(run), note=str(note))
        runs.append(entry)
        if not (run/'selected_checkpoint.json').exists():
            continue
        selection, pre, summary = (read_json(run/p) for p in ('selected_checkpoint.json','pretraining_record.json','artifacts/formal_training_summary.json'))
        assert summary['epochs'] == 100 and summary['optimizer_updates'] == 5400 and summary['resize_scale'] == scale
        assert not selection['test_evaluated'] and sha(selection['best_checkpoint']) == selection['best_checkpoint_sha256']
        assert pre['annotation_hashes']['val'] == sha(DATA/'03_coco_dataset/annotations/instances_val.json')
        review = read_json(FRCNN / 'workspace/audit' / (stage + '_review') / 'acceptance.json')
        assert review['status'] == 'PASS' and not review['visual_review_pending'] and Path(review['run_dir']) == run
        assert review['selected_checkpoint_sha256'] == selection['best_checkpoint_sha256']
        if anchor4:
            assert review['only_anchor_scale_scientific_diff'] and pre['anchor_scales'] == [4]
        if scale1600:
            assert review['only_resize_scientific_diff'] and pre['anchor_scales'] == [8]
        epoch = selection['selected_epoch']
        entry.update(status='COMPLETED_VAL_REVIEWED',epochs=100,training_iterations=5400,optimizer_updates=5400,selected_epoch=epoch)
        row = base_row(entry['experiment_id'],run,entry['model'],'val')
        row.update({k:template[k] for k in shared})
        row.update(evaluation_status='VERIFIED_RESULT_REVIEW',
            comparison_scope=comparison,
            pretrained_source=pre['checkpoint']['url'],pretrained_path=pre['checkpoint']['path'],pretrained_sha256=pre['checkpoint']['sha256'],
            pretrained_load_evidence=str(run/'artifacts/actual_training_weight_load.json'),
            resize_mode=f'Resize scale=({scale[0]},{scale[1]}), keep_ratio=True; pad to divisor32',input_size=bounds,
            batch=2,learning_rate=pre['lr'],training_iterations=5400,optimizer_updates=5400,iteration_count_status='5400 finite batch records verified',
            training_seconds=summary['wall_seconds_including_validation_checkpoints_snapshots'],training_time_scope='training + val + checkpoint + fixed-train snapshots',
            checkpoint_path=selection['best_checkpoint'],checkpoint_sha256=selection['best_checkpoint_sha256'],selected_epoch=epoch,
            selection_rule='max original val coco/bbox_mAP rounded to 3 decimals; first best on tie; no retrospective replacement',
            annotation_path=str(DATA/'03_coco_dataset/annotations/instances_val.json'),annotation_sha256=pre['annotation_hashes']['val'],
            config_source=str(run/resolved),config_sha256=sha(run/resolved),
            log_source=str(FRCNN/'workspace/logs'/run.name/(stage+'_console.txt')),original_log_source=str(FRCNN/'workspace/logs'/run.name/'train_console.log'),
            training_command_source=str(run/'code'/command),
            evaluation_command=f'native validation each epoch inside saved {stage.upper()} training command; exact recomputation uses saved predictions only',
            experiment_note=str(note),visualization_source=str(run/'artifacts'),evidence_bundle=str(run),
            source_verification=f'{stage.upper()} independent saved-prediction COCO, hashes/config/LR/selection and visual review passed; test not run')
        pred=Path(selection['validation_predictions'])
        native_source=run/f'artifacts/validation/epoch_{epoch:03d}/metrics.json'
        rows.append(coco_metric(row.copy(),read_json(native_source)['metrics'],None,native_source,pred,'selected_epoch_validation'))
        rows.append(coco_metric(row.copy(),None,selection['validation_metrics'],run/'selected_checkpoint.json',pred,'selected_validation_offline_exact'))
        rows += scale_test_rows(row, run, selection, pre, entry)
    return rows,runs


def scale_test_rows(template, run, selection, pre, entry):
    """Supplemental test never mutates the original training selection snapshot."""
    epoch = selection['selected_epoch']
    summaries = sorted((run / f'test_selected_epoch{epoch}').glob('*/test_summary.json'))
    assert len(summaries) <= 1, 'Multiple test results need explicit review'
    rows = []
    for path in summaries:
        dest = path.parent
        stage = dest.name.split('_')[0]
        review_path = FRCNN / 'workspace/audit' / f'{stage}_review' / 'acceptance.json'
        assert review_path.exists(), 'Supplemental test result review is required before registration'
        review, summary = read_json(review_path), read_json(path)
        assert review['status'] == summary['status'] == 'PASS' and not review['visual_review_pending']
        assert Path(review['result_dir']) == dest and summary['selected_epoch'] == epoch
        assert summary['images'] == 23 and summary['boxes'] == 36 and summary['broken'] == 8 and summary['loose'] == 28
        assert review['selected_checkpoint_sha256'] == summary['selected_checkpoint_sha256'] == selection['best_checkpoint_sha256']
        assert review['metrics'] == summary['metrics'] == read_json(dest / 'test_metrics.json')
        for source, digest in review['source_files_sha256'].items():
            assert sha(source) == digest, 'Reviewed supplemental test source changed'
        pretest = read_json(dest / 'pretest_record.json')
        manifest = read_json(dest / 'test_evidence_manifest.json')
        for source, digest in manifest['files_sha256'].items():
            assert sha(source) == digest, 'Supplemental test evidence changed'
        row = template.copy()
        row.update(evaluation_split='test', eval_images=23, eval_boxes=36, eval_broken=8, eval_loose=28,
            evaluation_status='VERIFIED_RESULT_REVIEW',
            comparison_scope=template['comparison_scope'].replace('no test evaluation', 'fixed selected checkpoint supplemental test; prior test exposure disclosed'),
            annotation_path=str(DATA/'03_coco_dataset/annotations/instances_test.json'), annotation_sha256=pre['annotation_hashes']['test'],
            label_files_manifest_sha256=label_manifest('test'),
            config_source=str(dest/'resolved_test_config.py'), config_sha256=sha(dest/'resolved_test_config.py'),
            log_source=str(FRCNN/'workspace/logs'/dest.name/(stage+'_console.txt')),
            evaluation_command=pretest['command_line'], visualization_source=str(dest/'predicted_images'), evidence_bundle=str(dest),
            source_verification=f'{stage.upper()} hashes/checkpoint/class/coordinate/saved-prediction COCO/visual review PASS; completion={summary.get("completion_mode", "USER_TEST_COMPLETED")}; original logs preserved')
        assert sha(row['annotation_path']) == row['annotation_sha256']
        predictions = dest/'predictions.bbox.json'
        assert len(read_json(dest/'predictions_details.json')) == 23 and len(list((dest/'predicted_images').glob('*.jpg'))) == 23
        native = summary['native_coco_metrics']
        if native:
            rows.append(coco_metric(row.copy(), native, None, dest/'native_coco_metrics.json', predictions, 'fixed_selected_test'))
        rows.append(coco_metric(row.copy(), None, summary['metrics'], dest/'test_metrics.json', predictions, 'fixed_test_offline_exact'))
        entry.update(status='COMPLETED_REVIEWED', test_status='COMPLETED_REVIEWED')
    return rows


def verify_book(plan):
    ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(BOOK) as z:
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        names = [s.attrib['name'] for s in wb.find('s:sheets', ns)]
        assert names == [s['name'] for s in plan['sheets']]
        styles = ET.fromstring(z.read('xl/styles.xml'))
        numfmts = {int(x.attrib['numFmtId']): x.attrib['formatCode'] for x in styles.find('s:numFmts', ns) or []}
        formats = [int(x.attrib.get('numFmtId', 0)) for x in styles.find('s:cellXfs', ns)]
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            strings = [''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        count = 0
        for index, sheet in enumerate(plan['sheets'], 1):
            xml = ET.fromstring(z.read(f'xl/worksheets/sheet{index}.xml'))
            cells = {c.attrib['r']: c for c in xml.findall('.//s:sheetData/s:row/s:c', ns)}
            for r, values in enumerate(sheet['rows'], 5):
                for col, expected in enumerate(values):
                    letter = column_name(col + 1)
                    cell = cells.get(f'{letter}{r}')
                    text = '' if cell is None else (cell.findtext('s:v', default='', namespaces=ns))
                    if cell is not None and cell.attrib.get('t') == 's':
                        text = strings[int(text)]
                    elif cell is not None and cell.attrib.get('t') == 'inlineStr':
                        text = ''.join(cell.find('s:is', ns).itertext())
                    if expected is None or expected == '':
                        assert text == '', (sheet['name'], r, letter, 'blank became nonblank', text)
                    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
                        assert cell is not None and cell.attrib.get('t', 'n') == 'n'
                        assert math.isclose(float(text), expected, rel_tol=1e-12, abs_tol=1e-14)
                        if col in sheet.get('ratio_columns', []):
                            fmt = numfmts.get(formats[int(cell.attrib.get('s', 0))], '')
                            assert '%' in fmt, (sheet['name'], r, letter, fmt)
                    else:
                        assert text == str(expected), (sheet['name'], r, letter, text, expected)
                    count += 1
        return count


def column_name(n):
    value = ''
    while n:
        n, rem = divmod(n - 1, 26)
        value = chr(65 + rem) + value
    return value


def make_plan(records, runs):
    events = {'formal_independent_evaluation':'正式独立评价', 'selected_epoch_validation':'训练中选中轮次的val',
              'automatic_best_checkpoint_revalidation':'训练结束后best重验证', 'selected_validation_offline_exact':'选中val预测的精确COCO复算',
              'fixed_selected_test':'固定选中checkpoint的test', 'fixed_test_offline_exact':'保存test预测的精确COCO复算',
              'fixed_last_epoch_test':'合并130训练后固定epoch100的test', 'fixed_last_test_offline_exact':'合并130的保存test预测精确COCO复算'}
    def brief_missing(row):
        missing = json.loads(row['missing_reasons'])
        parts = []
        if row['framework'] == 'Ultralytics':
            parts += ['AR100未计算', 'P/R最佳F1阈值数值未保存', '机器可读预测未保存']
        else:
            parts += ['普通P/R未计算']
            if 'coco_ar100' in missing:
                parts += ['AR100未在原生JSON输出']
        if 'broken_ap50' in missing:
            parts += ['分类AP50未在该来源保存']
        if 'broken_ap50_95' in missing:
            parts += ['分类AP50-95未在该来源保存']
        return '；'.join(parts)
    def value(row, key):
        if key == 'model' and row['experiment_id'] == 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001':
            return row['model'] + ' (anchor4)'
        if key == 'evaluation_event':
            return events.get(row[key], row[key])
        if key == 'missing_reasons':
            return brief_missing(row)
        v = row.get(key, '')
        if v == '':
            return None
        return float(v) if key in RATIO or key in ('learning_rate', 'training_seconds') else int(float(v)) if key in ('train_images', 'selected_epoch', 'eval_images', 'eval_boxes', 'eval_broken', 'eval_loose') else v
    def table(name, subtitle, headers, keys, data, ratios, widths):
        return dict(name=name, subtitle=subtitle, headers=headers,
                    rows=[[value(row, key) for key in keys] for row in data], ratio_columns=ratios, widths=widths)
    settings = dict(name='实验设置', subtitle='训练设置与完成状态。2700 batches 不等于实际优化器更新次数。',
        headers=['实验编号', '运行编号', '模型', '训练状态', 'train图片', 'val图片', 'test图片', 'epoch', 'batch', '训练迭代数', '优化器更新数', '基础学习率', '输入', '选定epoch', 'test状态'],
        rows=[[r.get(k) for k in ('experiment_id', 'run_id', 'model', 'status', 'train_images', 'val_images', 'test_images', 'epochs', 'batch', 'training_iterations', 'optimizer_updates', 'learning_rate', 'input_size', 'selected_epoch', 'test_status')] for r in runs],
        ratio_columns=[], widths=[36, 44, 28, 26, 12, 12, 12, 12, 12, 16, 18, 16, 23, 15, 36])
    for r in settings['rows']:
        if r[5] is None:
            r[5] = '不适用'
    headers = ['评价编号', '模型', '训练图片数', '输入', '评价事件', '选定epoch', '图片', '框', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'COCO AR100', '评价口径', '数值精度', '空值说明']
    keys = ['evaluation_id', 'model', 'train_images', 'input_size', 'evaluation_event', 'selected_epoch', 'eval_images', 'eval_boxes', 'map50', 'map50_95', 'precision', 'recall', 'coco_ar100', 'evaluation_protocol', 'metric_precision', 'missing_reasons']
    tables = [settings]
    for split, name in [('val', '验证集结果'), ('test', '测试集结果')]:
        tables.append(table(name, '比例底层为0~1，显示为百分数。test和val分开，复算不改变选模。', headers, keys,
                            [r for r in records if r['evaluation_split'] == split], [8, 9, 10, 11, 12], [26, 28, 15, 22, 48, 14, 12, 12, 16, 16, 16, 16, 17, 51, 55, 76]))
    class_rows = []
    for r in records:
        for cls in ('broken', 'loose'):
            class_rows.append([r['evaluation_id'], value(r, 'model'), r['evaluation_split'], events[r['evaluation_event']], cls,
                value(r, 'eval_' + cls)] + [value(r, cls + '_' + key) for key in ('ap50', 'ap50_95', 'precision', 'recall', 'ar100')] + [r['precision_recall_rule'], r['metric_precision']])
    tables.append(dict(name='分类别结果', subtitle='COCO class *_precision 为 AP50-95。普通P/R、AR100分别列示。',
        headers=['评价编号', '模型', '集合', '评价事件', '类别', '类别框数', 'AP50', 'AP50-95', 'Precision', 'Recall', 'AR100', 'P/R规则', '精度与缺项'],
        rows=class_rows, ratio_columns=[6, 7, 8, 9, 10], widths=[26, 28, 12, 48, 14, 15, 16, 16, 16, 16, 16, 70, 65]))
    tables.append(table('证据索引', '完整路径和哈希用于追溯。配置/版本/数据哈希等全部字段在统一CSV。',
        ['评价编号', '指标来源完整路径', '指标来源SHA256', '预测文件完整路径', '原始证据目录', '实验说明完整路径', '核查状态'],
        ['evaluation_id', 'metric_source', 'metric_source_sha256', 'predictions_source', 'evidence_bundle', 'experiment_note', 'source_verification'], records, [], [26, 90, 72, 90, 90, 85, 72]))
    rules = [
        ['统一CSV', str(CSV), '自动从原始证据更新；一行一评价编号；不手工修改数字'],
        ['测试待评价', 'FRCNN-R50FPN-VAL-001', '原23张test尚未运行；当前没有填0或创建虚假评价行'],
        ['MERGE130', 'FRCNN-R50FPN-MERGE130-001', '未开始；train130、无独立val、test23；同一COCO初始权重、固定100轮、选最后轮'],
        ['自检', 'FRCNN-DEBUG-OVERFIT-8', '诊断实验，不加入正式泛化CSV或val/test表'],
        ['自检原始目录', str(FRCNN / 'workspace/runs/FRCNN-DEBUG-OVERFIT-8/step06_24506_3904'), '仅用于训练流程诊断'],
        ['Precision/Recall', 'Ultralytics: IoU0.50，平滑平均F1最大处；阈值数值未保存', 'FRCNN普通P/R未计算，留空；AR100单独列示'],
        ['历史1600实验', '选中轮次 CSV 和自动 best.pt 重验证终端输出分别登记', '终端只保存显示精度；不拿较高结果替换较低结果'],
        ['原生比较', 'YOLO/RTDETR native AP vs COCO AP；输入缩放/后处理/最大检测数量不同', '只作参考对照；单种子不能宣称统计显著性'],
        ['失败或中断', '保留各运行目录及workspace/logs；不覆盖、不自动重训或重测', '只整理已有文件；缺证据明确标记'],
    ]
    if any(r['evaluation_split'] == 'test' and r['framework'] == 'MMDetection' for r in records):
        complete = any(r['experiment_id'] == 'FRCNN-R50FPN-VAL-001' and r['evaluation_split'] == 'test'
                       and r['evaluation_status'] == 'VERIFIED_RESULT_REVIEW' for r in records)
        rules[1][0] = '测试已完成并核查' if complete else '测试完成待复核'
        rules[1][2] = '固定epoch66 test原始输出/哈希/映射/坐标/预测指标及可视化已核查' if complete else '已保存并登记固定epoch66 test；等待用户反馈后的结果及可视化复核'
    if any(r['experiment_id'] == 'FRCNN-R50FPN-MERGE130-001' and r.get('run_id') for r in runs):
        rules[2][2] = '运行状态见实验设置；无独立val；train130、test23；固定epoch100；test计算后仍需复核'
    if any(r['experiment_id'] == 'FRCNN-R50FPN-MERGE130-001' and r['evaluation_status'] == 'VERIFIED_RESULT_REVIEW' for r in records):
        rules[2][2] = '训练及固定epoch100 test已完成并核查；train130、无独立val、test23；保存预测精确COCO复算一致；未据test调参'
    if any(r['experiment_id'] == 'FRCNN-R50FPN-SCALE1280-VAL-001' for r in records):
        rules.append(['输入尺寸对照','FRCNN-R50FPN-SCALE1280-VAL-001','107/23，(2133,1280)，100轮5400更新，val选第5轮；已核查；test未运行，不能填0'])
    if any(r['experiment_id'] == 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001' for r in records):
        rules.append(['anchor尺寸对照','FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001','固定(2133,1280)，anchor scale8→4，其余科学设置不变；107/23、100轮5400更新，val选第15轮；test未运行。属于诊断支持的消融，不是创新模块'])
    if any(r['experiment_id'] == 'FRCNN-R50FPN-SCALE1600-VAL-001' for r in records):
        rules.append(['最后一次尺寸对照','FRCNN-R50FPN-SCALE1600-VAL-001','相对STEP11默认anchor8，只改为(2666,1600)；107/23、100轮5400更新，val选第12轮，已核查；主要val AP未提高；test未运行；结束本轮尺寸尝试'])
    for rule in rules:
        complete = any(r['experiment_id'] == rule[1] and r['evaluation_split'] == 'test' and
                       r['evaluation_status'] == 'VERIFIED_RESULT_REVIEW' for r in records)
        if complete and rule[1].startswith('FRCNN-R50FPN-SCALE'):
            rule[2] = rule[2].replace('test未运行，不能填0', '固定val选中权重的test已完成并核查')
            rule[2] = rule[2].replace('test未运行', '固定val选中权重的test已完成并核查')
    rules.append(['补测顺序', '2026-10-14用户指定五项已有权重', '1280默认anchor第5轮→anchor4第15轮→1600默认anchor第12轮→YOLO1600第54轮→RTDETR1600第42轮；每项先验收登记再继续'])
    tables.append(dict(name='登记说明', subtitle='待评价状态与正式评价数据分开。历史文件保留。',
                       headers=['项目', '来源或规则', '状态或说明'], rows=rules, ratio_columns=[], widths=[26, 108, 85]))
    return dict(output_path=str(BOOK), preview_dir=str(AUDIT / 'workbook_previews'), sheets=tables)


def diagnostic_entries():
    entries = [dict(experiment_id='FRCNN-DEBUG-OVERFIT-8', status='DIAGNOSTIC_ONLY',
                    source=str(FRCNN / 'workspace/runs/FRCNN-DEBUG-OVERFIT-8/step06_24506_3904'))]
    path = FRCNN / 'workspace/audit/step12_review/acceptance.json'
    if path.exists():
        review = read_json(path)
        if review['status'] == 'PASS':
            entries.append(dict(experiment_id='FRCNN-DIAG-RPNROI-TRAIN8', status='DIAGNOSTIC_REVIEWED',
                run_id=Path(review['run_dir']).name, source=review['run_dir'], acceptance=str(path),
                note=str(FRCNN / 'workspace/notes/FRCNN-DIAG-RPNROI-TRAIN8.md'),
                evaluation_scope='original train8 / 15 GT; epoch100 fit diagnosis, not val/test',
                models=review['models']))
    return entries


def diagnostic_report_lines():
    path = FRCNN / 'workspace/audit/step12_review/acceptance.json'
    if not path.exists():
        return []
    a = read_json(path)
    if a['status'] != 'PASS':
        return []
    models = {m['label']: m for m in a['models']}
    old, new = models['baseline800'], models['scale1280']
    return ['## 固定训练图RPN与ROI诊断（不列入泛化成绩）', '',
        f'FRCNN-DIAG-RPNROI-TRAIN8/{Path(a["run_dir"]).name}已完成并复核。两份epoch100权重各在固定原train8/15GT上诊断，16份保存数组独立复算、权重哈希、FPN形状和原末轮快照一致性检查通过。没有训练更新、val/test推理，也未改变原val选中的epoch66/5。', '',
        f'按IoU≥0.50的候选覆盖诊断，800的RPN/过滤前同类ROI/最终同类框分别覆盖{old["rpn_covered"]}/{old["roi_before_score_covered"]}/{old["final_covered"]}个GT；1280分别为{new["rpn_covered"]}/{new["roi_before_score_covered"]}/{new["final_covered"]}个。它们是逐GT是否存在候选的覆盖计数，不是普通Recall或独立泛化指标。', '',
        '1280剩余GT87/123/124/160均为loose；RPN交给ROI的候选及ROI同类回归在过滤前最大IoU均低于0.50，不能只通过降低显示阈值解决。它们恰好对应默认anchor下零正样本目标，支持anchor几何匹配假设，但不是唯一原因的证明。RPN保存的是NMS/top1000后的输出，未恢复历史训练随机采样或全量特征图。', '',
        f'完整诊断说明：`{FRCNN / "workspace/notes/FRCNN-DIAG-RPNROI-TRAIN8.md"}`；复核：`{path}`。本诊断不向正式CSV/Excel新增AP记录；后续准备状态与实际成绩分开记录。', '']


def write_reports(records, runs):
    lines = ['# 输电导线缺陷检测实验持续汇总', '', '本文件由已保存原始指标生成。统一CSV为数值登记依据，Excel为显示视图。历史09报告与原四实验Excel保留。', '',
             '## 评价口径', '', 'val与test分别报告；107张训练与130张训练分别注明。当前各框架原生AP只作参考对照，输入缩放、增强、预训练、后处理和评价实现并不相同。单seed结果不能证明统计显著性。', '',
             'Ultralytics原生AP使用101点插值积分；普通P/R在IoU0.50、平滑平均分类F1最大处计算，实际阈值数值未保存。MMDetection使用pycocotools COCO AP，普通P/R未计算；AR100不能代替普通Recall。', '',
             'YOLO：等比例缩放和LetterBox；RT-DETR：直接方形缩放；Faster R-CNN：保持比例的Resize边界，原基线(1333,800)、尺寸敏感性实验(2133,1280)和(2666,1600)。YOLO NMS IoU0.7、max300；RT-DETR不使用NMS、max300；Faster R-CNN ROI NMS0.5、max100。评价预测过滤0.001与FRCNN展示阈值0.30分别记录。', '',
             '原生MMDetection输出保留三位小数，类别字段`*_precision`实际为AP50-95。保存预测的完整精度COCO复算另列，不改变已经固定的第66轮选模。历史1600实验选中轮次结果和自动重验证结果是两次评价，分别列示；终端输出不增加虚假小数位。', '']
    for split, title in [('val', '验证集结果（23图片、37框）'), ('test', '测试集结果（23图片、36框）')]:
        lines += [f'## {title}', '', '|评价编号|模型/输入|训练图片数|epoch|评价事件/口径|mAP50|mAP50-95|P|R|AR100|', '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
        for r in records:
            if r['evaluation_split'] != split:
                continue
            fmt = lambda k: f'{float(r[k])*100:.4f}%' if r[k] != '' else '未计算/未保存'
            lines.append(f'|{r["evaluation_id"]}|{r["model"]}/{r["input_size"]}|{r["train_images"]}|{r["selected_epoch"]}|{r["evaluation_event"]}; {r["evaluation_protocol"]}|{fmt("map50")}|{fmt("map50_95")}|{fmt("precision")}|{fmt("recall")}|{fmt("coco_ar100")}|')
        lines += ['']
    lines += ['## 分类别AP', '', '|评价编号|集合|broken AP50|broken AP50-95|loose AP50|loose AP50-95|', '|---|---|---:|---:|---:|---:|']
    for r in records:
        fmt = lambda k: f'{float(r[k])*100:.4f}%' if r[k] != '' else '该来源未保存'
        lines.append('|'+ '|'.join([r['evaluation_id'], r['evaluation_split']] + [fmt(k) for k in ('broken_ap50', 'broken_ap50_95', 'loose_ap50', 'loose_ap50_95')]) + '|')
    lines += ['', 'RT-DETR部分类别在原生P/R曲线插值中记录P=1、R=0。这是原生数值约定，不能解释为该类检测可靠；分类AP仍为0。', '']
    lines += ['', '## 训练与评价状态', '', '|实验|运行|train/val/test|迭代数/更新数|test状态|', '|---|---|---|---|---|']
    for r in runs:
        lines.append(f'|{r["experiment_id"]}|{r.get("run_id") or "尚未运行"}|{r["train_images"]}/{r.get("val_images") if r.get("val_images") is not None else "不适用"}/{r["test_images"]}|{r.get("training_iterations") or "未运行"}/{r.get("optimizer_updates") or "未保存或未运行"}|{r["test_status"]}|')
    lines += ['', '四次Ultralytics实验为2700训练batch。梯度累积和AMP使它们不等于2700优化器更新；实际更新数历史未保存。Faster R-CNN VAL-001为5400更新，MERGE130-001为6500更新。同样100epoch不代表相同计算量。训练时间和显存采集口径见原始说明，不做精确效率排名。', '',
              'YOLO1280和RTDETR1280的test各已使用一次；1600实验未做test。FRCNN VAL实验选模只依据val，后续测试分数不得用于挑epoch或调参。MERGE130从同一官方COCO初始权重重新训练，100epoch选最后轮；原val已进入train后不能作为独立val。MERGE130的固定epoch100 test已保存，核查状态见上表。', '',
              'DEBUG8为诊断实验。最终train-fit mAP50约63.93%、mAP50-95约53.51%，未完全过拟合；不能放入正式泛化排名。固定官方anchor下有部分小框无正RPN anchor，这是已观察到的流程限制，不能直接认定为正式AP低的唯一原因。', '',
              '## 已确认事实与解释限制', '', '当前Faster R-CNN选中epoch66，val精确COCO mAP50=3.7739%、mAP50-95=1.2734%，仍较低。损失下降不能直接证明泛化良好。不能把没有显示框解释成完全没有预测，须同时查看低阈值保存预测。数据量、小目标缩放、类别不平衡等是可能因素，现有结果不能确定单一原因。', '',
              '## 原始证据索引', '', f'- 数值CSV：`{CSV}`', f'- 总Excel：`{BOOK}`', f'- 登记验收：`{AUDIT / "registration_acceptance.json"}`', '']
    merge_review_path = FRCNN / 'workspace/audit/step10_review/acceptance.json'
    if any(r['experiment_id'] == 'FRCNN-R50FPN-SCALE1280-VAL-001' for r in records):
        a=read_json(FRCNN/'workspace/audit/step11_review/acceptance.json')
        m=a['selected_metrics']; b=a['baseline_selected_metrics']; insertion=lines.index('## 原始证据索引')
        lines[insertion:insertion]=['## 输入分辨率敏感性实验', '',
            f'FRCNN-R50FPN-SCALE1280-VAL-001保持107/23、COCO初始化及训练设置，改为Resize(2133,1280)。100轮5400更新，val选第{a["selected_epoch"]}轮。精确val mAP50={m["mAP50"]*100:.4f}%、mAP50-95={m["mAP50_95"]*100:.4f}%、AR100={m["AR100"]*100:.4f}%。较原基线mAP50变化{(m["mAP50"]-b["mAP50"])*100:+.4f}个百分点，mAP50-95变化{(m["mAP50_95"]-b["mAP50_95"])*100:+.4f}个百分点，broken AP仍为0。', '',
            f'末轮固定train8的mAP50-95由{a["baseline_train8_final"]["mAP50_95"]*100:.4f}%升至{a["train8_final"]["mAP50_95"]*100:.4f}%，这是训练拟合诊断，不能作为val或泛化成绩。验证最优出现在早期，之后下降；仅放大整图没有解决低泛化问题，不能据此确定唯一原因或宣布尺寸无影响。', '',
             '尺寸训练开发与选模依据val。用户随后指定固定第5轮补测；当前test完成状态见运行表与下方补测记录。旧test已被查看，不能称其为未接触盲测集。固定train8的RPN与ROI诊断见单独说明。', '']
    if any(r['experiment_id'] == 'FRCNN-R50FPN-MERGE130-001' and r['evaluation_status'] == 'VERIFIED_RESULT_REVIEW' for r in records):
        merge_review = read_json(merge_review_path)
        insertion = lines.index('## 原始证据索引')
        lines[insertion:insertion] = [
            'Faster R-CNN两次固定test均已完成并核查。107图训练、val选epoch66：精确COCO mAP50=0.082508%、mAP50-95=0.049505%、AR100=1.071429%；130图训练、固定epoch100：mAP50=0.028289%、mAP50-95=0.016973%、AR100=1.071429%。两次broken AP50/AP50-95均为0。MERGE130本次结果未显示泛化提升；训练规模、更新数及checkpoint选择规则不同，单seed不能证明合并训练必然降低效果。', '',
            f'MERGE130保存129个预测，score>0.30有6个；按现有标注，这6个与同类GT均未达到IoU0.50。36个GT中只有{merge_review["gt_with_same_class_candidate_iou_at_least_0_5"]}个有低阈值同类候选IoU≥0.50，这是候选覆盖诊断，不是普通Recall。原生三位小数指标保留为0.000，同时单列保存预测的精确COCO复算结果，不将舍入的0解释为精确全零。', '',
            f'最终结果核查：`{merge_review_path}`。原计划的DEBUG8、VAL-001和MERGE130-001已完成；没有据test成绩新增训练、挑epoch或调参。跨框架评价实现仍未统一，因此与前四次属于参考对照。', '',
        ]
    for r in records:
        lines += [f'### {r["evaluation_id"]}', '', f'- 指标：`{r["metric_source"]}`', f'- 指标SHA256：`{r["metric_source_sha256"]}`',
                  f'- checkpoint：`{r["checkpoint_path"]}`', f'- checkpoint SHA256：`{r["checkpoint_sha256"]}`',
                  f'- 预测：`{r["predictions_source"]}`' if r['predictions_source'] else '- 机器可读预测未保存。引用已有图片，不为填表重跑推理。',
                  f'- 配置：`{r["config_source"]}`', f'- 日志：`{r["log_source"]}`', f'- 说明：`{r["experiment_note"]}`', f'- 核查状态：{r["evaluation_status"]}', '']
    if any(r['experiment_id'] == 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001' for r in records):
        a = read_json(FRCNN / 'workspace/audit/step13_review/acceptance.json')
        m, b = a['selected_metrics'], a['baseline_selected_metrics']
        insertion = lines.index('## 原始证据索引')
        lines[insertion:insertion] = ['## 固定1280输入的anchor尺寸对照', '',
            f'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001仅把RPN anchor scale8改为4，其余实际配置及5400条学习率轨迹与SCALE1280对照一致，同一官方COCO重新初始化。100轮完成，val按原规则选第{a["selected_epoch"]}轮。精确val mAP50={m["mAP50"]*100:.4f}%（对照{b["mAP50"]*100:.4f}%）、mAP50-95={m["mAP50_95"]*100:.4f}%（对照{b["mAP50_95"]*100:.4f}%）、AR100={m["AR100"]*100:.4f}%；broken AP仍0。', '',
            f'固定train8末轮mAP50={a["train8_final"]["mAP50"]*100:.4f}%、mAP50-95={a["train8_final"]["mAP50_95"]*100:.4f}%，15/15GT存在IoU≥0.50同类框。此前4个loose训练漏检都获得准确的最终框，但这是107张训练后的8图拟合诊断，不能称全107张已完全过拟合，也不能声称恢复了原8图专用过拟合实验。', '',
            '本次支持anchor尺寸匹配曾限制部分小训练目标拟合；主要val mAP50-95未提升、mAP50上升，不能宣布泛化问题已解决，不能只取有利指标。扩大整图和缩小anchor是两个不同因素，不能把本次当作纯输入尺寸实验或论文创新模块。它提供消融/诊断证据，不是论文必做项目。', '',
             '当前停止自动追加尺寸、anchor或其他训练。用户随后授权固定第15轮补测test，完成状态见运行表。训练拟合与验证表现的差距仍需结合数据和标注标准解释；不把数据问题当成已经证实的结论。', '',
            f'说明及原始证据：`{FRCNN / "workspace/notes/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md"}`。', '']
    if any(r['experiment_id'] == 'FRCNN-R50FPN-SCALE1600-VAL-001' for r in records):
        a = read_json(FRCNN / 'workspace/audit/step14_review/acceptance.json')
        m, b = a['selected_metrics'], a['baseline_selected_metrics']
        insertion = lines.index('## 原始证据索引')
        lines[insertion:insertion] = ['## 最后一次整图尺寸对照：1600', '',
            f'FRCNN-R50FPN-SCALE1600-VAL-001/step14_3575_9657完成100轮5400更新，原107/23划分，同一官方COCO重新初始化。相对STEP11默认anchor8的1280对照，仅训练/val/快照Resize提高为(2666,1600)，其余配置和5400条学习率一致。按原val规则选第{a["selected_epoch"]}轮；不与STEP13的anchor4混称尺寸单因素比较。', '',
            f'精确val mAP50={m["mAP50"]*100:.4f}%（1280为{b["mAP50"]*100:.4f}%），主要mAP50-95={m["mAP50_95"]*100:.4f}%（1280为{b["mAP50_95"]*100:.4f}%），AR100={m["AR100"]*100:.4f}%。宽松IoU=0.50下有收益，主要AP未改善，不能仅挑有利指标。broken AP50={m["per_class"]["broken"]["AP50"]*100:.6f}%、AP50-95={m["per_class"]["broken"]["AP50_95"]*100:.6f}%，原生类别AP显示0.000只是舍入，精确值并非零。', '',
            f'末轮固定train8 mAP50-95={a["train8_final"]["mAP50_95"]*100:.4f}%（1280为{a["baseline_train8_final"]["mAP50_95"]*100:.4f}%），15/15GT有IoU≥0.50同类候选；仅为训练子集拟合，不是全107图完全过拟合或泛化成绩。早期val较好、后期降低，训练/val差距仍大。', '',
            f'耗时{a["wall_seconds"]/60:.2f}分钟，峰值CUDA allocated {a["peak_cuda_allocated_mib"]/1024:.2f}GiB，无OOM或中断。选定val保存2021个低阈值框、17张达到100框上限；显示>0.30有10框，一对一同类IoU≥0.50匹配TP3/FP7/未匹配GT34，仅作显示阈值描述，不改变AP或选模。', '',
             '800→1280→1600默认anchor系列未显示主要val AP随尺寸持续提升，未解决低泛化问题；不能据此断言尺寸无影响或低分只有一个原因。本轮尺寸训练已结束，不追加训练。用户随后授权固定第12轮补测test。单seed、23张val且多轮开发、旧test已被查看的限制仍保留。', '',
             f'本次说明：`{FRCNN / "workspace/notes/FRCNN-R50FPN-SCALE1600-VAL-001.md"}`。test状态见运行表，未完成项留空。', '']
    supplemental = [r for r in records if r['experiment_id'].startswith('FRCNN-R50FPN-SCALE') and
                    r['evaluation_split'] == 'test' and r['evaluation_protocol'] == 'pycocotools-exact-from-saved-predictions']
    if supplemental:
        insertion = lines.index('## 原始证据索引')
        added = ['## 固定已有权重的最终test补齐', '',
                 '按用户指定顺序补测，每项保持已固定的val选模权重与评价配置。每次先核查并登记，再进行下一项；不根据test改选epoch或阈值。原test已有历史使用，不能重新宣称盲测。', '']
        for r in supplemental:
            s = read_json(Path(r['evidence_bundle'])/'test_summary.json')
            added += [f'{r["experiment_id"]}，第{r["selected_epoch"]}轮，原23图test：mAP50={float(r["map50"])*100:.6f}%、mAP50-95={float(r["map50_95"])*100:.6f}%、COCO AR100={float(r["coco_ar100"])*100:.6f}%。保存{s["returned_predictions"]}个低阈值预测，score>0.30显示{s["visible_predictions_above_0_3"]}个；没有显示框不能解释为零预测。', '',
                      f'核查状态：{r["evaluation_status"]}；完整结果：`{r["evidence_bundle"]}`。', '']
            if s.get('completion_mode') == 'RECOVERED_SAVED_RESULTS_ONLY':
                added += ['原测试在23图预测和指标已保存后，因哈希函数收到字符串路径而在末尾归档报错。已仅用保存预测完成复核和补齐归档，没有再次模型推理；原FAIL日志保留。该次未落盘的推理耗时/显存留空，不估算或填0。', '']
        lines[insertion:insertion] = added
    insertion = lines.index('## 原始证据索引')
    lines[insertion:insertion] = diagnostic_report_lines()
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    formal_note = Path(next(r['note'] for r in runs if r['experiment_id'] == 'FRCNN-R50FPN-VAL-001'))
    vrows = [r for r in records if r['experiment_id'] == 'FRCNN-R50FPN-VAL-001']
    text = ['# FRCNN-R50FPN-VAL-001', '', '目的：官方Faster R-CNN R50-FPN在固定107/23/23导线缺陷数据上的参考基线。', '',
            f'运行编号：`{FRUN.name}`；原始目录：`{FRUN}`。', '',
            '训练107图/168框（broken39、loose129）；val23图/37框（9、28）；test23图/36框（8、28）。沿用原图片级划分。YOLO0/1 → COCO1/2 → 模型0/1；num_classes=2。数据和划分哈希登记于CSV及pretraining_record。', '',
            '实际设置：100epoch、batch2、workers0、seed42、FP32、SGD lr0.0025/momentum0.9/decay0.0001；warmup500 iter、start_factor0.001；milestones67/92、gamma0.1。Resize边界(1333,800)、keep_ratio=True、flip0.5。实际5400更新，训练含验证/保存/固定训练图预测耗时2193.604秒，峰值allocated约3769.94MiB。', '',
            '官方完整COCO checkpoint加载：348项中344项一致；ROI fc_cls weight/bias和fc_reg weight/bias因80→2类重新初始化。其余backbone、FPN、RPN和ROI共享层确实加载。初始权重路径、URL、哈希见pretraining_record及CSV。', '',
            '选模规则：原val native coco/bbox_mAP的三位小数最高值，同值保留首次；epoch66已固定。完整精度复算仅核查，不更换checkpoint。', '',
            '## 已保存评价', '', '|集合|事件|口径|mAP50|mAP50-95|broken AP50/50-95|loose AP50/50-95|AR100|', '|---|---|---|---:|---:|---|---|---:|']
    for r in vrows:
        fmt = lambda k: f'{float(r[k])*100:.4f}%' if r[k] != '' else '未保存/未计算'
        text.append(f'|{r["evaluation_split"]}|{r["evaluation_event"]}|{r["evaluation_protocol"]}|{fmt("map50")}|{fmt("map50_95")}|{fmt("broken_ap50")}/{fmt("broken_ap50_95")}|{fmt("loose_ap50")}/{fmt("loose_ap50_95")}|{fmt("coco_ar100")}|')
    has_test = any(r['evaluation_split'] == 'test' for r in vrows)
    reviewed_test = any(r['evaluation_status'] == 'VERIFIED_RESULT_REVIEW' for r in vrows)
    text += ['', 'test状态：'+ ('已完成并核查原始指标、哈希、映射、坐标、保存预测及可视化。' if reviewed_test else '计算完成、已登记，等待用户反馈后复核结果与可视化。' if has_test else '尚未运行，保持待评价；没有填0。'), '',
             '## 异常、可视化与解释', '', '5400批次总损失及RPN/ROI分量有限，首100次总损失均值约0.35665，末100次约0.01357；val AP仍低。原训练已完成，无需为了整理再运行。', '',
             'checkpoint首次受限读取拒绝HistoryBuffer元数据，STEP08用限定可信类型的临时safe_globals解决，348/348严格CPU加载通过，无缺项和尺寸不匹配，未进行额外测试推理。早期CUDA脚本将正确nvcc12.8.93误写为12.8.90导致假失败，已更正。', '',
             '已保存训练损失/学习率图、原图GT和管线图，以及固定train8图片epoch0/25/50/75/100预测。正式test可视化阈值0.30，AP预测过滤0.001；显示无框不能证明零预测。', '',
             '已确认事实：val mAP50约3.7739%、mAP50-95约1.2734%，性能仍低。尚待验证原因：数据量、目标尺寸、缩放/增强及anchor匹配等；不将这些推测当成实验已经证明的结论。', '',
             '比较限制：框架AP实现、输入、后处理、maxDet、优化器及batch不同；100epoch计算量不同。参考对照，单seed不做显著性结论。train-fit自检不作为泛化指标。', '',
             '## 完整路径索引', '', f'- 训练配置：`{FRUN / "resolved_val_001.py"}`', f'- 代码/版本/哈希：`{FRUN / "pretraining_record.json"}`', f'- 依赖：`{FRUN / "requirements_freeze.txt"}`',
             f'- 实际预训练加载：`{FRUN / "artifacts/actual_training_weight_load.json"}`', f'- 损失及学习率：`{FRUN / "artifacts/loss_curves.png"}`', f'- 完整损失：`{FRUN / "artifacts/losses.csv"}`',
             f'- 选中权重记录：`{FRUN / "selected_checkpoint.json"}`', f'- 验证预测：`{FRUN / "artifacts/validation/epoch_066/predictions.bbox.json"}`',
             f'- val原生指标：`{FRUN / "artifacts/validation/epoch_066/metrics.json"}`', f'- val完整精度：`{FRUN / "artifacts/best_validation_metrics.json"}`',
             f'- 训练日志：`{FRCNN / "workspace/logs" / FRUN.name}`', f'- test结果父目录：`{FRUN / "test_selected_epoch66"}`',
             f'- 数据转换核查：`{DATA / "03_coco_dataset/audit"}`', f'- 管线核查：`{FRCNN / "workspace/notes/STEP05_debug8_static_check.txt"}`',
             f'- 统一CSV：`{CSV}`', f'- 汇总：`{REPORT}`', '',
             '核查状态：训练完成和val指标、选中权重哈希已核验；test '+('已完成并核查。' if reviewed_test else '已自动登记、待结果复核。' if has_test else '待评价。')]
    if reviewed_test:
        text += ['', 'test事实：123个返回框、score>0.30仅1个。cond_000031的loose框score=0.6756，与同类GT最大IoU=0，按现有标注为误检。36个GT中仅1个有同类候选IoU≥0.50；此项是候选覆盖诊断，不作为普通Recall。test精确mAP50=0.082508%、mAP50-95=0.049505%、AR100=1.071429%；broken AP=0。', '',
                 f'结果核查：`{FRCNN / "workspace/audit/step08_review/acceptance.json"}`；画框总览：`{FRCNN / "workspace/audit/step08_review/test_gt_and_visible_predictions.jpg"}`。', '',
                 'test泛化性能很低。现有检查通过不等于已排除所有可能问题，也不能确定低分的唯一原因。MERGE130沿用训练前固定设置，不依据本次test调参。']
    formal_note.write_text('\n'.join(text) + '\n', encoding='utf-8')


def write_merged_note(runs, records):
    path = FRCNN / 'workspace/notes/FRCNN-R50FPN-MERGE130-001.md'
    entries = [r for r in runs if r['experiment_id'] == 'FRCNN-R50FPN-MERGE130-001']
    lines = ['# FRCNN-R50FPN-MERGE130-001', '',
        '目的：执行原train107与val23合并为130图的既定实验。该设置在VAL-001 test前已固定。', '',
        '训练130图/205框（broken48、loose157），原test23图/36框（8、28）保留。没有独立val；原val已经成为训练数据，不能报告其独立验证成绩。YOLO0/1→COCO1/2→模型0/1（broken/loose）。', '',
        '固定设置：同一官方完整COCO初始checkpoint重新开始，不接着VAL-001私人数据checkpoint、不resume。100epoch、batch2、workers0、FP32、seed42、SGD0.0025/0.9/0.0001、warmup500、milestones67/92、gamma0.1；Resize边界(1333,800)保持比例、flip0.5、官方anchor。每轮65更新，共6500；实际运行状态及核查记录见下方。', '',
        'checkpoint规则：固定最后epoch100，不按val/test或固定train8拟合分数挑epoch。本训练阶段不做test。原test已有前两次Ultralytics基础实验和FRCNN VAL-001的使用记录，后续分数不得用于调参。', '',
        '## 运行记录', '']
    for r in entries:
        lines += [f'- 运行编号：`{r.get("run_id") or "尚未启动"}`', f'- 状态：{r["status"]}', f'- 原始目录：`{r["source"]}`']
        if r.get('run_id'):
            source = Path(r['source'])
            lines += [f'- 完整日志：`{FRCNN / "workspace/logs" / r["run_id"] / "step09_console.txt"}`',
                      f'- 配置：`{source / "resolved_merge130_001.py"}`', f'- 代码/环境/命令/数据哈希/初始权重来源哈希：`{source / "pretraining_record.json"}`']
            summary = source / 'artifacts/formal_training_summary.json'
            if summary.exists():
                stats = read_json(summary)
                lines += [f'- 实际训练：{stats["epochs"]}轮，{stats["optimizer_updates"]}更新；训练含保存/快照耗时{stats["wall_seconds_including_checkpoints_snapshots"]:.3f}秒；allocated峰值{stats["observed_peak_cuda_allocated_mib"]:.2f}MiB。',
                          f'- 完整COCO加载证据：`{source / "artifacts/actual_training_weight_load.json"}`',
                          f'- 损失/LR曲线：`{source / "artifacts/loss_curves.png"}`', f'- 逐次loss：`{source / "artifacts/losses.csv"}`',
                          f'- epoch0/25/50/75/100固定train8预测：`{source / "artifacts"}`', f'- 最后一轮权重/哈希：`{source / "selected_checkpoint.json"}`']
        lines += ['']
    lines += ['## 评价及核查状态', '', 'val：不适用。test：待评价，当前不填0、没有虚假评价行。每类AP和AR100待epoch100的固定test评价。训练完成后先核查损失、初始化、6500次更新和最后权重，再准备test。', '',
              '异常及可视化：当前训练未运行，暂无结果发现；未来以实际日志和保存图片填写，不把计划写成已完成事实。', '',
              '解释与比较限制：训练规模由107变130，更新数由5400变6500；无独立val。不能视为严格同规模单变量对照，也不能保证合并数据提高性能。单seed不做统计显著性结论。', '',
              f'- 阶段命令与文件：`{FRCNN / "workspace/notes/STEP09_merge130_training.txt"}`', f'- 统一CSV：`{CSV}`', f'- 总Excel：`{BOOK}`', f'- 汇总：`{REPORT}`']
    if any(r.get('run_id') for r in entries):
        lines = ['异常及可视化：训练状态及产物已登记；待用户提供终端输出后核查损失和图片，不能以保存成功直接证明拟合成功。'
                 if line.startswith('异常及可视化：') else line for line in lines]
    review_path = FRCNN / 'workspace/audit/step09_review/acceptance.json'
    if review_path.exists():
        review = read_json(review_path)
        if review['status'] == 'PASS':
            lines += ['', '## 训练核查', '',
                f'100轮/6500更新的损失及学习率核查通过。首100次总loss均值{review["first_100_mean_loss"]:.6f}，末100次{review["last_100_mean_loss"]:.6f}。344个兼容COCO张量加载与4个类别输出层重新初始化已确认，未接着VAL-001 checkpoint训练。', '',
                '损失/LR曲线已查看，总损失下降，训练快照部分图片出现高置信框，仍有图片在0.30下无可见框。固定train8的最终拟合mAP50约60.396%、mAP50-95约59.777%，只作过程诊断，不能作为130图整体或泛化成绩。终端ROI acc不能当作缺陷检测Recall。', '',
                f'核查证据：`{review_path}`。训练流程验收通过；最终test成绩尚需计算，不能提前判断模型泛化改善。']
            lines = ['异常及可视化：本次100轮训练已完成并核查，未发现非有限训练损失；已检查损失/LR曲线和固定训练图片快照。'
                     if line.startswith('异常及可视化：') else line for line in lines]
    test_rows = [r for r in records if r['experiment_id'] == 'FRCNN-R50FPN-MERGE130-001']
    if test_rows:
        reviewed_test = any(r['evaluation_status'] == 'VERIFIED_RESULT_REVIEW' for r in test_rows)
        test_status = ('val：不适用。test：已完成并核查指标、checkpoint/文件哈希、类别映射、原图坐标、保存预测及可视化。'
                       if reviewed_test else 'val：不适用。test：已计算并登记，待用户反馈后核查原始预测、指标和可视化。')
        lines = [test_status if line.startswith('val：不适用。test：') else line for line in lines]
        lines = [line.replace('最终test成绩尚需计算，不能提前判断模型泛化改善。',
                             '固定epoch100 test已完成并核查，未根据test改变训练设置或checkpoint。' if reviewed_test
                             else 'test已计算并登记，结果解释等待原始输出和可视化核查。') for line in lines]
        lines += ['', '## 固定epoch100测试结果', '', '|评价编号|口径|mAP50|mAP50-95|AR100|broken AP50/AP50-95|loose AP50/AP50-95|', '|---|---|---:|---:|---:|---|---|']
        for row in test_rows:
            fmt = lambda key: f'{float(row[key])*100:.6f}%' if row[key] != '' else '未保存/未计算'
            lines.append(f'|{row["evaluation_id"]}|{row["evaluation_protocol"]}|{fmt("map50")}|{fmt("map50_95")}|{fmt("coco_ar100")}|{fmt("broken_ap50")}/{fmt("broken_ap50_95")}|{fmt("loose_ap50")}/{fmt("loose_ap50_95")}|')
        lines += ['']
        for row in test_rows:
            lines += [f'指标（{row["evaluation_id"]}）：`{row["metric_source"]}`；预测：`{row["predictions_source"]}`；状态：{row["evaluation_status"]}。', '']
        if reviewed_test:
            review_path = FRCNN / 'workspace/audit/step10_review/acceptance.json'
            review = read_json(review_path)
            destination = Path(review['result_dir'])
            lines += ['', '## 最终测试核查与解释', '',
                '原test23图/36框的129个预测已核查。2张在score>0.001后无返回框；score>0.30只有6个显示框，分布在4张图。6个显示框与同类GT均未达到IoU0.50；cond_000026的broken最大同类IoU约0.184，其余5个为0，按现有标注为误检。不能把未画出框解释成完全没有预测。', '',
                f'36个GT中{review["gt_with_same_class_candidate_iou_at_least_0_5"]}个有同类候选IoU≥0.50；这只是候选覆盖，不是普通Recall或一对一检出率。普通Precision/Recall本轮未计算，不填0；COCO AR100单列。', '',
                '精确COCO mAP50=0.028289%、mAP50-95=0.016973%、AR100=1.071429%；broken AP50/AP50-95均为0，loose分别0.056577%/0.033946%。原生mAP输出0.000属于三位小数舍入，保存预测复算精确值非全零；两种口径均保留，不改变权重选择。', '',
                '与107图训练、val选epoch66的原test精确结果0.082508%/0.049505%相比，本次AP数值更低，AR100相同。本次合并未显示泛化提升；训练规模、更新数、checkpoint规则不同且仅一个seed，不能推断合并数据普遍降低性能。', '',
                '流程核查通过，最终泛化表现很差，两者应分别表述。数据量、小目标缩放、anchor匹配、类别不平衡等只能列为尚待验证的可能因素，现有证据无法确定唯一原因。没有依据test新增训练、挑epoch或调整参数。', '',
                f'- 完整测试结果目录：`{destination}`', f'- 原生指标：`{destination / "native_coco_metrics.json"}`',
                f'- 精确指标：`{destination / "test_metrics.json"}`', f'- 预测坐标、类别及置信度：`{destination / "predictions_details.json"}`',
                f'- COCO预测：`{destination / "predictions.bbox.json"}`', f'- 23张测试可视化：`{destination / "predicted_images"}`',
                f'- 测试命令/环境/数据和选择哈希：`{destination / "pretest_record.json"}`',
                f'- 完整测试终端日志：`{FRCNN / "workspace/logs" / destination.name / "step10_console.txt"}`',
                f'- 结果核查：`{review_path}`', f'- GT与已保存预测总览：`{review_path.parent / "test_gt_and_visible_predictions.jpg"}`',
                f'- 离线独立COCO核对：`{review_path.parent / "independent_coco_check.txt"}`', '',
                '核查状态：COMPLETED_REVIEWED。原计划的两个正式Faster R-CNN实验及小样本诊断已完成；无需为了登记重新训练或推理。跨框架仍按原生口径参考比较，未统一评价的限制继续保留。']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    assert Path(sys.executable).resolve() == (FRCNN / '.venv/Scripts/python.exe').resolve(), 'Use dedicated Faster R-CNN interpreter'
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    rows, runs = legacy_rows()
    more, run = faster_rows()
    rows += more
    runs.append(run)
    runs += merged_runs()
    rows += merged_test_rows(runs, more[0])
    scale_rows, scale_runs = scale1280_rows(more[0])
    rows += scale_rows
    runs += scale_runs
    anchor_rows, anchor_runs = scale1280_rows(more[0], anchor4=True)
    rows += anchor_rows
    runs += anchor_runs
    larger_rows, larger_runs = scale1280_rows(more[0], scale1600=True)
    rows += larger_rows
    runs += larger_runs
    rows = [finish_row(r) for r in rows]
    assert len(set(r['evaluation_id'] for r in rows)) == len(rows), 'Duplicate evaluation identity'
    if CSV.exists():
        existing = read_csv(CSV)
        assert set(r['evaluation_id'] for r in existing) <= set(r['evaluation_id'] for r in rows), 'Unrecognized existing evaluations; refuse to drop records'
        for prior in existing:
            now = next(r for r in rows if r['evaluation_id'] == prior['evaluation_id'])
            assert prior['metric_source_sha256'] == now['metric_source_sha256'], 'A previously indexed source changed'
    for row in rows:
        for field in ('checkpoint_path', 'metric_source', 'config_source', 'log_source', 'training_command_source',
                      'visualization_source', 'predictions_source', 'pretrained_path', 'pretrained_load_evidence'):
            if row.get(field):
                assert Path(row[field]).exists(), f'Evidence path missing: {field}: {row[field]}'
    with CSV.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    # Read back the canonical CSV: every downstream view uses these values.
    records = read_csv(CSV)
    plan = make_plan(records, runs)
    save_json(AUDIT / 'workbook_plan.json', plan)
    save_json(AUDIT / 'run_status_index.json', dict(runs=runs, diagnostics=diagnostic_entries(),
        log_manifests=[str(p) for p in sorted((FRCNN / 'workspace/logs').glob('*/step*manifest.json'))]))
    write_reports(records, runs)
    write_merged_note(runs, records)
    print(f'RECORDS_EXTRACTED={len(records)} unique_evaluations; no inference/training', flush=True)
    if args.extract_only:
        return 0
    subprocess.run([str(NODE), str(BASE / 'scripts/build_experiment_workbook.mjs')], check=True)
    cells = verify_book(plan)
    merge_pending = []
    for entry in runs:
        if entry['experiment_id'] != 'FRCNN-R50FPN-MERGE130-001':
            continue
        if entry['test_status'] == 'COMPLETED_REVIEWED':
            continue
        if entry['test_status'] == 'COMPLETED_AWAITING_REVIEW':
            merge_pending.append('MERGE130 test result/visual review pending')
        elif entry['status'] == 'TRAINING_REVIEWED_TEST_PENDING':
            merge_pending.append('MERGE130 final epoch100 test pending')
        else:
            merge_pending.append('MERGE130 training review/test pending')
    receipt = dict(status='PASS', updated_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_rows=len(records), unique_evaluation_ids=len(records), verified_exported_cells=cells,
        csv_sha256=sha(CSV), excel_sha256=sha(BOOK), report_sha256=sha(REPORT),
        ratios_stored_0_to_1=True, percent_display_checked=True, blanks_not_replaced_by_zero=True,
        historical_sources_preserved=True, training_started=False, model_inference_performed=False,
        pending=([] if run['test_status'] == 'COMPLETED_REVIEWED' else ['FRCNN original test pending' if run['test_status'] == 'PENDING' else 'FRCNN test result/visual review pending']) +
                merge_pending + [r['experiment_id'] + ' fixed checkpoint test pending/review pending'
                                 for r in scale_runs + anchor_runs + larger_runs if r['test_status'] != 'COMPLETED_REVIEWED'] +
                ['YOLO11s1600 epoch54 and RT-DETR-L1600 epoch42 fixed best.pt test pending'] +
                ['Cross-framework unified evaluation not performed', 'Legacy raw predictions / actual optimizer update counts not saved'])
    save_json(AUDIT / 'registration_acceptance.json', receipt)
    # User requested a separate concise teacher view; keep it derived from the same CSV.
    if (BASE / 'scripts/prepare_teacher_simple.py').exists():
        subprocess.run([sys.executable, '-I', '-X', 'utf8', str(BASE / 'scripts/prepare_teacher_simple.py')], check=True)
        subprocess.run([str(NODE), str(BASE / 'scripts/build_teacher_simple.mjs')], check=True)
        subprocess.run([sys.executable, '-I', '-X', 'utf8', str(BASE / 'scripts/verify_teacher_simple.py')], check=True)
    print(f'EXPERIMENT_RECORDS_PASS: CSV/Excel matched {cells} cells; unique evaluation IDs; ratio/blank checks passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
