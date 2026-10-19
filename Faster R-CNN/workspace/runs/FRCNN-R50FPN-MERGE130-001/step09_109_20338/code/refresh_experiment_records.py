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
        entries.append(entry)
    if not entries:
        entries.append(dict(experiment_id='FRCNN-R50FPN-MERGE130-001', run_id=None, model='Faster R-CNN R50-FPN',
            status='PREPARED_NOT_STARTED', train_images=130, val_images=None, test_images=23, epochs=100, batch=2,
            training_iterations=None, optimizer_updates=None, learning_rate=0.0025, input_size='1333x800 bounds',
            selected_epoch=None, test_status='NOT_STARTED', source=str(folder), note=str(notes)))
    return entries


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
    row['missing_reasons'] = json.dumps(reasons, ensure_ascii=False, separators=(',', ':'))
    return {key: row.get(key, '') for key in FIELDS}


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
              'fixed_selected_test':'固定选中checkpoint的test', 'fixed_test_offline_exact':'保存test预测的精确COCO复算'}
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
        if key == 'evaluation_event':
            return events.get(row[key], row[key])
        if key == 'missing_reasons':
            return brief_missing(row)
        v = row.get(key, '')
        if v == '':
            return None
        return float(v) if key in RATIO or key in ('learning_rate', 'training_seconds') else int(float(v)) if key in ('selected_epoch', 'eval_images', 'eval_boxes', 'eval_broken', 'eval_loose') else v
    def table(name, subtitle, headers, keys, data, ratios, widths):
        return dict(name=name, subtitle=subtitle, headers=headers,
                    rows=[[value(row, key) for key in keys] for row in data], ratio_columns=ratios, widths=widths)
    settings = dict(name='实验设置', subtitle='已完成训练和下一阶段计划。2700 batches 不等于实际优化器更新次数。',
        headers=['实验编号', '运行编号', '模型', '训练状态', 'train图片', 'val图片', 'test图片', 'epoch', 'batch', '训练迭代数', '优化器更新数', '基础学习率', '输入', '选定epoch', 'test状态'],
        rows=[[r.get(k) for k in ('experiment_id', 'run_id', 'model', 'status', 'train_images', 'val_images', 'test_images', 'epochs', 'batch', 'training_iterations', 'optimizer_updates', 'learning_rate', 'input_size', 'selected_epoch', 'test_status')] for r in runs],
        ratio_columns=[], widths=[36, 44, 28, 26, 12, 12, 12, 12, 12, 16, 18, 16, 23, 15, 36])
    for r in settings['rows']:
        if r[5] is None:
            r[5] = '不适用'
    headers = ['评价编号', '模型', '输入', '评价事件', '选定epoch', '图片', '框', 'mAP50', 'mAP50-95', 'Precision', 'Recall', 'COCO AR100', '评价口径', '数值精度', '空值说明']
    keys = ['evaluation_id', 'model', 'input_size', 'evaluation_event', 'selected_epoch', 'eval_images', 'eval_boxes', 'map50', 'map50_95', 'precision', 'recall', 'coco_ar100', 'evaluation_protocol', 'metric_precision', 'missing_reasons']
    tables = [settings]
    for split, name in [('val', '验证集结果'), ('test', '测试集结果')]:
        tables.append(table(name, '比例底层为0~1，显示为百分数。test和val分开，复算不改变选模。', headers, keys,
                            [r for r in records if r['evaluation_split'] == split], [7, 8, 9, 10, 11], [26, 28, 22, 48, 14, 12, 12, 16, 16, 16, 16, 17, 51, 55, 76]))
    class_rows = []
    for r in records:
        for cls in ('broken', 'loose'):
            class_rows.append([r['evaluation_id'], r['model'], r['evaluation_split'], events[r['evaluation_event']], cls,
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
        complete = any(r['evaluation_status'] == 'VERIFIED_RESULT_REVIEW' for r in records)
        rules[1][0] = '测试已完成并核查' if complete else '测试完成待复核'
        rules[1][2] = '固定epoch66 test原始输出/哈希/映射/坐标/预测指标及可视化已核查' if complete else '已保存并登记固定epoch66 test；等待用户反馈后的结果及可视化复核'
    if any(r['experiment_id'] == 'FRCNN-R50FPN-MERGE130-001' and r.get('run_id') for r in runs):
        rules[2][2] = '运行状态见实验设置；无独立val；train130、test23；固定epoch100；本阶段不测试'
    tables.append(dict(name='登记说明', subtitle='待评价状态与正式评价数据分开。历史文件保留。',
                       headers=['项目', '来源或规则', '状态或说明'], rows=rules, ratio_columns=[], widths=[26, 108, 85]))
    return dict(output_path=str(BOOK), preview_dir=str(AUDIT / 'workbook_previews'), sheets=tables)


def write_reports(records, runs):
    lines = ['# 输电导线缺陷检测实验持续汇总', '', '本文件由已保存原始指标生成。统一CSV为数值登记依据，Excel为显示视图。历史09报告与原四实验Excel保留。', '',
             '## 评价口径', '', 'val与test分别报告；107张训练与未来130张训练分别注明。当前各框架原生AP只作参考对照，输入缩放、增强、预训练、后处理和评价实现并不相同。单seed结果不能证明统计显著性。', '',
             'Ultralytics原生AP使用101点插值积分；普通P/R在IoU0.50、平滑平均分类F1最大处计算，实际阈值数值未保存。MMDetection使用pycocotools COCO AP，普通P/R未计算；AR100不能代替普通Recall。', '',
             'YOLO：等比例缩放和LetterBox；RT-DETR：直接方形缩放；Faster R-CNN：Resize(1333,800)边界并保持比例。YOLO NMS IoU0.7、max300；RT-DETR不使用NMS、max300；Faster R-CNN ROI NMS0.5、max100。评价预测过滤0.001与FRCNN展示阈值0.30分别记录。', '',
             '原生MMDetection输出保留三位小数，类别字段`*_precision`实际为AP50-95。保存预测的完整精度COCO复算另列，不改变已经固定的第66轮选模。历史1600实验选中轮次结果和自动重验证结果是两次评价，分别列示；终端输出不增加虚假小数位。', '']
    for split, title in [('val', '验证集结果（23图片、37框）'), ('test', '测试集结果（23图片、36框）')]:
        lines += [f'## {title}', '', '|评价编号|模型/输入|epoch|评价事件/口径|mAP50|mAP50-95|P|R|AR100|', '|---|---|---:|---|---:|---:|---:|---:|---:|']
        for r in records:
            if r['evaluation_split'] != split:
                continue
            fmt = lambda k: f'{float(r[k])*100:.4f}%' if r[k] != '' else '未计算/未保存'
            lines.append(f'|{r["evaluation_id"]}|{r["model"]}/{r["input_size"]}|{r["selected_epoch"]}|{r["evaluation_event"]}; {r["evaluation_protocol"]}|{fmt("map50")}|{fmt("map50_95")}|{fmt("precision")}|{fmt("recall")}|{fmt("coco_ar100")}|')
        lines += ['']
    lines += ['## 分类别AP', '', '|评价编号|集合|broken AP50|broken AP50-95|loose AP50|loose AP50-95|', '|---|---|---:|---:|---:|---:|']
    for r in records:
        fmt = lambda k: f'{float(r[k])*100:.4f}%' if r[k] != '' else '该来源未保存'
        lines.append('|'+ '|'.join([r['evaluation_id'], r['evaluation_split']] + [fmt(k) for k in ('broken_ap50', 'broken_ap50_95', 'loose_ap50', 'loose_ap50_95')]) + '|')
    lines += ['', 'RT-DETR部分类别在原生P/R曲线插值中记录P=1、R=0。这是原生数值约定，不能解释为该类检测可靠；分类AP仍为0。', '']
    lines += ['', '## 训练与评价状态', '', '|实验|运行|train/val/test|迭代数/更新数|test状态|', '|---|---|---|---|---|']
    for r in runs:
        lines.append(f'|{r["experiment_id"]}|{r.get("run_id") or "尚未运行"}|{r["train_images"]}/{r.get("val_images") if r.get("val_images") is not None else "不适用"}/{r["test_images"]}|{r.get("training_iterations") or "未运行"}/{r.get("optimizer_updates") or "未保存或未运行"}|{r["test_status"]}|')
    lines += ['', '四次Ultralytics实验为2700训练batch。梯度累积和AMP使它们不等于2700优化器更新；实际更新数历史未保存。Faster R-CNN为5400更新。同样100epoch不代表相同计算量。训练时间和显存采集口径见原始说明，不做精确效率排名。', '',
              'YOLO1280和RTDETR1280的test各已使用一次；1600实验未做test。FRCNN VAL实验选模只依据val，后续测试分数不得用于挑epoch或调参。MERGE130未来从同一官方COCO初始权重重新训练，100epoch选最后轮；原val已进入train后不能作为独立val。', '',
              'DEBUG8为诊断实验。最终train-fit mAP50约63.93%、mAP50-95约53.51%，未完全过拟合；不能放入正式泛化排名。固定官方anchor下有部分小框无正RPN anchor，这是已观察到的流程限制，不能直接认定为正式AP低的唯一原因。', '',
              '## 已确认事实与解释限制', '', '当前Faster R-CNN选中epoch66，val精确COCO mAP50=3.7739%、mAP50-95=1.2734%，仍较低。损失下降不能直接证明泛化良好。不能把没有显示框解释成完全没有预测，须同时查看低阈值保存预测。数据量、小目标缩放、类别不平衡等是可能因素，现有结果不能确定单一原因。', '',
              '## 原始证据索引', '', f'- 数值CSV：`{CSV}`', f'- 总Excel：`{BOOK}`', f'- 登记验收：`{AUDIT / "registration_acceptance.json"}`', '']
    for r in records:
        lines += [f'### {r["evaluation_id"]}', '', f'- 指标：`{r["metric_source"]}`', f'- 指标SHA256：`{r["metric_source_sha256"]}`',
                  f'- checkpoint：`{r["checkpoint_path"]}`', f'- checkpoint SHA256：`{r["checkpoint_sha256"]}`',
                  f'- 预测：`{r["predictions_source"]}`' if r['predictions_source'] else '- 机器可读预测未保存。引用已有图片，不为填表重跑推理。',
                  f'- 配置：`{r["config_source"]}`', f'- 日志：`{r["log_source"]}`', f'- 说明：`{r["experiment_note"]}`', f'- 核查状态：{r["evaluation_status"]}', '']
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    formal_note = Path(next(r['note'] for r in runs if r['experiment_id'] == 'FRCNN-R50FPN-VAL-001'))
    vrows = [r for r in records if r['framework'] == 'MMDetection']
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


def write_merged_note(runs):
    path = FRCNN / 'workspace/notes/FRCNN-R50FPN-MERGE130-001.md'
    entries = [r for r in runs if r['experiment_id'] == 'FRCNN-R50FPN-MERGE130-001']
    lines = ['# FRCNN-R50FPN-MERGE130-001', '',
        '目的：执行原train107与val23合并为130图的既定实验。该设置在VAL-001 test前已固定。', '',
        '训练130图/205框（broken48、loose157），原test23图/36框（8、28）保留。没有独立val；原val已经成为训练数据，不能报告其独立验证成绩。YOLO0/1→COCO1/2→模型0/1（broken/loose）。', '',
        '固定设置：同一官方完整COCO初始checkpoint重新开始，不接着VAL-001私人数据checkpoint、不resume。100epoch、batch2、workers0、FP32、seed42、SGD0.0025/0.9/0.0001、warmup500、milestones67/92、gamma0.1；Resize边界(1333,800)保持比例、flip0.5、官方anchor。每轮65更新，计划6500；实际完成后以原始日志核对。', '',
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
    save_json(AUDIT / 'run_status_index.json', dict(runs=runs, diagnostics=[dict(
        experiment_id='FRCNN-DEBUG-OVERFIT-8', status='DIAGNOSTIC_ONLY',
        source=str(FRCNN / 'workspace/runs/FRCNN-DEBUG-OVERFIT-8/step06_24506_3904'))],
        log_manifests=[str(p) for p in sorted((FRCNN / 'workspace/logs').glob('*/step0[6789]*manifest.json'))]))
    write_reports(records, runs)
    write_merged_note(runs)
    print(f'RECORDS_EXTRACTED={len(records)} unique_evaluations; no inference/training', flush=True)
    if args.extract_only:
        return 0
    subprocess.run([str(NODE), str(BASE / 'scripts/build_experiment_workbook.mjs')], check=True)
    cells = verify_book(plan)
    receipt = dict(status='PASS', updated_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_rows=len(records), unique_evaluation_ids=len(records), verified_exported_cells=cells,
        csv_sha256=sha(CSV), excel_sha256=sha(BOOK), report_sha256=sha(REPORT),
        ratios_stored_0_to_1=True, percent_display_checked=True, blanks_not_replaced_by_zero=True,
        historical_sources_preserved=True, training_started=False, model_inference_performed=False,
        pending=([] if run['test_status'] == 'COMPLETED_REVIEWED' else ['FRCNN original test pending' if run['test_status'] == 'PENDING' else 'FRCNN test result/visual review pending']) +
                ['MERGE130 training/test review not completed', 'Cross-framework unified evaluation not performed', 'Legacy raw predictions / actual optimizer update counts not saved'])
    save_json(AUDIT / 'registration_acceptance.json', receipt)
    print(f'EXPERIMENT_RECORDS_PASS: CSV/Excel matched {cells} cells; unique evaluation IDs; ratio/blank checks passed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
