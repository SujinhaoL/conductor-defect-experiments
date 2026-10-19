"""Resolution sensitivity launcher; original train107/val23. --check-preparation never builds/trains a detector."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'workspace/scripts'
CONFIG = ROOT / 'workspace/configs/frcnn_r50fpn_scale1280_val_001.py'
EXPERIMENT = 'FRCNN-R50FPN-SCALE1280-VAL-001'
DATA = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
COMMIT = '44ebd17b145c2372c4b700bfb9cb20dbd28ab64a'
helpers = runpy.run_path(str(SCRIPTS / 'step04_convert_coco.py'))
require, sha, save_json = (helpers[key] for key in ('require', 'sha', 'save_json'))


def check_preparation():
    sys.path.insert(0, str(SCRIPTS))
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    import mmdet
    import torch
    import yaml
    runpy.run_path(str(SCRIPTS / 'step01_check_cuda.py'))['check_identity']()
    register_all_modules(init_default_scope=True)
    versions = {key: importlib.metadata.version(key) for key in ('torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'numpy', 'pycocotools')}
    require(versions == dict(torch='2.7.1+cu128', torchvision='0.22.1+cu128', mmcv='2.1.0', mmengine='0.10.7',
                             mmdet='3.3.0', numpy='1.26.4', pycocotools='2.0.10'), 'Environment versions changed')
    source = ROOT / 'mmdetection'
    require(Path(mmdet.__file__).resolve().is_relative_to(source), 'Wrong MMDetection import')
    git = ['git', '-c', 'safe.directory=' + str(source)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=source, text=True).strip() == COMMIT, 'Official commit changed')
    require(not subprocess.check_output(git + ['diff', 'HEAD', '--name-only'], cwd=source), 'Official tracked code changed')
    cfg = Config.fromfile(str(CONFIG))
    # Verify all inherited configuration values against the frozen baseline.
    baseline_path = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py'
    baseline = Config.fromfile(str(baseline_path))
    import copy
    normalized = copy.deepcopy(cfg.to_dict())
    reference = copy.deepcopy(baseline.to_dict())
    for key in ('experiment_id', 'snapshot_scale', 'work_dir'):
        normalized.pop(key, None)
        reference.pop(key, None)
    pipelines = [normalized['train_pipeline'], normalized['test_pipeline'],
                 normalized['train_dataloader']['dataset']['pipeline'], normalized['val_dataloader']['dataset']['pipeline']]
    for pipeline in pipelines:
        resize = [item for item in pipeline if item['type'] == 'Resize']
        require(len(resize) == 1 and tuple(resize[0]['scale']) == (2133, 1280) and resize[0]['keep_ratio'], 'All actual pipelines must use the new scale')
        resize[0]['scale'] = (1333, 800)
    require(normalized == reference, 'A setting besides resize/record labels differs from the frozen baseline')
    require(tuple(cfg.snapshot_scale) == (2133, 1280), 'Snapshot resize differs from train/val')
    print('SCALE_ONLY_CONFIG_DIFF_PASS: train/val/snapshots=(2133,1280); all other scientific settings match frozen VAL-001', flush=True)
    provenance = json.loads((ROOT / 'workspace/configs/step03_preparation.json').read_text(encoding='utf-8'))
    require(Path(cfg.load_from).resolve() == Path(provenance['checkpoint']['path']).resolve(), 'Must start from the original official detector checkpoint')
    require(sha(Path(cfg.load_from)) == provenance['checkpoint']['sha256'], 'Official checkpoint hash changed')
    require(not cfg.resume and cfg.model.backbone.init_cfg is None and cfg.model.roi_head.bbox_head.num_classes == 2, 'Initialization/class setup changed')
    require(cfg.model.rpn_head.anchor_generator.scales == [8] and cfg.model.train_cfg.rpn.assigner.min_pos_iou == 0.3, 'Default anchors/assignment changed')
    require(cfg.train_cfg.max_epochs == 100 and cfg.train_cfg.val_interval == 1, 'Expected 100 epochs with validation each epoch')
    require(cfg.test_cfg is None and cfg.test_dataloader is None and cfg.test_evaluator is None, 'No test evaluation in this training stage')
    require(cfg.optim_wrapper.type == 'OptimWrapper' and cfg.optim_wrapper.accumulative_counts == 1, 'Expected FP32 / no accumulation')
    require(dict(cfg.optim_wrapper.optimizer) == dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001), 'Optimizer changed')
    require([p.type for p in cfg.train_pipeline] == ['LoadImageFromFile', 'LoadAnnotations', 'Resize', 'RandomFlip', 'PackDetInputs'], 'Unexpected training image transform')
    require(tuple(cfg.train_pipeline[2].scale) == (2133, 1280) and cfg.train_pipeline[2].keep_ratio and cfg.train_pipeline[3].prob == 0.5, 'Resize/flip changed')
    require(cfg.default_hooks.checkpoint.save_best == 'coco/bbox_mAP' and cfg.default_hooks.checkpoint.rule == 'greater', 'Wrong best checkpoint criterion')
    require(list(cfg.val_evaluator.proposal_nums) == [1, 10, 100] and cfg.model.test_cfg.rcnn.max_per_img == 100 and cfg.model.test_cfg.rcnn.score_thr == 0.001,
            'Prediction/evaluation filtering changed')
    require(cfg.param_scheduler[0].end == 500 and list(cfg.param_scheduler[1].milestones) == [67, 92], 'LR schedule changed')
    selection = json.loads((ROOT / 'workspace/configs/frcnn_debug8_selection.json').read_text(encoding='utf-8'))
    split_summary = {}
    for split, expected in selection['all_coco_hashes'].items():
        path = DATA / 'annotations' / f'instances_{split}.json'
        require(sha(path) == expected, 'Existing COCO annotations changed: ' + split)
        data = json.loads(path.read_text(encoding='utf-8'))
        split_summary[split] = dict(images=len(data['images']), boxes=len(data['annotations']))
    require(split_summary == dict(train=dict(images=107, boxes=168), val=dict(images=23, boxes=37),
                                   test=dict(images=23, boxes=36), trainval=dict(images=130, boxes=205)), 'Dataset counts changed')
    prior = json.loads((ROOT / 'workspace/logs/step05_20493_32385/step05_summary.json').read_text(encoding='utf-8'))
    require(sha(DATA / 'annotations/instances_debug8.json') == prior['debug_annotation_sha256'], 'Fixed snapshot annotation changed')
    for sample in selection['samples']:
        require(sha(DATA / 'images' / sample['file_name']) == sample['image_sha256'], 'Fixed snapshot image changed')
    # Only construct dataloaders: do not create Runner/model/optimizer or run training.
    train = Runner.build_dataloader(cfg.train_dataloader, seed=42)
    val = Runner.build_dataloader(cfg.val_dataloader, seed=42)
    require(len(train.dataset) == 107 and len(train) == 54 and train.dataset.cat2label == {1: 0, 2: 1}, 'Unexpected train dataset/batches')
    require(len(val.dataset) == 23 and len(val) == 23 and val.dataset.cat2label == {1: 0, 2: 1}, 'Unexpected validation dataset/batches')
    require(all('/images/train/' in train.dataset.get_data_info(i)['img_path'].replace('\\', '/') for i in range(107)), 'Train dataset points outside original train')
    require(all('/images/val/' in val.dataset.get_data_info(i)['img_path'].replace('\\', '/') for i in range(23)), 'Validation dataset points outside original val')
    require(torch.cuda.is_available() and torch.version.cuda == '12.8', 'CUDA environment unavailable')
    gpu = torch.cuda.get_device_properties(0)
    require((gpu.major, gpu.minor) == (12, 0), 'Expected previously checked sm_120 GPU')
    require(shutil.disk_usage(ROOT).free > 3 * 1024**3, 'Need 3 GiB free for artifacts/checkpoints')
    previous_paths = [
        ROOT.parent / 'yolo11/workspace/runs/Y11-BASE-001-R1_20260917_203044_883451/args.yaml',
        ROOT.parent / 'RT-DETR/workspace/runs/RTD-BASE-001-R1_20260918_172120/args.yaml',
        ROOT.parent / 'yolo11/workspace/runs/Y11-SENS-IMGSZ1600-001/args.yaml',
        ROOT.parent / 'RT-DETR/workspace/runs/RTD-SENS-IMGSZ1600-001/args.yaml',
    ]
    previous_args = []
    keys = ('epochs', 'batch', 'imgsz', 'workers', 'optimizer', 'lr0', 'lrf', 'nbs', 'warmup_epochs', 'amp', 'seed', 'deterministic', 'conf', 'iou', 'max_det')
    for path in previous_paths:
        args = yaml.safe_load(path.read_text(encoding='utf-8'))
        previous_args.append(dict(path=str(path), sha256=sha(path), configured_args={key: args.get(key) for key in keys},
                                  optimizer_note='optimizer=auto; lr0 is the requested field, not proof of actual optimizer learning rate'))
    record = dict(status='PASS', experiment=EXPERIMENT, training_started=False, detector_forward_performed=False,
                  versions=versions, mmdet_commit=COMMIT, gpu=gpu.name, gpu_total_bytes=gpu.total_memory,
                  dataset_counts=split_summary, annotation_hashes=selection['all_coco_hashes'],
                  checkpoint=provenance['checkpoint'], epochs=100, iterations_per_epoch=54, optimizer_updates=5400,
                  batch_size=2, amp=False, workers=0, seed=42, lr=0.0025, momentum=0.9, weight_decay=0.0001,
                  warmup_iterations=500, warmup_start_factor=0.001, epoch_lr_milestones=[67, 92], gamma=0.1,
                  resize_scale=[2133, 1280], keep_ratio=True, horizontal_flip_probability=0.5,
                  model_prediction_score_threshold=0.001, visualization_score_threshold=0.30,
                  rcnn_nms_iou=0.5, max_predictions_per_image=100, coco_max_dets=[1, 10, 100],
                  best_selection_metric='coco/bbox_mAP', previous_experiments_args=previous_args,
                  comparison_scope='Resolution sensitivity versus frozen VAL-001: same train107/val23, batch/steps/optimizer/seed; only resize changed; no test-driven tuning.')
    print('STEP11_PREPARATION_PASS: train=107 val=23 test_reserved=23; batch=2 epochs=100 updates=5400', flush=True)
    return cfg, record


def training_child(run_dir):
    require(run_dir.is_relative_to(ROOT / 'workspace/runs' / EXPERIMENT), 'Unexpected run path')
    sys.path.insert(0, str(run_dir / 'code'))
    official = ROOT / 'mmdetection/tools/train.py'
    sys.argv = [str(official), str(run_dir / 'resolved_scale1280_val_001.py'), '--work-dir', str(run_dir), '--launcher', 'none']
    runpy.run_path(str(official), run_name='__main__')
    return 0


def finish_review(run_dir):
    from frcnn_debug_hooks import evaluate_train_fit
    summary_file = run_dir / 'artifacts/formal_training_summary.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    require(summary['epochs'] == 100 and summary['optimizer_updates'] == 5400, 'Training did not complete')
    best_epoch = summary['best_validation']['epoch']
    best_files = list(run_dir.glob('best_coco_bbox_mAP_epoch_*.pth'))
    require(len(best_files) == 1, 'Expected one native best validation checkpoint')
    match = re.fullmatch(r'best_coco_bbox_mAP_epoch_(\d+)\.pth', best_files[0].name)
    require(match and int(match.group(1)) == best_epoch, 'Saved checkpoint disagrees with recorded validation selection')
    final = run_dir / 'epoch_100.pth'
    require(final.is_file(), 'Last checkpoint missing')
    annotation = DATA / 'annotations/instances_val.json'
    truth = json.loads(annotation.read_text(encoding='utf-8'))
    selected_predictions = run_dir / 'artifacts/validation' / f'epoch_{best_epoch:03d}' / 'predictions.bbox.json'
    predictions = json.loads(selected_predictions.read_text(encoding='utf-8'))
    exact = evaluate_train_fit(annotation, predictions, [image['id'] for image in truth['images']],
                               run_dir / 'artifacts/best_validation',
                               scope='original 23-image validation set; selected by validation mAP50-95; NOT test')
    require(abs(exact['mAP50_95'] - summary['best_validation']['metrics']['coco/bbox_mAP']) <= 0.00051, 'Independent saved-prediction evaluation disagrees with native COCO metric')
    selection = dict(status='SELECTED_VAL_REVIEW_PENDING', criterion='coco/bbox_mAP', selected_epoch=best_epoch,
                     best_checkpoint=str(best_files[0]), best_checkpoint_sha256=sha(best_files[0]),
                     last_checkpoint=str(final), last_checkpoint_sha256=sha(final),
                     validation_predictions=str(selected_predictions), validation_metrics=exact,
                     test_evaluated=False, native_coco_selection_metrics=summary['best_validation']['metrics'])
    save_json(run_dir / 'selected_checkpoint.json', selection)
    print('BEST_CHECKPOINT=' + str(best_files[0]), flush=True)
    print(f'BEST_VAL_EPOCH={best_epoch} VAL_mAP50={exact["mAP50"]:.6f} VAL_mAP50_95={exact["mAP50_95"]:.6f}', flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-preparation', action='store_true')
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--train-process', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train_process is not None:
        return training_child(args.train_process.resolve())
    if args.check_preparation:
        cfg, record = check_preparation()
        audit = ROOT / 'workspace/audit/step11_preflight'
        audit.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(audit / 'resolved_scale1280_val_001.py'))
        save_json(audit / 'preparation_check.json', record)
        return 0
    require(args.log_dir is not None, '--log-dir required')
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / 'workspace/logs') and logs.name.startswith('step11_'), 'Unexpected log path')
    logs.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / 'workspace/runs' / EXPERIMENT / logs.name
    report = dict(status='FAIL', started_utc=datetime.now(timezone.utc).isoformat(),
                  training_process_launched=False, experiment=EXPERIMENT, run_dir=str(run_dir), log_dir=str(logs))
    started = time.perf_counter()
    try:
        cfg, record = check_preparation()
        require(not run_dir.exists(), 'Run path already exists; no overwrite or automatic resume')
        (run_dir / 'code').mkdir(parents=True)
        cfg.work_dir = str(run_dir)
        cfg.dump(str(run_dir / 'resolved_scale1280_val_001.py'))
        inputs = [CONFIG, SCRIPTS / 'frcnn_formal_hooks.py', SCRIPTS / 'frcnn_debug_hooks.py', Path(__file__),
                  SCRIPTS / 'run_step11_train_scale1280.cmd', ROOT / 'workspace/configs/frcnn_debug8_selection.json',
                  ROOT.parent / 'AGENTS.md', ROOT / 'workspace/audit/step11_resolution/resolution_audit.json',
                  ROOT / 'workspace/notes/STEP11_resolution1280_plan.md',
                  DATA / 'annotations/instances_train.json', DATA / 'annotations/instances_val.json',
                  DATA / 'annotations/instances_debug8.json']
        record['baseline_config_sha256'] = sha(ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py')
        record['launch_cmd'] = 'call \"' + str(SCRIPTS / 'run_step11_train_scale1280.cmd') + '\"'
        record['command_line'] = subprocess.list2cmdline([sys.executable, '-I', '-X', 'utf8', str(Path(__file__)), '--log-dir', str(logs)])
        record['saved_inputs_sha256'] = {str(path): sha(path) for path in inputs}
        for path in inputs:
            shutil.copy2(path, run_dir / 'code' / path.name)
        save_json(run_dir / 'pretraining_record.json', record)
        (run_dir / 'requirements_freeze.txt').write_bytes(subprocess.check_output(
            [sys.executable, '-I', '-X', 'utf8', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'freeze', '--all']))
        env = os.environ.copy()
        for key in list(env):
            if key.upper() in {'PYTHONHOME', 'PYTHONPATH'}:
                del env[key]
        env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT / 'workspace/cache/matplotlib'),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0')
        print('RUN_DIR=' + str(run_dir), flush=True)
        print('Starting user-run SCALE1280: original train107/val23, Resize(2133,1280), fresh COCO; test excluded.', flush=True)
        report['training_process_launched'] = True
        save_json(logs / 'step11_manifest.json', report)
        run = runpy.run_path(str(SCRIPTS / 'step02c_build_mmcv.py'))['run']
        run([sys.executable, '-I', '-X', 'utf8', Path(__file__), '--train-process', run_dir],
            ROOT, env, logs / 'train_console.log')
        report.update(status='TRAINING_COMPLETED_VAL_REVIEW_PENDING', selected_checkpoint=finish_review(run_dir))
        print('STEP11_VAL_TRAIN_COMPLETED', flush=True)
        print('TEST_NOT_RUN; send the terminal tail, RUN_DIR and LOG_DIR for result review.', flush=True)
    except BaseException:
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
        print('STEP11_FAIL; send complete error and LOG_DIR. Do not automatically retry/resume.', flush=True)
    report['launcher_wall_seconds'] = time.perf_counter() - started
    save_json(logs / 'step11_manifest.json', report)
    print('RUN_DIR=' + str(run_dir), flush=True)
    return 0 if report['status'] == 'TRAINING_COMPLETED_VAL_REVIEW_PENDING' else 1


if __name__ == '__main__':
    # Persist the complete outer console as well as the child train_console.log.
    if '--log-dir' in sys.argv:
        from contextlib import redirect_stdout, redirect_stderr
        position = sys.argv.index('--log-dir')
        console_dir = Path(sys.argv[position + 1]).resolve()
        require(console_dir.is_relative_to(ROOT / 'workspace/logs') and console_dir.name.startswith('step11_'), 'Unexpected console directory')
        console_dir.mkdir(parents=True, exist_ok=True)
        class Tee:
            def __init__(self, terminal, file):
                self.terminal, self.file = terminal, file
            def write(self, value):
                self.terminal.write(value)
                self.file.write(value)
                self.file.flush()
                return len(value)
            def flush(self):
                self.terminal.flush()
                self.file.flush()
        with (console_dir / 'step11_console.txt').open('x', encoding='utf-8') as console_file:
            with redirect_stdout(Tee(sys.stdout, console_file)), redirect_stderr(Tee(sys.stderr, console_file)):
                code = main()
        raise SystemExit(code)
    raise SystemExit(main())
