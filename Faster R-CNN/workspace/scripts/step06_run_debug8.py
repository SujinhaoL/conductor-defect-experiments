"""Launch DEBUG8 only when the user runs the batch file; --check-preparation never trains."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'workspace/scripts'
CONFIG = ROOT / 'workspace/configs/frcnn_debug_overfit8.py'
DATA = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
STEP05 = ROOT / 'workspace/logs/step05_20493_32385/step05_summary.json'
COMMIT = '44ebd17b145c2372c4b700bfb9cb20dbd28ab64a'
HELPERS = runpy.run_path(str(SCRIPTS / 'step04_convert_coco.py'))
require, sha, save_json = (HELPERS[n] for n in ('require', 'sha', 'save_json'))


def check_preparation():
    # Restrict imports to this experiment's explicit custom-code directory.
    sys.path.insert(0, str(SCRIPTS))
    from mmengine.config import Config
    from mmdet.utils import register_all_modules
    from mmdet.registry import DATASETS, HOOKS
    import mmdet
    import torch
    register_all_modules(init_default_scope=True)
    previous = json.loads(STEP05.read_text(encoding='utf-8'))
    require(previous['status'] == 'PASS', 'STEP05 must have passed first')
    plan, _, provenance = runpy.run_path(str(SCRIPTS / 'step05_check_debug8.py'))['check_preparation']()
    for path, digest in [
        (DATA / 'annotations/instances_debug8.json', previous['debug_annotation_sha256']),
        (ROOT / 'workspace/configs/frcnn_debug8_selection.json', previous['selection_sha256']),
        (ROOT / 'workspace/configs/frcnn_debug8_pipeline_check.py', previous['config_sha256']),
    ]:
        require(sha(path) == digest, 'STEP05 input changed: ' + str(path))
    cfg = Config.fromfile(str(CONFIG))
    require(cfg.train_cfg.type == 'EpochBasedTrainLoop' and cfg.train_cfg.max_epochs == 100, 'Unexpected training loop')
    require(all(cfg.get(k) is None for k in ('val_cfg', 'test_cfg', 'val_dataloader', 'test_dataloader',
                                            'val_evaluator', 'test_evaluator')), 'Independent evaluation must stay disabled')
    require(cfg.train_dataloader.batch_size == 2 and cfg.train_dataloader.num_workers == 0, 'Unexpected batch/workers')
    require(cfg.optim_wrapper.type == 'OptimWrapper' and cfg.optim_wrapper.accumulative_counts == 1, 'Expected FP32 / no accumulation')
    require(dict(cfg.optim_wrapper.optimizer) == dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001), 'Optimizer changed')
    require(len(cfg.param_scheduler) == 1 and cfg.param_scheduler[0].end == 20, 'Warmup configuration changed')
    require(not cfg.resume and not cfg.auto_scale_lr.enable, 'Unexpected resume/auto LR scaling')
    require(cfg.model.test_cfg.rcnn.score_thr == 0.001 and cfg.model.test_cfg.rcnn.max_per_img == 100, 'Prediction filter changed')
    require(HOOKS.get('Debug8ArtifactsHook') is not None, 'Custom artifact hook was not imported')
    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    require(len(dataset) == 8 and dataset.cat2label == {1: 0, 2: 1}, 'Unexpected debug dataset')
    require([dataset.get_data_info(i)['img_id'] for i in range(8)] == [x['image_id'] for x in plan['samples']], 'Debug image order changed')
    versions = {key: importlib.metadata.version(key) for key in ('torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'numpy', 'pycocotools')}
    expected = dict(torch='2.7.1+cu128', torchvision='0.22.1+cu128', mmcv='2.1.0', mmengine='0.10.7',
                    mmdet='3.3.0', numpy='1.26.4', pycocotools='2.0.10')
    require(versions == expected, 'Pinned environment versions differ: ' + str(versions))
    source = ROOT / 'mmdetection'
    require(Path(mmdet.__file__).resolve().is_relative_to(source), 'Wrong MMDetection import location')
    git = ['git', '-c', 'safe.directory=' + str(source)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=source, text=True).strip() == COMMIT, 'MMDetection commit changed')
    require(not subprocess.check_output(git + ['diff', 'HEAD', '--name-only'], cwd=source), 'Official tracked source changed')
    require(torch.cuda.is_available() and torch.version.cuda == '12.8', 'Expected CUDA 12.8 build and accessible GPU')
    device = torch.cuda.get_device_properties(0)
    require((device.major, device.minor) == (12, 0), 'Expected the checked sm_120 GPU')
    require(shutil.disk_usage(ROOT).free > 3 * 1024**3, 'Need at least 3 GiB free for checkpoints and predictions')
    record = dict(status='PASS', experiment='FRCNN-DEBUG-OVERFIT-8', training_started=False,
                  detector_forward_performed=False, versions=versions, mmdet_commit=COMMIT,
                  gpu=device.name, gpu_total_bytes=device.total_memory,
                  epochs=100, iterations_per_epoch=4, optimizer_updates=400,
                  batch_size=2, workers=0, amp=False, seed=42, base_lr=0.0025,
                  checkpoint=provenance['checkpoint'], prior_static_check=str(STEP05),
                  debug_annotation_sha256=previous['debug_annotation_sha256'],
                  selection_sha256=previous['selection_sha256'])
    print('STEP06_PREPARATION_PASS: images=8 boxes=15 epochs=100 batch=2 updates=400; NO training in this check', flush=True)
    return cfg, record


def training_child(run_dir):
    # Only the user's normal batch command reaches this branch. The preparation
    # check and developer validation do not construct a Runner or call train().
    require(run_dir.is_relative_to(ROOT / 'workspace/runs/FRCNN-DEBUG-OVERFIT-8'), 'Unexpected run path')
    sys.path.insert(0, str(run_dir / 'code'))
    official = ROOT / 'mmdetection/tools/train.py'
    sys.argv = [str(official), str(run_dir / 'resolved_debug8.py'), '--work-dir', str(run_dir), '--launcher', 'none']
    runpy.run_path(str(official), run_name='__main__')
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-preparation', action='store_true')
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--train-process', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.train_process is not None:
        return training_child(args.train_process.resolve())
    if args.check_preparation:
        _, record = check_preparation()
        audit = ROOT / 'workspace/audit/step06_preflight'
        audit.mkdir(parents=True, exist_ok=True)
        save_json(audit / 'preparation_check.json', record)
        return 0
    require(args.log_dir is not None, '--log-dir required')
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / 'workspace/logs') and logs.name.startswith('step06_'), 'Unexpected log path')
    logs.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / 'workspace/runs/FRCNN-DEBUG-OVERFIT-8' / logs.name
    report = dict(status='FAIL', started_utc=datetime.now(timezone.utc).isoformat(),
                  training_process_launched=False, run_dir=str(run_dir), log_dir=str(logs))
    started = time.perf_counter()
    try:
        cfg, preparation = check_preparation()
        require(not run_dir.exists(), 'Run directory already exists; no overwrite or automatic resume')
        (run_dir / 'code').mkdir(parents=True)
        cfg.work_dir = str(run_dir)
        cfg.dump(str(run_dir / 'resolved_debug8.py'))
        inputs = [CONFIG, ROOT / 'workspace/configs/frcnn_debug8_pipeline_check.py',
                  ROOT / 'workspace/configs/frcnn_debug8_selection.json',
                  SCRIPTS / 'frcnn_debug_hooks.py', Path(__file__), SCRIPTS / 'run_step06_train_debug8.cmd',
                  DATA / 'annotations/instances_debug8.json']
        preparation['saved_inputs_sha256'] = {str(path): sha(path) for path in inputs}
        for path in inputs:
            shutil.copy2(path, run_dir / 'code' / path.name)
        save_json(run_dir / 'pretraining_record.json', preparation)
        (run_dir / 'requirements_freeze.txt').write_bytes(subprocess.check_output(
            [sys.executable, '-I', '-X', 'utf8', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'freeze', '--all']))
        env = os.environ.copy()
        for key in list(env):
            if key.upper() in {'PYTHONHOME', 'PYTHONPATH'}:
                del env[key]
        env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT / 'workspace/cache/matplotlib'),
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0')
        print('RUN_DIR=' + str(run_dir), flush=True)
        print('Starting USER-REQUESTED debug training: 8 train images, 100 epochs, 400 updates; no val/test.', flush=True)
        report['training_process_launched'] = True
        save_json(logs / 'step06_manifest.json', report)
        run = runpy.run_path(str(SCRIPTS / 'step02c_build_mmcv.py'))['run']
        run([sys.executable, '-I', '-X', 'utf8', Path(__file__), '--train-process', run_dir],
            ROOT, env, logs / 'train_console.log')
        summary_file = run_dir / 'diagnostics/debug8_training_summary.json'
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        require(summary['epochs'] == 100 and summary['optimizer_updates'] == 400, 'Training did not complete the planned updates')
        require(summary['status'] == 'COMPLETED_REQUIRES_RESULT_REVIEW', 'Unexpected result status')
        final = run_dir / 'epoch_100.pth'
        require(final.is_file(), 'Final checkpoint not saved')
        report.update(status='COMPLETED_REQUIRES_RESULT_REVIEW', final_checkpoint=str(final),
                      final_checkpoint_sha256=sha(final), summary=str(summary_file))
        print('STEP06_DEBUG_TRAIN_COMPLETED', flush=True)
        print('RESULT_REVIEW_REQUIRED; these are training-subset metrics, not val/test results.', flush=True)
    except BaseException:
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
        print('STEP06_FAIL; send complete output and LOG_DIR. Do not automatically rerun or resume.', flush=True)
    report['launcher_wall_seconds'] = time.perf_counter() - started
    save_json(logs / 'step06_manifest.json', report)
    print('RUN_DIR=' + str(run_dir), flush=True)
    return 0 if report['status'] == 'COMPLETED_REQUIRES_RESULT_REVIEW' else 1


if __name__ == '__main__':
    raise SystemExit(main())
