"""User-run merged training only. --check-preparation cannot start training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
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
CONFIG = ROOT / 'workspace/configs/frcnn_r50fpn_merge130_001.py'
EXPERIMENT = 'FRCNN-R50FPN-MERGE130-001'
PREVIOUS = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
DATA = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
RECORDS = ROOT.parent / 'baseline_analysis/scripts/refresh_experiment_records.py'
helpers = runpy.run_path(str(SCRIPTS / 'step04_convert_coco.py'))
require, sha, save_json, Tee = (helpers[key] for key in ('require', 'sha', 'save_json', 'Tee'))


def check_preparation():
    sys.path.insert(0, str(SCRIPTS))
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet.utils import register_all_modules
    import mmdet
    import torch
    import importlib.metadata
    runpy.run_path(str(SCRIPTS / 'step01_check_cuda.py'))['check_identity']()
    register_all_modules(init_default_scope=True)
    previous_record = json.loads((PREVIOUS / 'pretraining_record.json').read_text(encoding='utf-8'))
    versions = {name: importlib.metadata.version(name) for name in previous_record['versions']}
    require(versions == previous_record['versions'], 'Dedicated environment versions changed')
    source = ROOT / 'mmdetection'
    require(Path(mmdet.__file__).resolve().is_relative_to(source), 'Wrong MMDetection import')
    git = ['git', '-c', 'safe.directory=' + str(source)]
    require(subprocess.check_output(git+['rev-parse', 'HEAD'], cwd=source, text=True).strip() == previous_record['mmdet_commit'], 'Official code commit changed')
    require(not subprocess.check_output(git+['diff', 'HEAD', '--name-only'], cwd=source), 'Official tracked source changed')
    reviewed = json.loads((ROOT / 'workspace/audit/step08_review/acceptance.json').read_text(encoding='utf-8'))
    require(reviewed['status'] == 'PASS' and not reviewed['test_results_used_to_change_training_settings'], 'VAL-001 test review not completed')
    cfg = Config.fromfile(str(CONFIG))
    frozen = Config.fromfile(str(PREVIOUS / 'resolved_val_001.py'), import_custom_modules=False)
    for name in ('model', 'optim_wrapper', 'param_scheduler', 'randomness', 'train_pipeline', 'auto_scale_lr', 'env_cfg'):
        require(cfg[name] == frozen[name], f'Previously frozen training setting changed: {name}')
    require('Fresh same COCO initialization' in previous_record['merged_stage_frozen_settings'], 'Missing prior frozen merged-stage plan')
    require(cfg.load_from == frozen.load_from and sha(Path(cfg.load_from)) == previous_record['checkpoint']['sha256'], 'Must restart from SAME official full COCO checkpoint')
    require(not cfg.resume and cfg.model.backbone.init_cfg is None and cfg.model.roi_head.bbox_head.num_classes == 2, 'Fresh initialization/classes changed')
    require(cfg.train_cfg.max_epochs == 100, 'Merged training epochs must remain100')
    require(all(cfg[name] is None for name in ('val_dataloader', 'val_cfg', 'val_evaluator', 'test_dataloader', 'test_cfg', 'test_evaluator')), 'This stage must not validate/test')
    require(cfg.default_hooks.checkpoint.save_best is None and cfg.default_hooks.checkpoint.save_last, 'Select final checkpoint only')
    hashes, source_jsons = {}, {}
    for split, expected in previous_record['annotation_hashes'].items():
        path = DATA / 'annotations' / f'instances_{split}.json'
        hashes[split] = sha(path)
        require(hashes[split] == expected, 'Existing annotations changed: '+split)
        source_jsons[split] = json.loads(path.read_text(encoding='utf-8'))
    merged = source_jsons['trainval']
    require(merged['images'] == source_jsons['train']['images'] + source_jsons['val']['images'], 'Merged images/order must be the original train+val union')
    require(merged['annotations'] == source_jsons['train']['annotations'] + source_jsons['val']['annotations'], 'Merged boxes/IDs must be unchanged train+val union')
    require(len(merged['images']) == 130 and len(merged['annotations']) == 205, 'Expected130 images205boxes')
    require(Counter(a['category_id'] for a in merged['annotations']) == {1:48, 2:157}, 'Merged class counts changed')
    require(len({i['id'] for i in merged['images']}) == 130 and len({a['id'] for a in merged['annotations']}) == 205, 'Merged IDs not unique')
    require(len({i['file_name'] for i in merged['images']}) == 130, 'Merged filename conflict')
    require(all(i['file_name'].startswith(('train/', 'val/')) for i in merged['images']), 'Merged dataset includes non-train/val file')
    # Build dataloader and iterate INDEX batches only: no model, optimizer,
    # image pipeline execution or training. Confirm130 distinct images per epoch.
    loader = Runner.build_dataloader(cfg.train_dataloader, seed=42)
    batches = list(loader.batch_sampler)
    indices = [i for batch in batches for i in batch]
    require(len(loader.dataset) == 130 and loader.dataset.cat2label == {1:0, 2:1}, 'Merged MMDetection mapping changed')
    require(len(loader) == len(batches) == 65 and len(indices) == len(set(indices)) == 130, 'Unexpected batch or image coverage')
    image_hashes = {}
    for image in merged['images']:
        name = image['file_name']
        copied = DATA / 'images' / name
        original = DATA.parent / '02_yolo11_dataset/images' / name
        digest = sha(copied)
        require(digest == sha(original), 'Copied training image content differs: '+name)
        image_hashes[name] = digest
    gpu = torch.cuda.get_device_properties(0)
    require(torch.cuda.is_available() and torch.version.cuda == '12.8' and (gpu.major, gpu.minor) == (12,0), 'Previously checked CUDA/sm120 environment changed')
    require(shutil.disk_usage(ROOT).free > 3*1024**3, 'Need3GiB free for checkpoints and artifacts')
    record = dict(status='PASS', experiment=EXPERIMENT, training_started=False, detector_forward_performed=False,
        versions=versions, python=sys.version, mmdet_commit=previous_record['mmdet_commit'],
        gpu=gpu.name, gpu_total_bytes=gpu.total_memory, checkpoint=previous_record['checkpoint'],
        frozen_setting_source=str(PREVIOUS / 'pretraining_record.json'), frozen_setting_source_sha256=sha(PREVIOUS / 'pretraining_record.json'),
        annotation_hashes=hashes, image_hashes=image_hashes,
        dataset_counts=dict(train=dict(images=130, boxes=205, broken=48, loose=157), val=None, test=dict(images=23, boxes=36, broken=8, loose=28)),
        epochs=100, iterations_per_epoch=65, optimizer_updates=6500, batch_size=2, workers=0, amp=False, seed=42,
        optimizer='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001,
        warmup_iterations=500, warmup_start_factor=0.001, epoch_lr_milestones=[67,92], gamma=0.1,
        resize_scale=[1333,800], keep_ratio=True, horizontal_flip_probability=0.5,
        selected_epoch=100, selection_rule='fixed last epoch100; no val/test-based selection',
        independent_validation=False, original_val_in_training=True, test_evaluation_in_this_stage=False,
        continued_from_val001_checkpoint=False, frozen_before_val001_test=True,
        fixed_train_visualization_threshold=0.30, model_prediction_score_threshold=0.001,
        rcnn_nms_iou=0.5, max_predictions_per_image=100,
        model_created=False, optimizer_created=False, pipeline_or_model_forward_performed=False)
    print('STEP09_PREPARATION_PASS: train=130 boxes=205 broken=48 loose=157; val=NONE test_reserved=23; batch=2 epochs=100 updates=6500', flush=True)
    return cfg, record


def training_child(run_dir):
    require(run_dir.is_relative_to(ROOT / 'workspace/runs' / EXPERIMENT), 'Unexpected run path')
    sys.path.insert(0, str(run_dir / 'code'))
    official = ROOT / 'mmdetection/tools/train.py'
    sys.argv = [str(official), str(run_dir / 'resolved_merge130_001.py'), '--work-dir', str(run_dir), '--launcher', 'none']
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
        cfg, record = check_preparation()
        audit = ROOT / 'workspace/audit/step09_preflight'
        audit.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(audit / 'resolved_merge130_001.py'))
        save_json(audit / 'preparation_check.json', record)
        print('STEP09_STATIC_CHECK_PASS; no model/optimizer/pipeline forward/training/test inference', flush=True)
        return 0
    require(args.log_dir is not None, '--log-dir required')
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / 'workspace/logs') and logs.name.startswith('step09_'), 'Unexpected log directory')
    logs.mkdir(parents=True, exist_ok=True)
    run_dir = ROOT / 'workspace/runs' / EXPERIMENT / logs.name
    report = dict(status='FAIL', started_utc=datetime.now(timezone.utc).isoformat(),
                  experiment=EXPERIMENT, run_dir=str(run_dir), log_dir=str(logs), training_process_launched=False,
                  validation_or_test_evaluated=False)
    stdout, stderr = sys.stdout, sys.stderr
    started = time.perf_counter()
    with (logs / 'step09_console.txt').open('w', encoding='utf-8') as output:
        sys.stdout, sys.stderr = Tee(stdout, output), Tee(stderr, output)
        try:
            cfg, record = check_preparation()
            require(not list((ROOT / 'workspace/runs' / EXPERIMENT).glob('*/selected_checkpoint.json')), 'A completed MERGE130 run exists; inspect before any new training')
            require(not run_dir.exists(), 'Run directory exists; no overwrite or automatic resume')
            (run_dir / 'code').mkdir(parents=True)
            cfg.work_dir = str(run_dir)
            cfg.dump(str(run_dir / 'resolved_merge130_001.py'))
            inputs = [CONFIG, Path(__file__), SCRIPTS / 'frcnn_merge130_hooks.py', SCRIPTS / 'frcnn_debug_hooks.py',
                      SCRIPTS / 'run_step09_train_merge130.cmd', ROOT / 'workspace/notes/STEP09_merge130_training.txt',
                      ROOT / 'workspace/notes/EXPERIMENT_RECORDING_RULES.md',
                      PREVIOUS / 'pretraining_record.json', PREVIOUS / 'resolved_val_001.py',
                      DATA / 'annotations/instances_trainval.json', DATA / 'annotations/instances_debug8.json',
                      RECORDS, RECORDS.with_name('build_experiment_workbook.mjs')]
            record['saved_inputs_sha256'] = {str(p):sha(p) for p in inputs}
            record['launch_cmd'] = 'call "'+str(SCRIPTS / 'run_step09_train_merge130.cmd')+'"'
            record['command_line'] = subprocess.list2cmdline([sys.executable, '-I', '-X', 'utf8', str(Path(__file__).resolve()), '--log-dir', str(logs)])
            record['actual_python_argv'] = sys.orig_argv
            for path in inputs:
                shutil.copy2(path, run_dir / 'code' / path.name)
            save_json(run_dir / 'pretraining_record.json', record)
            (run_dir / 'requirements_freeze.txt').write_bytes(subprocess.check_output([sys.executable,'-I','-X','utf8','-m','pip','--isolated','--disable-pip-version-check','freeze','--all']))
            env = os.environ.copy()
            for key in list(env):
                if key.upper() in {'PYTHONHOME','PYTHONPATH'}:
                    del env[key]
            env.update(MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT / 'workspace/cache/matplotlib'),
                       CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_VISIBLE_DEVICES='0')
            print('RUN_DIR='+str(run_dir), flush=True)
            print('User-run MERGE130: fresh same official COCO initialization; no val/test; last epoch100 fixed.', flush=True)
            report['training_process_launched'] = True
            save_json(logs / 'step09_manifest.json', report)
            run = runpy.run_path(str(SCRIPTS / 'step02c_build_mmcv.py'))['run']
            run([sys.executable,'-I','-X','utf8',Path(__file__),'--train-process',run_dir], ROOT, env, logs / 'train_console.log')
            summary = json.loads((run_dir / 'artifacts/formal_training_summary.json').read_text(encoding='utf-8'))
            require(summary['epochs'] == 100 and summary['optimizer_updates'] == 6500, 'Training did not finish100epochs6500updates')
            final = run_dir / 'epoch_100.pth'
            require(final.exists() and not list(run_dir.glob('best_*.pth')), 'Must keep final checkpoint; no val best selection')
            selected = dict(status='SELECTED_LAST_EPOCH_TEST_PENDING', selected_epoch=100,
                checkpoint=str(final), checkpoint_sha256=sha(final), selection_rule=record['selection_rule'],
                train_images=130, independent_validation=False, test_evaluated=False,
                fresh_official_coco_initialization=True, initial_checkpoint_sha256=record['checkpoint']['sha256'])
            save_json(run_dir / 'selected_checkpoint.json', selected)
            report.update(status='TRAINING_COMPLETED_TEST_PENDING', selected_checkpoint=selected)
            print('FINAL_CHECKPOINT='+str(final), flush=True)
            print('STEP09_MERGE130_TRAIN_COMPLETED; TEST_NOT_RUN', flush=True)
            try:
                registered = subprocess.run([sys.executable,'-I','-X','utf8',str(RECORDS)], stdout=subprocess.PIPE,stderr=subprocess.STDOUT,encoding='utf-8')
                print(registered.stdout, end='', flush=True)
                require(registered.returncode == 0, 'Registration failed; do not rerun training')
                report['registration_status']='PASS_AWAITING_TRAINING_RESULT_REVIEW'
                print('STEP09_RECORDS_PASS; send terminal tail, RUN_DIR and LOG_DIR for review', flush=True)
            except Exception:
                report['registration_status']='FAIL_TRAINING_EVIDENCE_SAVED'
                print(traceback.format_exc(), flush=True)
                print('STEP09_RECORDS_FAIL; raw training/checkpoint saved. Do NOT rerun training.', flush=True)
        except BaseException:
            report['traceback']=traceback.format_exc()
            print(report['traceback'], flush=True)
            print('STEP09_FAIL; send complete error and LOG_DIR. Do not automatically retry/resume.', flush=True)
        finally:
            report['launcher_wall_seconds']=time.perf_counter()-started
            save_json(logs / 'step09_manifest.json', report)
            print('RUN_DIR='+str(run_dir), flush=True)
            sys.stdout, sys.stderr = stdout, stderr
    if report['status'] != 'TRAINING_COMPLETED_TEST_PENDING':
        return 1
    return 0 if report.get('registration_status','').startswith('PASS') else 2


if __name__ == '__main__':
    raise SystemExit(main())
