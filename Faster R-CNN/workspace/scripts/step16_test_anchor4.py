"""User-run test of the fixed SCALE1280-ANCHOR4-VAL-001 epoch15 checkpoint. Never starts training."""
import argparse
import builtins
import copy
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
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001/step13_27622_23378'
CONFIG = ROOT / 'workspace/configs/frcnn_r50fpn_anchor4_test.py'
SELECTION = RUN / 'selected_checkpoint.json'
TEST_ROOT = RUN / 'test_selected_epoch15'
REVIEW = ROOT / 'workspace/audit/step13_review/acceptance.json'
QUEUE = ROOT.parent / 'baseline_analysis/audit/test_completion_review_20261014/fixed_test_queue.json'
INVENTORY = Path('D:/小论文/conductor_defect_dataset/03_coco_dataset/audit/source_inventory.json')
helpers = runpy.run_path(str(SCRIPTS / 'step04_convert_coco.py'))
require, sha, save_json, Tee = (helpers[key] for key in ('require', 'sha', 'save_json', 'Tee'))


def load_verified_checkpoint(selected):
    """Restricted loading, scoped to this hash-verified locally trained checkpoint."""
    import numpy as np
    import torch
    from mmengine.logging.history_buffer import HistoryBuffer
    checkpoint = Path(selected['best_checkpoint']).resolve()
    require(checkpoint == RUN / 'best_coco_bbox_mAP_epoch_15.pth', 'Unexpected selected checkpoint path')
    require(sha(checkpoint) == selected['best_checkpoint_sha256'], 'Selected checkpoint changed')
    known = {
        'mmengine.logging.history_buffer.HistoryBuffer': HistoryBuffer,
        'numpy.core.multiarray._reconstruct': np.core.multiarray._reconstruct,
        'numpy.core.multiarray.scalar': np.core.multiarray.scalar,
        'numpy.ndarray': np.ndarray, 'numpy.dtype': np.dtype, 'builtins.getattr': builtins.getattr,
    }
    present = torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint))
    require(set(present) <= set(known), 'Checkpoint contains an unexpected metadata type: ' + str(present))
    # NumPy dtype classes are dynamically reconstructed and are not all listed
    # by the static GLOBAL scanner. These are trusted installed NumPy types.
    dtype_classes = [type(np.dtype(name)) for name in ('float64', 'float32', 'int64', 'int32', 'uint8', 'bool')]
    before = set(torch.serialization.get_safe_globals())
    with torch.serialization.safe_globals([known[name] for name in present] + dtype_classes):
        data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(set(torch.serialization.get_safe_globals()) == before, 'Scoped load modified persistent allowlist')
    require(data['meta']['epoch'] == 15 and data['meta']['iter'] == 15 * 54, 'Checkpoint epoch/iteration differs')
    require(tuple(data['meta']['dataset_meta']['classes']) == ('broken', 'loose'), 'Checkpoint class metadata differs')
    require(len(data['state_dict']) == 348, 'Unexpected state tensor count')
    return data, present


def check_preparation():
    from mmengine.config import Config
    from mmdet.utils import register_all_modules
    from mmdet.registry import DATASETS
    import mmdet
    import torch
    runpy.run_path(str(SCRIPTS / 'step01_check_cuda.py'))['check_identity']()
    register_all_modules(init_default_scope=True)
    selected = json.loads(SELECTION.read_text(encoding='utf-8'))
    record = json.loads((RUN / 'pretraining_record.json').read_text(encoding='utf-8'))
    review = json.loads(REVIEW.read_text(encoding='utf-8'))
    queued = json.loads(QUEUE.read_text(encoding='utf-8'))['items'][1]
    require(queued['order'] == 2 and queued['epoch'] == 15 and queued['training_run_id'] == RUN.name,
            'Fixed test order/epoch/run changed')
    require(review['status'] == 'PASS' and review['selected_epoch'] == 15, 'Training review did not pass')
    require(Path(queued['checkpoint']).resolve() == Path(selected['best_checkpoint']).resolve(), 'Queue checkpoint differs')
    require(selected['best_checkpoint_sha256'] == queued['checkpoint_sha256'] == review['selected_checkpoint_sha256'],
            'Selected checkpoint hash differs from accepted review/fixed queue')
    require(sha(SELECTION) == queued['selection_record_sha256'] == review['source_files_sha256'][str(SELECTION)],
            'Validation selection record changed')
    frozen_path = RUN / 'resolved_scale1280_anchor4_val_001.py'
    require(sha(frozen_path) == review['source_files_sha256'][str(frozen_path)], 'Frozen training config changed')
    require(selected['selected_epoch'] == 15 and selected['criterion'] == 'coco/bbox_mAP', 'Validation selection changed')
    versions = {name: importlib.metadata.version(name) for name in record['versions']}
    require(versions == record['versions'], 'Dedicated environment versions changed')
    source = ROOT / 'mmdetection'
    require(Path(mmdet.__file__).resolve().is_relative_to(source), 'Wrong MMDetection import')
    git = ['git', '-c', 'safe.directory=' + str(source)]
    require(subprocess.check_output(git + ['rev-parse', 'HEAD'], cwd=source, text=True).strip() == record['mmdet_commit'], 'Official source commit changed')
    require(not subprocess.check_output(git + ['diff', 'HEAD', '--name-only'], cwd=source), 'Official tracked code changed')
    cfg = Config.fromfile(str(CONFIG))
    frozen = Config.fromfile(str(frozen_path), import_custom_modules=False)
    require(cfg.model.to_dict() == frozen.model.to_dict(), 'Model or prediction settings changed after validation selection')
    require(Path(cfg.load_from).resolve() == Path(selected['best_checkpoint']).resolve(), 'Test config does not use selected checkpoint')
    require(cfg.train_cfg is None and cfg.train_dataloader is None and cfg.val_cfg is None and cfg.val_dataloader is None,
            'Evaluation config unexpectedly enables training/validation')
    require(cfg.optim_wrapper is None and cfg.param_scheduler is None and not cfg.resume and not cfg.custom_hooks,
            'Evaluation config enables optimization, resume or training artifacts')
    require(list(cfg.model.rpn_head.anchor_generator.scales) == [4], 'Expected anchor4')
    require(cfg.test_dataloader.dataset.pipeline == frozen.val_dataloader.dataset.pipeline,
            'Test preprocessing differs from selected validation preprocessing')
    current_metric, original_metric = cfg.test_evaluator.to_dict(), frozen.val_evaluator.to_dict()
    current_metric.pop('ann_file')
    original_metric.pop('ann_file')
    require(current_metric == original_metric, 'Test evaluator differs from validation beyond annotation split')
    require(cfg.model.test_cfg.rcnn.score_thr == 0.001 and cfg.model.test_cfg.rcnn.max_per_img == 100,
            'Fixed prediction thresholds/count changed')
    require(list(cfg.test_evaluator.proposal_nums) == [1, 10, 100], 'COCO evaluation cap changed')
    require(tuple(cfg.test_dataloader.dataset.pipeline[1].scale) == (2133, 1280)
            and cfg.test_dataloader.dataset.pipeline[1].keep_ratio, 'Test resize changed')
    annotation = Path(cfg.test_evaluator.ann_file)
    require(sha(annotation) == record['annotation_hashes']['test'], 'Original test annotation changed')
    truth = json.loads(annotation.read_text(encoding='utf-8'))
    require(len(truth['images']) == 23 and len(truth['annotations']) == 36, 'Expected original 23 images / 36 boxes')
    require(sum(a['category_id'] == 1 for a in truth['annotations']) == 8
            and sum(a['category_id'] == 2 for a in truth['annotations']) == 28, 'Test class counts differ')
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    require(len(dataset) == 23 and dataset.cat2label == {1: 0, 2: 1}, 'Test reader class mapping differs')
    require([dataset.get_data_info(i)['img_id'] for i in range(23)] == [image['id'] for image in truth['images']], 'Test image IDs/order differ')
    require(all('/images/test/' in dataset.get_data_info(i)['img_path'].replace('\\', '/') for i in range(23)), 'Dataset references a non-test split')
    inventory = {row['filename']: row for row in json.loads(INVENTORY.read_text(encoding='utf-8')) if row['split'] == 'test'}
    require(len(inventory) == 23, 'Original image inventory does not contain 23 test images')
    image_hashes = {}
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        path = Path(info['img_path'])
        expected = inventory[path.name]
        digest = sha(path)
        require(digest == expected['image_sha256'], 'Test image changed: ' + str(path))
        require((info['width'], info['height']) == (expected['width'], expected['height']), 'Test image size metadata changed')
        image_hashes[str(path)] = digest
    require(torch.cuda.is_available() and torch.version.cuda == '12.8', 'CUDA environment unavailable')
    metadata = dict(status='PASS', selected_checkpoint=str(cfg.load_from), selected_epoch=15,
                    selected_checkpoint_sha256=selected['best_checkpoint_sha256'], selection_sha256=sha(SELECTION),
                    test_annotation_sha256=record['annotation_hashes']['test'], images=23, boxes=36,
                    broken=8, loose=28, versions=versions, source_commit=record['mmdet_commit'],
                    score_threshold=0.001, visualization_score_threshold=0.30, max_per_image=100,
                    rcnn_nms_iou=0.5, coco_max_dets=[1, 10, 100], resize=[2133, 1280], keep_ratio=True, anchor_scales=[4],
                    training_started=False, test_inference_performed=False, model_created=False,
                    prior_test_results_already_seen=True, queue_manifest_sha256=sha(QUEUE),
                    reviewed_selection_sha256=sha(REVIEW), source_inventory_sha256=sha(INVENTORY),
                    test_image_sha256=image_hashes,
                    training_config_sha256=sha(RUN / 'resolved_scale1280_anchor4_val_001.py'))
    print('STEP16_PREPARATION_PASS: selected_epoch=15 test_images=23 boxes=36', flush=True)
    return cfg, dataset, selected, metadata


def load_model(cfg, selected, destination):
    import torch
    from mmdet.registry import MODELS
    data, allowed = load_verified_checkpoint(selected)
    model = MODELS.build(cfg.model).cpu()
    incompatible = model.load_state_dict(data['state_dict'], strict=True)
    actual = model.state_dict()
    require(not incompatible.missing_keys and not incompatible.unexpected_keys, 'Checkpoint keys do not fully match')
    require(all(torch.equal(actual[key], value) for key, value in data['state_dict'].items()), 'A selected checkpoint tensor failed to load identically')
    audit = dict(status='PASS', checkpoint=str(cfg.load_from), sha256=selected['best_checkpoint_sha256'],
                 selected_epoch=15, loaded_tensors=len(actual), missing_keys=[], unexpected_keys=[],
                 shape_mismatches=[], class_names=['broken', 'loose'], restricted_weights_only=True,
                 scoped_metadata_types=allowed, persistent_allowlist_modified=False,
                 detector_forward_performed=False, training_started=False)
    save_json(destination / 'selected_checkpoint_load_audit.json', audit)
    model.cfg = cfg
    model.dataset_meta = dict(cfg.metainfo)
    model.eval()
    del data, actual
    print('SELECTED_CHECKPOINT_FULL_LOAD_PASS: matched=348 missing=0 unexpected=0', flush=True)
    return model


def evaluate_test(cfg, dataset, model, destination):
    import numpy as np
    import torch
    from PIL import Image
    from mmcv.transforms import Compose
    from mmdet.apis import inference_detector
    from mmdet.registry import METRICS
    from mmengine.runner import set_random_seed
    from frcnn_debug_hooks import draw_predictions, evaluate_train_fit
    evaluator_cfg = cfg.test_evaluator.to_dict()
    evaluator_cfg['outfile_prefix'] = str(destination / 'predictions')
    metric = METRICS.build(evaluator_cfg)
    metric.dataset_meta = dict(cfg.metainfo)
    # Ground-truth annotations are only read by the metric, never passed to the detector.
    pipeline_spec = [copy.deepcopy(dict(step)) for step in cfg.test_dataloader.dataset.pipeline
                     if step['type'] != 'LoadAnnotations']
    pipeline_spec[0] = dict(type='mmdet.LoadImageFromNDArray')
    pipeline = Compose(pipeline_spec)
    (destination / 'predicted_images').mkdir()
    details, image_ids = [], []
    # Apply the same deterministic runtime used for validation in STEP13.
    set_random_seed(42, deterministic=True)
    torch.backends.cudnn.benchmark = False
    model.to('cuda:0')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        with Image.open(info['img_path']) as image:
            rgb = np.array(image.convert('RGB'))
        require(rgb.shape[:2] == (info['height'], info['width']), 'Test image dimensions changed')
        result = inference_detector(model, np.ascontiguousarray(rgb[:, :, ::-1]), test_pipeline=pipeline)
        result.set_metainfo(dict(img_id=int(info['img_id']), img_path=info['img_path']))
        prediction = result.pred_instances.cpu()
        require(torch.isfinite(prediction.bboxes).all().item() and torch.isfinite(prediction.scores).all().item(), 'Non-finite prediction')
        boxes, labels, scores = prediction.bboxes.tolist(), prediction.labels.tolist(), prediction.scores.tolist()
        require(set(labels) <= {0, 1} and len(scores) <= 100, 'Prediction mapping/count differs')
        metric.process({}, [result.cpu().to_dict()])
        row = dict(image_id=int(info['img_id']), image_path=info['img_path'], boxes_xyxy=boxes,
                   model_labels=labels, classes=[('broken', 'loose')[label] for label in labels], scores=scores,
                   original_shape=list(result.ori_shape), resized_shape=list(result.img_shape),
                   scale_factor=[float(value) for value in result.scale_factor],
                   returned_count=len(scores), visible_count=sum(score > 0.30 for score in scores))
        details.append(row)
        image_ids.append(row['image_id'])
        # Append before drawing so partial output remains usable after an I/O error.
        with (destination / 'predictions_details.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        draw_predictions(rgb, boxes, labels, scores, 0.30,
                         destination / 'predicted_images' / (Path(info['img_path']).stem + '.jpg'))
        print(f'TEST_IMAGE={index + 1}/23 {Path(info["img_path"]).name} '
              f'returned={len(scores)} visible_above_0.3={row["visible_count"]}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    native = metric.evaluate(23)
    save_json(destination / 'native_coco_metrics.json', native)
    predictions = json.loads((destination / 'predictions.bbox.json').read_text(encoding='utf-8'))
    exact = evaluate_train_fit(cfg.test_evaluator.ann_file, predictions, image_ids, destination / 'test',
                               scope='original held-out test23; checkpoint fixed by val before this test; other experiments on this test have already been viewed')
    if native:
        require(abs(exact['mAP50_95'] - native['coco/bbox_mAP']) <= 0.00051
                and abs(exact['mAP50'] - native['coco/bbox_mAP_50']) <= 0.00051, 'Native/offline AP disagreement')
    else:
        require(not predictions and exact['mAP50_95'] == 0 and exact['mAP50'] == 0, 'Unexpected empty native metrics')
    save_json(destination / 'predictions_details.json', details)
    return dict(status='PASS', experiment='FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001', split='original test',
                selected_epoch=15, images=23, boxes=36, broken=8, loose=28,
                metrics=exact, native_coco_metrics=native,
                returned_predictions=sum(row['returned_count'] for row in details),
                visible_predictions_above_0_3=sum(row['visible_count'] for row in details),
                resize_scale=[2133, 1280], anchor_scales=[4],
                model_score_threshold=0.001, visualization_score_threshold=0.30, max_per_image=100,
                rcnn_nms_iou=0.5, inference_and_image_saving_seconds=elapsed,
                peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                training_started=False, test_used_for_epoch_selection=False,
                prior_test_results_already_seen=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-preparation', action='store_true')
    parser.add_argument('--log-dir', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(SCRIPTS))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.check_preparation:
        cfg, _, selected, metadata = check_preparation()
        audit = ROOT / 'workspace/audit/step16_preflight'
        audit.mkdir(parents=True, exist_ok=True)
        data, allowed = load_verified_checkpoint(selected)
        save_json(audit / 'selected_checkpoint_metadata.json', dict(status='PASS', selected_epoch=15,
            checkpoint_sha256=selected['best_checkpoint_sha256'], state_dict_items=len(data['state_dict']),
            epoch=data['meta']['epoch'], iterations=data['meta']['iter'], scoped_metadata_types=allowed,
            model_created=False, detector_forward_performed=False))
        del data
        cfg.dump(str(audit / 'resolved_test_config.py'))
        save_json(audit / 'preparation_check.json', metadata)
        print('STEP16_CPU_METADATA_CHECK_PASS; no model created, no detector forward, no test inference, no training', flush=True)
        return 0
    require(args.log_dir is not None, '--log-dir required')
    logs = args.log_dir.resolve()
    require(logs.is_relative_to(ROOT / 'workspace/logs') and logs.name.startswith('step16_'), 'Unexpected log path')
    logs.mkdir(parents=True, exist_ok=True)
    destination = TEST_ROOT / logs.name
    stdout, stderr = sys.stdout, sys.stderr
    report = dict(status='FAIL', started_utc=datetime.now(timezone.utc).isoformat(),
                  result_dir=str(destination), log_dir=str(logs), training_started=False,
                  test_inference_started=False)
    with (logs / 'step16_console.txt').open('x', encoding='utf-8') as output:
        sys.stdout, sys.stderr = Tee(stdout, output), Tee(stderr, output)
        try:
            cfg, dataset, selected, metadata = check_preparation()
            require(not TEST_ROOT.exists(), 'A previous test attempt exists; inspect it before any retry')
            TEST_ROOT.mkdir(exist_ok=False)
            require(not destination.exists(), 'Result directory exists; no overwrite')
            destination.mkdir(parents=True)
            (destination / 'code').mkdir()
            files = [CONFIG, Path(__file__), SCRIPTS / 'run_step16_test_anchor4.cmd',
                     SCRIPTS / 'frcnn_debug_hooks.py', SELECTION, RUN / 'resolved_scale1280_anchor4_val_001.py',
                     Path(cfg.test_evaluator.ann_file), REVIEW, QUEUE, INVENTORY,
                     ROOT / 'workspace/notes/STEP16_anchor4_final_test.txt',
                     ROOT / 'workspace/notes/EXPERIMENT_RECORDING_RULES.md']
            metadata['saved_inputs_sha256'] = {str(path): sha(path) for path in files}
            metadata['launch_cmd'] = 'call "' + str(SCRIPTS / 'run_step16_test_anchor4.cmd') + '"'
            metadata['command_line'] = subprocess.list2cmdline([
                sys.executable, '-I', '-X', 'utf8', str(Path(__file__).resolve()), '--log-dir', str(logs)])
            metadata['actual_python_argv'] = sys.orig_argv
            metadata['training_run_id'] = RUN.name
            metadata['evaluation_run_id'] = logs.name
            metadata['registration_status'] = 'PENDING_RESULT_REVIEW'
            for path in files:
                shutil.copy2(path, destination / 'code' / path.name)
            cfg.work_dir = str(destination)
            cfg.dump(str(destination / 'resolved_test_config.py'))
            save_json(destination / 'pretest_record.json', metadata)
            model = load_model(cfg, selected, destination)
            print('RESULT_DIR=' + str(destination), flush=True)
            report['test_inference_started'] = True
            save_json(logs / 'step16_manifest.json', report)
            summary = evaluate_test(cfg, dataset, model, destination)
            require(sha(Path(selected['best_checkpoint'])) == selected['best_checkpoint_sha256'], 'Checkpoint file changed during test')
            summary.update(selected_checkpoint=selected['best_checkpoint'],
                           selected_checkpoint_sha256=selected['best_checkpoint_sha256'])
            save_json(destination / 'test_summary.json', summary)
            report.update(status='PASS', summary=str(destination / 'test_summary.json'))
            evidence_files = [destination / name for name in (
                'test_summary.json', 'native_coco_metrics.json', 'test_metrics.json',
                'predictions.bbox.json', 'predictions_details.json', 'pretest_record.json',
                'selected_checkpoint_load_audit.json', 'resolved_test_config.py')]
            evidence_files += sorted((destination / 'predicted_images').glob('*.jpg'))
            save_json(destination / 'test_evidence_manifest.json', dict(
                status='PASS', files_sha256={str(path): sha(path) for path in evidence_files},
                checkpoint_path=selected['best_checkpoint'], checkpoint_sha256=selected['best_checkpoint_sha256'],
                no_training=True, selected_before_test=True))
            print(f'TEST_mAP50={summary["metrics"]["mAP50"]:.6f} '
                  f'TEST_mAP50_95={summary["metrics"]["mAP50_95"]:.6f} TEST_AR100={summary["metrics"]["AR100"]:.6f}', flush=True)
            print('RETURNED_PREDICTIONS=' + str(summary['returned_predictions']), flush=True)
            print('VISIBLE_PREDICTIONS_ABOVE_0.3=' + str(summary['visible_predictions_above_0_3']), flush=True)
            print('STEP16_TEST_EVAL_PASS', flush=True)
            report['registration_status'] = 'PENDING_RESULT_REVIEW'
            print('STEP16_RAW_RESULTS_SAVED; send output for review and CSV/Excel registration. Do not rerun test.', flush=True)
        except BaseException:
            report['traceback'] = traceback.format_exc()
            print(report['traceback'], flush=True)
            print('STEP16_FAIL; send full error and LOG_DIR. No training was started.', flush=True)
        finally:
            save_json(logs / 'step16_manifest.json', report)
            print('RESULT_DIR=' + str(destination), flush=True)
            sys.stdout, sys.stderr = stdout, stderr
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
