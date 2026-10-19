"""Validate native COCO metric and empty-prediction handling without a model/training."""
import ast
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'workspace/scripts'))
import torch
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmdet.registry import METRICS
from frcnn_formal_hooks import FormalBaselineArtifactsHook
from frcnn_debug_hooks import evaluate_train_fit, write_json

register_all_modules(init_default_scope=True)
audit = ROOT / 'workspace/audit/step07_preflight'
audit.mkdir(parents=True, exist_ok=True)
for name in ('frcnn_formal_hooks.py', 'frcnn_debug_hooks.py', 'step07_run_val_baseline.py', 'step07_validate_preparation.py'):
    ast.parse((ROOT / 'workspace/scripts' / name).read_text(encoding='utf-8'))
cfg = Config.fromfile(str(ROOT / 'workspace/configs/frcnn_r50fpn_val_001.py'))
cfg.dump(str(audit / 'resolved_val_001.py'))
readback = Config.fromfile(str(audit / 'resolved_val_001.py'))
assert cfg.to_dict() == readback.to_dict()
truth = json.loads(Path(cfg.val_evaluator.ann_file).read_text(encoding='utf-8'))
assert len(truth['images']) == 23 and len(truth['annotations']) == 37
with tempfile.TemporaryDirectory(prefix='native_coco_', dir=audit) as temporary:
    temporary = Path(temporary)
    for epoch, empty in [(1, False), (2, True)]:
        hook = FormalBaselineArtifactsHook()
        hook.output = temporary
        hook.validation_rows = []
        metric = METRICS.build(copy.deepcopy(cfg.val_evaluator))
        metric.dataset_meta = dict(cfg.metainfo)
        runner = SimpleNamespace(epoch=epoch, cfg=cfg, val_evaluator=SimpleNamespace(metrics=[metric]))
        hook.before_val_epoch(runner)
        for image in truth['images']:
            anns = [] if empty else [ann for ann in truth['annotations'] if ann['image_id'] == image['id']]
            xyxy = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2], a['bbox'][1] + a['bbox'][3]] for a in anns]
            # Synthetic exact GT predictions exercise mapping/metric indexing.
            # They are deliberately not retained as model performance results.
            sample = dict(img_id=image['id'], ori_shape=(image['height'], image['width']),
                          pred_instances=dict(bboxes=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
                                              labels=torch.tensor([a['category_id'] - 1 for a in anns], dtype=torch.long),
                                              scores=torch.full((len(anns),), 0.99)))
            metric.process({}, [sample])
        with (temporary / f'epoch_{epoch}_stdout.txt').open('w', encoding='utf-8') as output:
            with contextlib.redirect_stdout(output):
                metrics = metric.evaluate(23)
                hook.after_val_epoch(runner, metrics)
                prediction_file = temporary / 'validation' / f'epoch_{epoch:03d}' / 'predictions.bbox.json'
                predictions = json.loads(prediction_file.read_text(encoding='utf-8'))
                offline = evaluate_train_fit(cfg.val_evaluator.ann_file, predictions,
                                             [image['id'] for image in truth['images']], temporary / f'offline_{epoch}',
                                             scope='synthetic helper verification; not experiment results')
        assert metrics['coco/bbox_mAP'] == (0 if empty else 1)
        assert metrics['coco/bbox_mAP_50'] == (0 if empty else 1)
        assert abs(offline['mAP50_95'] - metrics['coco/bbox_mAP']) < 1e-6
        assert hook.validation_rows[0]['epoch'] == epoch
        assert all(offline['per_class'][label]['AP50'] == (0 if empty else 1) or
                   abs(offline['per_class'][label]['AP50'] - 1) < 1e-6 for label in ('broken', 'loose'))
    assert FormalBaselineArtifactsHook.priority < 60
write_json(audit / 'validation_helpers_check.json', {
    'status': 'PASS', 'config_serialization': True, 'native_coco_class_mapping': True,
    'native_and_offline_coco_agreement': True, 'zero_prediction_validation_handled': True,
    'validation_epoch_directory_numbering': True, 'peak_capture_before_logger_reset': True,
    'model_created': False, 'optimizer_created': False, 'detector_forward_performed': False,
    'training_started': False, 'note': 'Synthetic exact/empty metric values are not saved as experimental results.'})
print('STEP07_HELPERS_VALIDATION_PASS: no model, no optimizer, no detector forward, no training')
