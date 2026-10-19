"""Check DetDataSample->CocoMetric using saved VAL predictions; no model/test inference."""
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'workspace/scripts'))
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.utils import register_all_modules
from mmdet.registry import METRICS
from frcnn_debug_hooks import write_json
import torch

register_all_modules(init_default_scope=True)
run = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
selected = json.loads((run / 'selected_checkpoint.json').read_text(encoding='utf-8'))
predictions = json.loads(Path(selected['validation_predictions']).read_text(encoding='utf-8'))
truth = json.loads(Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset\annotations\instances_val.json').read_text(encoding='utf-8'))
cfg = Config.fromfile(str(ROOT / 'workspace/configs/frcnn_r50fpn_val001_test.py'))
audit = ROOT / 'workspace/audit/step08_preflight'
with tempfile.TemporaryDirectory(prefix='saved_val_', dir=audit) as temporary:
    spec = cfg.test_evaluator.to_dict()
    spec['ann_file'] = str(Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset\annotations\instances_val.json'))
    spec['outfile_prefix'] = str(Path(temporary) / 'validation_handoff')
    metric = METRICS.build(spec)
    metric.dataset_meta = dict(cfg.metainfo)
    for info in truth['images']:
        rows = [row for row in predictions if row['image_id'] == info['id']]
        sample = DetDataSample(metainfo=dict(img_id=info['id'], ori_shape=(info['height'], info['width'])))
        boxes = [[r['bbox'][0], r['bbox'][1], r['bbox'][0] + r['bbox'][2], r['bbox'][1] + r['bbox'][3]] for r in rows]
        sample.pred_instances = InstanceData(
            bboxes=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            labels=torch.tensor([r['category_id'] - 1 for r in rows], dtype=torch.long),
            scores=torch.tensor([r['score'] for r in rows], dtype=torch.float32))
        metric.process({}, [sample.cpu().to_dict()])
    result = metric.evaluate(23)
    assert result['coco/bbox_mAP'] == selected['native_coco_selection_metrics']['coco/bbox_mAP']
    assert result['coco/bbox_mAP_50'] == selected['native_coco_selection_metrics']['coco/bbox_mAP_50']
write_json(audit / 'metric_handoff_check.json', {
    'status': 'PASS', 'saved_validation_predictions_only': True,
    'native_metric_matches_original_selected_validation': True,
    'DetDataSample_id_class_shape_handoff_verified': True,
    'model_created': False, 'test_inference_performed': False,
    'test_metrics_computed': False, 'training_started': False})
print('STEP08_SAVED_VAL_HANDOFF_PASS; no new inference, no test evaluation, no training')
