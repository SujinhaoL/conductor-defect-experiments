"""CPU evaluation of already saved epoch15 VAL predictions; no detector/test inference."""
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
from step16_test_anchor4 import RUN, CONFIG, REVIEW, SELECTION, sha, require
import torch

register_all_modules(init_default_scope=True)
selected = json.loads(SELECTION.read_text(encoding='utf-8'))
review = json.loads(REVIEW.read_text(encoding='utf-8'))
prediction_path = Path(selected['validation_predictions'])
require(sha(prediction_path) == review['source_files_sha256'][str(prediction_path)], 'Saved val predictions changed')
predictions = json.loads(prediction_path.read_text(encoding='utf-8'))
annotation = Path('D:/小论文/conductor_defect_dataset/03_coco_dataset/annotations/instances_val.json')
record = json.loads((RUN / 'pretraining_record.json').read_text(encoding='utf-8'))
require(sha(annotation) == record['annotation_hashes']['val'], 'Original val annotations changed')
truth = json.loads(annotation.read_text(encoding='utf-8'))
cfg = Config.fromfile(str(CONFIG))
audit = ROOT / 'workspace/audit/step16_preflight'
audit.mkdir(exist_ok=True, parents=True)
with tempfile.TemporaryDirectory(prefix='saved_val_', dir=audit) as temporary:
    spec = cfg.test_evaluator.to_dict()
    spec['ann_file'] = str(annotation)
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
    expected = selected['native_coco_selection_metrics']
    for key in ('coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75',
                'coco/broken_precision', 'coco/loose_precision'):
        require(result[key] == expected[key], 'Saved val metric handoff mismatch: ' + key)
write_json(audit / 'metric_handoff_check.json', {
    'status': 'PASS', 'saved_validation_predictions_only': True,
    'native_metric_matches_original_selected_validation': True,
    'DetDataSample_id_class_shape_handoff_verified': True,
    'source_predictions': str(prediction_path), 'source_predictions_sha256': sha(prediction_path),
    'computed_val_metrics': result,
    'model_created': False, 'test_inference_performed': False,
    'test_metrics_computed': False, 'training_started': False})
print('STEP16_SAVED_VAL_HANDOFF_PASS; no new inference, no test evaluation, no training')
