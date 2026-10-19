"""No model/Runner/optimizer training: check artifact math and the inference transform."""
import ast
import contextlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'workspace/scripts'))
from frcnn_debug_hooks import evaluate_train_fit, draw_predictions, plot_losses, write_json
from mmengine.config import Config
from mmdet.utils import register_all_modules
from mmcv.transforms import Compose
import numpy as np
from PIL import Image

register_all_modules(init_default_scope=True)
audit = ROOT / 'workspace/audit/step06_preflight'
audit.mkdir(parents=True, exist_ok=True)
for name in ('frcnn_debug_hooks.py', 'step06_run_debug8.py', 'step06_validate_preparation.py'):
    ast.parse((ROOT / 'workspace/scripts' / name).read_text(encoding='utf-8'))
cfg = Config.fromfile(str(ROOT / 'workspace/configs/frcnn_debug_overfit8.py'))
assert cfg.val_dataloader is None and cfg.test_dataloader is None
assert cfg.train_cfg.max_epochs == 100
assert cfg.model.roi_head.bbox_head.num_classes == 2
assert cfg.model.backbone.init_cfg is None
assert cfg.custom_hooks[0].visual_threshold == 0.3
cfg.dump(str(audit / 'resolved_debug8_preflight.py'))
readback = Config.fromfile(str(audit / 'resolved_debug8_preflight.py'))
assert cfg.to_dict() == readback.to_dict()
source = Path(cfg.data_root) / 'annotations/instances_debug8.json'
truth = json.loads(source.read_text(encoding='utf-8'))
ids = [image['id'] for image in truth['images']]
# Exact GT and empty predictions exercise ID/category mapping and COCO array
# indexing. These synthetic helper checks are NOT model performance results.
perfect = [dict(image_id=a['image_id'], category_id=a['category_id'],
                bbox=a['bbox'], score=0.99) for a in truth['annotations']]
with tempfile.TemporaryDirectory(prefix='artifact_math_', dir=audit) as temp:
    temp = Path(temp)
    with (temp / 'evaluation_stdout.txt').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            exact = evaluate_train_fit(source, perfect, ids, temp / 'synthetic_exact')
            empty = evaluate_train_fit(source, [], ids, temp / 'synthetic_empty')
    for key in ('mAP50', 'mAP50_95', 'AR100'):
        assert np.isclose(exact[key], 1) and empty[key] == 0, key
    for label in ('broken', 'loose'):
        for key in ('AP50', 'AP50_95', 'AR100'):
            assert np.isclose(exact['per_class'][label][key], 1)
            assert empty['per_class'][label][key] == 0
    rows = [dict(iteration=i + 1, loss=1 / (i + 1), loss_rpn_cls=0.1,
                 loss_rpn_bbox=0.2, loss_cls=0.3, loss_bbox=0.4, learning_rate=0.0025)
            for i in range(4)]
    plot_losses(rows, temp / 'synthetic_loss_plot.png')
    assert (temp / 'synthetic_loss_plot.png').stat().st_size > 1000

image_info = truth['images'][0]
image_file = Path(cfg.data_root) / 'images' / image_info['file_name']
with Image.open(image_file) as image:
    rgb = np.array(image.convert('RGB'))
pipeline = Compose([
    dict(type='mmdet.LoadImageFromNDArray'),
    dict(type='mmdet.Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='mmdet.PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
])
packed = pipeline(dict(img=np.ascontiguousarray(rgb[:, :, ::-1]), img_id=0))
assert tuple(packed['inputs'].shape) == (3, 750, 1333)
assert packed['data_samples'].ori_shape == (3078, 5472)
# Clearly labelled render preview, using the known annotation, not a detector.
ann = next(a for a in truth['annotations'] if a['image_id'] == image_info['id'])
x, y, w, h = ann['bbox']
draw_predictions(rgb, [[x, y, x + w, y + h]], [ann['category_id'] - 1], [0.987], 0.3,
                 audit / 'synthetic_annotation_render_NOT_MODEL.jpg')
write_json(audit / 'artifact_helpers_validation.json', {
    'status': 'PASS', 'config_roundtrip': True, 'inference_resize_without_detector': True,
    'synthetic_exact_and_empty_coco_math': True, 'loss_plot_generation': True,
    'model_created': False, 'detector_forward_performed': False,
    'optimizer_created': False, 'training_started': False,
    'note': 'Synthetic metric values intentionally not retained as experiment results.'})
print('STEP06_ARTIFACT_HELPERS_PASS; no model, no optimizer, no forward, no training')
