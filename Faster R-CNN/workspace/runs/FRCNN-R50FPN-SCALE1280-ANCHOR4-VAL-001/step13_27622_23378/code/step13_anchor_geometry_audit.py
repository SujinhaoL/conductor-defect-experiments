"""Compare one smaller anchor scale using train annotations on CPU; no detector."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
OUT = ROOT / 'workspace/audit/step13_anchor_geometry'
BASE = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115/resolved_scale1280_val_001.py'


def main():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    sys.path.insert(0, str(ROOT / 'workspace/scripts'))
    import mmcv
    import torch
    from mmengine.config import Config
    from mmengine.structures import InstanceData
    from mmdet.registry import TASK_UTILS
    from mmdet.structures.bbox import bbox_overlaps
    from mmdet.utils import register_all_modules
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(BASE))
    assert tuple(cfg.train_pipeline[2].scale) == (2133, 1280)
    assert cfg.model.train_cfg.rpn.allowed_border == -1
    ann_path = DATA / 'annotations/instances_train.json'
    ann_hash = hashlib.sha256(ann_path.read_bytes()).hexdigest()
    selection = json.loads((ROOT / 'workspace/configs/frcnn_debug8_selection.json').read_text(encoding='utf-8'))
    assert ann_hash == selection['all_coco_hashes']['train']
    truth = json.loads(ann_path.read_text(encoding='utf-8'))
    assert (len(truth['images']), len(truth['annotations'])) == (107, 168)
    assigner = TASK_UTILS.build(cfg.model.train_cfg.rpn.assigner)
    generators = {}
    for scale in (8, 4):
        spec = cfg.model.rpn_head.anchor_generator.copy()
        spec['scales'] = [scale]
        generators[scale] = TASK_UTILS.build(spec)
        assert generators[scale].num_base_priors == [3] * 5
    rows = []
    for index, img in enumerate(truth['images'], 1):
        annotations = [a for a in truth['annotations'] if a['image_id'] == img['id']]
        width, height = mmcv.rescale_size((img['width'], img['height']), (2133, 1280))
        # Match CocoDataset/LoadAnnotations: xywh -> xyxy in Python before float32.
        boxes = torch.tensor([[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                               a['bbox'][1] + a['bbox'][3]] for a in annotations], dtype=torch.float32)
        boxes *= torch.tensor([width / img['width'], height / img['height']] * 2)
        padded = (math.ceil(height / 32) * 32, math.ceil(width / 32) * 32)
        sizes = [(math.ceil(padded[0] / s), math.ceil(padded[1] / s)) for s in (4, 8, 16, 32, 64)]
        for scale, generator in generators.items():
            anchors = torch.cat(generator.grid_priors(sizes, device='cpu'))
            flags = torch.cat(generator.valid_flags(sizes, padded, device='cpu'))
            anchors = anchors[flags]
            for flipped in (False, True):
                gt = boxes.clone()
                if flipped:
                    gt[:, 0] = width - boxes[:, 2]
                    gt[:, 2] = width - boxes[:, 0]
                result = assigner.assign(InstanceData(priors=anchors),
                    InstanceData(bboxes=gt, labels=torch.zeros(len(gt), dtype=torch.long)))
                max_iou = bbox_overlaps(gt, anchors).max(dim=1).values.tolist()
                for j, ann in enumerate(annotations):
                    rows.append(dict(anchor_scale=scale, flipped=flipped, image_id=img['id'],
                        file_name=img['file_name'], annotation_id=ann['id'],
                        category=('broken', 'loose')[ann['category_id'] - 1],
                        resized_width=width, resized_height=height,
                        box_width=float(gt[j, 2] - gt[j, 0]), box_height=float(gt[j, 3] - gt[j, 1]),
                        max_anchor_iou=max_iou[j],
                        positive_anchors_before_sampling=int((result.gt_inds == j + 1).sum())))
        if index % 20 == 0 or index == 107:
            print(f'ANCHOR_GEOMETRY_TRAIN={index}/107; CPU only', flush=True)
    assert len(rows) == 168 * 4
    stats = {}
    for scale in (8, 4):
        for flipped in (False, True):
            part = [r for r in rows if r['anchor_scale'] == scale and r['flipped'] == flipped]
            zero = [r for r in part if r['positive_anchors_before_sampling'] == 0]
            stats[f'scale{scale}_flip{int(flipped)}'] = dict(boxes=len(part),
                zero_positive_gt=len(zero), zero_positive_ids=[r['annotation_id'] for r in zero],
                zero_positive_by_class={c: sum(r['category'] == c for r in zero) for c in ('broken', 'loose')})
    changes = {}
    for flipped in (False, True):
        old = {r['annotation_id']: r for r in rows if r['anchor_scale'] == 8 and r['flipped'] == flipped}
        new = {r['annotation_id']: r for r in rows if r['anchor_scale'] == 4 and r['flipped'] == flipped}
        changes[str(flipped)] = dict(
            gained_positive_ids=[i for i in old if old[i]['positive_anchors_before_sampling'] == 0 and new[i]['positive_anchors_before_sampling'] > 0],
            lost_positive_ids=[i for i in old if old[i]['positive_anchors_before_sampling'] > 0 and new[i]['positive_anchors_before_sampling'] == 0])
    # Independent consistency check against previous debug8 default-anchor audit.
    old_rows = list(csv.DictReader((ROOT / 'workspace/audit/step11_resolution/debug8_anchor_sizes.csv').open(encoding='utf-8-sig', newline='')))
    current = {r['annotation_id']: r for r in rows if r['anchor_scale'] == 8 and not r['flipped']}
    for old in old_rows:
        if old['scale_short'] != '1280':
            continue
        new = current[int(old['annotation_id'])]
        assert abs(new['max_anchor_iou'] - float(old['max_anchor_iou'])) < 1e-6, (old['annotation_id'], new['max_anchor_iou'], old['max_anchor_iou'])
        assert new['positive_anchors_before_sampling'] == int(old['positive_anchors_before_sampling'])
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'train107_anchor_comparison.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    result = dict(status='PASS', annotation_path=str(ann_path), annotation_sha256=ann_hash,
        source_config=str(BASE), source_config_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_created=False, detector_forward_performed=False, training_started=False,
        evaluation_images_used=False, train_images=107, train_boxes=168, resize_scale=[2133, 1280],
        anchor_scales_compared=[8, 4], anchor_ratios=[0.5, 1.0, 2.0],
        stats=stats, changes=changes,
        four_diagnostic_targets=[r for r in rows if r['annotation_id'] in (87, 123, 124, 160)],
        notes=['Actual MMDetection generator/assigner on CPU; both flip states; before random sampling.',
               'Valid flags are based on each image padded to 32; extra batch padding contributes invalid anchors.',
               'Counts are geometry diagnostics, not measured proposal recall, AP, or a guarantee of learning.',
               'Only one preselected candidate scale4 compared to default scale8; no test set used.'])
    (OUT / 'anchor_geometry.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(stats=stats, changes=changes), ensure_ascii=False, indent=2))
    print('STEP13_ANCHOR_GEOMETRY_PASS; no model, no training, no inference')


if __name__ == '__main__':
    main()
