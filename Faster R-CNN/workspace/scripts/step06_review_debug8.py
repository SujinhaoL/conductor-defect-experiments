"""Review saved DEBUG8 predictions and static RPN assignment; never trains a model."""
import csv
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/FRCNN-DEBUG-OVERFIT-8/step06_24506_3904'
OUT = ROOT / 'workspace/audit/step06_review'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / 'workspace/scripts'))
from frcnn_debug_hooks import write_json
from mmengine.config import Config
from mmengine.structures import InstanceData
from mmdet.utils import register_all_modules
from mmdet.registry import DATASETS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps
import torch

register_all_modules(init_default_scope=True)
cfg = Config.fromfile(str(RUN / 'resolved_debug8.py'))
dataset = DATASETS.build(cfg.train_dataloader.dataset)
generator = TASK_UTILS.build(cfg.model.rpn_head.anchor_generator)
assigner = TASK_UTILS.build(cfg.model.train_cfg.rpn.assigner)
truth = json.loads(Path(dataset.ann_file).read_text(encoding='utf-8'))
predictions = json.loads((RUN / 'diagnostics/epoch_100/predictions_coco.json').read_text(encoding='utf-8'))
pipeline = json.loads(Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset\audit\debug8\step05_20493_32385\pipeline_report.json').read_text(encoding='utf-8'))
rows = []
for index in range(8):
    info = dataset.get_data_info(index)
    sample = dataset[index]['data_samples']
    boxes = sample.gt_instances.bboxes.tensor
    # FPN maps derive from the known batch padding; no detector forward needed.
    batch_hw = pipeline['images'][index]['padded_batch_hw']
    sizes = [(math.ceil(batch_hw[0] / s), math.ceil(batch_hw[1] / s)) for s in (4, 8, 16, 32, 64)]
    padded_hw = tuple(math.ceil(v / 32) * 32 for v in sample.img_shape[:2])
    anchors = torch.cat(generator.grid_priors(sizes, device='cpu'))
    flags = torch.cat(generator.valid_flags(sizes, padded_hw, device='cpu'))
    anchors = anchors[flags]  # official allowed_border=-1 keeps all valid anchors
    assigned = assigner.assign(InstanceData(priors=anchors),
                               InstanceData(bboxes=boxes, labels=torch.zeros(len(boxes), dtype=torch.long)))
    anchor_iou = bbox_overlaps(boxes, anchors).max(dim=1).values.tolist()
    anns = [a for a in truth['annotations'] if a['image_id'] == info['img_id']]
    assert len(anns) == len(boxes)
    for k, ann in enumerate(anns):
        x, y, w, h = ann['bbox']
        gt = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        candidates = [p for p in predictions if p['image_id'] == ann['image_id'] and p['category_id'] == ann['category_id']]
        def best_iou(items):
            if not items:
                return 0.0, None
            tensor = torch.tensor([[p['bbox'][0], p['bbox'][1], p['bbox'][0] + p['bbox'][2],
                                    p['bbox'][1] + p['bbox'][3]] for p in items], dtype=torch.float32)
            values = bbox_overlaps(gt, tensor)[0]
            arg = int(values.argmax())
            return float(values[arg]), float(items[arg]['score'])
        best, score = best_iou(candidates)
        visible, _ = best_iou([p for p in candidates if p['score'] > 0.3])
        row = dict(image=Path(info['img_path']).name, annotation_id=ann['id'],
                   category=('broken', 'loose')[ann['category_id'] - 1],
                   resized_width=float(boxes[k, 2] - boxes[k, 0]),
                   resized_height=float(boxes[k, 3] - boxes[k, 1]),
                   max_default_anchor_iou=anchor_iou[k],
                   assigned_positive_anchors_before_sampling=int((assigned.gt_inds == k + 1).sum()),
                   best_returned_same_class_iou=best, score_of_best_iou_box=score,
                   best_visible_same_class_iou=visible)
        rows.append(row)
        print(f'{row["image"]} {row["category"]}: size={row["resized_width"]:.2f}x{row["resized_height"]:.2f} '
              f'anchorIoU={anchor_iou[k]:.3f} positive_anchors={row["assigned_positive_anchors_before_sampling"]} '
              f'predictionIoU={best:.3f} score={score}', flush=True)
with (OUT / 'per_box_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
losses = [json.loads(line) for line in (RUN / 'diagnostics/losses.jsonl').read_text(encoding='utf-8').splitlines()]
console = (ROOT / 'workspace/logs/step06_24506_3904/train_console.log').read_text(encoding='utf-8')
memory_values = [int(value) for value in re.findall(r'memory:\s*(\d+)', console)]
result = dict(training_completed=len(losses) == 400,
              finite_losses=all(math.isfinite(row['loss']) for row in losses),
              zero_rpn_positive_gt=sum(row['assigned_positive_anchors_before_sampling'] == 0 for row in rows),
              zero_rpn_positive_gt_by_class={label:sum(row['assigned_positive_anchors_before_sampling'] == 0 and row['category'] == label for row in rows)
                                            for label in ('broken', 'loose')},
              gt_with_same_class_iou50=sum(row['best_returned_same_class_iou'] >= 0.5 for row in rows),
              training_log_max_memory_mib=max(memory_values),
              memory_note='Use logged per-iteration peak. End-of-run max_memory_allocated was reset by MMEngine and is not the whole training peak.',
              rows=rows, training_started_by_this_review=False, model_created=False, detector_forward_performed=False)
write_json(OUT / 'review.json', result)
print('STATIC_RESULT_REVIEW_COMPLETED; no detector forward, no training')
print('ZERO_RPN_POSITIVE_GT=' + str(result['zero_rpn_positive_gt']))
print('GT_WITH_SAME_CLASS_PREDICTION_IOU50=' + str(result['gt_with_same_class_iou50']))
