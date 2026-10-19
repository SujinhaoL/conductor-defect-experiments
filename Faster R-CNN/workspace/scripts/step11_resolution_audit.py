"""CPU-only scale/box/anchor audit on original train and val. No model or training."""
from collections import Counter
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
OUT = ROOT / 'workspace/audit/step11_resolution'
BASE_RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
SCALES = [(1333, 800), (2133, 1280), (2666, 1600)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    sys.path.insert(0, str(ROOT / 'workspace/scripts'))
    import mmcv
    import numpy as np
    import torch
    from mmengine.config import Config
    from mmengine.structures import InstanceData
    from mmcv.transforms import Compose
    from mmdet.registry import TASK_UTILS
    from mmdet.structures.bbox import bbox_overlaps
    from mmdet.utils import register_all_modules
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(BASE_RUN / 'resolved_val_001.py'))
    annotation_hashes, rows, summaries, dimensions = {}, [], {}, {}
    for split, counts in [('train', (107, 168)), ('val', (23, 37))]:
        path = DATA / 'annotations' / f'instances_{split}.json'
        annotation_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        truth = read(path)
        assert (len(truth['images']), len(truth['annotations'])) == counts
        images = {im['id']: im for im in truth['images']}
        dimensions[split] = dict(Counter(f'{im["width"]}x{im["height"]}' for im in truth['images']))
        for scale in SCALES:
            for ann in truth['annotations']:
                img = images[ann['image_id']]
                width, height = mmcv.rescale_size((img['width'], img['height']), scale)
                bw, bh = ann['bbox'][2] * width / img['width'], ann['bbox'][3] * height / img['height']
                rows.append(dict(split=split, image_id=img['id'], file_name=img['file_name'], annotation_id=ann['id'],
                    category=('broken', 'loose')[ann['category_id']-1], scale_long=scale[0], scale_short=scale[1],
                    original_width=img['width'], original_height=img['height'], resized_width=width, resized_height=height,
                    box_width=bw, box_height=bh, box_short_edge=min(bw, bh), box_area=bw*bh))
            sample = [r for r in rows if r['split'] == split and r['scale_short'] == scale[1]]
            key = f'{split}_{scale[1]}'
            summaries[key] = {}
            for label in ('all', 'broken', 'loose'):
                group = [r for r in sample if label == 'all' or r['category'] == label]
                sides = [r['box_short_edge'] for r in group]
                summaries[key][label] = dict(boxes=len(group), median_short_edge=float(np.median(sides)),
                    p10_short_edge=float(np.percentile(sides, 10)), min_short_edge=min(sides),
                    **{f'short_edge_lt_{n}': sum(x < n for x in sides) for n in (2, 4, 8, 16)})
    with (OUT / 'box_sizes_train_val.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    selection = read(ROOT / 'workspace/configs/frcnn_debug8_selection.json')
    debug = read(DATA / 'annotations/instances_debug8.json')
    generator = TASK_UTILS.build(cfg.model.rpn_head.anchor_generator)
    assigner = TASK_UTILS.build(cfg.model.train_cfg.rpn.assigner)
    assert cfg.model.train_cfg.rpn.allowed_border == -1
    fixed_images = sorted(debug['images'], key=lambda im: im['id'])
    anchor_rows, views = [], {}
    for scale in SCALES:
        pipeline = Compose([
            dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=scale, keep_ratio=True)])
        transformed = []
        for img in fixed_images:
            annotations = [a for a in debug['annotations'] if a['image_id'] == img['id']]
            instances = [dict(bbox=[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]],
                              bbox_label=a['category_id']-1, ignore_flag=0) for a in annotations]
            data = pipeline(dict(img_path=str(DATA / 'images' / img['file_name']), img_id=img['id'],
                                 height=img['height'], width=img['width'], instances=instances))
            expected_wh = mmcv.rescale_size((img['width'], img['height']), scale)
            assert data['img'].shape[:2] == (expected_wh[1], expected_wh[0])
            expected = torch.tensor([a['bbox'] for a in annotations], dtype=torch.float32)
            expected[:, 2:] += expected[:, :2]
            expected *= torch.tensor([expected_wh[0]/img['width'], expected_wh[1]/img['height']]*2)
            assert torch.allclose(data['gt_bboxes'].tensor, expected, atol=0.001, rtol=0)
            transformed.append((img, annotations, data))
        for index, (img, annotations, data) in enumerate(transformed):
            pair = transformed[(index//2)*2:(index//2)*2+2]
            padded_hw = tuple(math.ceil(max(p[2]['img'].shape[j] for p in pair) / 32)*32 for j in (0, 1))
            sizes = [(math.ceil(padded_hw[0]/s), math.ceil(padded_hw[1]/s)) for s in (4, 8, 16, 32, 64)]
            anchors = torch.cat(generator.grid_priors(sizes, device='cpu'))
            own_pad = tuple(math.ceil(data['img'].shape[j]/32)*32 for j in (0, 1))
            flags = torch.cat(generator.valid_flags(sizes, own_pad, device='cpu'))
            anchors = anchors[flags]
            boxes = data['gt_bboxes'].tensor
            assignment = assigner.assign(InstanceData(priors=anchors), InstanceData(bboxes=boxes, labels=torch.zeros(len(boxes), dtype=torch.long)))
            maximum = bbox_overlaps(boxes, anchors).max(dim=1).values.tolist()
            for j, ann in enumerate(annotations):
                anchor_rows.append(dict(scale_short=scale[1], image_id=img['id'], file_name=img['file_name'],
                    annotation_id=ann['id'], category=('broken','loose')[ann['category_id']-1],
                    resized_box_width=float(boxes[j,2]-boxes[j,0]), resized_box_height=float(boxes[j,3]-boxes[j,1]),
                    max_anchor_iou=maximum[j], positive_anchors_before_sampling=int((assignment.gt_inds == j+1).sum())))
            if scale[1] in (800, 1280):
                views[(img['id'], scale[1])] = (Image.fromarray(np.ascontiguousarray(data['img'][:, :, ::-1])), boxes.numpy())
        print(f'SCALE_AUDIT short={scale[1]} DEBUG8_ZERO_POSITIVE_GT=' + str(sum(r['positive_anchors_before_sampling']==0 for r in anchor_rows if r['scale_short']==scale[1])), flush=True)
    with (OUT / 'debug8_anchor_sizes.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(anchor_rows[0])); writer.writeheader(); writer.writerows(anchor_rows)
    # Same-size windows from already-resized train images: 1 input pixel = 1 output pixel.
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 17)
    sheet = Image.new('RGB', (864, 8*245+40), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), 'TRAIN only | 1:1 resized-pixel crops | green: GT | 800 vs 1280 short edge', fill='black', font=font)
    for i, img in enumerate(fixed_images):
        for c, size in enumerate((800,1280)):
            image, boxes = views[(img['id'], size)]
            target = boxes[np.argmin(np.minimum(boxes[:,2]-boxes[:,0], boxes[:,3]-boxes[:,1]))]
            cx, cy = (target[0]+target[2])/2, (target[1]+target[3])/2
            left, top = int(round(cx-210)), int(round(cy-94))
            crop = image.crop((left, top, left+420, top+188))
            d = ImageDraw.Draw(crop)
            for b in boxes:
                if b[2]>=left and b[0]<left+420 and b[3]>=top and b[1]<top+188:
                    d.rectangle((b[0]-left,b[1]-top,b[2]-left,b[3]-top), outline='#00EE00', width=1)
            x,y=c*432,40+i*245
            sheet.paste(crop,(x+6,y+45))
            draw.text((x+6,y+5), f'{Path(img["file_name"]).stem} short={size}', fill='black',font=font)
            draw.text((x+6,y+25), f'smallest GT: {target[2]-target[0]:.2f} x {target[3]-target[1]:.2f} px', fill='black',font=font)
    sheet.save(OUT / 'train_debug8_resize_800_vs_1280.png')
    fig, ax = plt.subplots(figsize=(8,4.8))
    for scale in SCALES:
        x=np.sort([r['box_short_edge'] for r in rows if r['split']=='train' and r['scale_short']==scale[1]])
        ax.plot(x, np.arange(1,len(x)+1)/len(x), label=f'short edge {scale[1]}')
    ax.set_xscale('log'); ax.set_xlabel('GT box short edge after resize (pixels)'); ax.set_ylabel('Fraction of training GT boxes')
    ax.set_title('Original train107 / 168 boxes; keep_ratio=True'); ax.set_ylim(0,1)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'train_box_short_edge_cdf.png',dpi=180); plt.close(fig)
    result=dict(status='PASS', annotation_hashes=annotation_hashes, source_dimensions=dimensions,
        scales=SCALES, statistics=summaries,
        debug8_zero_positive_gt={str(s[1]):sum(r['positive_anchors_before_sampling']==0 for r in anchor_rows if r['scale_short']==s[1]) for s in SCALES},
        debug8_boxes=15, selected_experiment_scale=[2133,1280], future_1600_is_diagnostic_size_only_not_an_authorized_training_batch=True,
        training_started=False, model_created=False, detector_forward_performed=False, test_images_or_scores_used=False,
        notes=['Anchor counts are for fixed train8 and fixed nonflipped pairs before sampling, not all train or ROI proposal recall.',
               'GT short-edge thresholds 2/4/8/16 are descriptive; they are not universal detection success thresholds.',
               'Scale statistics alone cannot establish the cause of low AP.'])
    (OUT/'resolution_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))
    print('STEP11_RESOLUTION_AUDIT_PASS; CPU data transforms/anchor assignment only; NO TRAINING')


if __name__=='__main__':
    main()
