"""CPU-only verification of the new Resize configuration; no detector or optimizer."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
OUT = ROOT / 'workspace/audit/step14_resolution'
CONTROL = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    sys.path.insert(0, str(ROOT / 'workspace/scripts'))
    import mmcv
    import numpy as np
    from mmcv.transforms import Compose
    from mmengine.config import Config
    from mmdet.utils import register_all_modules
    from PIL import Image, ImageDraw, ImageFont
    register_all_modules(init_default_scope=True)
    config_path = ROOT / 'workspace/configs/frcnn_r50fpn_scale1600_val_001.py'
    cfg = Config.fromfile(str(config_path))
    assert tuple(cfg.train_pipeline[2].scale) == (2666, 1600)
    old_audit_path = ROOT / 'workspace/audit/step11_resolution/resolution_audit.json'
    previous = read(old_audit_path)
    assert previous['status'] == 'PASS'
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    annotation_hashes = {}
    for split in ('train', 'val'):
        path = DATA / f'annotations/instances_{split}.json'
        annotation_hashes[split] = sha(path)
        assert annotation_hashes[split] == previous['annotation_hashes'][split]
        for im in read(path)['images']:
            sizes = [mmcv.rescale_size((im['width'], im['height']), scale)
                     for scale in ((2133, 1280), (2666, 1600))]
            ow, oh = sizes[0]
            nw, nh = sizes[1]
            rows.append(dict(split=split, image_id=im['id'], file_name=im['file_name'],
                original_width=im['width'], original_height=im['height'],
                width1280=ow, height1280=oh, width1600=nw, height1600=nh,
                own_pad_width1600=math.ceil(nw/32)*32, own_pad_height1600=math.ceil(nh/32)*32,
                pixel_ratio=nw*nh/(ow*oh)))
    with (OUT / 'image_dimensions_train_val.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Actual LoadImage/LoadAnnotations/Resize on original train8, preserving
    # CocoDataset's xywh->xyxy before casting to float32. No random flip here.
    debug = read(DATA / 'annotations/instances_debug8.json')
    fixed = sorted(debug['images'], key=lambda x: x['id'])
    sheet = Image.new('RGB', (864, len(fixed)*245+40), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 17)
    draw.text((10, 10), 'TRAIN only | 1 input pixel = 1 crop pixel | green: GT | 1280 vs 1600', fill='black', font=font)
    checked = []
    for i, im in enumerate(fixed):
        annotations = [a for a in debug['annotations'] if a['image_id'] == im['id']]
        instances = [dict(bbox=[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]],
                          bbox_label=a['category_id']-1, ignore_flag=0) for a in annotations]
        for c, scale in enumerate(((2133, 1280), (2666, 1600))):
            pipeline = Compose([dict(type='LoadImageFromFile', backend_args=None),
                dict(type='LoadAnnotations', with_bbox=True), dict(type='Resize', scale=scale, keep_ratio=True)])
            data = pipeline(dict(img_path=str(DATA / 'images' / im['file_name']), img_id=im['id'],
                height=im['height'], width=im['width'], instances=instances))
            nw, nh = mmcv.rescale_size((im['width'], im['height']), scale)
            assert data['img'].shape[:2] == (nh, nw)
            expected = np.array([x['bbox'] for x in instances], dtype=np.float32)
            expected *= np.array([nw/im['width'], nh/im['height']]*2, dtype=np.float32)
            actual = data['gt_bboxes'].tensor.numpy()
            np.testing.assert_allclose(actual, expected, atol=0.001, rtol=0)
            assert data['gt_bboxes_labels'].tolist() == [x['bbox_label'] for x in instances]
            checked.append(dict(image_id=im['id'], file_name=im['file_name'], scale=list(scale),
                resized_wh=[nw, nh], boxes=actual.tolist(), labels=data['gt_bboxes_labels'].tolist(),
                maximum_coordinate_error=float(np.max(np.abs(actual-expected)))))
            image = Image.fromarray(np.ascontiguousarray(data['img'][:, :, ::-1]))
            target = actual[np.argmin(np.minimum(actual[:, 2]-actual[:, 0], actual[:, 3]-actual[:, 1]))]
            left = int(round((target[0]+target[2])/2-210))
            top = int(round((target[1]+target[3])/2-94))
            crop = image.crop((left, top, left+420, top+188))
            painter = ImageDraw.Draw(crop)
            for box in actual:
                if box[2] >= left and box[0] < left+420 and box[3] >= top and box[1] < top+188:
                    painter.rectangle((box[0]-left, box[1]-top, box[2]-left, box[3]-top), outline='#00EE00', width=1)
            x, y = c*432+6, i*245+40
            sheet.paste(crop, (x, y+45))
            draw.text((x, y+5), f'{Path(im["file_name"]).stem} short={scale[1]}', fill='black', font=font)
            draw.text((x, y+25), f'smallest GT: {target[2]-target[0]:.2f} x {target[3]-target[1]:.2f} px', fill='black', font=font)
    sheet.save(OUT / 'train8_resize_1280_vs_1600.png')
    summary = read(CONTROL / 'artifacts/formal_training_summary.json')
    assert summary['epochs'] == 100 and summary['optimizer_updates'] == 5400
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version',
                                   '--format=csv,noheader'], text=True).strip()
    result = dict(status='PASS', experiment='FRCNN-R50FPN-SCALE1600-VAL-001',
        model_created=False, detector_forward_performed=False, training_started=False,
        config_sha256=sha(config_path), script_sha256=sha(Path(__file__)),
        annotation_hashes=annotation_hashes, debug_annotation_sha256=sha(DATA / 'annotations/instances_debug8.json'),
        prior_geometry_audit_path=str(old_audit_path), prior_geometry_audit_sha256=sha(old_audit_path),
        prior_geometry_note='Historical STEP11 authorization flags describe that past stage; user now explicitly requested one final size experiment.',
        control_run=str(CONTROL), control_summary_sha256=sha(CONTROL / 'artifacts/formal_training_summary.json'),
        resize_scale=[2666, 1600], anchor_scales=[8], actual_resize_checks=checked,
        shapes1600=dict(Counter(f'{r["width1600"]}x{r["height1600"]}' for r in rows)),
        maximum_own_padded_wh=[max(r['own_pad_width1600'] for r in rows), max(r['own_pad_height1600'] for r in rows)],
        padding_note='Individual padding is calculated geometrically; actual batch padding uses batch maximum dimensions.',
        train_box_statistics1280=previous['statistics']['train_1280'],
        train_box_statistics1600=previous['statistics']['train_1600'],
        debug8_zero_positive_gt_reused={'1280': previous['debug8_zero_positive_gt']['1280'], '1600': previous['debug8_zero_positive_gt']['1600']},
        pixel_ratio_range=[min(r['pixel_ratio'] for r in rows), max(r['pixel_ratio'] for r in rows)],
        gpu_snapshot=gpu, control_peak_cuda_allocated_mib=summary['observed_peak_cuda_allocated_mib'],
        crude_area_scaled_peak_allocated_mib=summary['observed_peak_cuda_allocated_mib']*(1600/1280)**2,
        memory_note='Area-scaled allocated-memory estimate only; not a measured peak, not total/reserved/OS GPU memory, and not proof that training fits.',
        test_images_or_scores_used=False)
    (OUT / 'resolution_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('STEP14_CPU_RESIZE_AUDIT_PASS: 130 image dimensions; train8 x 2 scales; no detector/training/test')


if __name__ == '__main__':
    main()
