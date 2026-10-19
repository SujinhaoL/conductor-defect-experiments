"""CPU-only replay of the frozen STEP11 input pipeline. No detector or weights.

New diagnostic tool: it checks preprocessing, not training or detection metrics.
Run without --preflight for train107 (both flip branches) and val23.
"""
import argparse
import copy
import csv
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'
CONFIG = RUN / 'resolved_scale1280_val_001.py'
DATA = Path('D:/小论文/conductor_defect_dataset/03_coco_dataset')
OUT = ROOT / 'workspace/audit/FRCNN-DIAG-INPUT1280-001'
CLASSES = ('broken（断股）', 'loose（松股）')
BRANCHES = ('train_no_flip', 'train_flip', 'val')
TITLES = ('训练：不翻转', '训练：水平翻转', '验证：不翻转')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def array(boxes):
    return getattr(boxes, 'tensor', boxes).detach().cpu().numpy().copy()


def git(repo, *args):
    # Per-process trusted project path; never changes global Git configuration.
    return subprocess.check_output(['git', '-c', 'safe.directory=' + str(repo.resolve()),
                                    '-C', str(repo), *args], encoding='utf-8').strip()


def prepare():
    """Read frozen inputs without importing training hooks or building a detector."""
    require(Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve(),
            'Use the Faster R-CNN .venv Python, not another model environment.')
    import numpy as np
    import torch
    import mmdet
    from mmengine.config import Config
    from mmdet.utils import register_all_modules
    from mmdet.registry import DATASETS, MODELS
    torch.set_num_threads(4)
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(CONFIG), import_custom_modules=False)
    record = read(RUN / 'pretraining_record.json')
    selection = read(ROOT / 'workspace/configs/frcnn_debug8_selection.json')
    require(Path(mmdet.__file__).resolve().is_relative_to(ROOT / 'mmdetection'), 'Wrong mmdet source')
    require(git(ROOT / 'mmdetection', 'rev-parse', 'HEAD') == record['mmdet_commit'], 'mmdet commit changed')
    require(not git(ROOT / 'mmdetection', 'status', '--porcelain', '--untracked-files=no'), 'Official tracked source changed')
    accepted = read(ROOT / 'workspace/audit/step11_review/acceptance.json')
    config_hash = sha(CONFIG)
    require(config_hash == accepted['source_files_sha256'][str(CONFIG)], 'Frozen config changed since STEP11 acceptance')
    for package in ('torch', 'mmcv', 'mmengine', 'mmdet', 'numpy'):
        require(importlib.metadata.version(package) == record['versions'][package], 'Version changed: ' + package)
    require(cfg.train_dataloader.batch_size == 2 and cfg.val_dataloader.batch_size == 1, 'Unexpected batch sizes')
    require(cfg.train_dataloader.batch_sampler.type == 'AspectRatioBatchSampler', 'Unexpected batch sampler')
    require(cfg.model.roi_head.bbox_head.num_classes == 2, 'Expected two target classes')
    require(tuple(cfg.model.rpn_head.anchor_generator.scales) == (8,), 'Not default-anchor1280 run')
    expected_types = {'train': ['LoadImageFromFile', 'LoadAnnotations', 'Resize', 'RandomFlip', 'PackDetInputs'],
                      'val': ['LoadImageFromFile', 'Resize', 'LoadAnnotations', 'PackDetInputs']}
    datasets, truth, files = {}, {}, [CONFIG, RUN / 'pretraining_record.json', Path(__file__),
        ROOT / 'workspace/configs/frcnn_debug8_selection.json', DATA / 'audit/source_inventory.json']
    for split, counts in (('train', (107, 168)), ('val', (23, 37))):
        ann = DATA / 'annotations' / f'instances_{split}.json'
        require(sha(ann) == record['annotation_hashes'][split], split + ' annotation hash changed')
        truth[split] = read(ann)
        require((len(truth[split]['images']), len(truth[split]['annotations'])) == counts, 'Counts changed')
        dc = copy.deepcopy(cfg[f'{split}_dataloader'].dataset)
        require(Path(dc.data_root).resolve() == DATA.resolve(), 'Dataset root changed')
        require(dc.ann_file == f'annotations/instances_{split}.json' and dc.data_prefix.img == 'images/', 'Unexpected dataset paths')
        pipeline = dc.pipeline
        require([t['type'] for t in pipeline] == expected_types[split], 'Unexpected transform sequence')
        resize = next(t for t in pipeline if t['type'] == 'Resize')
        require(tuple(resize['scale']) == (2133, 1280) and resize['keep_ratio'], 'Unexpected resize')
        if split == 'train':
            flip = next(t for t in pipeline if t['type'] == 'RandomFlip')
            require(flip['prob'] == 0.5 and flip.get('direction', 'horizontal') == 'horizontal', 'Unexpected flip')
        datasets[split] = DATASETS.build(dc)
        require(len(datasets[split]) == counts[0] and datasets[split].cat2label == {1: 0, 2: 1}, 'Dataset filtering/mapping changed')
        require({datasets[split].get_data_info(i)['img_id'] for i in range(len(datasets[split]))}
                == {im['id'] for im in truth[split]['images']}, 'Dataset image IDs differ from COCO')
        files.append(ann)
    # Build ONLY the configured data preprocessor. Never MODELS.build(cfg.model).
    processor = MODELS.build(copy.deepcopy(cfg.model.data_preprocessor)).cpu()
    require(processor.batch_augments is None and processor.pad_size_divisor == 32, 'Unexpected batch augmentation/padding')
    require(processor.pad_value == 0 and not list(processor.parameters()), 'Unexpected processor state')
    require(cfg.model.data_preprocessor.bgr_to_rgb, 'Expected BGR to RGB')
    np.testing.assert_allclose(processor.mean.numpy().reshape(-1), [123.675, 116.28, 103.53], atol=1e-5, rtol=0)
    np.testing.assert_allclose(processor.std.numpy().reshape(-1), [58.395, 57.12, 57.375], atol=1e-5, rtol=0)
    for relative in ('mmdet/datasets/transforms/transforms.py', 'mmdet/datasets/transforms/loading.py',
                     'mmdet/datasets/transforms/formatting.py', 'mmdet/datasets/coco.py',
                     'mmdet/datasets/samplers/batch_sampler.py', 'mmdet/models/data_preprocessors/data_preprocessor.py'):
        files.append(ROOT / 'mmdetection' / relative)
    import mmcv.transforms.processing
    import mmengine.model.base_model.data_preprocessor
    files.extend([Path(mmcv.transforms.processing.__file__), Path(mmengine.model.base_model.data_preprocessor.__file__)])
    hashes = {str(p): sha(p) for p in files}
    print('DIAG02_PREPARATION_PASS: frozen1280; CPU data preprocessor only; no checkpoint', flush=True)
    return cfg, datasets, truth, selection, processor, hashes


def expected_geometry(annotations, width, height, flip):
    """Independent arithmetic from original COCO: xywh -> xyxy -> scale -> flip."""
    import numpy as np
    ratio = min(2133 / max(width, height), 1280 / min(width, height))
    new_w, new_h = int(width * ratio + 0.5), int(height * ratio + 0.5)
    raw = np.array([a['bbox'] for a in annotations], dtype=np.float64).reshape(-1, 4)
    raw[:, 2:] += raw[:, :2]
    scaled = raw * np.array([new_w / width, new_h / height] * 2)
    if flip:
        scaled[:, [0, 2]] = new_w - scaled[:, [2, 0]]
    return raw, scaled, (new_h, new_w)


def prepare_item(dataset, index, pipeline, truth, branch, inventory, image_hashes):
    """Official pipeline vs COCO arithmetic and independently resized pixels."""
    import cv2
    import numpy as np
    raw_info = dataset.get_data_info(index)
    image_id = raw_info['img_id']
    im = next(i for i in truth['images'] if i['id'] == image_id)
    anns = [a for a in truth['annotations'] if a['image_id'] == image_id]
    image_path = DATA / 'images' / im['file_name']
    require(Path(raw_info['img_path']).resolve() == image_path.resolve(), 'Dataset image path mismatch')
    if str(image_path) not in image_hashes:
        image_hashes[str(image_path)] = sha(image_path)
    require(image_hashes[str(image_path)] == inventory[im['file_name']], 'Image differs from conversion hash')
    raw_boxes, expected, hw = expected_geometry(anns, im['width'], im['height'], branch == 'train_flip')
    labels = [a['category_id'] - 1 for a in anns]
    require([i['bbox_label'] for i in raw_info['instances']] == labels, 'COCO label order/mapping mismatch')
    np.testing.assert_allclose([i['bbox'] for i in raw_info['instances']], raw_boxes, atol=1e-6, rtol=0)
    packed = pipeline(copy.deepcopy(raw_info))
    sample = packed['data_samples']
    actual = array(sample.gt_instances.bboxes)
    require(sample.gt_instances.labels.tolist() == labels and len(sample.ignored_instances) == 0, 'Labels lost or ignored')
    require(tuple(sample.img_shape) == hw and tuple(sample.ori_shape) == (im['height'], im['width']), 'Image dimensions mismatch')
    require(tuple(packed['inputs'].shape) == (3, *hw), 'PackDetInputs CHW shape mismatch')
    np.testing.assert_allclose(sample.scale_factor, [hw[1] / im['width'], hw[0] / im['height']], rtol=0, atol=1e-8)
    # Validation annotations intentionally remain ORIGINAL for original-space COCO evaluation.
    expected_sample = raw_boxes if branch == 'val' else expected
    np.testing.assert_allclose(actual, expected_sample, atol=0.001, rtol=0, err_msg='Image/GT coordinate mismatch')
    if branch != 'val':
        require(bool(sample.flip) == (branch == 'train_flip'), 'Flip branch mismatch')
    original = cv2.imdecode(np.frombuffer(image_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    require(original is not None and original.shape[:2] == tuple(sample.ori_shape), 'Decode dimensions mismatch')
    reference_pixels = cv2.resize(original, (hw[1], hw[0]), interpolation=cv2.INTER_LINEAR)
    if branch == 'train_flip':
        reference_pixels = reference_pixels[:, ::-1]
    actual_pixels = packed['inputs'].permute(1, 2, 0).numpy()
    require(np.array_equal(reference_pixels, actual_pixels), 'Resize/flip pixel mismatch')
    row = dict(branch=branch, image_id=image_id, file_name=im['file_name'],
               gt_ids=[a['id'] for a in anns], labels=labels, original_hw=[im['height'], im['width']],
               resized_hw=list(hw), scale_factor_xy=list(sample.scale_factor),
               gt_count=len(anns), sample_gt_coordinate_space='original' if branch == 'val' else 'resized_flipped',
               original_xyxy=raw_boxes.tolist(), expected_input_xyxy=expected.tolist(),
               actual_sample_gt_xyxy=actual.tolist(), box_max_error_px=float(np.max(np.abs(actual - expected_sample))),
               resize_flip_pixels_equal=True)
    # RGB original used only for selected visualizations; no model forward occurs.
    return packed, row, original[:, :, ::-1].copy()


def check_batch(processor, packed, rows, training):
    """Use the real processor and check its outgoing tensor/GT before any detector."""
    import numpy as np
    import torch
    from mmengine.dataset import pseudo_collate
    before = [array(p['data_samples'].gt_instances.bboxes) for p in packed]
    with torch.no_grad():
        out = processor(pseudo_collate(copy.deepcopy(packed)), training=training)
    pixels = out['inputs']
    require(pixels.device.type == 'cpu' and pixels.dtype == torch.float32 and torch.isfinite(pixels).all(), 'Invalid outgoing tensor')
    expected_hw = tuple(math.ceil(max(r['resized_hw'][j] for r in rows) / 32) * 32 for j in (0, 1))
    require(tuple(pixels.shape) == (len(rows), 3, *expected_hw), 'Batch padding shape mismatch')
    restored = []
    for i, (p, r) in enumerate(zip(packed, rows)):
        h, w = r['resized_hw']
        tensor = pixels[i]
        rgb = p['inputs'][[2, 1, 0]].numpy().astype(np.float32)
        # Independent NumPy normalization; do not just invert the same wrong formula.
        expected = (rgb - np.array([123.675, 116.28, 103.53], np.float32)[:, None, None]) / np.array([58.395, 57.12, 57.375], np.float32)[:, None, None]
        norm_error = float(np.max(np.abs(tensor[:, :h, :w].numpy() - expected)))
        require(norm_error < 1e-5, 'RGB/normalization value mismatch')
        require(torch.count_nonzero(tensor[:, h:, :]).item() == 0 and torch.count_nonzero(tensor[:, :, w:]).item() == 0,
                'Padding must be zero in normalized space')
        after = out['data_samples'][i]
        require(np.array_equal(array(after.gt_instances.bboxes), before[i]), 'Preprocessor unexpectedly moved GT')
        require(after.gt_instances.labels.tolist() == r['labels'], 'Preprocessor changed labels')
        require(tuple(after.batch_input_shape) == expected_hw, 'Batch metadata mismatch')
        require(tuple(after.pad_shape) == tuple(math.ceil(s / 32) * 32 for s in (h, w)), 'Per-image padding metadata mismatch')
        visible = tensor * processor.std + processor.mean
        inverse_error = float(np.max(np.abs(visible[:, :h, :w].numpy() - rgb)))
        require(inverse_error < 1e-4, 'Inverse normalization mismatch')
        r.update(batch_input_hw=list(expected_hw), tensor_shape=list(pixels.shape),
                 pad_bottom=expected_hw[0] - h, pad_right=expected_hw[1] - w,
                 normalization_max_error=norm_error, inverse_pixel_max_error=inverse_error,
                 sample_gt_unchanged_by_preprocessor=True, labels_after_preprocessor=after.gt_instances.labels.tolist())
        # Validation overlay is a separate display copy; actual original-space GT stays unchanged.
        display_boxes = array(after.gt_instances.bboxes).astype(np.float64)
        if not training:
            display_boxes *= np.array(r['scale_factor_xy'] * 2)
        np.testing.assert_allclose(display_boxes, r['expected_input_xyxy'], atol=0.001, rtol=0)
        r['display_input_xyxy'] = display_boxes.tolist()
        restored.append(visible.round().clamp(0, 255).byte().permute(1, 2, 0).numpy())
    return restored


def draw(rgb, boxes, row, path, boundary=None, max_width=None):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb)
    scale = min(1.0, max_width / im.width) if max_width else 1.0
    if scale < 1:
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    d = ImageDraw.Draw(im)
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
    for box, gt, label in zip(boxes, row['gt_ids'], row['labels']):
        x1, y1, x2, y2 = [float(v) * scale for v in box]
        d.rectangle((x1, y1, x2, y2), outline=(0, 180, 90), width=3)
        text = f'GT{gt} {CLASSES[label]}'
        tw = d.textbbox((0, 0), text, font=font)[2]
        tx, ty = max(0, min(x1, im.width - tw)), max(0, y1 - 26)
        d.rectangle((tx, ty, tx + tw, ty + 26), fill=(248, 255, 248))
        d.text((tx, ty), text, font=font, fill=(0, 110, 50))
    if boundary:
        h, w = boundary
        d.rectangle((0, 0, (w - 1) * scale, (h - 1) * scale), outline=(240, 140, 0), width=3)
    im.save(path)
    return im


def save_views(out, row, packed, original, restored):
    from PIL import Image, ImageDraw, ImageFont
    stem = row['branch'] + '_' + Path(row['file_name']).stem
    paths = [f'views/{stem}_{stage}.png' for stage in ('original', 'packed', 'input')]
    draw(original, row['original_xyxy'], row, out / paths[0], max_width=1600)
    rgb = packed['inputs'][[2, 1, 0]].permute(1, 2, 0).numpy()
    draw(rgb, row['display_input_xyxy'], row, out / paths[1])
    final = draw(restored, row['display_input_xyxy'], row, out / paths[2], boundary=row['resized_hw'])
    crops = []
    for gt, box in zip(row['gt_ids'], row['display_input_xyxy']):
        if gt not in (123, 177):
            continue
        x1, y1, x2, y2 = box
        margin = max(45, (x2 - x1) * 0.3, (y2 - y1) * 0.3)
        roi = (max(0, math.floor(x1 - margin)), max(0, math.floor(y1 - margin)),
               min(final.width, math.ceil(x2 + margin)), min(final.height, math.ceil(y2 + margin)))
        # Crop unlabelled tensor pixels, then draw only this GT to avoid clipped text.
        crop = Image.fromarray(restored).crop(roi)
        ImageDraw.Draw(crop).rectangle((x1 - roi[0], y1 - roi[1], x2 - roi[0], y2 - roi[1]), outline=(0, 180, 90), width=1)
        crop = crop.resize((crop.width * 3, crop.height * 3), Image.Resampling.NEAREST)
        panel = Image.new('RGB', (max(620, crop.width), crop.height + 45), 'white')
        panel.paste(crop, ((panel.width - crop.width) // 2, 45))
        ImageDraw.Draw(panel).text((10, 8), f'GT{gt} 人工标注｜输入局部仅显示放大3倍',
                                  font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20), fill=(0, 100, 50))
        name = f'views/{stem}_GT{gt}_input_zoom3.png'
        panel.save(out / name)
        crops.append(dict(gt=gt, path=name, display_zoom_only=3))
    row['views'] = paths
    row['crops'] = crops


def report(out, rows, summary):
    cards = []
    for r in rows:
        if 'views' not in r:
            continue
        images = ''.join(f'<figure><figcaption>{title}</figcaption><a href="{p}"><img loading="lazy" src="{p}"></a></figure>'
                         for title, p in zip(('① 原图＋人工标注（页面缩略图）', '② 缩放/翻转后的像素＋标注', '③ 真实预处理器输出，逆归一化显示＋标注'), r['views']))
        crops = ''.join(f'<figure><figcaption>GT{c["gt"]}输入局部：仅显示放大3倍，不增加原有细节</figcaption><img src="{c["path"]}"></figure>' for c in r['crops'])
        cards.append(f'<section><h2>{TITLES[BRANCHES.index(r["branch"])]} · {r["file_name"]}</h2>'
                     f'<p>尺寸均为高×宽：原图{r["original_hw"]} → 缩放{r["resized_hw"]} → 批次输入{r["batch_input_hw"]}；'
                     f'底部补{r["pad_bottom"]}、右侧补{r["pad_right"]}像素。框坐标最大误差{r["box_max_error_px"]:.6f}像素。</p>'
                     f'<div class="grid">{images}</div>{crops}<details><summary>查看这张图的数值证据（含原图框、实际样本框、显示框）</summary>'
                     f'<pre>{html.escape(json.dumps(r, ensure_ascii=False, indent=2))}</pre></details></section>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DIAG02 · 模型输入前的数据检查</title>
<style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#f1f5f9;color:#23364a;margin:24px auto;max-width:1500px;padding:16px}section,header{background:white;border-radius:12px;padding:24px;margin:20px 0}h1{font-size:28px}h2{font-size:21px}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}figure{margin:0}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}.notice{background:#fff5dc;padding:16px}a{color:#14699c}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
<header><h1>1280默认anchor：真正进入检测网络前，图像与标注是否对应？</h1>
<p>这里没有预测框，没有训练，也没有检测网络前向。绿色框全部来自人工标注GT。橙色外边界表示有效图像范围，右侧/下方可能是补边。</p>
<p>归一化后的数值无法直接当照片看；第三列把真实输出张量逆归一化，供人看清。补边是归一化空间的0，逆变换后呈均值色，并不要求是黑色。</p>
<p class="notice">训练框跟随缩放和翻转。验证样本里的GT按官方约定保持原图坐标；本页仅给显示框单独乘缩放系数，不修改验证标注。不能把这一约定误判为漏缩放。</p>
<p>核查采用固定顺序分批；训练翻转两种分支分别遍历，原训练prob=0.5没有改。不能恢复过去100轮的随机顺序和每个历史batch。</p>
<p>先看cond_000006/000008的翻转，再看cond_000108的GT123和cond_000060的GT177局部。检查框是否跟着缺陷移动、颜色是否自然、补边是否仅出现在有效图像外。</p>
<p><a href="summary.json">汇总</a> · <a href="per_image.csv">逐图表</a> · <a href="records.json">完整坐标</a> · <a href="manifest.json">来源与哈希</a></p>'''
    page += '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></header>' + ''.join(cards) + '</html>'
    (out / 'index.html').write_text(page, encoding='utf-8')


def run(args, out):
    cfg, datasets, truth, selection, processor, sources = prepare()
    from mmcv.transforms import Compose
    from mmengine.dataset import DefaultSampler
    from mmdet.datasets.samplers import AspectRatioBatchSampler
    inventory = {r['split'] + '/' + r['filename']: r['image_sha256']
                 for r in read(DATA / 'audit/source_inventory.json') if r['split'] in ('train', 'val')}
    view_ids = {s['image_id'] for s in selection['samples']} | {108, 115}
    rows, image_hashes = [], {}
    save(out / 'effective_input_config.json', dict(
        train_pipeline=cfg.train_dataloader.dataset.pipeline,
        val_pipeline=cfg.val_dataloader.dataset.pipeline,
        data_preprocessor=cfg.model.data_preprocessor,
        diagnostic_override='Copy of RandomFlip prob=0.5 enumerated as 0.0 and 1.0; fixed sampler order. Original configuration unchanged.',
        validation_gt='Original-space GT is retained; only overlay copies are scaled.'))
    for branch in BRANCHES:
        split = 'val' if branch == 'val' else 'train'
        dataset = datasets[split]
        pipeline_config = copy.deepcopy(cfg[f'{split}_dataloader'].dataset.pipeline)
        # Only the diagnostic copy forces one branch; original config remains prob=0.5.
        if split == 'train':
            next(t for t in pipeline_config if t['type'] == 'RandomFlip')['prob'] = 1.0 if branch == 'train_flip' else 0.0
        pipeline = Compose(pipeline_config)
        batches = list(AspectRatioBatchSampler(DefaultSampler(dataset, shuffle=False, seed=42), batch_size=2, drop_last=False)) if split == 'train' else [[i] for i in range(len(dataset))]
        if args.preflight:
            chosen = {6, 8} if split == 'train' else {108, 115}
            inds = [i for i in range(len(dataset)) if dataset.get_data_info(i)['img_id'] in chosen]
            batches = [inds] if split == 'train' else [[i] for i in inds]
        done = 0
        for batch_id, inds in enumerate(batches, 1):
            items = [prepare_item(dataset, i, pipeline, truth[split], branch, inventory, image_hashes) for i in inds]
            packed, batch_rows, originals = map(list, zip(*items))
            restored = check_batch(processor, packed, batch_rows, training=split == 'train')
            for p, row, orig, rgb in zip(packed, batch_rows, originals, restored):
                row['batch_id'] = batch_id
                if row['image_id'] in view_ids:
                    save_views(out, row, p, orig, rgb)
                rows.append(row)
            done += len(inds)
            if done % 20 == 0 or batch_id == len(batches):
                print(f'DIAG02_PROGRESS branch={branch} images={done}', flush=True)
            del items, packed, originals, restored
    counts = {b: sum(r['branch'] == b for r in rows) for b in BRANCHES}
    require(counts == (dict(zip(BRANCHES, (2, 2, 2))) if args.preflight else dict(zip(BRANCHES, (107, 107, 23)))), 'Incomplete checks')
    for p, expected in sources.items():
        require(sha(p) == expected, 'Source changed during diagnostic: ' + p)
    summary = dict(status='PREFLIGHT_ONLY_PASS' if args.preflight else 'DIAG02_INPUT_PIPELINE_PASS',
                   unique_images=len(image_hashes), branch_image_counts=counts,
                   branch_gt_counts={b: sum(r['gt_count'] for r in rows if r['branch'] == b) for b in BRANCHES},
                   max_box_error_px=max(r['box_max_error_px'] for r in rows),
                   max_normalization_error=max(r['normalization_max_error'] for r in rows),
                   max_inverse_pixel_error=max(r['inverse_pixel_max_error'] for r in rows),
                   visualized_cases=sum('views' in r for r in rows), detector_built=False,
                   checkpoint_loaded=False, training_started=False, model_inference=False, test_images_read=False,
                   limitation='CPU replay with fixed batching; not captured historical random training batches. Visual review pending.')
    save(out / 'records.json', rows)
    save(out / 'summary.json', summary)
    fields = ['branch', 'image_id', 'file_name', 'gt_count', 'original_hw', 'resized_hw', 'batch_input_hw',
              'pad_bottom', 'pad_right', 'box_max_error_px', 'normalization_max_error', 'inverse_pixel_max_error', 'sample_gt_coordinate_space']
    with (out / 'per_image.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    report(out, rows, summary)
    save(out / 'manifest.json', dict(status=summary['status'], config=str(CONFIG), sources_sha256=sources,
         images_sha256=image_hashes, source_git_commit=git(ROOT.parent, 'rev-parse', 'HEAD'),
         log_dir=str(ROOT / 'workspace/logs' / out.name),
         source_git_status=git(ROOT.parent, 'status', '--short'), command=subprocess.list2cmdline([sys.executable, '-I', '-X', 'utf8', *sys.argv]),
         output_sha256={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file() and p.name != 'console.txt'}))
    if not args.preflight:
        (OUT / 'LATEST_REPORT.txt').write_text(str(out / 'index.html') + '\n', encoding='utf-8')
        (OUT / 'latest.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=' + out.name + '/index.html"><a href="' + out.name + '/index.html">打开最新检查报告</a>', encoding='utf-8')
    print(summary['status'], flush=True)
    print('REPORT=' + str(out / 'index.html'), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true', help='Developer preparation: only 2 train + 2 val images; not the full audit')
    args = parser.parse_args()
    name = ('preflight_' if args.preflight else 'diag02_') + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out = OUT / name
    (out / 'views').mkdir(parents=True, exist_ok=False)
    (out / 'code').mkdir()
    shutil.copy2(__file__, out / 'code' / Path(__file__).name)
    log_dir = ROOT / 'workspace/logs' / name
    log_dir.mkdir(parents=True, exist_ok=False)
    console = (log_dir / 'console.txt').open('w', encoding='utf-8', buffering=1)
    original_stdout, original_stderr = sys.stdout, sys.stderr
    class Tee:
        def write(self, text):
            original_stdout.write(text); console.write(text)
        def flush(self):
            original_stdout.flush(); console.flush()
    sys.stdout = sys.stderr = Tee()
    code = 0
    try:
        print('DIAG02_CPU_INPUT_PIPELINE; no training/inference/checkpoint/test', flush=True)
        print('RESULT_DIR=' + str(out), flush=True)
        print('LOG_DIR=' + str(log_dir), flush=True)
        run(args, out)
    except Exception:
        code = 1
        traceback.print_exc()
        save(out / 'failure.json', dict(status='FAIL', traceback=traceback.format_exc(), no_training=True))
        print('DIAG02_FAIL: keep this output; do not alter training settings or rerun blindly.', flush=True)
    finally:
        print(f'DIAG02_EXIT_CODE={code}', flush=True)
        sys.stdout, sys.stderr = original_stdout, original_stderr
        console.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
