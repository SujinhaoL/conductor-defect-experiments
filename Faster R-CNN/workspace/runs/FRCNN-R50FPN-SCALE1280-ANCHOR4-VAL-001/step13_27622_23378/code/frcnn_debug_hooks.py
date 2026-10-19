"""Artifacts for the user-run DEBUG8 training; importing this does not train."""
from collections import Counter
import contextlib
import copy
import csv
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

CLASSES = ('broken', 'loose')
OUTPUT_KEYS = {
    'roi_head.bbox_head.fc_cls.weight', 'roi_head.bbox_head.fc_cls.bias',
    'roi_head.bbox_head.fc_reg.weight', 'roi_head.bbox_head.fc_reg.bias',
}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                   allow_nan=False) + '\n', encoding='utf-8')


def mean_valid(values):
    values = np.asarray(values)
    valid = values[values >= 0]
    return float(valid.mean()) if valid.size else None


def evaluate_train_fit(annotation_file, predictions, image_ids, output_prefix,
                       scope='training-subset fit diagnostic; NOT independent validation or test'):
    """Saved-prediction COCO evaluation; default scope is the DEBUG8 train subset."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(annotation_file))
    if predictions:
        dt = gt.loadRes(copy.deepcopy(predictions))
    else:
        dt = COCO()
        dt.dataset = dict(images=copy.deepcopy(gt.dataset['images']),
                          categories=copy.deepcopy(gt.dataset['categories']),
                          annotations=[])
        dt.createIndex()
    evaluator = COCOeval(gt, dt, 'bbox')
    evaluator.params.imgIds = list(image_ids)
    evaluator.params.catIds = [1, 2]
    evaluator.params.maxDets = [1, 10, 100]
    with Path(str(output_prefix) + '_coco_eval.txt').open('w', encoding='utf-8') as log:
        with contextlib.redirect_stdout(log):
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
    precision = evaluator.eval['precision']  # [IoU, recall, category, area, maxDets]
    recall = evaluator.eval['recall']        # [IoU, category, area, maxDets]
    at50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.5))[0])
    classes = {}
    for index, cat_id in enumerate(evaluator.params.catIds):
        classes[gt.cats[cat_id]['name']] = {
            'AP50_95': mean_valid(precision[:, :, index, 0, -1]),
            'AP50': mean_valid(precision[at50, :, index, 0, -1]),
            'AR100': mean_valid(recall[:, index, 0, -1]),
        }
    result = {
        'scope': scope,
        'mAP50_95': float(evaluator.stats[0]), 'mAP50': float(evaluator.stats[1]),
        'AR100': float(evaluator.stats[8]), 'per_class': classes,
        'image_ids': list(image_ids), 'prediction_count': len(predictions),
        'coco_max_dets': [1, 10, 100], 'iou_thresholds': evaluator.params.iouThrs.tolist(),
        'area_ranges': evaluator.params.areaRng, 'use_categories': True,
    }
    write_json(str(output_prefix) + '_metrics.json', result)
    return result


def draw_predictions(rgb, boxes, labels, scores, threshold, destination):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', max(18, image.width // 120))
    for box, label, score in zip(boxes, labels, scores):
        if score <= threshold:
            continue
        color = ((240, 60, 60), (20, 210, 230))[label]
        draw.rectangle(box, outline=color, width=max(2, image.width // 600))
        text = f'{CLASSES[label]} {score:.3f}'
        bounds = draw.textbbox((0, 0), text, font=font)
        tw, th = bounds[2] + 8, bounds[3] + 8
        x = max(0, min(box[0], image.width - tw))
        y = max(0, box[1] - th)
        draw.rectangle((x, y, x + tw, y + th), fill=color)
        draw.text((x + 4, y + 2), text, font=font, fill=(0, 0, 0))
    image.save(destination, quality=95)


def make_contact_sheet(images, destination, caption):
    from PIL import Image, ImageDraw, ImageFont, ImageOps
    contact = Image.new('RGB', (1600, 4 * 320), (245, 245, 245))
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 19)
    for idx, path in enumerate(images):
        x, y = (idx % 2) * 800, (idx // 2) * 320
        with Image.open(path) as image:
            thumb = ImageOps.contain(image, (790, 285))
        contact.paste(thumb, (x + (800 - thumb.width) // 2, y + 30))
        ImageDraw.Draw(contact).text((x + 8, y + 4), f'{path.stem} / {caption}',
                                    font=font, fill=(0, 0, 0))
    contact.save(destination, quality=95)


def plot_losses(rows, destination, title='FRCNN-DEBUG-OVERFIT-8 / train-only diagnostic'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    xs = [row['iteration'] for row in rows]
    for axis, keys, panel_title in zip(axes.flat,
            [('loss',), ('loss_rpn_cls', 'loss_rpn_bbox'),
             ('loss_cls', 'loss_bbox'), ('learning_rate',)],
            ['Total loss', 'RPN losses', 'ROI losses', 'Learning rate used']):
        for key in keys:
            axis.plot(xs, [row[key] for row in rows], linewidth=1, label=key)
        axis.set(title=panel_title, xlabel='Optimizer update')
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.savefig(destination, dpi=150)
    plt.close(fig)


@HOOKS.register_module()
class Debug8ArtifactsHook(Hook):
    # Record the LR used before ParamSchedulerHook (LOW=70) changes it.
    # Run before LoggerHook (60), which reads and resets CUDA peak statistics.
    priority = 55

    def __init__(self, interval=25, visual_threshold=0.30):
        self.interval = interval
        self.visual_threshold = visual_threshold
        self.rows = []
        self.snapshots = []
        self.snapshot_seconds = 0.0
        self.observed_cuda_peak_mib = 0.0

    def before_train(self, runner):
        self.started = time.perf_counter()
        self.output = Path(runner.work_dir) / 'diagnostics'
        self.output.mkdir(exist_ok=False)
        self.dataset = runner.train_dataloader.dataset
        assert len(self.dataset) == 8 and self.dataset.cat2label == {1: 0, 2: 1}
        assert len(runner.train_dataloader) == 4 and runner.max_epochs == 100
        assert runner.world_size == 1 and not runner.cfg.resume
        self.model = runner.model
        self.model.cfg = runner.cfg
        self.model.dataset_meta = self.dataset.metainfo
        # This hook runs AFTER Runner has initialized and loaded the official checkpoint.
        original = torch.load(runner.cfg.load_from, map_location='cpu', weights_only=True)['state_dict']
        loaded = self.model.state_dict()
        assert original.keys() == loaded.keys(), 'Unexpected model/checkpoint key mismatch'
        changed = {k for k in original if original[k].shape != loaded[k].shape}
        assert changed == OUTPUT_KEYS, 'Unexpected checkpoint shape mismatch'
        equal = [k for k in original if k not in changed and torch.equal(original[k], loaded[k].cpu())]
        assert len(equal) == 344, 'Not all compatible COCO tensors loaded before training'
        audit = {
            'status': 'PASS', 'loaded_tensors': len(equal),
            'loaded_by_module': dict(Counter(k.split('.')[0] for k in equal)),
            'expected_new_outputs': {k: {'coco_shape': list(original[k].shape),
                                       'two_class_shape': list(loaded[k].shape)} for k in sorted(changed)},
            'checkpoint': runner.cfg.load_from, 'num_target_classes': 2,
            'initialization_audited_before_first_optimizer_update': True,
        }
        write_json(self.output / 'actual_training_weight_load.json', audit)
        del original, loaded
        print('DEBUG8_ACTUAL_WEIGHT_LOAD_PASS: matched=344 expected_new_outputs=4', flush=True)
        torch.cuda.reset_peak_memory_stats()
        self.snapshot(runner, 0)

    def before_train_iter(self, runner, batch_idx, data_batch=None):
        self.iter_started = time.perf_counter()

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        self.observed_cuda_peak_mib = max(self.observed_cuda_peak_mib,
                                         torch.cuda.max_memory_allocated() / 1024**2)
        row = {'epoch': runner.epoch + 1, 'iteration': runner.iter + 1,
               'batch_index': batch_idx, 'learning_rate': runner.optim_wrapper.optimizer.param_groups[0]['lr']}
        for key, value in outputs.items():
            row[key] = float(value.detach().cpu().item() if torch.is_tensor(value) else value)
        row['iteration_seconds_excluding_data_loading'] = time.perf_counter() - self.iter_started
        required = ('loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox')
        if not all(key in row and math.isfinite(row[key]) for key in required):
            raise RuntimeError('Missing or non-finite detection loss; stop and inspect logs')
        self.rows.append(row)
        with (self.output / 'losses.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
        with (self.output / 'losses.csv').open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if len(self.rows) == 1:
                writer.writeheader()
            writer.writerow(row)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch % self.interval == 0 or epoch == runner.max_epochs:
            self.snapshot(runner, epoch)
            plot_losses(self.rows, self.output / 'loss_curves.png')

    def snapshot(self, runner, epoch):
        from mmcv.transforms import Compose
        from mmdet.apis import inference_detector
        from PIL import Image
        torch.cuda.synchronize()
        started = time.perf_counter()
        py_rng, np_rng = random.getstate(), np.random.get_state()
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        module_modes = [(module, module.training) for module in self.model.modules()]
        destination = self.output / f'epoch_{epoch:03d}'
        destination.mkdir(exist_ok=False)
        coco_predictions, details, image_paths, ids = [], [], [], []
        try:
            self.model.eval()
            pipeline = Compose([
                dict(type='mmdet.LoadImageFromNDArray'),
                dict(type='mmdet.Resize', scale=tuple(runner.cfg.get('snapshot_scale', (1333, 800))), keep_ratio=True),
                dict(type='mmdet.PackDetInputs', meta_keys=(
                    'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
            ])
            for index in range(len(self.dataset)):
                info = self.dataset.get_data_info(index)
                ids.append(int(info['img_id']))
                with Image.open(info['img_path']) as image:
                    rgb = np.array(image.convert('RGB'))
                bgr = np.ascontiguousarray(rgb[:, :, ::-1])
                prediction = inference_detector(self.model, bgr, test_pipeline=pipeline)
                instances = prediction.pred_instances.cpu()
                boxes, labels, scores = instances.bboxes.tolist(), instances.labels.tolist(), instances.scores.tolist()
                assert set(labels) <= {0, 1}
                assert np.isfinite(np.asarray(boxes)).all() and np.isfinite(scores).all()
                for box, label, score in zip(boxes, labels, scores):
                    x1, y1, x2, y2 = box
                    coco_predictions.append(dict(image_id=int(info['img_id']), category_id=label + 1,
                                                 bbox=[x1, y1, x2 - x1, y2 - y1], score=score))
                details.append(dict(image_id=int(info['img_id']), image_path=info['img_path'],
                                    boxes_xyxy=boxes, model_labels=labels, scores=scores,
                                    returned_count=len(scores), visible_count=sum(s > self.visual_threshold for s in scores)))
                path = destination / (Path(info['img_path']).stem + '.jpg')
                draw_predictions(rgb, boxes, labels, scores, self.visual_threshold, path)
                image_paths.append(path)
            write_json(destination / 'predictions_coco.json', coco_predictions)
            write_json(destination / 'predictions_details.json', details)
            metrics = evaluate_train_fit(self.dataset.ann_file, coco_predictions, ids, destination / 'train_fit')
            metrics.update(epoch=epoch, model_score_threshold=0.001, rcnn_nms_iou=0.5,
                           max_per_image=100, visualization_score_threshold=self.visual_threshold,
                           resize_scale=list(runner.cfg.get('snapshot_scale', (1333, 800))))
            write_json(destination / 'train_fit_metrics.json', metrics)
            make_contact_sheet(image_paths, destination / 'predictions_contact_sheet.jpg',
                               f'epoch {epoch}; score > {self.visual_threshold}')
            self.snapshots.append(metrics)
            write_json(self.output / 'train_fit_progress.json', self.snapshots)
            print(f'DEBUG8_SNAPSHOT epoch={epoch} train_mAP50={metrics["mAP50"]:.4f} '
                  f'train_mAP50_95={metrics["mAP50_95"]:.4f} train_AR100={metrics["AR100"]:.4f}', flush=True)
        finally:
            for module, training in module_modes:
                module.training = training
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.synchronize()
            self.observed_cuda_peak_mib = max(self.observed_cuda_peak_mib,
                                             torch.cuda.max_memory_allocated() / 1024**2)
            self.snapshot_seconds += time.perf_counter() - started

    def after_train(self, runner):
        torch.cuda.synchronize()
        assert len(self.rows) == 400 and runner.epoch == 100 and runner.iter == 400
        assert [x['epoch'] for x in self.snapshots] == [0, 25, 50, 75, 100]
        elapsed = time.perf_counter() - self.started
        summary = {
            'status': 'COMPLETED_REQUIRES_RESULT_REVIEW', 'experiment': 'FRCNN-DEBUG-OVERFIT-8',
            'purpose': 'train-only memorization diagnostic; not generalization performance',
            'epochs': 100, 'images': 8, 'boxes': 15, 'broken': 5, 'loose': 10,
            'batch_size': 2, 'workers': 0, 'optimizer_updates': 400, 'amp': False,
            'optimizer': 'SGD', 'base_lr': 0.0025, 'momentum': 0.9, 'weight_decay': 0.0001,
            'warmup_iterations': 20, 'random_image_augmentation': False,
            'official_random_roi_rpn_sampling_retained': True, 'seed': 42,
            'wall_seconds_from_before_train': elapsed,
            'snapshot_seconds': self.snapshot_seconds,
            'wall_seconds_excluding_snapshots': elapsed - self.snapshot_seconds,
            'timing_note': 'Wall timing includes logging/checkpoints. Iteration times exclude data loading; CUDA synchronized by scalar loss reads.',
            'observed_peak_cuda_allocated_mib_including_snapshots': self.observed_cuda_peak_mib,
            'peak_memory_note': 'Maximum accumulated across iteration/snapshot observations before LoggerHook resets CUDA peak counters.',
            'first_20_mean_loss': float(np.mean([r['loss'] for r in self.rows[:20]])),
            'last_20_mean_loss': float(np.mean([r['loss'] for r in self.rows[-20:]])),
            'initial_train_fit': self.snapshots[0], 'final_train_fit': self.snapshots[-1],
            'independent_val_or_test_evaluated': False,
            'overfit_success_automatically_claimed': False,
        }
        write_json(self.output / 'debug8_training_summary.json', summary)
        print('DEBUG8_100_EPOCHS_400_UPDATES_COMPLETED; inspect losses and train predictions.', flush=True)
