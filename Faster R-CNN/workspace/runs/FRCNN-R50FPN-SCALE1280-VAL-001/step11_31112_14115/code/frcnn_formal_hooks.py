"""Record the user-run formal baseline; reuse the tested DEBUG8 artifact functions."""
import copy
import json
from pathlib import Path
import time

import torch
from mmdet.registry import DATASETS, HOOKS
from frcnn_debug_hooks import Debug8ArtifactsHook, OUTPUT_KEYS, plot_losses, write_json


@HOOKS.register_module()
class FormalBaselineArtifactsHook(Debug8ArtifactsHook):
    def before_train(self, runner):
        self.started = time.perf_counter()
        self.output = Path(runner.work_dir) / 'artifacts'
        self.output.mkdir(exist_ok=False)
        self.validation_rows = []
        assert len(runner.train_dataloader.dataset) == 107
        assert len(runner.train_dataloader) == 54 and runner.max_epochs == 100
        assert runner.world_size == 1 and not runner.cfg.resume
        assert len(runner.val_dataloader.dataset) == 23 and runner.cfg.test_dataloader is None
        self.model = runner.model
        self.model.cfg = runner.cfg
        self.model.dataset_meta = runner.train_dataloader.dataset.metainfo
        # Fixed train-only snapshots reuse the previously inspected eight images.
        subset = copy.deepcopy(runner.cfg.train_dataloader.dataset)
        subset.ann_file = 'annotations/instances_debug8.json'
        subset.filter_cfg = None
        self.dataset = DATASETS.build(subset)
        assert len(self.dataset) == 8 and self.dataset.cat2label == {1: 0, 2: 1}
        original = torch.load(runner.cfg.load_from, map_location='cpu', weights_only=True)['state_dict']
        loaded = self.model.state_dict()
        assert original.keys() == loaded.keys()
        changed = {k for k in original if original[k].shape != loaded[k].shape}
        assert changed == OUTPUT_KEYS
        equal = [k for k in original if k not in changed and torch.equal(original[k], loaded[k].cpu())]
        assert len(equal) == 344, 'Compatible official COCO weights were not fully loaded'
        write_json(self.output / 'actual_training_weight_load.json', {
            'status': 'PASS', 'loaded_tensors': 344, 'checkpoint': runner.cfg.load_from,
            'expected_shape_changes': {k: {'coco': list(original[k].shape),
                                          'two_class': list(loaded[k].shape)} for k in sorted(changed)},
            'loaded_before_first_optimizer_update': True})
        del original, loaded
        print('FORMAL_OFFICIAL_COCO_WEIGHT_LOAD_PASS: matched=344 expected_new_outputs=4', flush=True)
        torch.cuda.reset_peak_memory_stats()
        self.snapshot(runner, 0)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch % self.interval == 0 or epoch == runner.max_epochs:
            self.snapshot(runner, epoch)
            plot_losses(self.rows, self.output / 'loss_curves.png', title=runner.cfg.get('experiment_id', 'FRCNN-R50FPN-VAL-001') + ' / training')

    def before_val_epoch(self, runner):
        # EpochBasedTrainLoop increments epoch before launching ValLoop.
        epoch = runner.epoch
        assert 1 <= epoch <= 100
        destination = self.output / 'validation' / f'epoch_{epoch:03d}'
        destination.mkdir(parents=True, exist_ok=False)
        metrics = runner.val_evaluator.metrics
        assert len(metrics) == 1 and metrics[0].__class__.__name__ == 'CocoMetric'
        metrics[0].outfile_prefix = str(destination / 'predictions')

    def after_val_epoch(self, runner, metrics=None):
        assert metrics is not None
        if not metrics:
            # Native CocoMetric returns an empty dict when no detections exist.
            # Empty bbox predictions have AP=0; preserve that result so the
            # native checkpoint hook can still compare epochs without aborting.
            destination = self.output / 'validation' / f'epoch_{runner.epoch:03d}'
            assert json.loads((destination / 'predictions.bbox.json').read_text(encoding='utf-8')) == []
            truth = json.loads(Path(runner.cfg.val_evaluator.ann_file).read_text(encoding='utf-8'))
            areas = [ann['area'] for ann in truth['annotations']]
            metrics.update({'coco/bbox_mAP': 0.0, 'coco/bbox_mAP_50': 0.0, 'coco/bbox_mAP_75': 0.0,
                            'coco/broken_precision': 0.0, 'coco/loose_precision': 0.0})
            for name, low, high in [('s', 0, 1024), ('m', 1024, 9216), ('l', 9216, 1e10)]:
                metrics['coco/bbox_mAP_' + name] = 0.0 if any(low <= area <= high for area in areas) else -1.0
            write_json(destination / 'empty_prediction_note.json', {
                'predictions': 0, 'note': 'Native CocoMetric returned no metrics for empty predictions; recorded mathematically zero AP, with unavailable area groups -1.'})
            print('VALIDATION_EMPTY_PREDICTIONS: AP recorded as zero; training continues.', flush=True)
        assert 'coco/bbox_mAP' in metrics and 'coco/bbox_mAP_50' in metrics
        self.observed_cuda_peak_mib = max(self.observed_cuda_peak_mib,
                                         torch.cuda.max_memory_allocated() / 1024**2)
        row = dict(epoch=runner.epoch, metrics={key: float(value) for key, value in metrics.items()})
        self.validation_rows.append(row)
        destination = self.output / 'validation' / f'epoch_{runner.epoch:03d}'
        write_json(destination / 'metrics.json', row)
        with (self.output / 'validation_history.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, allow_nan=False) + '\n')

    def after_train(self, runner):
        torch.cuda.synchronize()
        assert runner.epoch == 100 and runner.iter == 5400 and len(self.rows) == 5400
        assert len(self.validation_rows) == 100
        assert [x['epoch'] for x in self.snapshots] == [0, 25, 50, 75, 100]
        # Match native CheckpointHook: greatest rounded CocoMetric mAP, first tie.
        best = max(self.validation_rows, key=lambda x: x['metrics']['coco/bbox_mAP'])
        elapsed = time.perf_counter() - self.started
        write_json(self.output / 'formal_training_summary.json', {
            'status': 'TRAINING_COMPLETED_VAL_REVIEW_PENDING' if runner.cfg.get('experiment_id') else 'TRAINING_COMPLETED_TEST_PENDING',
            'experiment': runner.cfg.get('experiment_id', 'FRCNN-R50FPN-VAL-001'),
            'resize_scale': list(runner.cfg.get('snapshot_scale', (1333, 800))),
            'train_images': 107, 'train_boxes': 168, 'validation_images': 23, 'validation_boxes': 37,
            'epochs': 100, 'iterations_per_epoch': 54, 'optimizer_updates': 5400,
            'batch_size': 2, 'workers': 0, 'amp': False, 'seed': 42,
            'optimizer': 'SGD', 'base_lr': 0.0025, 'momentum': 0.9, 'weight_decay': 0.0001,
            'warmup_iterations': 500, 'epoch_lr_milestones': [67, 92], 'lr_decay_factor': 0.1,
            'best_validation': best, 'selection_metric': 'coco/bbox_mAP',
            'selection_precision': 'Native CocoMetric rounds overall AP to 3 decimals; first best is kept on ties.',
            'wall_seconds_including_validation_checkpoints_snapshots': elapsed,
            'fixed_train_snapshot_seconds': self.snapshot_seconds,
            'observed_peak_cuda_allocated_mib': self.observed_cuda_peak_mib,
            'memory_note': 'Peak accumulated before LoggerHook resets counters. No claim about reserved memory or OS device memory.',
            'test_evaluated': False, 'training_fit_metrics_used_for_selection': False,
            'final_validation': self.validation_rows[-1]})
        plot_losses(self.rows, self.output / 'loss_curves.png', title=runner.cfg.get('experiment_id', 'FRCNN-R50FPN-VAL-001') + ' / training')
        print(f'FORMAL_100_EPOCHS_5400_UPDATES_COMPLETED; best_val_epoch={best["epoch"]} '
              f'val_mAP50_95={best["metrics"]["coco/bbox_mAP"]:.3f}', flush=True)
