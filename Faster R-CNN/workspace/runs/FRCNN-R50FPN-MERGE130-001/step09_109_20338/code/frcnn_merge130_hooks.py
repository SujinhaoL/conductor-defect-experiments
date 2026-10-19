"""Artifacts for user-run MERGE130. No validation or test inference."""
import copy
import json
from pathlib import Path
import time

import torch
from mmdet.registry import DATASETS, HOOKS
from frcnn_debug_hooks import Debug8ArtifactsHook, OUTPUT_KEYS, write_json


def plot_merge_losses(rows, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    x = [r['iteration'] for r in rows]
    for ax, keys, title in zip(axes.flat,
            [('loss',), ('loss_rpn_cls', 'loss_rpn_bbox'), ('loss_cls', 'loss_bbox'), ('learning_rate',)],
            ['Total loss', 'RPN losses', 'ROI losses', 'Learning rate used']):
        for key in keys:
            ax.plot(x, [r[key] for r in rows], linewidth=1, label=key)
        ax.set(title=title, xlabel='Optimizer update')
        ax.grid(alpha=0.25); ax.legend()
    fig.suptitle('FRCNN-R50FPN-MERGE130-001 / training only')
    fig.savefig(destination, dpi=150)
    plt.close(fig)


@HOOKS.register_module()
class Merge130ArtifactsHook(Debug8ArtifactsHook):
    # The inherited per-iteration loss/LR/memory recorder and fixed train8
    # snapshot restore RNG state; their train-fit scores never select an epoch.
    def before_train(self, runner):
        self.started = time.perf_counter()
        self.output = Path(runner.work_dir) / 'artifacts'
        self.output.mkdir(exist_ok=False)
        pre = json.loads((Path(runner.work_dir) / 'pretraining_record.json').read_text(encoding='utf-8'))
        self.expected_iters = pre['optimizer_updates']
        assert len(runner.train_dataloader.dataset) == 130 and len(runner.train_dataloader) == 65
        assert runner.max_epochs == 100 and runner.world_size == 1 and not runner.cfg.resume
        assert all(runner.cfg[key] is None for key in ('val_dataloader', 'val_cfg', 'val_evaluator', 'test_dataloader', 'test_cfg', 'test_evaluator'))
        self.model = runner.model
        self.model.cfg = runner.cfg
        self.model.dataset_meta = runner.train_dataloader.dataset.metainfo
        subset = copy.deepcopy(runner.cfg.train_dataloader.dataset)
        subset.ann_file = 'annotations/instances_debug8.json'
        subset.filter_cfg = None
        self.dataset = DATASETS.build(subset)
        assert len(self.dataset) == 8 and self.dataset.cat2label == {1: 0, 2: 1}
        original = torch.load(runner.cfg.load_from, map_location='cpu', weights_only=True)['state_dict']
        loaded = self.model.state_dict()
        assert original.keys() == loaded.keys()
        changed = {key for key in original if original[key].shape != loaded[key].shape}
        assert changed == OUTPUT_KEYS
        equal = [key for key in original if key not in changed and torch.equal(original[key], loaded[key].cpu())]
        assert len(equal) == 344, 'Full matching COCO tensors not loaded before first update'
        write_json(self.output / 'actual_training_weight_load.json', dict(
            status='PASS', loaded_tensors=344, checkpoint=runner.cfg.load_from,
            expected_shape_changes={key: dict(coco=list(original[key].shape), two_class=list(loaded[key].shape)) for key in sorted(changed)},
            loaded_before_first_optimizer_update=True, fresh_official_coco_initialization=True,
            continued_from_val001_checkpoint=False))
        del original, loaded
        print('MERGE130_OFFICIAL_COCO_WEIGHT_LOAD_PASS: matched=344 expected_new_outputs=4; fresh initialization', flush=True)
        torch.cuda.reset_peak_memory_stats()
        self.snapshot(runner, 0)

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if epoch % self.interval == 0 or epoch == runner.max_epochs:
            self.snapshot(runner, epoch)
            plot_merge_losses(self.rows, self.output / 'loss_curves.png')

    def after_train(self, runner):
        torch.cuda.synchronize()
        assert runner.epoch == 100 and runner.iter == self.expected_iters == 6500 and len(self.rows) == 6500
        assert [row['epoch'] for row in self.snapshots] == [0, 25, 50, 75, 100]
        write_json(self.output / 'formal_training_summary.json', dict(
            status='TRAINING_COMPLETED_TEST_PENDING', experiment='FRCNN-R50FPN-MERGE130-001',
            train_images=130, train_boxes=205, broken=48, loose=157,
            validation_images=None, validation_boxes=None, independent_validation=False,
            epochs=100, iterations_per_epoch=65, optimizer_updates=runner.iter,
            batch_size=2, workers=0, amp=False, seed=42, optimizer='SGD', base_lr=0.0025,
            momentum=0.9, weight_decay=0.0001, warmup_iterations=500,
            epoch_lr_milestones=[67, 92], lr_decay_factor=0.1,
            selected_epoch=100, selection_metric=None, selection_rule='last epoch100 fixed before training; no val/test selection',
            wall_seconds_including_checkpoints_snapshots=time.perf_counter()-self.started,
            fixed_train_snapshot_seconds=self.snapshot_seconds,
            observed_peak_cuda_allocated_mib=self.observed_cuda_peak_mib,
            test_evaluated=False, train_fit_metrics_used_for_selection=False,
            fresh_official_coco_initialization=True, continued_from_val001_checkpoint=False))
        plot_merge_losses(self.rows, self.output / 'loss_curves.png')
        print('MERGE130_100_EPOCHS_6500_UPDATES_COMPLETED; final_epoch100 selected; val/test not evaluated', flush=True)
