# FRCNN-DEBUG-OVERFIT-8: memorization diagnostic, NOT generalization results.
_base_ = './frcnn_debug8_pipeline_check.py'

custom_imports = dict(imports=['frcnn_debug_hooks'], allow_failed_imports=False)
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=100, val_interval=101)
# val/test loops and dataloaders remain None. Only the eight TRAIN images are used.
optim_wrapper = dict(
    _delete_=True, type='OptimWrapper', accumulative_counts=1,
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))
# Official base LR 0.02 at global batch 16 -> 0.0025 at batch 2.
# Brief warmup for this 400-update diagnostic; no late LR decay.
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=20),
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
               dist_cfg=dict(backend='gloo'))
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=25, by_epoch=True,
                    save_last=True, max_keep_ckpts=4, save_best=None),
    visualization=dict(type='DetVisualizationHook', draw=False))
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)

# Lower filtering for train-fit AP diagnostics; display threshold is separate.
# Other official RPN/ROI NMS settings remain inherited.
model = dict(test_cfg=dict(rcnn=dict(score_thr=0.001, max_per_img=100)))
custom_hooks = [dict(type='Debug8ArtifactsHook', interval=25, visual_threshold=0.30)]
resume = False
