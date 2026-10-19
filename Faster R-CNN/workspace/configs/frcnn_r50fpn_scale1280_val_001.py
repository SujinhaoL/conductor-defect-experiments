# Controlled sensitivity run: only image resize differs from frozen VAL-001.
_base_ = '../runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py'
experiment_id = 'FRCNN-R50FPN-SCALE1280-VAL-001'
snapshot_scale = (2133, 1280)
work_dir = None  # Launcher assigns a fresh unique run directory.

# Same annotation order, interpolation, flip, normalization and padding as baseline.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(2133, 1280), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(2133, 1280), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# test_pipeline is the inherited name for evaluation transforms; no test set is run.
test_dataloader = None
test_cfg = None
test_evaluator = None
resume = False
