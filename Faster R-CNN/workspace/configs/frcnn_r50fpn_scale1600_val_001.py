# Final size comparison: inherit the completed default-anchor STEP11 run.
# Only Resize changes; STEP13 (anchor4) is NOT the direct size-only control.
_base_ = '../runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115/resolved_scale1280_val_001.py'
experiment_id = 'FRCNN-R50FPN-SCALE1600-VAL-001'
snapshot_scale = (2666, 1600)
work_dir = None  # A unique directory is assigned only when the user starts.

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(2666, 1600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(2666, 1600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))

# The inherited evaluation-pipeline name does not enable test-set evaluation.
test_dataloader = None
test_cfg = None
test_evaluator = None
resume = False
