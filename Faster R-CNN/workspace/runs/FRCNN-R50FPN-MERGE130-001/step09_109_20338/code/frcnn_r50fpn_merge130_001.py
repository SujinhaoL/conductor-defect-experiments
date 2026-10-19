# FRCNN-R50FPN-MERGE130-001: settings frozen BEFORE VAL-001 test.
# Inherit the exact saved VAL-001 model, optimizer, Resize, augmentation and seed.
_base_ = '../runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py'
custom_imports = dict(_delete_=True, imports=['frcnn_merge130_hooks'], allow_failed_imports=False)

# Fresh official full COCO detector initialization. Never resume VAL-001 weights.
load_from = 'D:/小论文/model_projects/Faster R-CNN/workspace/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume = False
train_dataloader = dict(dataset=dict(ann_file='annotations/instances_trainval.json'))
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=100)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

# No validation-based best checkpoint. Final test uses epoch_100.pth only.
default_hooks = dict(checkpoint=dict(
    _delete_=True, type='CheckpointHook', interval=10, by_epoch=True,
    max_keep_ckpts=2, save_last=True, save_best=None))
custom_hooks = [dict(type='Merge130ArtifactsHook', interval=25, visual_threshold=0.30)]
