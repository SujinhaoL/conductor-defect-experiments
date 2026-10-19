# Diagnostic data/model config for FRCNN-DEBUG-OVERFIT-8.
# No training loop is enabled here. A later training config will inherit this file.
_base_ = '../../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

data_root = 'D:/小论文/conductor_defect_dataset/03_coco_dataset/'
metainfo = dict(classes=('broken', 'loose'), palette=[(240, 60, 60), (20, 210, 230)])
model = dict(backbone=dict(init_cfg=None), roi_head=dict(bbox_head=dict(num_classes=2)))
load_from = 'D:/小论文/model_projects/Faster R-CNN/workspace/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume = False
randomness = dict(seed=42, deterministic=True)

# Deliberately omit random image augmentations for the diagnostic subset.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
train_dataloader = dict(
    _delete_=True, batch_size=2, num_workers=0, persistent_workers=False,
    pin_memory=False, drop_last=False, sampler=dict(type='DefaultSampler', shuffle=False),
    batch_sampler=None,
    dataset=dict(type='CocoDataset', data_root=data_root,
                 ann_file='annotations/instances_debug8.json', data_prefix=dict(img='images/'),
                 metainfo=metainfo, filter_cfg=None, pipeline=train_pipeline))

# This config is for inspection only; do not accidentally inherit COCO training/evaluation.
train_cfg = None
val_cfg = None
test_cfg = None
val_dataloader = None
test_dataloader = None
val_evaluator = None
test_evaluator = None
