# FRCNN-R50FPN-VAL-001: official R50-FPN reference baseline on the fixed split.
_base_ = '../../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'
custom_imports = dict(imports=['frcnn_formal_hooks'], allow_failed_imports=False)
data_root = 'D:/小论文/conductor_defect_dataset/03_coco_dataset/'
metainfo = dict(classes=('broken', 'loose'), palette=[(240, 60, 60), (20, 210, 230)])
model = dict(backbone=dict(init_cfg=None), roi_head=dict(bbox_head=dict(num_classes=2)),
             test_cfg=dict(rcnn=dict(score_thr=0.001, max_per_img=100)))
load_from = 'D:/小论文/model_projects/Faster R-CNN/workspace/weights/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume = False
randomness = dict(seed=42, deterministic=True)

# Inherit official Resize, RandomFlip, sampler and aspect-ratio batching.
train_dataloader = dict(
    batch_size=2, num_workers=0, persistent_workers=False, pin_memory=False, drop_last=False,
    dataset=dict(data_root=data_root, ann_file='annotations/instances_train.json',
                 data_prefix=dict(img='images/'), metainfo=metainfo, filter_cfg=None))
val_dataloader = dict(
    batch_size=1, num_workers=0, persistent_workers=False, pin_memory=False,
    dataset=dict(data_root=data_root, ann_file='annotations/instances_val.json',
                 data_prefix=dict(img='images/'), metainfo=metainfo))
val_evaluator = dict(
    ann_file=data_root + 'annotations/instances_val.json', metric='bbox', classwise=True,
    proposal_nums=(1, 10, 100),
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'])
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)
val_cfg = dict(type='ValLoop')
# Test evaluation is a separate user-run step after reviewing the selected model.
test_cfg = None
test_dataloader = None
test_evaluator = None

optim_wrapper = dict(
    _delete_=True, type='OptimWrapper', accumulative_counts=1,
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    # Scale official 8/12 and 11/12 milestone positions to the 100-epoch run.
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=100, milestones=[67, 92], gamma=0.1),
]
auto_scale_lr = dict(enable=False, base_batch_size=16)
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
               dist_cfg=dict(backend='gloo'))
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=10),
    checkpoint=dict(type='CheckpointHook', interval=10, by_epoch=True, max_keep_ckpts=2,
                    save_last=True, save_best='coco/bbox_mAP', rule='greater'),
    visualization=dict(type='DetVisualizationHook', draw=False))
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
custom_hooks = [dict(type='FormalBaselineArtifactsHook', interval=25, visual_threshold=0.30)]
