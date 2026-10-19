# Test the checkpoint already selected by original val mAP50-95.
# Inherit the exact saved training model/preprocessing/inference settings.
_base_ = '../runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py'
custom_imports = None
custom_hooks = []
train_cfg = None
train_dataloader = None
val_cfg = None
val_dataloader = None
val_evaluator = None
optim_wrapper = None
param_scheduler = None
load_from = 'D:/小论文/model_projects/Faster R-CNN/workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/best_coco_bbox_mAP_epoch_66.pth'
resume = False

test_cfg = dict(_delete_=True, type='TestLoop')
test_dataloader = dict(
    _delete_=True, batch_size=1, num_workers=0, persistent_workers=False,
    drop_last=False, sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='CocoDataset',
                 data_root='D:/小论文/conductor_defect_dataset/03_coco_dataset/',
                 ann_file='annotations/instances_test.json', data_prefix=dict(img='images/'),
                 metainfo=dict(classes=('broken', 'loose'), palette=[(240, 60, 60), (20, 210, 230)]),
                 test_mode=True, pipeline=[
                     dict(type='LoadImageFromFile'),
                     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                     dict(type='LoadAnnotations', with_bbox=True),
                     dict(type='PackDetInputs', meta_keys=(
                         'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))]))
test_evaluator = dict(
    _delete_=True, type='CocoMetric',
    ann_file='D:/小论文/conductor_defect_dataset/03_coco_dataset/annotations/instances_test.json',
    metric='bbox', classwise=True, proposal_nums=(1, 10, 100),
    metric_items=['mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'])
