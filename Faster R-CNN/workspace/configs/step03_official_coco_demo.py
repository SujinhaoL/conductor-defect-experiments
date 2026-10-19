# Official 80-class COCO inference smoke check; not a conductor training config.
_base_ = '../../mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# The official dataset config explicitly says to omit LoadAnnotations when
# running an image without ground truth. All model and resize settings stay inherited.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=(
        'img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
]
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
