# One-factor anchor-size control against the completed frozen SCALE1280 run.
_base_ = '../runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115/resolved_scale1280_val_001.py'
experiment_id = 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001'
work_dir = None  # Assigned once by the launcher; old runs are never overwritten.
model = dict(rpn_head=dict(anchor_generator=dict(scales=[4])))

# Inherited: Resize(2133,1280), ratios/strides, num_classes=2, fresh official COCO
# load_from, resume=False, train107/val23, batch2, FP32, seed42 and 100 epochs.
# No test dataloader/evaluator. Snapshot scale is also (2133,1280).
