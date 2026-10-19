"""CPU synthetic geometry and native metric API checks. No checkpoint load, detector or inference."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'workspace/scripts'))
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'workspace/configs/ultralytics_settings')
os.environ['MPLCONFIGDIR']=str(ROOT/'workspace/cache/matplotlib')
os.environ['YOLO_AUTOINSTALL']='false'
import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import DetMetrics,smooth
from step18_test_yolo1600 import save_json

pred=dict(bboxes=torch.tensor([[80.,560.,800.,880.]]),conf=torch.tensor([.8]),cls=torch.tensor([0.]))
before=pred['bboxes'].clone()
prepared=dict(imgsz=(1600,1600),ori_shape=(100,200),ratio_pad=((8.,8.),(0.,400.)))
scaled=DetectionValidator.scale_preds(SimpleNamespace(),pred,prepared)
assert torch.equal(pred['bboxes'],before) and np.allclose(scaled['bboxes'].tolist(),[[10.,20.,100.,60.]])
json.dumps(dict(ratio_pad=prepared['ratio_pad'],scores=scaled['conf'].tolist()),allow_nan=False)
metric=DetMetrics(names={0:'broken',1:'loose'})
metric.update_stats(dict(tp=np.ones((2,10),dtype=bool),conf=np.array([.8,.7]),pred_cls=np.array([0.,1.]),
    target_cls=np.array([0.,1.]),target_img=np.array([0.,1.]),im_name='synthetic-not-a-dataset-image'))
metric.process(plot=False)
index=int(smooth(metric.box.f1_curve.mean(0),.1).argmax())
assert np.allclose(metric.box.p,metric.box.p_curve[:,index]) and np.allclose(metric.box.r,metric.box.r_curve[:,index])
assert len(metric.class_result(0))==4 and np.isfinite(list(metric.results_dict.values())).all()
audit=ROOT/'workspace/audit/step18_preflight';audit.mkdir(parents=True,exist_ok=True)
save_json(audit/'export_api_check.json',dict(status='PASS',synthetic_cpu_only=True,original_coordinates_verified=True,
    native_prediction_tensor_not_modified=True,native_metric_class_and_pr_threshold_api_verified=True,
    model_created=False,checkpoint_loaded=False,test_inference_performed=False,training_started=False))
print('STEP18_EXPORT_API_CPU_PASS; synthetic arrays only; no detector/model inference/training')
