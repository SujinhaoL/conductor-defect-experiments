"""Read only YOLO project environment/defaults. No checkpoint/model creation or inference."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'workspace/configs/ultralytics_settings')
os.environ['MPLCONFIGDIR']=str(ROOT/'workspace/cache/matplotlib')
os.environ['YOLO_AUTOINSTALL']='false'
import torch
import ultralytics
from ultralytics.cfg import get_cfg

assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
assert Path(ultralytics.__file__).resolve().is_relative_to(ROOT/'ultralytics_src')
git=['git','-c','safe.directory='+str(ROOT/'ultralytics_src')]
cfg=vars(get_cfg(overrides=dict(mode='val',task='detect',imgsz=1600,batch=4,workers=0,rect=True,quantize=16,
    split='test',conf=.001,iou=.7,max_det=300,save_json=False,save_txt=False,plots=True,augment=False,device='0')))
result=dict(python=sys.version.split()[0],versions={name:importlib.metadata.version(name) for name in
    ('torch','torchvision','ultralytics','numpy','pillow','pyyaml')},source_file=ultralytics.__file__,
    commit=subprocess.check_output(git+['rev-parse','HEAD'],cwd=ROOT/'ultralytics_src',text=True).strip(),
    tracked_diff=subprocess.check_output(git+['diff','HEAD','--name-only'],cwd=ROOT/'ultralytics_src',text=True).strip(),
    cuda_available=torch.cuda.is_available(),torch_cuda=torch.version.cuda,
    selected_defaults={k:cfg[k] for k in ('imgsz','batch','workers','quantize','rect','conf','iou','max_det','save_json','save_txt','augment','split','nms')},
    model_created=False,model_inference_performed=False,training_started=False)
print(json.dumps(result,ensure_ascii=False,indent=2))
