"""User-run inference-only diagnostic on fixed original train8. Never trains."""
import argparse
import builtins
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[2]
DATA=Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset')
ANN=DATA/'annotations/instances_debug8.json'
EXPERIMENT='FRCNN-DIAG-RPNROI-TRAIN8'
SOURCES=[('baseline800','FRCNN-R50FPN-VAL-001/step07_235_27043','resolved_val_001.py',(1333,800)),
         ('scale1280','FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115','resolved_scale1280_val_001.py',(2133,1280))]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'workspace/cache/matplotlib'))


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''): h.update(b)
    return h.hexdigest()


def preparation():
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    from mmengine.config import Config
    selection=read(ROOT/'workspace/configs/frcnn_debug8_selection.json')
    truth=read(ANN)
    assert len(truth['images'])==8 and Counter(a['category_id'] for a in truth['annotations'])=={1:5,2:10}
    assert all(im['file_name'].startswith('train/') for im in truth['images'])
    assert sha(ANN)=='3470758428dab44a2f9dfc9d468275ee71254fc985f6d9f08c6cb1ef774d1edb'
    for sample in selection['samples']:
        assert sha(DATA/'images'/sample['file_name'])==sample['image_sha256']
    review=read(ROOT/'workspace/audit/step11_review/acceptance.json')
    assert review['status']=='PASS' and not review['visual_review_pending']
    records=[]
    for label,relative,filename,scale in SOURCES:
        run=ROOT/'workspace/runs'/relative
        cfg=Config.fromfile(str(run/filename),import_custom_modules=False)
        selected=read(run/'selected_checkpoint.json'); pre=read(run/'pretraining_record.json')
        checkpoint=run/'epoch_100.pth'
        assert sha(checkpoint)==selected['last_checkpoint_sha256']
        assert {k:importlib.metadata.version(k) for k in pre['versions']}==pre['versions']
        assert cfg.model.roi_head.bbox_head.num_classes==2 and cfg.model.backbone.init_cfg is None
        assert tuple(cfg.val_dataloader.dataset.pipeline[1].scale)==scale
        assert cfg.model.test_cfg.rcnn.score_thr==.001 and cfg.model.test_cfg.rcnn.max_per_img==100
        records.append(dict(label=label,run=str(run),config=str(run/filename),config_sha256=sha(run/filename),
            checkpoint=str(checkpoint),checkpoint_sha256=selected['last_checkpoint_sha256'],epoch=100,iterations=5400,scale=list(scale),
            saved_predictions=str(run/'artifacts/epoch_100/predictions_details.json')))
    source=ROOT/'mmdetection'; git=['git','-c','safe.directory='+str(source)]
    assert subprocess.check_output(git+['rev-parse','HEAD'],cwd=source,text=True).strip()==pre['mmdet_commit']
    assert not subprocess.check_output(git+['diff','HEAD','--name-only'],cwd=source)
    print('STEP12_PREPARATION_PASS: 2 existing epoch100 checkpoints; fixed original train8 only; no training/test',flush=True)
    return truth,dict(status='PASS',sources=records,annotation_path=str(ANN),annotation_sha256=sha(ANN),
        versions=pre['versions'],mmdet_commit=pre['mmdet_commit'],training_started=False,detector_forward_performed=False,
        purpose='Locate RPN/ROI losses on fixed train8 at epoch100; not checkpoint selection, training replay, or generalization evaluation')


def checkpoint_data(path):
    import numpy as np
    import torch
    from mmengine.logging import HistoryBuffer
    known={'mmengine.logging.history_buffer.HistoryBuffer':HistoryBuffer,
        'numpy.core.multiarray._reconstruct':np.core.multiarray._reconstruct,'numpy.core.multiarray.scalar':np.core.multiarray.scalar,
        'numpy.ndarray':np.ndarray,'numpy.dtype':np.dtype,'builtins.getattr':builtins.getattr}
    present=torch.serialization.get_unsafe_globals_in_checkpoint(str(path))
    assert set(present)<=set(known),present
    types=[type(np.dtype(k)) for k in ['float64','float32','int64','int32','uint8','bool']]
    with torch.serialization.safe_globals([known[k] for k in present]+types):
        data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['meta']['epoch']==100 and data['meta']['iter']==5400
    assert tuple(data['meta']['dataset_meta']['classes'])==('broken','loose') and len(data['state_dict'])==348
    return data


def overlaps(boxes,gt):
    import numpy as np
    boxes=np.asarray(boxes).reshape(-1,4)
    if not len(boxes):return np.empty(0)
    wh=np.maximum(0,np.minimum(boxes[:,2:],gt[2:])-np.maximum(boxes[:,:2],gt[:2]))
    inter=wh.prod(1); area=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(1)
    return inter/np.maximum(area+(gt[2]-gt[0])*(gt[3]-gt[1])-inter,1e-12)


def inference(record,truth,dest):
    import numpy as np
    import torch
    from mmcv.transforms import Compose
    from mmengine.config import Config
    from mmengine.runner import set_random_seed
    from mmdet.utils import register_all_modules
    from mmdet.registry import MODELS
    from mmdet.apis import inference_detector
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    register_all_modules(init_default_scope=True)
    cfg=Config.fromfile(record['config'],import_custom_modules=False)
    data=checkpoint_data(record['checkpoint'])
    model=MODELS.build(cfg.model).cpu()
    model.load_state_dict(data['state_dict'],strict=True)
    assert all(torch.equal(model.state_dict()[k],v) for k,v in data['state_dict'].items())
    save(dest/'weight_load.json',dict(status='PASS',loaded_tensors=348,checkpoint_sha256=record['checkpoint_sha256'],epoch=100))
    del data
    model.cfg=cfg;model.dataset_meta=dict(cfg.metainfo);model.eval();model.requires_grad_(False);model.to('cuda:0')
    set_random_seed(42,deterministic=True);torch.backends.cudnn.benchmark=False
    pipeline=Compose([dict(type='mmdet.LoadImageFromNDArray'),dict(type='mmdet.Resize',scale=tuple(record['scale']),keep_ratio=True),
        dict(type='mmdet.PackDetInputs',meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor'))])
    captured={}
    def roi_input(module,args):captured['rois']=args[1].detach().clone()
    def head_output(module,args,outputs):captured['cls'],captured['reg']=(t.detach().clone() for t in outputs)
    def fpn_output(module,args,outputs):captured['fpn_shapes']=[list(t.shape) for t in outputs]
    handles=[model.roi_head.bbox_roi_extractor.register_forward_pre_hook(roi_input),model.roi_head.bbox_head.register_forward_hook(head_output),model.neck.register_forward_hook(fpn_output)]
    saved={p['image_id']:p for p in read(record['saved_predictions'])}
    rows=[];images=[]
    try:
        for index,im in enumerate(truth['images']):
            captured.clear()
            with Image.open(DATA/'images'/im['file_name']) as src:rgb=np.array(src.convert('RGB'))
            assert rgb.shape[:2]==(im['height'],im['width'])
            with torch.inference_mode():
                result=inference_detector(model,np.ascontiguousarray(rgb[:,:,::-1]),test_pipeline=pipeline)
                assert all(k in captured for k in ['rois','cls','reg','fpn_shapes'])
                # Decode the same cached ROI outputs without NMS. No second neural forward.
                pre=model.roi_head.bbox_head.predict_by_feat(rois=(captured['rois'],),cls_scores=(captured['cls'],),
                    bbox_preds=(captured['reg'],),batch_img_metas=[result.metainfo],rcnn_test_cfg=None,rescale=True)[0].cpu()
            pred=result.pred_instances.cpu(); boxes=pred.bboxes.numpy();scores=pred.scores.numpy();labels=pred.labels.numpy()
            original=saved[im['id']]
            assert labels.tolist()==original['model_labels'],'Final label/count differs from saved snapshot'
            assert np.allclose(boxes,np.asarray(original['boxes_xyxy']).reshape(-1,4),rtol=0,atol=.1),'Final boxes differ >0.1 original pixels'
            assert np.allclose(scores,original['scores'],rtol=0,atol=1e-4),'Final scores differ >1e-4'
            scale=np.array(list(result.metainfo['scale_factor'])*2,dtype=float)
            rois=captured['rois'][:,1:].cpu().numpy()/scale
            decoded=pre.bboxes.numpy().reshape(-1,2,4); probs=pre.scores.numpy()
            assert np.isfinite(rois).all() and np.isfinite(decoded).all() and np.isfinite(probs).all()
            # Bounded intermediate data sufficient to inspect every surviving RPN proposal and ROI output.
            np.savez_compressed(dest/(Path(im['file_name']).stem+'_intermediates.npz'),
                rpn_boxes_xyxy_original=rois,roi_boxes_xyxy_original=decoded,roi_class_scores=probs,
                final_boxes_xyxy_original=boxes,final_labels=labels,final_scores=scores)
            for a in [a for a in truth['annotations'] if a['image_id']==im['id']]:
                x,y,w,h=a['bbox'];gt=np.array([x,y,x+w,y+h]);cls=a['category_id']-1
                ri=overlaps(rois,gt);di=overlaps(decoded[:,cls,:],gt);fi=overlaps(boxes[labels==cls],gt)
                best=int(ri.argmax()); rpnmax=float(ri[best]);finalmax=float(fi.max()) if len(fi) else 0.
                eligible=probs[:,cls]>.001
                afterscore=float(di[eligible].max()) if eligible.any() else 0.
                # Descriptive stage flags, not a proof of a unique causal failure.
                flag='FINAL_COVERED' if finalmax>=.5 else ('NO_RPN_IOU50' if rpnmax<.5 else
                    'ROI_REGRESSED_BOX_NO_IOU50' if di.max()<.5 else 'ROI_SCORE_FILTER_LOSS' if afterscore<.5 else 'NMS_OR_TOP100_LOSS')
                row=dict(model=record['label'],image_id=im['id'],file_name=im['file_name'],annotation_id=a['id'],class_name=('broken','loose')[cls],
                    rpn_max_iou=rpnmax,rpn_proposals_iou_ge_0_5=int((ri>=.5).sum()),
                    roi_same_class_max_iou_before_filter=float(di.max()),roi_same_class_max_iou_after_score=afterscore,
                    final_same_class_max_iou=finalmax,best_rpn_roi_true_class_score=float(probs[best,cls]),
                    best_rpn_roi_background_score=float(probs[best,2]),diagnostic_flag=flag)
                rows.append(row)
                # Per-object crop: GT green, closest RPN blue, best final same-class prediction red.
                pad=max(64,w*.7,h*.7);crop=[max(0,int(x-pad)),max(0,int(y-pad)),min(im['width'],int(x+w+pad)),min(im['height'],int(y+h+pad))]
                canvas=Image.fromarray(rgb).crop(crop);d=ImageDraw.Draw(canvas)
                def box(b,color):d.rectangle([b[0]-crop[0],b[1]-crop[1],b[2]-crop[0],b[3]-crop[1]],outline=color,width=3)
                box(gt,'lime');box(rois[best],'blue')
                if len(fi):box(boxes[labels==cls][int(fi.argmax())],'red')
                thumb=ImageOps.contain(canvas,(720,400));panel=Image.new('RGB',(740,450),'white');panel.paste(thumb,((740-thumb.width)//2,45))
                font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',15)
                ImageDraw.Draw(panel).text((6,5),f'GT {a["id"]} {row["class_name"]} / {flag}\nRPN IoU {rpnmax:.3f}, final IoU {finalmax:.3f}',font=font,fill='black')
                panel.save(dest/f'gt_{a["id"]:03d}.jpg',quality=95)
            images.append(dict(image_id=im['id'],img_shape=list(result.metainfo['img_shape']),scale_factor=list(result.metainfo['scale_factor']),
                fpn_shapes=captured['fpn_shapes'],rpn_proposals=len(rois),final_predictions=len(scores),saved_predictions_match=True))
            print(f'DIAG_IMAGE_PASS model={record["label"]} image={index+1}/8',flush=True)
    finally:
        for h in handles:h.remove()
        del model
        torch.cuda.empty_cache()
    with (dest/'per_gt.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(dest/'images.json',images)
    return dict(label=record['label'],images=8,gt_boxes=15,flags=dict(Counter(r['diagnostic_flag'] for r in rows)),
        matched_saved_epoch100_predictions=True,source=record,per_gt_path=str(dest/'per_gt.csv'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-preparation',action='store_true');parser.add_argument('--log-dir',type=Path)
    args=parser.parse_args()
    truth,record=preparation()
    if args.check_preparation:
        dest=ROOT/'workspace/audit/step12_preflight';dest.mkdir(parents=True,exist_ok=True)
        save(dest/'preparation.json',record);return 0
    assert args.log_dir is not None
    logs=args.log_dir.resolve();assert logs.is_relative_to(ROOT/'workspace/logs') and logs.name.startswith('step12_')
    run=ROOT/'workspace/runs'/EXPERIMENT/logs.name
    assert not run.exists(),'Existing diagnostic output; refusing overwrite'
    (run/'code').mkdir(parents=True)
    for p in [Path(__file__),Path(__file__).with_name('run_step12_diagnose_rpn_roi.cmd'),ROOT/'workspace/notes/STEP12_rpn_roi_diagnostic_plan.md',ANN]:shutil.copy2(p,run/'code'/p.name)
    save(run/'preparation.json',record)
    print('RUN_DIR='+str(run),flush=True)
    results=[]
    for source in record['sources']:
        dest=run/source['label'];dest.mkdir();results.append(inference(source,truth,dest))
        assert sha(source['checkpoint'])==source['checkpoint_sha256']
    save(run/'summary.json',dict(status='DIAGNOSTIC_COMPLETED_REVIEW_PENDING',created_utc=datetime.now(timezone.utc).isoformat(),results=results,
        training_started=False,optimizer_created=False,detector_forward_performed=True,test_evaluated=False,
        interpretation='Train8 diagnostic only. Epoch100 used to examine final fit, not to replace val-selected epochs66/5. No historical sampling reconstruction.'))
    print('STEP12_DIAGNOSTIC_PASS; send LOG_DIR and RUN_DIR; no training or test evaluation',flush=True)
    return 0


if __name__=='__main__':
    if '--log-dir' in sys.argv:
        logs=Path(sys.argv[sys.argv.index('--log-dir')+1]).resolve()
        assert logs.is_relative_to(ROOT/'workspace/logs') and logs.name.startswith('step12_')
        logs.mkdir(parents=True,exist_ok=True)
        class Tee:
            def __init__(self,terminal,file):self.terminal,self.file=terminal,file
            def write(self,v):self.terminal.write(v);self.file.write(v);self.file.flush();return len(v)
            def flush(self):self.terminal.flush();self.file.flush()
        with (logs/'step12_console.txt').open('x',encoding='utf-8') as file:
            with redirect_stdout(Tee(sys.stdout,file)),redirect_stderr(Tee(sys.stderr,file)):
                try:code=main()
                except Exception:
                    traceback.print_exc();print('STEP12_DIAGNOSTIC_FAIL; retain outputs and do not retry automatically');code=1
                print('LOG_DIR='+str(logs))
        raise SystemExit(code)
    raise SystemExit(main())
