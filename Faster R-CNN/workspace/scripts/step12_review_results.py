"""Independent STEP12 saved-array review. No model creation or inference."""
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'workspace/runs/FRCNN-DIAG-RPNROI-TRAIN8/step12_24867_6686'
OUT=ROOT/'workspace/audit/step12_review'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()
def csvrows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def overlaps(boxes,gt):
    b=np.asarray(boxes,dtype=np.float64).reshape(-1,4)
    if not len(b):return np.empty(0)
    inter=np.maximum(0,np.minimum(b[:,2:],gt[2:])-np.maximum(b[:,:2],gt[:2])).prod(1)
    union=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])+(gt[2]-gt[0])*(gt[3]-gt[1])-inter
    return inter/np.maximum(union,1e-12)
def maxval(a):return float(a.max()) if len(a) else 0.

def main():
    assert Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    summary=read(RUN/'summary.json');pre=read(RUN/'preparation.json')
    assert not summary['training_started'] and not summary['optimizer_created'] and not summary['test_evaluated']
    assert summary['detector_forward_performed'] and len(summary['results'])==2
    assert sha(pre['annotation_path'])==pre['annotation_sha256']
    truth=read(pre['annotation_path']);assert len(truth['images'])==8 and len(truth['annotations'])==15
    anchors=csvrows(ROOT/'workspace/audit/step11_resolution/debug8_anchor_sizes.csv')
    joined=[];all_stats=[];hashes={}
    for source,result in zip(pre['sources'],summary['results']):
        assert source==result['source']
        assert sha(source['checkpoint'])==source['checkpoint_sha256'] and sha(source['config'])==source['config_sha256']
        name=source['label']; dest=RUN/name;meta=read(dest/'images.json');load=read(dest/'weight_load.json')
        assert load['status']=='PASS' and load['loaded_tensors']==348 and load['checkpoint_sha256']==source['checkpoint_sha256']
        saved={p['image_id']:p for p in read(source['saved_predictions'])}
        rows={int(r['annotation_id']):r for r in csvrows(dest/'per_gt.csv')}
        short=source['scale'][1];anchor={int(r['annotation_id']):r for r in anchors if int(r['scale_short'])==short}
        stats=dict(label=name,rpn_covered=0,roi_before_score_covered=0,final_covered=0,flags=Counter(),missing_ids=[],zero_positive_anchor_ids=[])
        for im,info in zip(truth['images'],meta):
            assert im['id']==info['image_id'] and im['file_name'].startswith('train/')
            file=dest/(Path(im['file_name']).stem+'_intermediates.npz')
            with np.load(file,allow_pickle=False) as data:
                arrays={k:data[k] for k in data.files}
            rpn=arrays['rpn_boxes_xyxy_original'];roi=arrays['roi_boxes_xyxy_original'];prob=arrays['roi_class_scores']
            final=arrays['final_boxes_xyxy_original'];labels=arrays['final_labels'];scores=arrays['final_scores']
            n=len(rpn)
            assert rpn.shape==(n,4) and roi.shape==(n,2,4) and prob.shape==(n,3)
            assert n==info['rpn_proposals']==1000 and len(final)==info['final_predictions']<=100
            assert all(np.isfinite(v).all() for v in arrays.values())
            assert np.allclose(prob.sum(1),1,atol=1e-6) and np.all((prob>=0)&(prob<=1))
            assert set(labels.tolist())<={0,1} and np.all(scores>.001) and np.all(scores<=1)
            for b in [rpn,roi.reshape(-1,4),final]:
                assert np.all(b[:,2:]>=b[:,:2]) and b.min()>=-.001
                assert np.max(b[:,[0,2]])<=im['width']+.001 and np.max(b[:,[1,3]])<=im['height']+.001
            old=saved[im['id']]
            assert labels.tolist()==old['model_labels']
            assert np.allclose(final,np.asarray(old['boxes_xyxy']).reshape(-1,4),atol=.1,rtol=0)
            assert np.allclose(scores,old['scores'],atol=1e-4,rtol=0)
            ih,iw=info['img_shape'];ph,pw=math.ceil(ih/32)*32,math.ceil(iw/32)*32
            assert info['fpn_shapes']==[[1,256,math.ceil(ph/s),math.ceil(pw/s)] for s in [4,8,16,32,64]]
            for a in [a for a in truth['annotations'] if a['image_id']==im['id']]:
                x,y,w,h=a['bbox'];gt=np.array([x,y,x+w,y+h]);c=a['category_id']-1
                ri=overlaps(rpn,gt);di=overlaps(roi[:,c,:],gt);fi=overlaps(final[labels==c],gt)
                mxr,mxd,mxf=maxval(ri),maxval(di),maxval(fi);mxs=maxval(di[prob[:,c]>.001]);bi=int(ri.argmax())
                values=dict(rpn_max_iou=mxr,rpn_proposals_iou_ge_0_5=int((ri>=.5).sum()),
                    roi_same_class_max_iou_before_filter=mxd,roi_same_class_max_iou_after_score=mxs,final_same_class_max_iou=mxf,
                    best_rpn_roi_true_class_score=float(prob[bi,c]),best_rpn_roi_background_score=float(prob[bi,2]))
                for k,v in values.items():assert abs(v-float(rows[a['id']][k]))<1e-6,(a['id'],k)
                flag='FINAL_COVERED' if mxf>=.5 else 'NO_RPN_IOU50' if mxr<.5 else 'ROI_REGRESSED_BOX_NO_IOU50' if mxd<.5 else 'ROI_SCORE_FILTER_LOSS' if mxs<.5 else 'NMS_OR_TOP100_LOSS'
                assert flag==rows[a['id']]['diagnostic_flag']
                positive=int(anchor[a['id']]['positive_anchors_before_sampling'])
                joined.append(dict(model=name,annotation_id=a['id'],image_id=im['id'],file_name=im['file_name'],category=('broken','loose')[c],
                    resized_width=float(anchor[a['id']]['resized_box_width']),resized_height=float(anchor[a['id']]['resized_box_height']),
                    default_anchor_max_iou=float(anchor[a['id']]['max_anchor_iou']),positive_anchors_before_sampling=positive,
                    **values,diagnostic_flag=flag))
                stats['rpn_covered']+=mxr>=.5;stats['roi_before_score_covered']+=mxd>=.5;stats['final_covered']+=mxf>=.5;stats['flags'][flag]+=1
                if mxf<.5:stats['missing_ids'].append(a['id'])
                if positive==0:stats['zero_positive_anchor_ids'].append(a['id'])
        assert dict(stats['flags'])==result['flags']
        all_stats.append(stats)
        for f in dest.iterdir():
            if f.is_file():hashes[str(f)]=sha(f)
    for f in [RUN/'summary.json',RUN/'preparation.json',*sorted((RUN/'code').iterdir())]:
        if f.is_file():hashes[str(f)]=sha(f)
    missed=all_stats[1]['missing_ids']
    assert missed==[87,123,124,160] and missed==all_stats[1]['zero_positive_anchor_ids']
    with (OUT/'anchor_proposal_roi_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(joined[0]));writer.writeheader();writer.writerows(joined)
    # Assemble existing diagnostic crops without changing any image content.
    canvas=Image.new('RGB',(1480,1860),'white');font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',22)
    draw=ImageDraw.Draw(canvas);draw.text((12,8),'Epoch100 train-only diagnostic | GT green, nearest RPN blue, nearest final red',font=font,fill='black')
    draw.text((12,35),'Left: baseline800 | Right: scale1280 | These are enlarged original-image crops, not input pixels',font=font,fill='black')
    for i,aid in enumerate(missed):
        for j,name in enumerate(['baseline800','scale1280']):
            with Image.open(RUN/name/f'gt_{aid:03d}.jpg') as image:canvas.paste(image,(j*740,60+i*450))
    canvas.save(OUT/'four_uncovered_targets_comparison.jpg',quality=95)
    acceptance=dict(status='PASS',reviewed_utc=datetime.now(timezone.utc).isoformat(),run_dir=str(RUN),
        models=all_stats,images_per_model=8,gt_per_model=15,independent_saved_array_recompute=True,
        saved_snapshot_comparison_pass=True,fpn_dimensions_pass=True,checkpoint_hashes_pass=True,
        roi_pre_filter_below_0_5_for_all_four=True,four_zero_positive_anchor_match=True,
        evidence_files_sha256=hashes,training_started=False,detector_forward_performed=False,test_used=False,
        visual_review_completed=True,visual_review_scope='All four scale1280 miss crops inspected; GT semantics and annotation boundaries require domain interpretation, not assumed wrong.',
        limitations=['RPN outputs are after RPN NMS and top1000; raw RPN scoring/pre-NMS is not localized.',
            'No positive-anchor geometry is a supported hypothesis, not causal proof or an explanation of all val/test errors.',
            'The original train8 subset and epoch100 describe fit, not independent generalization or historical random sampling.'])
    (OUT/'acceptance.json').write_text(json.dumps(acceptance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('STEP12_RESULT_REVIEW_PASS: 16 saved NPZ files recomputed; hashes/classes/coordinates/snapshot/FPN checks passed')
    print(json.dumps(all_stats,ensure_ascii=False))


if __name__=='__main__':main()
