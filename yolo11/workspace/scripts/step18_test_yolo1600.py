"""User-run test of YOLO11s1600 epoch54 best.pt. Check branch has no model creation/forward."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/'workspace/scripts'
RUN=ROOT/'workspace/runs/Y11-SENS-IMGSZ1600-001'
CHECKPOINT=RUN/'weights/best.pt'
TEST_ROOT=RUN/'test_selected_epoch54'
CONFIG=ROOT/'workspace/configs/step18_yolo1600_test.yaml'
DATA_YAML=ROOT/'workspace/configs/conductor_bl_v1.yaml'
QUEUE=ROOT.parent/'baseline_analysis/audit/test_completion_review_20261014/fixed_test_queue.json'
DATA=Path('D:/小论文/conductor_defect_dataset/02_yolo11_dataset')
INVENTORY=Path('D:/小论文/conductor_defect_dataset/03_coco_dataset/audit/source_inventory.json')
COCO_TEST=INVENTORY.parents[1]/'annotations/instances_test.json'
PINNED_VERSIONS={'torch':'2.8.0+cu128','torchvision':'0.23.0+cu128','ultralytics':'8.4.152',
                 'numpy':'2.5.2','pillow':'12.3.0','pyyaml':'6.0.3'}
COMMIT='9aa41d679d121afb8b4cc512d7a162f4f36886c3'


def require(value,message):
    if not value:raise AssertionError(message)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path,value):
    def convert(obj):
        if isinstance(obj,Path):return str(obj)
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8')


class Tee:
    def __init__(self,*handles):self.handles=handles
    def write(self,text):
        for handle in self.handles:handle.write(text);handle.flush()
    def flush(self):
        for handle in self.handles:handle.flush()
    def isatty(self):return False


def configure():
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(SCRIPTS))
    os.environ['YOLO_CONFIG_DIR']=str(ROOT/'workspace/configs/ultralytics_settings')
    os.environ['MPLCONFIGDIR']=str(ROOT/'workspace/cache/matplotlib')
    os.environ['TORCH_HOME']=str(ROOT/'workspace/cache/torch')
    os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'


def check_preparation():
    import yaml
    import torch
    import ultralytics
    from ultralytics.cfg import get_cfg
    require(Path(sys.executable).resolve()==(ROOT/'.venv/Scripts/python.exe').resolve(),'Use the YOLO11 dedicated interpreter')
    require(sys.version_info[:2]==(3,12),'Unexpected Python major/minor')
    versions={name:importlib.metadata.version(name) for name in PINNED_VERSIONS}
    require(versions==PINNED_VERSIONS,'YOLO environment versions changed')
    source=ROOT/'ultralytics_src'
    require(Path(ultralytics.__file__).resolve().is_relative_to(source),'Wrong Ultralytics import')
    git=['git','-c','safe.directory='+str(source)]
    require(subprocess.check_output(git+['rev-parse','HEAD'],cwd=source,text=True).strip()==COMMIT,'Official source commit changed')
    require(not subprocess.check_output(git+['diff','HEAD','--name-only'],cwd=source),'Official tracked source changed')
    queue=read(QUEUE)['items'][3]
    require(queue['order']==4 and queue['epoch']==54 and Path(queue['checkpoint']).resolve()==CHECKPOINT,'Fixed checkpoint queue differs')
    require(sha(CHECKPOINT)==queue['checkpoint_sha256'],'Epoch54 best.pt changed')
    training=yaml.safe_load((RUN/'args.yaml').read_text(encoding='utf-8'))
    require(training['imgsz']==1600 and training['batch']==4 and training['workers']==0 and training['amp'] is True,'Original1600 training settings differ')
    with (RUN/'results.csv').open(encoding='utf-8-sig',newline='') as handle:
        rows=[{key.strip():value.strip() for key,value in row.items()} for row in csv.DictReader(handle)]
    require(len(rows)==100 and int(float(rows[-1]['epoch']))==100,'100-epoch training not complete')
    best=max(rows,key=lambda row:float(row['metrics/mAP50-95(B)']))
    require(int(float(best['epoch']))==54,'Expected already fixed selection epoch54')
    config=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    require(config['imgsz']==1600 and config['batch']==4 and config['workers']==0 and config['quantize']==16
        and config['conf']==.001 and config['iou']==.7 and config['max_det']==300 and config['rect'] is True,
        'Fixed evaluation config differs')
    require(config['split']=='test' and config['save_json'] is False and config['save_txt'] is False and config['augment'] is False,'Native test metric protocol changed')
    # Resolve config without constructing a validator or detector.
    get_cfg(overrides={**config,'data':str(DATA_YAML),'model':str(CHECKPOINT),'mode':'val','task':'detect'})
    dataset=yaml.safe_load(DATA_YAML.read_text(encoding='utf-8'))
    require(Path(dataset['path']).resolve()==DATA and dataset['test']=='images/test' and dataset['names']=={0:'broken',1:'loose'},'Data YAML/class mapping differs')
    inventory=[row for row in read(INVENTORY) if row['split']=='test']
    require(len(inventory)==23,'Original test inventory differs')
    image_hashes,label_hashes={},{}
    for row in inventory:
        image=DATA/'images/test'/row['filename']
        label=DATA/'labels/test'/(image.stem+'.txt')
        require(sha(image)==row['image_sha256'] and sha(label)==row['label_sha256'],'Original test image/label changed: '+image.name)
        image_hashes[str(image)]=sha(image);label_hashes[str(label)]=sha(label)
    require({p.name for p in (DATA/'images/test').iterdir() if p.is_file()}=={row['filename'] for row in inventory},'Extra/missing original test images')
    truth=read(COCO_TEST)
    require(len(truth['images'])==23 and len(truth['annotations'])==36 and
        sum(a['category_id']==1 for a in truth['annotations'])==8 and sum(a['category_id']==2 for a in truth['annotations'])==28,'Test GT counts differ')
    require(torch.cuda.is_available() and torch.version.cuda=='12.8','YOLO CUDA environment unavailable')
    metadata=dict(status='PASS',experiment_id='Y11-SENS-IMGSZ1600-001',selected_epoch=54,
        selected_checkpoint=str(CHECKPOINT),selected_checkpoint_sha256=sha(CHECKPOINT),
        python_version=sys.version.split()[0],versions=versions,source_commit=COMMIT,
        test_images=23,test_boxes=36,broken=8,loose=28,image_sha256=image_hashes,label_sha256=label_hashes,
        dataset_yaml_sha256=sha(DATA_YAML),test_annotation_sha256=sha(COCO_TEST),source_inventory_sha256=sha(INVENTORY),
        training_args_sha256=sha(RUN/'args.yaml'),training_results_sha256=sha(RUN/'results.csv'),
        evaluation_config=config,queue_manifest_sha256=sha(QUEUE),prior_test_results_already_seen=True,
        model_created=False,model_inference_performed=False,training_started=False,
        checkpoint_metadata_unpickled=False,
        precision_note='FP16 quantize=16 and batch4/workers0 match CUDA AMP training-end validator; YOLO1280 standalone test used default FP32, disclose this difference.',
        native_ap_note='save_json=False keeps original Ultralytics AP; separate full-precision export never invokes coco_evaluate')
    print('STEP18_PREPARATION_PASS: YOLO11s1600 selected_epoch=54 test_images=23 boxes=36',flush=True)
    return config,truth,metadata


def evaluate(config,truth,metadata,destination):
    import numpy as np
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.metrics import smooth
    from readonly_dataset import install_write_guard,redirect_label_caches
    from step18_yolo1600_evidence import make_validator,draw_saved_predictions
    install_write_guard();redirect_label_caches()
    # User-run branch only. Native checkpoint loader constructs the entire saved detector.
    best=YOLO(str(CHECKPOINT),task='detect')
    require(best.names=={0:'broken',1:'loose'} and best.model.model[-1].nc==2,'Loaded detector is not two-class YOLO11')
    save_json(destination/'checkpoint_load_record.json',dict(status='PASS',checkpoint=str(CHECKPOINT),sha256=sha(CHECKPOINT),
        names=best.names,num_classes=2,state_dict_items=len(best.model.state_dict()),selected_epoch=54,
        load_method='native YOLO complete serialized detector load; no partial pretrained transfer',training_started=False))
    print('SELECTED_CHECKPOINT_NATIVE_LOAD_PASS: YOLO11s classes=broken,loose selected_epoch=54',flush=True)
    image_ids={Path(image['file_name']).name:image['id'] for image in truth['images']}
    Validator=make_validator(destination,image_ids,save_json,require)
    observed={}
    def on_start(validator):
        ds=validator.dataloader.dataset
        require(len(ds)==23 and all(Path(path).resolve().is_relative_to(DATA/'images/test') for path in ds.im_files),'Native test dataset includes wrong split')
        require(validator.args.quantize==16 and validator.args.batch==4 and validator.args.imgsz==1600 and
            validator.args.conf==.001 and validator.args.iou==.7 and validator.args.max_det==300,'Native runtime evaluation settings drifted')
        observed['validator']=validator
    best.reset_callbacks();best.add_callback('on_val_start',on_start)
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    metrics=best.val(validator=Validator,**config,data=str(DATA_YAML),project=str(TEST_ROOT),name=destination.name,exist_ok=True)
    elapsed=time.perf_counter()-started
    validator=observed['validator'];rows=validator.exported_rows
    require(len(rows)==23 and len({r['image_id'] for r in rows})==23 and set(r['image_id'] for r in rows)==set(image_ids.values()),'Exported test image records incomplete')
    raw={key:float(value) for key,value in metrics.results_dict.items()}
    require(all(math.isfinite(value) for value in raw.values()),'Native test metrics not finite')
    overall={name:raw[key] for name,key in [('precision','metrics/precision(B)'),('recall','metrics/recall(B)'),('map50','metrics/mAP50(B)'),('map50_95','metrics/mAP50-95(B)')]}
    classes={}
    for index,cls in enumerate(metrics.ap_class_index.tolist()):
        p,r,ap50,ap=metrics.class_result(index)
        classes[('broken','loose')[cls]]=dict(precision=float(p),recall=float(r),map50=float(ap50),map50_95=float(ap))
    require(set(classes)=={'broken','loose'},'Native category metrics missing')
    threshold_index=int(smooth(metrics.box.f1_curve.mean(0),.1).argmax())
    threshold=float(metrics.box.px[threshold_index])
    require(np.allclose(metrics.box.p,metrics.box.p_curve[:,threshold_index]) and np.allclose(metrics.box.r,metrics.box.r_curve[:,threshold_index]),'Native P/R threshold reconstruction differs')
    native=dict(overall=overall,per_class=classes,results_dict=raw,precision_recall_threshold=threshold,
        precision_recall_rule='IoU0.50; class-mean Precision/Recall at maximum smoothed mean-F1 over the native1000-point confidence curve; reporting rule only, not a new inference threshold',
        evaluator='Ultralytics8.4.152 native ap_per_class/compute_ap,101-point interpolation trapezoid; no COCO replacement',coco_ar100=None)
    save_json(destination/'native_metrics.json',native)
    predictions=[]
    for row in rows:
        for box,label,score in zip(row['boxes_xyxy'],row['model_labels'],row['scores']):
            x1,y1,x2,y2=box
            predictions.append(dict(image_id=row['image_id'],category_id=label+1,bbox=[x1,y1,x2-x1,y2-y1],score=score))
    save_json(destination/'predictions.bbox.json',predictions)
    draw_saved_predictions(destination,rows)
    return dict(status='PASS',experiment_id='Y11-SENS-IMGSZ1600-001',selected_epoch=54,split='original test',
        images=23,boxes=36,broken=8,loose=28,native_metrics=native,selected_checkpoint=str(CHECKPOINT),selected_checkpoint_sha256=sha(CHECKPOINT),
        returned_predictions=len(predictions),visible_predictions_above_0_3=sum(row['visible_count'] for row in rows),
        evaluation_config=config,precision='FP16 quantize=16',visualization_score_threshold=.30,
        native_test_seconds_including_model_backend_and_plots=elapsed,peak_cuda_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
        training_started=False,test_used_for_epoch_selection=False,prior_test_results_already_seen=True,
        prediction_export_changes_native_metric=False,registration_status='PENDING_RESULT_REVIEW')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check-preparation',action='store_true')
    parser.add_argument('--log-dir',type=Path)
    args=parser.parse_args();configure()
    if args.check_preparation:
        config,truth,metadata=check_preparation()
        audit=ROOT/'workspace/audit/step18_preflight';audit.mkdir(parents=True,exist_ok=True)
        save_json(audit/'preparation_check.json',metadata)
        print('STEP18_CPU_PREPARATION_PASS; no model created, no checkpoint unpickle, no test inference, no training',flush=True)
        return 0
    require(args.log_dir is not None,'--log-dir required')
    logs=args.log_dir.resolve()
    require(logs.is_relative_to(ROOT/'workspace/logs') and logs.name.startswith('step18_'),'Unexpected log directory')
    logs.mkdir(parents=True,exist_ok=True)
    destination=TEST_ROOT/logs.name
    report=dict(status='FAIL',started_utc=datetime.now(timezone.utc).isoformat(),log_dir=str(logs),result_dir=str(destination),training_started=False,test_inference_started=False)
    stdout,stderr=sys.stdout,sys.stderr
    with (logs/'step18_console.txt').open('x',encoding='utf-8') as handle:
        sys.stdout,sys.stderr=Tee(stdout,handle),Tee(stderr,handle)
        try:
            config,truth,metadata=check_preparation()
            require(not TEST_ROOT.exists(),'A prior test attempt exists; inspect before retry, no overwrite')
            TEST_ROOT.mkdir(exist_ok=False);destination.mkdir();(destination/'code').mkdir()
            inputs=[Path(__file__),SCRIPTS/'step18_yolo1600_evidence.py',SCRIPTS/'readonly_dataset.py',SCRIPTS/'run_step18_test_yolo1600.cmd',
                CONFIG,DATA_YAML,QUEUE,INVENTORY,COCO_TEST,RUN/'args.yaml',RUN/'results.csv',ROOT/'workspace/notes/STEP18_yolo1600_final_test.txt']
            metadata['saved_inputs_sha256']={str(path):sha(path) for path in inputs}
            metadata['command_line']=subprocess.list2cmdline([sys.executable,'-I','-X','utf8',str(Path(__file__).resolve()),'--log-dir',str(logs)])
            metadata['training_run_id']=RUN.name;metadata['evaluation_run_id']=logs.name
            metadata['archived_inputs']={str(path):str(destination/'code'/f'{index:02d}_{path.name}') for index,path in enumerate(inputs)}
            for index,path in enumerate(inputs):shutil.copy2(path,destination/'code'/f'{index:02d}_{path.name}')
            save_json(destination/'pretest_record.json',metadata)
            print('RESULT_DIR='+str(destination),flush=True)
            report['test_inference_started']=True;save_json(logs/'step18_manifest.json',report)
            summary=evaluate(config,truth,metadata,destination)
            require(sha(CHECKPOINT)==metadata['selected_checkpoint_sha256'],'Checkpoint changed during test')
            for path,digest in metadata['image_sha256'].items():require(sha(path)==digest,'Original test image changed')
            for path,digest in metadata['label_sha256'].items():require(sha(path)==digest,'Original test label changed')
            save_json(destination/'test_summary.json',summary)
            files=[destination/name for name in ('test_summary.json','native_metrics.json','native_match_stats.npz','predictions.bbox.json','predictions_details.json',
                'pretest_record.json','actual_evaluation_args.json','checkpoint_load_record.json')]+sorted((destination/'predicted_images').glob('*.jpg'))
            save_json(destination/'test_evidence_manifest.json',dict(status='PASS',files_sha256={str(path):sha(path) for path in files},checkpoint_sha256=sha(CHECKPOINT),native_metric_preserved=True,training_started=False))
            report.update(status='PASS',summary=str(destination/'test_summary.json'),registration_status='PENDING_RESULT_REVIEW')
            overall=summary['native_metrics']['overall']
            print(f'TEST_mAP50={overall["map50"]:.6f} TEST_mAP50_95={overall["map50_95"]:.6f} TEST_PRECISION={overall["precision"]:.6f} TEST_RECALL={overall["recall"]:.6f}',flush=True)
            print('RETURNED_PREDICTIONS='+str(summary['returned_predictions']),flush=True)
            print('VISIBLE_PREDICTIONS_ABOVE_0.3='+str(summary['visible_predictions_above_0_3']),flush=True)
            print('STEP18_TEST_EVAL_PASS',flush=True)
            print('STEP18_RAW_RESULTS_SAVED; send output for result review/registration. Do not rerun test.',flush=True)
        except BaseException:
            report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
            print('STEP18_FAIL; send complete output and LOG_DIR. No training started.',flush=True)
        finally:
            save_json(logs/'step18_manifest.json',report);print('RESULT_DIR='+str(destination),flush=True)
            sys.stdout,sys.stderr=stdout,stderr
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
