"""Verify handoff facts, prepared files and baseline preservation. No model forward."""
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(r'D:\小论文\conductor_defect_dataset')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): digest.update(block)
    return digest.hexdigest()


def main():
    assert Path(sys.executable).resolve() == (ROOT/'.venv/Scripts/python.exe').resolve()
    guide=ROOT.parent/'AGENTS.md'
    content=guide.read_text(encoding='utf-8')
    assert len(content.encode('utf-8')) < 32768
    referenced=[]
    for value in re.findall(r'`(D:\\[^`]+)`', content):
        if '<' in value: continue
        path=Path(value)
        assert path.exists(), f'Missing AGENTS path: {path}'
        referenced.append(str(path))
    provenance=read(ROOT/'workspace/configs/step03_preparation.json')['checkpoint']
    assert provenance['sha256'] in content and sha(Path(provenance['path'])) == provenance['sha256']
    with (DATA/'02_yolo11_dataset/split_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        manifest=list(csv.DictReader(f))
    assert len(manifest)==153
    groups=Counter(row['split_group_id'] for row in manifest)
    assert len(groups)==138 and sum(n==1 for n in groups.values())==127
    assert sum(n>1 for n in groups.values())==11 and sum(n for n in groups.values() if n>1)==26
    group_splits=defaultdict(set)
    for row in manifest: group_splits[row['split_group_id']].add(row['split'])
    assert all(len(v)==1 for v in group_splits.values()) and all(not row['scene_id'].strip() for row in manifest)
    counts={}
    for split, expected in [('train',(107,168,39,129)),('val',(23,37,9,28)),('test',(23,36,8,28)),('trainval',(130,205,48,157))]:
        value=read(DATA/'03_coco_dataset/annotations'/f'instances_{split}.json')
        categories=Counter(a['category_id'] for a in value['annotations'])
        actual=(len(value['images']),len(value['annotations']),categories[1],categories[2])
        assert actual==expected
        counts[split]=actual
    for package, expected in [('torch','2.7.1+cu128'),('torchvision','0.22.1+cu128'),('mmcv','2.1.0'),('mmengine','0.10.7'),('mmdet','3.3.0'),('numpy','1.26.4'),('pycocotools','2.0.10')]:
        assert importlib.metadata.version(package)==expected
    baserun=ROOT/'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
    selection=read(baserun/'selected_checkpoint.json')
    assert selection['selected_epoch']==66 and sha(Path(selection['best_checkpoint']))==selection['best_checkpoint_sha256']
    mergerun=ROOT/'workspace/runs/FRCNN-R50FPN-MERGE130-001/step09_109_20338'
    merged=read(mergerun/'selected_checkpoint.json')
    assert merged['selected_epoch']==100 and sha(Path(merged['checkpoint']))==merged['checkpoint_sha256']
    assert read(ROOT/'workspace/audit/step08_review/acceptance.json')['status']=='PASS'
    assert read(ROOT/'workspace/audit/step10_review/acceptance.json')['status']=='PASS'
    preflight=read(ROOT/'workspace/audit/step11_preflight/preparation_check.json')
    assert preflight['status']=='PASS' and preflight['resize_scale']==[2133,1280]
    assert not preflight['training_started'] and not preflight['detector_forward_performed']
    assert not (ROOT/'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001').exists(), 'Inspect an existing new run before marking PREPARED_NOT_STARTED'
    files=[guide,ROOT/'workspace/configs/frcnn_r50fpn_scale1280_val_001.py',
        ROOT/'workspace/scripts/step11_run_scale1280.py',ROOT/'workspace/scripts/run_step11_train_scale1280.cmd',
        ROOT/'workspace/scripts/frcnn_formal_hooks.py',ROOT/'workspace/scripts/frcnn_debug_hooks.py',
        ROOT/'workspace/scripts/step11_resolution_audit.py',ROOT/'workspace/notes/STEP11_resolution1280_plan.md']
    for path in files:
        assert path.is_file()
        if path.suffix=='.py': compile(path.read_text(encoding='utf-8'),str(path),'exec')
    # Preserve the old raw plot; correct only its accidentally overwritten title
    # in a separate review figure using the already saved numeric loss records.
    sys.path.insert(0,str(ROOT/'workspace/scripts'))
    from frcnn_debug_hooks import plot_losses
    losses=[json.loads(line) for line in (baserun/'artifacts/losses.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(losses)==5400
    corrected_plot=ROOT/'workspace/audit/step11_resolution/baseline107_loss_curves_title_corrected.png'
    plot_losses(losses,corrected_plot,title='FRCNN-R50FPN-VAL-001 / original 5400 updates')
    report=dict(status='PASS', stage='PREPARED_NOT_STARTED', project_agents_path=str(guide),
        checked_existing_agent_paths=referenced, dataset_counts=counts,
        groups=dict(total=138,singletons=127,multi_image_groups=11,multi_image_photos=26,registered_cross_split_groups=0),
        baseline_selected_epoch=66, merge_selected_epoch=100, old_results_preserved=True,
        files_sha256={str(path):sha(path) for path in files},
        corrected_title_plot=str(corrected_plot), loss_source_sha256=sha(baserun/'artifacts/losses.jsonl'),
        training_started=False, detector_forward_performed=False, gpu_training_memory_tested=False,
        note='Scope facts checked against actual manifests/configs/versions/weight hashes. This is not proof that all model internals are error-free.')
    dest=ROOT/'workspace/audit/step11_preflight/handoff_validation.json'
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PROJECT_AGENTS_FACTS_AND_PATHS_PASS')
    print('STEP11_HANDOFF_PASS: prepared, no training or detector forward; GPU memory at new scale not yet tested')


if __name__=='__main__': main()
