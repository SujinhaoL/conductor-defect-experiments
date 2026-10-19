"""Prepare fixed STEP15 files from the already reviewed STEP08 evaluator; no model execution."""
import json
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[2]
S = ROOT / 'workspace/scripts'
C = ROOT / 'workspace/configs'
replacements = [
    ('FRCNN-R50FPN-VAL-001', 'FRCNN-R50FPN-SCALE1280-VAL-001'),
    ('step07_235_27043', 'step11_31112_14115'),
    ('frcnn_r50fpn_val001_test.py', 'frcnn_r50fpn_scale1280_test.py'),
    ('resolved_val_001.py', 'resolved_scale1280_val_001.py'),
    ('step08', 'step15'), ('STEP08', 'STEP15'),
    ('test_val_baseline', 'test_scale1280'),
    ('epoch66', 'epoch5'), ('epoch_66', 'epoch_5'),
    ("== 66", "== 5"), ('66 * 54', '5 * 54'), ('selected_epoch=66', 'selected_epoch=5'),
    ('(1333, 800)', '(2133, 1280)'), ('[1333, 800]', '[2133, 1280]'),
]
def adapt(text):
    for old,new in replacements:
        text=text.replace(old,new)
    return text

script=adapt((S/'step08_test_val_baseline.py').read_text(encoding='utf-8'))
# STEP15 results are reviewed before the records importer is extended/refreshed.
# A completed evaluation must never be repeated to repair the workbook.
start=script.index('            # Registration reads saved results only.')
end=script.index('        except BaseException:',start)
script=script[:start]+'''            report['registration_status'] = 'PENDING_RESULT_REVIEW'
            print('STEP15_RAW_RESULTS_SAVED; send output for review and CSV/Excel registration. Do not rerun test.', flush=True)
'''+script[end:]
script=script.replace("return (0 if report.get('registration_status') == 'PASS_AWAITING_RESULT_REVIEW' else 2) if report['status'] == 'PASS' else 1", "return 0 if report['status'] == 'PASS' else 1")
script=script.replace('VAL-001 best checkpoint', 'SCALE1280-VAL-001 epoch5 checkpoint')
script=script.replace("RECORDS = ROOT.parent / 'baseline_analysis/scripts/refresh_experiment_records.py'\nWORKBOOK_BUILDER = RECORDS.with_name('build_experiment_workbook.mjs')", "REVIEW = ROOT / 'workspace/audit/step11_review/acceptance.json'\nQUEUE = ROOT.parent / 'baseline_analysis/audit/test_completion_review_20261014/fixed_test_queue.json'")
script=script.replace('Path(cfg.test_evaluator.ann_file), RECORDS, WORKBOOK_BUILDER,', 'Path(cfg.test_evaluator.ann_file), REVIEW, QUEUE,\n                     ROOT / \'workspace/notes/STEP15_scale1280_final_test.txt\',')
script=script.replace("metadata['registration_script'] = str(RECORDS)", "metadata['registration_status'] = 'PENDING_RESULT_REVIEW'")
script=script.replace('        load_model(cfg, selected, audit)  # CPU load only; no detector forward.', '''        data, allowed = load_verified_checkpoint(selected)
        save_json(audit / 'selected_checkpoint_metadata.json', dict(status='PASS', selected_epoch=5,
            checkpoint_sha256=selected['best_checkpoint_sha256'], state_dict_items=len(data['state_dict']),
            epoch=data['meta']['epoch'], iterations=data['meta']['iter'], scoped_metadata_types=allowed,
            model_created=False, detector_forward_performed=False))
        del data''')
script=script.replace('STEP15_CPU_LOAD_CHECK_PASS; no detector forward, no test inference, no training', 'STEP15_CPU_METADATA_CHECK_PASS; no model created, no detector forward, no test inference, no training')
script=script.replace("            require(not list(TEST_ROOT.glob('*/test_summary.json')), 'This selected checkpoint already has a completed test; inspect existing results first')", "            require(not TEST_ROOT.exists(), 'A previous test attempt exists; inspect it before any retry')\n            TEST_ROOT.mkdir(exist_ok=False)")
script=script.replace("with (logs / 'step15_console.txt').open('w', encoding='utf-8')", "with (logs / 'step15_console.txt').open('x', encoding='utf-8')")
script=script.replace("            report['test_inference_started'] = True", "            report['test_inference_started'] = True\n            save_json(logs / 'step15_manifest.json', report)")
script=script.replace("            summary.update(selected_checkpoint=", "            require(sha(selected['best_checkpoint']) == selected['best_checkpoint_sha256'], 'Checkpoint file changed during test')\n            summary.update(selected_checkpoint=")
script=script.replace("                model_score_threshold=0.001", "                resize_scale=[2133, 1280], anchor_scales=[8],\n                model_score_threshold=0.001")
script=script.replace("                    training_started=False, test_inference_performed=False)", "                    training_started=False, test_inference_performed=False, model_created=False,\n                    prior_test_results_already_seen=True, queue_manifest_sha256=sha(QUEUE),\n                    training_config_sha256=sha(RUN / 'resolved_scale1280_val_001.py'))")
for target,content in [
    (S/'step15_test_scale1280.py',script),
    (S/'run_step15_test_scale1280.cmd',adapt((S/'run_step08_test_val_baseline.cmd').read_text(encoding='utf-8')).replace('VAL_001_SELECTED_CHECKPOINT_TEST','SCALE1280_EPOCH5_USER_TEST')),
    (C/'frcnn_r50fpn_scale1280_test.py',adapt((C/'frcnn_r50fpn_val001_test.py').read_text(encoding='utf-8'))),
]:
    with target.open('x',encoding='utf-8',newline='\n') as f:
        f.write(content)
    print('PREPARED='+str(target))

specs=[
    ('FRCNN-R50FPN-SCALE1280-VAL-001','step11_31112_14115',5),
    ('FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001','step13_27622_23378',15),
    ('FRCNN-R50FPN-SCALE1600-VAL-001','step14_3575_9657',12),
]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
items=[]
for exp,rid,ep in specs:
    selection_path=ROOT/'workspace/runs'/exp/rid/'selected_checkpoint.json'
    selected=json.loads(selection_path.read_text(encoding='utf-8'))
    cp=Path(selected['best_checkpoint'])
    assert selected['selected_epoch']==ep and sha(cp)==selected['best_checkpoint_sha256']
    items.append(dict(order=len(items)+1,experiment_id=exp,training_run_id=rid,epoch=ep,
        checkpoint=str(cp),checkpoint_sha256=sha(cp),selection_record=str(selection_path),
        selection_record_sha256=sha(selection_path),status_at_freeze='TEST_PENDING'))
for project,exp,ep,expected in [
    ('yolo11','Y11-SENS-IMGSZ1600-001',54,'2cc998a09da3eb36d717df79dfa28adf2a41cebb1121ecd43d984ff32ea4c92e'),
    ('RT-DETR','RTD-SENS-IMGSZ1600-001',42,'29d23c8bb93bd650800ea439c9f43755b81f1763d0550ce4e14119dee4121a13')]:
    cp=ROOT.parent/project/'workspace/runs'/exp/'weights/best.pt'
    assert sha(cp)==expected
    items.append(dict(order=len(items)+1,experiment_id=exp,training_run_id=exp,epoch=ep,
        checkpoint=str(cp),checkpoint_sha256=expected,status_at_freeze='TEST_PENDING'))
queue=ROOT.parent/'baseline_analysis/audit/test_completion_review_20261014/fixed_test_queue.json'
with queue.open('x',encoding='utf-8') as f:
    json.dump(dict(purpose='User requested final comparison test completion in this fixed order',
        prior_test_results_already_seen=True,selection_not_changed_using_new_test=True,
        only_first_entry_prepared=True,no_training=True,assistant_inference_performed=False,
        items=items),f,ensure_ascii=False,indent=2)
print('FIVE_CHECKPOINTS_FROZEN='+str(queue))
