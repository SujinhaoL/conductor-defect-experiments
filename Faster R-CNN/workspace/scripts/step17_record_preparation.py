"""Record checked STEP17 handoff and completion queue; no model imports."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT/'workspace/audit/step17_preflight'
read = lambda path:json.loads(path.read_text(encoding='utf-8'))
save = lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
checks = [AUDIT/'preparation_check.json',AUDIT/'selected_checkpoint_metadata.json',AUDIT/'metric_handoff_check.json']
assert all(read(path)['status']=='PASS' for path in checks)
assert read(checks[1])['epoch']==12 and read(checks[1])['iterations']==648
assert not read(checks[0])['model_created'] and not read(checks[2])['test_inference_performed']
files = checks+[ROOT/'workspace/scripts/step17_test_scale1600.py',ROOT/'workspace/scripts/run_step17_test_scale1600.cmd',
    ROOT/'workspace/configs/frcnn_r50fpn_scale1600_test.py',ROOT/'workspace/notes/STEP17_scale1600_final_test.txt']
ready = dict(status='PREPARED_CPU_CHECK_PASS_NOT_TESTED',selected_epoch=12,anchor_scales=[8],resize=[2666,1600],
    files_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    model_created=False,model_inference_performed=False,training_started=False)
save(AUDIT/'handoff_ready.json',ready)
progress_path = ROOT.parent/'baseline_analysis/audit/test_completion_review_20261014/test_completion_progress.json'
progress = read(progress_path)
assert {row['order'] for row in progress['completed']}=={1,2}
assert all(row['status']=='COMPLETED_REVIEWED_REGISTERED' for row in progress['completed'])
progress.update(updated_utc=datetime.now(timezone.utc).isoformat(),next_order=3,
    next_preparation=dict(order=3,epoch=12,status='PREPARED_CPU_CHECK_PASS_NOT_TESTED',
        experiment_id='FRCNN-R50FPN-SCALE1600-VAL-001',command=str(ROOT/'workspace/scripts/run_step17_test_scale1600.cmd'),
        evidence=str(AUDIT/'handoff_ready.json')))
save(progress_path,progress)
print('STEP17_HANDOFF_READY; STEP15/16 completed/reviewed/registered; STEP17 not tested; no inference/training')
