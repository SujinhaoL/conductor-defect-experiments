"""Record the checked, user-run-only handoff and current completion queue. No model imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT/'workspace/audit/step16_preflight'
read = lambda path: json.loads(path.read_text(encoding='utf-8'))
checks = [AUDIT/'preparation_check.json', AUDIT/'selected_checkpoint_metadata.json', AUDIT/'metric_handoff_check.json']
assert all(read(path)['status'] == 'PASS' for path in checks)
assert read(checks[1])['epoch'] == 15 and read(checks[1])['iterations'] == 810
assert not read(checks[0])['model_created'] and not read(checks[2])['test_inference_performed']
files = checks + [ROOT/'workspace/scripts/step16_test_anchor4.py', ROOT/'workspace/scripts/run_step16_test_anchor4.cmd',
    ROOT/'workspace/configs/frcnn_r50fpn_anchor4_test.py', ROOT/'workspace/notes/STEP16_anchor4_final_test.txt']
save = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
ready = dict(status='PREPARED_CPU_CHECK_PASS_NOT_TESTED', selected_epoch=15, anchor_scales=[4], resize=[2133,1280],
    files_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    model_created=False, model_inference_performed=False, training_started=False)
save(AUDIT/'handoff_ready.json', ready)
progress_path = ROOT.parent/'baseline_analysis/audit/test_completion_review_20261014/test_completion_progress.json'
progress = read(progress_path)
assert progress['completed'][0]['status'] == 'COMPLETED_REVIEWED_REGISTERED'
progress.update(updated_utc=datetime.now(timezone.utc).isoformat(), next_order=2,
    next_preparation=dict(order=2, epoch=15, status='PREPARED_CPU_CHECK_PASS_NOT_TESTED',
        experiment_id='FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001',
        command=str(ROOT/'workspace/scripts/run_step16_test_anchor4.cmd'), evidence=str(AUDIT/'handoff_ready.json')))
save(progress_path, progress)
print('STEP16_HANDOFF_READY; STEP15 completed/reviewed/registered; STEP16 not tested; no model inference/training')
