"""Record checked YOLO user-run test handoff; no model/inference imports."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'workspace/audit/step18_preflight'
read=lambda path:json.loads(path.read_text(encoding='utf-8'))
save=lambda path,value:path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
checks=[AUDIT/'preparation_check.json',AUDIT/'export_api_check.json']
assert all(read(path)['status']=='PASS' for path in checks)
assert all(not read(path)['model_created'] and not read(path)['training_started'] for path in checks)
files=checks+[ROOT/'workspace/scripts/step18_test_yolo1600.py',ROOT/'workspace/scripts/step18_yolo1600_evidence.py',
    ROOT/'workspace/scripts/run_step18_test_yolo1600.cmd',ROOT/'workspace/configs/step18_yolo1600_test.yaml',
    ROOT/'workspace/notes/STEP18_yolo1600_final_test.txt']
for path in files:
    if path.suffix=='.py':ast.parse(path.read_text(encoding='utf-8'))
ready=dict(status='PREPARED_CPU_CHECK_PASS_NOT_TESTED',selected_epoch=54,experiment_id='Y11-SENS-IMGSZ1600-001',
    files_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    model_created=False,model_inference_performed=False,training_started=False)
save(AUDIT/'handoff_ready.json',ready)
progress_path=ROOT.parent/'baseline_analysis/audit/test_completion_review_20261014/test_completion_progress.json'
progress=read(progress_path)
assert {entry['order'] for entry in progress['completed']}=={1,2,3}
progress.update(updated_utc=datetime.now(timezone.utc).isoformat(),next_order=4,
    next_preparation=dict(order=4,epoch=54,status=ready['status'],experiment_id=ready['experiment_id'],
        command=str(ROOT/'workspace/scripts/run_step18_test_yolo1600.cmd'),evidence=str(AUDIT/'handoff_ready.json')))
save(progress_path,progress)
print('STEP18_HANDOFF_READY;3 Faster tests registered;YOLO1600 not tested;no model inference/training')
