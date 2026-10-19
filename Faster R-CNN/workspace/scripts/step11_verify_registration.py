"""Final STEP11 record reconciliation; standard library only."""
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'workspace/audit/step11_review'
BASE=ROOT.parent/'baseline_analysis'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=rows(OUT/'records_before_update/10_all_experiments_metrics.csv')
new=rows(BASE/'10_all_experiments_metrics.csv')
byid={r['evaluation_id']:r for r in new}
assert len(new)==len(byid)==16 and len(old)==14
for r in old:assert r==byid[r['evaluation_id']],r['evaluation_id']
added=[r for r in new if r['evaluation_id'] not in {s['evaluation_id'] for s in old}]
assert len(added)==2 and all(r['evaluation_split']=='val' and int(r['selected_epoch'])==5 and int(r['train_images'])==107 for r in added)
a=read(OUT/'acceptance.json');receipt=read(BASE/'audit/registration_acceptance.json')
assert a['status']=='PASS' and a['visual_review_completed']
assert receipt['status']=='PASS' and receipt['verified_exported_cells']==919
for k,p in [('csv_sha256',BASE/'10_all_experiments_metrics.csv'),('excel_sha256',BASE/'outputs/模型实验结果总表.xlsx'),('report_sha256',BASE/'11_all_experiments_summary_for_teacher.md')]:assert sha(p)==receipt[k]
exact=next(r for r in added if r['evaluation_protocol']=='pycocotools-exact-from-saved-predictions')
assert float(exact['map50'])==a['selected_metrics']['mAP50'] and float(exact['map50_95'])==a['selected_metrics']['mAP50_95']
pre=read(ROOT/'workspace/audit/step12_preflight/preparation.json')
assert pre['status']=='PASS' and not pre['detector_forward_performed'] and not pre['training_started']
assert not (ROOT/'workspace/runs/FRCNN-DIAG-RPNROI-TRAIN8').exists()
result=dict(status='PASS',old_evaluation_rows_unchanged=14,new_val_rows=2,total_unique_rows=16,
    csv_excel_cells_verified=919,step11_visual_review_completed=True,step12_prepared_not_started=True,
    training_started=False,detector_forward_performed=False,
    reviewed_paths_sha256={str(p):sha(p) for p in [ROOT.parent/'AGENTS.md',ROOT/'workspace/scripts/step12_diagnose_rpn_roi.py',ROOT/'workspace/scripts/run_step12_diagnose_rpn_roi.cmd',ROOT/'workspace/notes/STEP12_rpn_roi_diagnostic_plan.md']})
(OUT/'handoff_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('STEP11_REGISTRATION_HANDOFF_PASS: prior14 rows unchanged; added2 val rows; 919 CSV/Excel cells matched; STEP12 not started')
