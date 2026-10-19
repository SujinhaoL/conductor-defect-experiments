"""One-time preparation acceptance; do not rerun after the user starts STEP14."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT.parent / 'baseline_analysis'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior = read(BASELINE / 'audit/registration_acceptance.json')
    files = {
        'csv_sha256': BASELINE / '10_all_experiments_metrics.csv',
        'excel_sha256': BASELINE / 'outputs/模型实验结果总表.xlsx',
        'report_sha256': BASELINE / '11_all_experiments_summary_for_teacher.md',
    }
    assert all(sha(path) == prior[key] for key, path in files.items())
    assert not (ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1600-VAL-001').exists()
    assert not list((ROOT / 'workspace/logs').glob('step14_*'))
    out = ROOT / 'workspace/audit/step14_preflight'
    prep = read(out / 'preparation_check.json')
    assert prep['status'] == 'PASS'
    assert not prep['training_started'] and not prep['detector_forward_performed']
    assert not prep['preflight_cuda_context_created']
    new_files = [ROOT.parent / 'AGENTS.md',
        ROOT / 'workspace/configs/frcnn_r50fpn_scale1600_val_001.py',
        ROOT / 'workspace/scripts/step14_run_scale1600.py',
        ROOT / 'workspace/scripts/run_step14_train_scale1600.cmd',
        ROOT / 'workspace/scripts/step14_resolution_audit.py',
        ROOT / 'workspace/notes/STEP14_resolution1600_plan.md',
        ROOT / 'workspace/audit/step14_resolution/resolution_audit.json',
        ROOT / 'workspace/audit/step14_resolution/train8_resize_1280_vs_1600.png',
        out / 'preparation_check.json', Path(__file__)]
    result = dict(status='PASS', training_started=False, detector_forward_performed=False,
        existing_evaluation_rows=prior['evaluation_rows'], prior_result_files_unchanged=True,
        resize_coordinates_and_preview_reviewed=True,
        files={str(p): sha(p) for p in list(files.values())+new_files})
    (out / 'handoff_acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('STEP14_HANDOFF_PASS: preparation passed; no training/inference; previous 18 evaluation rows and CSV/Excel/report unchanged')


if __name__ == '__main__':
    main()
