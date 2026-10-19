"""Final read-only file/config handoff checks; never builds/runs a detector."""
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / 'baseline_analysis'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    assert Path(sys.executable).resolve() == (ROOT / '.venv/Scripts/python.exe').resolve()
    pre = read(ROOT / 'workspace/audit/step13_preflight/preparation_check.json')
    assert pre['status'] == 'PASS' and pre['anchor_scales'] == [4]
    assert not pre['training_started'] and not pre['detector_forward_performed']
    assert not (ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001').exists(), 'Review existing run before declaring not started'
    files = [ROOT / 'workspace/scripts/step13_run_anchor4.py', ROOT / 'workspace/scripts/run_step13_train_anchor4.cmd',
        ROOT / 'workspace/scripts/step13_anchor_geometry_audit.py', ROOT / 'workspace/configs/frcnn_r50fpn_scale1280_anchor4_val_001.py',
        ROOT / 'workspace/notes/STEP13_anchor4_plan.md', ROOT / 'workspace/notes/FRCNN-DIAG-RPNROI-TRAIN8.md',
        ROOT / 'workspace/scripts/step12_review_results.py', ROOT / 'workspace/scripts/step12_record_review.py',
        ROOT / 'workspace/audit/step12_review/acceptance.json', ROOT / 'workspace/audit/step12_review/recording_receipt.json',
        ROOT / 'workspace/audit/step13_anchor_geometry/anchor_geometry.json', ROOT.parent / 'AGENTS.md',
        BASE / 'scripts/refresh_experiment_records.py']
    for p in files:
        assert p.is_file(), p
        if p.suffix == '.py':
            ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    guide = (ROOT.parent / 'AGENTS.md').read_text(encoding='utf-8')
    assert len(guide.encode('utf-8')) < 32768
    for value in re.findall(r'`(D:\\[^`]+)`', guide):
        if '<' not in value:
            assert Path(value).exists(), value
    for value in ('step12_24867_6686', 'STEP12已完成', 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001', '尚未运行'):
        assert value in guide
    # Shared artifact hooks must match those used in the already completed control.
    frozen = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115/code'
    for name in ('frcnn_debug_hooks.py', 'frcnn_formal_hooks.py'):
        assert sha(ROOT / 'workspace/scripts' / name) == sha(frozen / name), name
    receipt = read(BASE / 'audit/registration_acceptance.json')
    for key, p in [('csv_sha256', BASE / '10_all_experiments_metrics.csv'),
                   ('excel_sha256', BASE / 'outputs/模型实验结果总表.xlsx'),
                   ('report_sha256', BASE / '11_all_experiments_summary_for_teacher.md')]:
        assert receipt[key] == sha(p), key
    assert receipt['evaluation_rows'] == 16
    index = read(BASE / 'audit/run_status_index.json')
    matches = [v for v in index['diagnostics'] if v['experiment_id'] == 'FRCNN-DIAG-RPNROI-TRAIN8']
    assert len(matches) == 1 and matches[0]['status'] == 'DIAGNOSTIC_REVIEWED'
    assert len(index['runs']) == 7
    # Confirm all files scheduled for freezing exist, without executing launch code.
    source = (ROOT / 'workspace/scripts/step13_run_anchor4.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inputs' for t in n.targets)]
    assert len(assignments) == 1
    expr = ast.Expression(assignments[0].value)
    planned = eval(compile(expr, '<archive-input-list>', 'eval'), {'ROOT': ROOT, 'SCRIPTS': ROOT / 'workspace/scripts',
        'CONFIG': ROOT / 'workspace/configs/frcnn_r50fpn_scale1280_anchor4_val_001.py',
        'DATA': Path(r'D:\小论文\conductor_defect_dataset\03_coco_dataset'), 'Path': Path,
        '__file__': str(ROOT / 'workspace/scripts/step13_run_anchor4.py')})
    assert len({p.name for p in planned}) == len(planned), 'Archive input basenames collide'
    assert all(p.is_file() for p in planned)
    result = dict(status='PASS', preparation='PREPARED_NOT_STARTED', only_scientific_change='anchor_generator.scales [8] -> [4]',
        formal_tables_unchanged=True, existing_formal_evaluation_rows=16, diagnostic_registered=True,
        shared_hooks_match_completed_control=True, archive_input_files_checked=len(planned),
        file_sha256={str(p): sha(p) for p in files}, training_started=False, detector_forward_performed=False)
    (ROOT / 'workspace/audit/step13_preflight/handoff_validation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('STEP13_HANDOFF_PASS: scripts/config/archive paths/AGENTS/diagnostic records checked; no training or model forward')


if __name__ == '__main__':
    main()
