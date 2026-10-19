"""Verify that STEP13 adds exactly two val records, preserving previous evidence."""
import csv
import json
from pathlib import Path
import re
import runpy

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / 'workspace/audit/step13_review'
BASE = ROOT.parent / 'baseline_analysis'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p): return list(csv.DictReader(p.open(encoding='utf-8-sig', newline='')))


def main():
    old = rows(AUDIT / 'records_before_update/10_all_experiments_metrics.csv')
    new = rows(BASE / '10_all_experiments_metrics.csv')
    assert len(old) == 16 and len(new) == 18
    by_id = {r['evaluation_id']: r for r in new}
    assert len(by_id) == 18
    for r in old:
        assert r == by_id[r['evaluation_id']], f'Old record changed: {r["evaluation_id"]}'
    extra = [r for r in new if r['evaluation_id'] not in {r['evaluation_id'] for r in old}]
    assert len(extra) == 2
    review = read(AUDIT / 'acceptance.json')
    exact = review['selected_metrics']
    for r in extra:
        assert r['experiment_id'] == 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001'
        assert r['run_id'] == 'step13_27622_23378'
        assert r['evaluation_split'] == 'val' and int(r['selected_epoch']) == 15
        assert int(r['train_images']) == 107 and int(r['eval_images']) == 23
        assert int(r['eval_broken']) == 9 and int(r['eval_loose']) == 28
        assert r['checkpoint_sha256'] == review['selected_checkpoint_sha256']
        assert r['precision'] == '' and r['recall'] == ''
        if r['evaluation_event'] == 'selected_validation_offline_exact':
            assert float(r['map50']) == exact['mAP50'] and float(r['map50_95']) == exact['mAP50_95']
            assert float(r['coco_ar100']) == exact['AR100']
    helper = runpy.run_path(str(BASE / 'scripts/refresh_experiment_records.py'))
    receipt = read(BASE / 'audit/registration_acceptance.json')
    assert receipt['status'] == 'PASS' and receipt['evaluation_rows'] == 18
    assert helper['verify_book'](read(BASE / 'audit/workbook_plan.json')) == receipt['verified_exported_cells'] == 1035
    for key, p in [('csv_sha256', BASE / '10_all_experiments_metrics.csv'), ('excel_sha256', BASE / 'outputs/模型实验结果总表.xlsx'), ('report_sha256', BASE / '11_all_experiments_summary_for_teacher.md')]:
        assert helper['sha'](p) == receipt[key]
    guide = (ROOT.parent / 'AGENTS.md').read_text(encoding='utf-8')
    assert 'STEP13已完成并验收' in guide and '当前没有准备下一次训练' in guide
    assert len(guide.encode('utf-8')) < 32768
    for value in re.findall(r'`(D:\\[^`]+)`', guide):
        if '<' not in value:
            assert Path(value).exists(), value
    index = read(BASE / 'audit/run_status_index.json')
    assert len(index['runs']) == 8 and len(index['diagnostics']) == 2
    latest = [r for r in index['runs'] if r['experiment_id'] == extra[0]['experiment_id']]
    assert len(latest) == 1 and latest[0]['status'] == 'COMPLETED_VAL_REVIEWED'
    assert latest[0]['test_status'] == 'NOT_RUN_DEVELOPMENT_VAL_ONLY'
    save = dict(status='PASS', old_records_preserved_exactly=16, new_val_records=2, new_test_records=0,
        formal_experiments=8, evaluation_records=18, verified_csv_excel_cells=1035,
        visual_review_completed=True, future_training_prepared=False, inference_or_training_started=False,
        experiment_note=str(ROOT / 'workspace/notes/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001.md'))
    (AUDIT / 'registration_handoff.json').write_text(json.dumps(save, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('STEP13_REGISTRATION_PASS: previous16 preserved; new2 val only; CSV/Excel1035 cells match; no next training prepared')


if __name__ == '__main__': main()
