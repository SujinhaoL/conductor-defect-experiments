"""Prepare the saved-prediction review from the exercised STEP11 review."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
s = (HERE / 'step11_review_results.py').read_text(encoding='utf-8')
s = s.replace('existing STEP11', 'existing STEP13')
s = s.replace("RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'", "RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001/step13_27622_23378'")
s = s.replace("BASE = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'", "BASE = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'")
s = s.replace("OUT = ROOT / 'workspace/audit/step11_review'", "OUT = ROOT / 'workspace/audit/step13_review'")
s = s.replace("logs/'step11_manifest.json'", "logs/'step13_manifest.json'")
s = s.replace("RUN/'resolved_scale1280_val_001.py'", "RUN/'resolved_scale1280_anchor4_val_001.py'")
s = s.replace("BASE/'resolved_val_001.py'", "BASE/'resolved_scale1280_val_001.py'")
start = s.index("    for k in ('experiment_id', 'snapshot_scale', 'work_dir'):")
end = s.index("    load = read(", start)
s = s[:start] + '''    for k in ('experiment_id', 'work_dir'):
        cfg.pop(k, None); ref.pop(k, None)
    assert cfg['model']['rpn_head']['anchor_generator']['scales'] == [4]
    assert ref['model']['rpn_head']['anchor_generator']['scales'] == [8]
    cfg['model']['rpn_head']['anchor_generator']['scales'] = [8]
    assert cfg == ref and cfg['test_dataloader'] is None and not cfg['resume']
    assert tuple(cfg['snapshot_scale']) == (2133,1280)
''' + s[end:]
s = s.replace("sel['selected_epoch'] == 5", "sel['selected_epoch'] == 15")
s = s.replace('selected_epoch=5,', 'selected_epoch=15,')
s = s.replace('only_resize_scientific_diff=True', 'only_anchor_scale_scientific_diff=True,anchor_scales=[4]')
s = s.replace('selected_prediction_inventory=inventory[4]', 'selected_prediction_inventory=inventory[14]')
s = s.replace("(bh,baseline_progress,'800 baseline'),(h,progress,'1280 input')", "(bh,baseline_progress,'1280 anchor8'),(h,progress,'1280 anchor4')")
s = s.replace('resolution_training_validation_comparison.png', 'anchor_training_validation_comparison.png')
s = s.replace('STEP11_RESULT_REVIEW_PASS', 'STEP13_RESULT_REVIEW_PASS')
compile(s, str(HERE / 'step13_review_results.py'), 'exec')
(HERE / 'step13_review_results.py').write_text(s, encoding='utf-8')
print('STEP13_SAVED_PREDICTION_REVIEW_PREPARED')
