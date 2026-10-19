"""Derive a user-run launcher from the checked STEP07 implementation. No training."""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
source = (SCRIPTS / 'step07_run_val_baseline.py').read_text(encoding='utf-8')
replacements = [
    ('Formal VAL-001 launcher.', 'Resolution sensitivity launcher; original train107/val23.'),
    ('frcnn_r50fpn_val_001.py', 'frcnn_r50fpn_scale1280_val_001.py'),
    ('FRCNN-R50FPN-VAL-001', 'FRCNN-R50FPN-SCALE1280-VAL-001'),
    ('resolved_val_001.py', 'resolved_scale1280_val_001.py'),
    ('step07', 'step11'), ('STEP07', 'STEP11'),
    ('(1333, 800)', '(2133, 1280)'), ('resize_scale=[1333, 800]', 'resize_scale=[2133, 1280]'),
    ('run_step11_train_val_baseline.cmd', 'run_step11_train_scale1280.cmd'),
    ('STEP11_formal_val_baseline.txt', 'STEP11_resolution1280_plan.md'),
    ('TRAINING_COMPLETED_TEST_PENDING', 'TRAINING_COMPLETED_VAL_REVIEW_PENDING'),
    ('SELECTED_TEST_PENDING', 'SELECTED_VAL_REVIEW_PENDING'),
    ('Starting user-run VAL-001: original train107/val23, fresh official COCO initialization, test reserved.',
     'Starting user-run SCALE1280: original train107/val23, Resize(2133,1280), fresh COCO; test excluded.'),
    ("merged_stage_frozen_settings='Keep the recorded model, optimizer, batch, augmentation, schedule and 100 epochs. Fresh same COCO initialization, no validation, last epoch selected. Actual iteration count will be recorded for 130 images.'",
     "comparison_scope='Resolution sensitivity versus frozen VAL-001: same train107/val23, batch/steps/optimizer/seed; only resize changed; no test-driven tuning.'"),
]
for old, new in replacements:
    assert old in source, old
    source = source.replace(old, new)
guard = '''    # Verify all inherited configuration values against the frozen baseline.
    baseline_path = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py'
    baseline = Config.fromfile(str(baseline_path))
    import copy
    normalized = copy.deepcopy(cfg.to_dict())
    reference = copy.deepcopy(baseline.to_dict())
    for key in ('experiment_id', 'snapshot_scale', 'work_dir'):
        normalized.pop(key, None)
        reference.pop(key, None)
    pipelines = [normalized['train_pipeline'], normalized['test_pipeline'],
                 normalized['train_dataloader']['dataset']['pipeline'], normalized['val_dataloader']['dataset']['pipeline']]
    for pipeline in pipelines:
        resize = [item for item in pipeline if item['type'] == 'Resize']
        require(len(resize) == 1 and tuple(resize[0]['scale']) == (2133, 1280) and resize[0]['keep_ratio'], 'All actual pipelines must use the new scale')
        resize[0]['scale'] = (1333, 800)
    require(normalized == reference, 'A setting besides resize/record labels differs from the frozen baseline')
    require(tuple(cfg.snapshot_scale) == (2133, 1280), 'Snapshot resize differs from train/val')
    print('SCALE_ONLY_CONFIG_DIFF_PASS: train/val/snapshots=(2133,1280); all other scientific settings match frozen VAL-001', flush=True)
'''
anchor = "    provenance = json.loads("
assert anchor in source
source = source.replace(anchor, guard + anchor, 1)
source = source.replace("record['saved_inputs_sha256'] = {str(path): sha(path) for path in inputs}",
    "record['baseline_config_sha256'] = sha(ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py')\n"
    "        record['launch_cmd'] = 'call \\\"' + str(SCRIPTS / 'run_step11_train_scale1280.cmd') + '\\\"'\n"
    "        record['command_line'] = subprocess.list2cmdline([sys.executable, '-I', '-X', 'utf8', str(Path(__file__)), '--log-dir', str(logs)])\n"
    "        record['saved_inputs_sha256'] = {str(path): sha(path) for path in inputs}")
source = source.replace("SCRIPTS / 'run_step11_train_scale1280.cmd', ROOT / 'workspace/configs/frcnn_debug8_selection.json',",
    "SCRIPTS / 'run_step11_train_scale1280.cmd', ROOT / 'workspace/configs/frcnn_debug8_selection.json',\n"
    "                  ROOT.parent / 'AGENTS.md', ROOT / 'workspace/audit/step11_resolution/resolution_audit.json',")
target = SCRIPTS / 'step11_run_scale1280.py'
assert not target.exists(), 'Do not replace an existing launcher'
compile(source, str(target), 'exec')
target.write_text(source, encoding='utf-8')
batch = (SCRIPTS / 'run_step07_train_val_baseline.cmd').read_text(encoding='utf-8')
batch = batch.replace('step07_run_val_baseline.py', 'step11_run_scale1280.py').replace('step07_', 'step11_').replace('STEP07', 'STEP11').replace('FRCNN_R50FPN_VAL_001', 'FRCNN_R50FPN_SCALE1280_VAL_001')
cmd = SCRIPTS / 'run_step11_train_scale1280.cmd'
assert not cmd.exists()
cmd.write_text(batch, encoding='utf-8', newline='\r\n')
print('STEP11_LAUNCHER_PREPARED; no training was started')
