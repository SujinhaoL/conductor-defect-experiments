"""Prepare a sibling of the previously exercised STEP11 launcher; no training."""
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
source = (SCRIPTS / 'step11_run_scale1280.py').read_text(encoding='utf-8')
text = source
for old, new in [
    ('Resolution sensitivity launcher', 'Anchor-size sensitivity launcher'),
    ('frcnn_r50fpn_scale1280_val_001.py', 'frcnn_r50fpn_scale1280_anchor4_val_001.py'),
    ('FRCNN-R50FPN-SCALE1280-VAL-001', 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001'),
    ('step11_run_scale1280.py', 'step13_run_anchor4.py'),
    ('run_step11_train_scale1280.cmd', 'run_step13_train_anchor4.cmd'),
    ('resolved_scale1280_val_001.py', 'resolved_scale1280_anchor4_val_001.py'),
    ('step11_', 'step13_'), ('STEP11_', 'STEP13_'),
]:
    text = text.replace(old, new)
start = text.index('    # Verify all inherited configuration values')
end = text.index('    provenance = ', start)
text = text[:start] + '''    # Freeze scientific settings against completed STEP11; only anchor size changes.
    baseline_path = ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115/resolved_scale1280_val_001.py'
    baseline = Config.fromfile(str(baseline_path))
    import copy
    normalized, reference = copy.deepcopy(cfg.to_dict()), copy.deepcopy(baseline.to_dict())
    for key in ('experiment_id', 'work_dir'):
        normalized.pop(key, None); reference.pop(key, None)
    require(normalized['model']['rpn_head']['anchor_generator']['scales'] == [4], 'Expected one anchor scale4')
    require(reference['model']['rpn_head']['anchor_generator']['scales'] == [8], 'Frozen control is not default scale8')
    normalized['model']['rpn_head']['anchor_generator']['scales'] = [8]
    require(normalized == reference, 'A scientific setting besides anchor scale differs from frozen SCALE1280')
    require(tuple(cfg.snapshot_scale) == (2133, 1280), 'Snapshot scale changed')
    from mmdet.registry import TASK_UTILS
    generator = TASK_UTILS.build(cfg.model.rpn_head.anchor_generator)
    require(generator.num_base_priors == [3] * 5, 'RPN output channel counts must stay unchanged')
    geometry_path = ROOT / 'workspace/audit/step13_anchor_geometry/anchor_geometry.json'
    geometry = json.loads(geometry_path.read_text(encoding='utf-8'))
    require(geometry['status'] == 'PASS' and geometry['source_config_sha256'] == sha(baseline_path), 'Geometry audit baseline changed')
    require(geometry['annotation_sha256'] == sha(DATA / 'annotations/instances_train.json'), 'Geometry annotation changed')
    require(geometry['code_sha256'] == sha(SCRIPTS / 'step13_anchor_geometry_audit.py'), 'Geometry audit code changed')
    for flip in (0, 1):
        require(geometry['stats'][f'scale8_flip{flip}']['zero_positive_gt'] == 33 and
                geometry['stats'][f'scale4_flip{flip}']['zero_positive_gt'] == 3, 'Review geometry before changing the experiment')
    diagnostic = json.loads((ROOT / 'workspace/audit/step12_review/acceptance.json').read_text(encoding='utf-8'))
    require(diagnostic['status'] == 'PASS' and diagnostic['four_zero_positive_anchor_match'], 'STEP12 review missing')
    print('ANCHOR_ONLY_CONFIG_DIFF_PASS: scale8 -> scale4; same Resize(2133,1280) and all other scientific settings', flush=True)
''' + text[end:]
text = text.replace("cfg.model.rpn_head.anchor_generator.scales == [8]", "cfg.model.rpn_head.anchor_generator.scales == [4]")
text = text.replace("'Default anchors/assignment changed'", "'Planned anchor scale4 / unchanged assignment required'")
text = text.replace("resize_scale=[2133, 1280], keep_ratio=True,", "resize_scale=[2133, 1280], keep_ratio=True, anchor_scales=[4], anchor_ratios=[0.5, 1.0, 2.0],\n                  anchor_strides=[4, 8, 16, 32, 64], anchor_geometry_audit=str(geometry_path),")
text = text.replace("Resolution sensitivity versus frozen VAL-001: same train107/val23, batch/steps/optimizer/seed; only resize changed; no test-driven tuning.",
                    "Anchor-size sensitivity versus frozen SCALE1280: same train107/val23, resize/batch/steps/optimizer/seed; only scale8 to scale4; no test-driven tuning.")
text = text.replace("ROOT / 'workspace/audit/step13_resolution/resolution_audit.json'", "ROOT / 'workspace/audit/step13_anchor_geometry/anchor_geometry.json'")
text = text.replace("ROOT / 'workspace/notes/STEP13_resolution1280_plan.md'", "ROOT / 'workspace/notes/STEP13_anchor4_plan.md',\n                  ROOT / 'workspace/audit/step12_review/acceptance.json', SCRIPTS / 'step13_anchor_geometry_audit.py'")
text = text.replace("ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py'", "ROOT / 'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115/resolved_scale1280_val_001.py'")
text = text.replace('Starting user-run SCALE1280:', 'Starting user-run SCALE1280 ANCHOR4:')
text = text.replace('Resize(2133,1280), fresh COCO; test excluded.', 'Resize(2133,1280), anchor scales=[4], fresh COCO; test excluded.')
assert 'step13_resolution' not in text and 'resolution1280_plan' not in text
compile(text, str(SCRIPTS / 'step13_run_anchor4.py'), 'exec')
(SCRIPTS / 'step13_run_anchor4.py').write_text(text, encoding='utf-8')
cmd = (SCRIPTS / 'run_step11_train_scale1280.cmd').read_text(encoding='utf-8')
for old, new in [('step11_', 'step13_'), ('STEP11_', 'STEP13_'),
                 ('step13_run_scale1280.py', 'step13_run_anchor4.py'),
                 ('SCALE1280_VAL_001_USER_TRAINING', 'SCALE1280_ANCHOR4_VAL_001_USER_TRAINING')]:
    cmd = cmd.replace(old, new)
(SCRIPTS / 'run_step13_train_anchor4.cmd').write_bytes(cmd.replace('\n', '\r\n').encode('utf-8'))
print('STEP13_FILES_PREPARED; training was not launched')
