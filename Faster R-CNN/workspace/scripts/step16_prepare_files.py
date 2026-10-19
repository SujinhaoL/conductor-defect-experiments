"""Prepare the second fixed-checkpoint test from the corrected STEP15 implementation; no inference."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT/'workspace/scripts'
CONFIGS = ROOT/'workspace/configs'
NOTES = ROOT/'workspace/notes'

source = (SCRIPTS/'step15_test_scale1280.py').read_text(encoding='utf-8')
replacements = [
    ('frcnn_r50fpn_scale1280_test.py', 'frcnn_r50fpn_anchor4_test.py'),
    ('run_step15_test_scale1280.cmd', 'run_step16_test_anchor4.cmd'),
    ('STEP15_scale1280_final_test.txt', 'STEP16_anchor4_final_test.txt'),
    ('FRCNN-R50FPN-SCALE1280-VAL-001', 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001'),
    ('step11_31112_14115', 'step13_27622_23378'),
    ('step11_review', 'step13_review'),
    ('resolved_scale1280_val_001.py', 'resolved_scale1280_anchor4_val_001.py'),
    ('epoch_5.pth', 'epoch_15.pth'), ('test_selected_epoch5', 'test_selected_epoch15'),
    ('epoch5', 'epoch15'), ('STEP11', 'STEP13'),
    ("['items'][0]", "['items'][1]"), ("queued['order'] == 1", "queued['order'] == 2"),
    ("data['meta']['epoch'] == 5", "data['meta']['epoch'] == 15"),
    ("data['meta']['iter'] == 5 * 54", "data['meta']['iter'] == 15 * 54"),
    ("queued['epoch'] == 5", "queued['epoch'] == 15"),
    ("review['selected_epoch'] == 5", "review['selected_epoch'] == 15"),
    ("selected['selected_epoch'] == 5", "selected['selected_epoch'] == 15"),
    ('selected_epoch=5', 'selected_epoch=15'),
    ('anchor_generator.scales) == [8]', 'anchor_generator.scales) == [4]'),
    ('Expected default anchor8', 'Expected anchor4'),
    ('anchor_scales=[8]', 'anchor_scales=[4]'),
    ('step15_', 'step16_'), ('STEP15_', 'STEP16_'),
    ("resize=[2133, 1280], keep_ratio=True,", "resize=[2133, 1280], keep_ratio=True, anchor_scales=[4],"),
]
for before, after in replacements:
    assert before in source, 'Missing replacement: ' + before
    source = source.replace(before, after)
ast.parse(source)
assert "sha(Path(selected['best_checkpoint']))" in source
assert 'step15' not in source.lower() and 'selected_epoch=5' not in source
target = SCRIPTS/'step16_test_anchor4.py'
assert not target.exists(), 'Preparation file already exists; inspect before overwrite'
target.write_text(source, encoding='utf-8')

config = (CONFIGS/'frcnn_r50fpn_scale1280_test.py').read_text(encoding='utf-8')
for before, after in replacements[:1] + replacements[3:9]:
    config = config.replace(before, after)
assert 'FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001' in config and 'best_coco_bbox_mAP_epoch_15.pth' in config
ast.parse(config)
config_target = CONFIGS/'frcnn_r50fpn_anchor4_test.py'
assert not config_target.exists()
config_target.write_text(config, encoding='utf-8')

command = (SCRIPTS/'run_step15_test_scale1280.cmd').read_text(encoding='utf-8')
command = command.replace('step15_test_scale1280.py', 'step16_test_anchor4.py').replace('step15_', 'step16_')
command = command.replace('STEP15_SCALE1280_EPOCH5', 'STEP16_ANCHOR4_EPOCH15').replace('STEP15_', 'STEP16_')
command_target = SCRIPTS/'run_step16_test_anchor4.cmd'
assert not command_target.exists()
command_target.write_text(command, encoding='utf-8', newline='\r\n')

handoff = (SCRIPTS/'step15_validate_metric_handoff.py').read_text(encoding='utf-8')
handoff = handoff.replace('step15_test_scale1280', 'step16_test_anchor4').replace('step15_', 'step16_').replace('STEP15_', 'STEP16_').replace('epoch5', 'epoch15')
handoff_target = SCRIPTS/'step16_validate_metric_handoff.py'
assert not handoff_target.exists()
handoff_target.write_text(handoff, encoding='utf-8')
print('STEP16_FILES_PREPARED; no model inference or training')
