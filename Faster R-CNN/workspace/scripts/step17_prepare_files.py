"""Prepare third fixed-checkpoint test; file preparation only, never starts inference/training."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT/'workspace/scripts'
CONFIGS = ROOT/'workspace/configs'
replacements = [
    ('frcnn_r50fpn_anchor4_test.py','frcnn_r50fpn_scale1600_test.py'),
    ('run_step16_test_anchor4.cmd','run_step17_test_scale1600.cmd'),
    ('STEP16_anchor4_final_test.txt','STEP17_scale1600_final_test.txt'),
    ('FRCNN-R50FPN-SCALE1280-ANCHOR4-VAL-001','FRCNN-R50FPN-SCALE1600-VAL-001'),
    ('SCALE1280-ANCHOR4-VAL-001 epoch15','SCALE1600-VAL-001 epoch12'),
    ('step13_27622_23378','step14_3575_9657'),
    ('step13_review','step14_review'),
    ('resolved_scale1280_anchor4_val_001.py','resolved_scale1600_val_001.py'),
    ('epoch_15.pth','epoch_12.pth'),('test_selected_epoch15','test_selected_epoch12'),
    ('2133, 1280','2666, 1600'),('STEP13','STEP14'),
    ("['items'][1]","['items'][2]"),("queued['order'] == 2","queued['order'] == 3"),
    ("data['meta']['epoch'] == 15","data['meta']['epoch'] == 12"),
    ("data['meta']['iter'] == 15 * 54","data['meta']['iter'] == 12 * 54"),
    ("queued['epoch'] == 15","queued['epoch'] == 12"),
    ("review['selected_epoch'] == 15","review['selected_epoch'] == 12"),
    ("selected['selected_epoch'] == 15","selected['selected_epoch'] == 12"),
    ('selected_epoch=15','selected_epoch=12'),
    ('anchor_generator.scales) == [4]','anchor_generator.scales) == [8]'),
    ('Expected anchor4','Expected default anchor8'),('anchor_scales=[4]','anchor_scales=[8]'),
    ('step16_','step17_'),('STEP16_','STEP17_'),
]
source = (SCRIPTS/'step16_test_anchor4.py').read_text(encoding='utf-8')
for before,after in replacements:
    assert before in source,'Missing replacement: '+before
    source = source.replace(before,after)
ast.parse(source)
assert 'step16' not in source.lower() and 'selected_epoch=15' not in source and 'anchor_scales=[4]' not in source
assert "sha(Path(selected['best_checkpoint']))" in source
target = SCRIPTS/'step17_test_scale1600.py'
assert not target.exists(),'Preparation exists; inspect before overwrite'
target.write_text(source,encoding='utf-8')
config = (CONFIGS/'frcnn_r50fpn_anchor4_test.py').read_text(encoding='utf-8')
for before,after in replacements:
    config = config.replace(before,after)
ast.parse(config)
assert 'best_coco_bbox_mAP_epoch_12.pth' in config and 'scale=(2666, 1600)' in config
target = CONFIGS/'frcnn_r50fpn_scale1600_test.py'
assert not target.exists()
target.write_text(config,encoding='utf-8')
command = (SCRIPTS/'run_step16_test_anchor4.cmd').read_text(encoding='utf-8')
command = command.replace('step16_test_anchor4.py','step17_test_scale1600.py').replace('STEP16_ANCHOR4_EPOCH15','STEP17_SCALE1600_EPOCH12')
command = command.replace('step16_','step17_').replace('STEP16_','STEP17_')
target = SCRIPTS/'run_step17_test_scale1600.cmd'
assert not target.exists()
target.write_text(command,encoding='utf-8',newline='\r\n')
handoff = (SCRIPTS/'step16_validate_metric_handoff.py').read_text(encoding='utf-8')
handoff = handoff.replace('step16_test_anchor4','step17_test_scale1600').replace('epoch15','epoch12').replace('step16_','step17_').replace('STEP16_','STEP17_')
target = SCRIPTS/'step17_validate_metric_handoff.py'
assert not target.exists()
target.write_text(handoff,encoding='utf-8')
print('STEP17_FILES_PREPARED; no model inference or training')
