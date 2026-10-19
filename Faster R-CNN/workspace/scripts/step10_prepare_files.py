"""Derive MERGE130 test files from the already verified STEP08 test implementation.

Administrative file preparation only; never loads a model or starts inference.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'workspace/scripts'
CONFIGS = ROOT / 'workspace/configs'
script = (SCRIPTS / 'step08_test_val_baseline.py').read_text(encoding='utf-8')
replacements = [
    ('User-run test of the fixed VAL-001 best checkpoint.', 'User-run test of the fixed MERGE130 last epoch100 checkpoint.'),
    ('run_step08_test_val_baseline.cmd', 'run_step10_test_merge130.cmd'),
    ('frcnn_r50fpn_val001_test.py', 'frcnn_r50fpn_merge130_test.py'),
    ('FRCNN-R50FPN-VAL-001/step07_235_27043', 'FRCNN-R50FPN-MERGE130-001/step09_109_20338'),
    ('test_selected_epoch66', 'test_selected_epoch100'),
    ('best_coco_bbox_mAP_epoch_66.pth', 'epoch_100.pth'),
    ('best_checkpoint_sha256', 'checkpoint_sha256'),
    ('best_checkpoint', 'checkpoint'),
    ("data['meta']['epoch'] == 66 and data['meta']['iter'] == 66 * 54", "data['meta']['epoch'] == 100 and data['meta']['iter'] == 6500"),
    ("require(selected['selected_epoch'] == 66 and selected['criterion'] == 'coco/bbox_mAP', 'Validation selection changed')",
     "require(selected['selected_epoch'] == 100 and selected['selection_rule'] == 'fixed last epoch100; no val/test-based selection' and not selected['independent_validation'], 'Fixed last-epoch selection changed')\n    review = json.loads((ROOT / 'workspace/audit/step09_review/acceptance.json').read_text(encoding='utf-8'))\n    require(review['status'] == 'PASS' and review['selected_checkpoint_sha256'] == selected['checkpoint_sha256'] and not review['test_evaluated'], 'Merged training review/checkpoint differs')"),
    ('resolved_val_001.py', 'resolved_merge130_001.py'),
    ('after validation selection', 'after fixed last-epoch selection'),
    ('selected_epoch=66', 'selected_epoch=100'),
    ('FRCNN-R50FPN-VAL-001', 'FRCNN-R50FPN-MERGE130-001'),
    ('original independent test23; fixed checkpoint selected only by original validation',
     'original independent test23; merged train130; fixed last epoch100; no independent val or test-based checkpoint selection'),
    ('deterministic runtime used for validation in STEP07', 'deterministic runtime and prediction settings fixed before merged training'),
    ('STEP08', 'STEP10'),
    ('step08_', 'step10_'),
]
for old, new in replacements:
    assert old in script, 'Verified template changed: '+old
    script = script.replace(old, new)
assert 'selected_epoch=66' not in script and 'best_coco_bbox_mAP' not in script and 'original validation' not in script
script = script.replace("training_started=False, test_inference_performed=False)",
    "training_images=130, independent_validation=False, selection_rule=selected['selection_rule'],\n                    checkpoint_choice_fixed_before_training=True, training_started=False, test_inference_performed=False)")
script = script.replace("selected_epoch=100, images=23, boxes=36, broken=8, loose=28,",
    "selected_epoch=100, training_images=130, independent_validation=False, images=23, boxes=36, broken=8, loose=28,")
script = script.replace("ROOT / 'workspace/notes/EXPERIMENT_RECORDING_RULES.md']",
    "ROOT / 'workspace/notes/EXPERIMENT_RECORDING_RULES.md',\n                     ROOT / 'workspace/notes/STEP10_merge130_final_test.txt',\n                     ROOT / 'workspace/audit/step09_review/acceptance.json']")
destination = SCRIPTS / 'step10_test_merge130.py'
assert not destination.exists(), 'Derived script exists; edit it in place instead of regenerating'
destination.write_text(script, encoding='utf-8')

config = (CONFIGS / 'frcnn_r50fpn_val001_test.py').read_text(encoding='utf-8')
config = config.replace('# Test the checkpoint already selected by original val mAP50-95.',
    '# Test only the last epoch100, fixed before MERGE130 training; no independent val.')
config = config.replace('FRCNN-R50FPN-VAL-001/step07_235_27043', 'FRCNN-R50FPN-MERGE130-001/step09_109_20338')
config = config.replace('resolved_val_001.py', 'resolved_merge130_001.py')
config = config.replace('best_coco_bbox_mAP_epoch_66.pth', 'epoch_100.pth')
cfg_file = CONFIGS / 'frcnn_r50fpn_merge130_test.py'
assert not cfg_file.exists()
cfg_file.write_text(config, encoding='utf-8')

cmd = (SCRIPTS / 'run_step08_test_val_baseline.cmd').read_text(encoding='utf-8')
cmd = cmd.replace('STEP08_VAL_001_SELECTED_CHECKPOINT_TEST', 'STEP10_MERGE130_LAST_EPOCH100_TEST')
cmd = cmd.replace('step08_test_val_baseline.py', 'step10_test_merge130.py').replace('step08_', 'step10_').replace('STEP08_', 'STEP10_')
cmd_file = SCRIPTS / 'run_step10_test_merge130.cmd'
assert not cmd_file.exists()
cmd_file.write_text(cmd, encoding='utf-8')
print('STEP10_FILES_PREPARED; template reused; no model, inference or training')
