"""Read-only acceptance of completed VAL-001 artifacts; no model forward/training."""
import json
import math
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043'
OUT = ROOT / 'workspace/audit/step07_review'
OUT.mkdir(parents=True, exist_ok=True)
helpers = runpy.run_path(str(ROOT / 'workspace/scripts/step04_convert_coco.py'))
sha, save_json = helpers['sha'], helpers['save_json']
selected = json.loads((RUN / 'selected_checkpoint.json').read_text(encoding='utf-8'))
summary = json.loads((RUN / 'artifacts/formal_training_summary.json').read_text(encoding='utf-8'))
losses = [json.loads(line) for line in (RUN / 'artifacts/losses.jsonl').read_text(encoding='utf-8').splitlines()]
validation = [json.loads(line) for line in (RUN / 'artifacts/validation_history.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(losses) == 5400 and [row['iteration'] for row in losses] == list(range(1, 5401))
assert [row['epoch'] for row in validation] == list(range(1, 101))
assert all(math.isfinite(row[key]) for row in losses for key in ('loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox'))
best = max(validation, key=lambda row: row['metrics']['coco/bbox_mAP'])
assert best['epoch'] == selected['selected_epoch'] == summary['best_validation']['epoch'] == 66
assert not selected['test_evaluated'] and not summary['test_evaluated']
checkpoint = Path(selected['best_checkpoint'])
assert checkpoint.is_relative_to(RUN) and sha(checkpoint) == selected['best_checkpoint_sha256']
import torch
unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(str(checkpoint))
try:
    data = torch.load(checkpoint, map_location='cpu', weights_only=True)
    reader = {'weights_only': True, 'state_tensors': len(data['state_dict']), 'metadata_epoch': data['meta'].get('epoch')}
    del data
except Exception as error:
    reader = {'weights_only': True, 'status': 'RESTRICTED_READER_REJECTED_METADATA',
              'exception_class': type(error).__name__, 'message': str(error), 'unsafe_globals': unsafe}
report = dict(status='PASS', epochs=100, iterations=5400, losses_finite=True,
              first_100_mean_loss=sum(row['loss'] for row in losses[:100]) / 100,
              last_100_mean_loss=sum(row['loss'] for row in losses[-100:]) / 100,
              selected_epoch=66, selected_by='native rounded validation coco/bbox_mAP; first tied best',
              selected_checkpoint_sha256=selected['best_checkpoint_sha256'],
              validation_metrics=selected['validation_metrics'],
              checkpoint_reader=reader, test_evaluated=False, training_started=False,
              detector_forward_performed=False)
save_json(OUT / 'acceptance.json', report)
print('STEP07_RESULT_ACCEPTED: epochs=100 updates=5400 best_val_epoch=66; test pending')
print('CHECKPOINT_STATIC_GLOBALS=' + json.dumps(unsafe))
print('RESTRICTED_READER=' + reader.get('status', 'PASS'))
