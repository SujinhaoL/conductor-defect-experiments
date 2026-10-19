"""Review saved MERGE130 training. No model, optimizer or detector inference."""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'workspace/runs/FRCNN-R50FPN-MERGE130-001/step09_109_20338'
AUDIT = ROOT / 'workspace/audit/step09_review'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    assert Path(sys.executable).resolve() == ROOT / '.venv/Scripts/python.exe'
    AUDIT.mkdir(parents=True, exist_ok=True)
    pre = read(RUN / 'pretraining_record.json')
    summary = read(RUN / 'artifacts/formal_training_summary.json')
    selection = read(RUN / 'selected_checkpoint.json')
    loaded = read(RUN / 'artifacts/actual_training_weight_load.json')
    from mmengine.config import Config
    cfg = Config.fromfile(str(RUN / 'resolved_merge130_001.py'), import_custom_modules=False)
    previous = Config.fromfile(str(ROOT / 'workspace/runs/FRCNN-R50FPN-VAL-001/step07_235_27043/resolved_val_001.py'), import_custom_modules=False)
    for key in ('model', 'optim_wrapper', 'param_scheduler', 'randomness', 'train_pipeline', 'env_cfg'):
        assert cfg[key] == previous[key], 'Previously frozen setting changed: '+key
    assert cfg.load_from == previous.load_from and not cfg.resume
    assert sha(cfg.load_from) == pre['checkpoint']['sha256'] == selection['initial_checkpoint_sha256']
    assert loaded['loaded_tensors'] == 344 and len(loaded['expected_shape_changes']) == 4
    assert loaded['loaded_before_first_optimizer_update'] and not loaded['continued_from_val001_checkpoint']
    assert all(cfg[k] is None for k in ('val_cfg', 'val_dataloader', 'val_evaluator', 'test_cfg', 'test_dataloader', 'test_evaluator'))
    assert not summary['independent_validation'] and not summary['test_evaluated']
    assert summary['epochs'] == 100 and summary['optimizer_updates'] == 6500 and summary['train_images'] == 130 and summary['train_boxes'] == 205
    assert selection['selected_epoch'] == 100 and Path(selection['checkpoint']).resolve() == RUN / 'epoch_100.pth'
    assert sha(selection['checkpoint']) == selection['checkpoint_sha256']
    assert not list(RUN.glob('best_*.pth'))
    frozen_files = {}
    for original, expected in pre['saved_inputs_sha256'].items():
        copy = RUN / 'code' / Path(original).name
        assert sha(copy) == expected, 'Frozen input differs: '+str(copy)
        frozen_files[str(copy)] = expected
    with (RUN / 'artifacts/losses.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6500
    assert [int(r['iteration']) for r in rows] == list(range(1, 6501))
    assert Counter(int(r['epoch']) for r in rows) == {e:65 for e in range(1,101)}
    keys = ('loss', 'loss_rpn_cls', 'loss_rpn_bbox', 'loss_cls', 'loss_bbox', 'learning_rate')
    assert all(math.isfinite(float(r[k])) for r in rows for k in keys)
    assert all(abs(float(r['learning_rate']) - (0.0025 if int(r['epoch']) <= 67 else 0.00025 if int(r['epoch']) <= 92 else 0.000025)) < 1e-12
               for r in rows[500:]), 'Recorded LR after warmup differs from fixed schedule'
    assert (RUN / 'artifacts/loss_curves.png').is_file()
    snapshots = read(RUN / 'artifacts/train_fit_progress.json')
    assert [r['epoch'] for r in snapshots] == [0,25,50,75,100]
    for e in (0,25,50,75,100):
        folder = RUN / 'artifacts' / f'epoch_{e:03d}'
        assert len(read(folder / 'predictions_details.json')) == 8
        assert len(list(folder.glob('cond_*.jpg'))) == 8
        assert (folder / 'predictions_contact_sheet.jpg').is_file()
    receipt = dict(status='PASS', reviewed_utc=datetime.now(timezone.utc).isoformat(), run_dir=str(RUN),
        experiment=summary['experiment'], epochs=100, iterations=6500, train_images=130, train_boxes=205,
        broken=48, loose=157, independent_validation=False, test_evaluated=False,
        selected_epoch=100, selected_checkpoint=selection['checkpoint'], selected_checkpoint_sha256=selection['checkpoint_sha256'],
        fresh_same_official_coco_initialization=True, compatible_initial_tensors_loaded=344, expected_new_outputs=4,
        frozen_shared_settings_match=True, frozen_files_sha256=frozen_files, losses_finite=True, lr_schedule_verified=True,
        first_100_mean_loss=sum(float(r['loss']) for r in rows[:100])/100,
        last_100_mean_loss=sum(float(r['loss']) for r in rows[-100:])/100,
        snapshots=snapshots, training_started=False, model_created=False, detector_forward_performed=False,
        snapshot_scope='fixed eight original training images only; not independent val or full130 metrics',
        selection_rule='last epoch100 fixed before training; no val/test selection')
    (AUDIT / 'acceptance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('STEP09_TRAINING_REVIEW_PASS: epochs=100 updates=6500 losses_finite; fresh_COCO344+4; final_epoch100; no val/test')
    print(f'LOSS_FIRST100={receipt["first_100_mean_loss"]:.6f} LOSS_LAST100={receipt["last_100_mean_loss"]:.6f}')
    print(f'FIXED_TRAIN8_FINAL_mAP50={snapshots[-1]["mAP50"]:.6f} mAP50_95={snapshots[-1]["mAP50_95"]:.6f}; NOT generalization')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
