"""Read and report the single completed baseline; never train or evaluate a model.

The pre-training three-seed implementation is preserved in the plan-change backup.
"""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import yaml
import run_formal_baseline as runner


def collect_completed(protocol=None):
    protocol = protocol or yaml.safe_load(runner.PROTOCOL.read_text(encoding='utf-8'))
    _, records, _ = runner.load_protocol_context(protocol)
    failed = [r for _, r in records if r['experiment_id'] == 'Y11-BASE-001']
    runner.require(len(failed) == 1 and failed[0]['status'] == 'Failed' and failed[0]['epochs_completed'] == 3,
                   '必须保留原 Y11-BASE-001 Failed 历史证据。')
    rows = [r for _, r in records if r['experiment_id'] == 'Y11-BASE-001-R1']
    runner.require(len(rows) == 1 and rows[0]['status'] == 'Completed',
                   'Y11-BASE-001-R1 缺失或未 Completed；不生成正式结果。')
    row = rows[0]
    notes = json.loads(row['notes'])
    path = Path(notes['status_json'])
    status = json.loads(path.read_text(encoding='utf-8'))
    runner.require(status['success'] and status['registry_synchronized'] and status['training_completed']
                   and status['epochs_completed'] == 100 and status['test_completed']
                   and status['test_attempts'] == 1 and row['test_used'] is True,
                   '训练、val、一次 test 或登记不完整。')
    run = Path(row['run_directory'])
    for name, digest in status['artifact_sha256'].items():
        runner.require(runner.sha256(run / name) == digest, f'产物发生变化：{name}')
    for split in ('val', 'test'):
        evidence = path.parent / f'{split}_metrics.json'
        runner.require(runner.sha256(evidence) == status[f'{split}_metrics_sha256']
                       and json.loads(evidence.read_text(encoding='utf-8')) == notes[split]
                       and notes[split] == status[f'{split}_evaluation'], f'{split} 指标证据不一致。')
    runner.require(status['dataset_integrity_after']['unchanged'] and status['git_after']['status'] == '',
                   '训练后完整性检查不通过。')
    return row, status


def write_summary():
    row, status = collect_completed()
    result = {k: row[k] for k in ('experiment_id', 'model', 'pretrained_weights', 'epochs_completed', 'seed',
              'best_epoch', 'runtime_seconds', 'peak_vram_mib', 'model_parameters', 'best_pt', 'run_directory')}
    result.update(native_best_checkpoint_epoch=status.get('native_best_checkpoint_epoch', 'NA'),
                  validation_nonfinite_epochs=status.get('validation_nonfinite_epochs', []),
                  validation_nonfinite_recovered=status.get('validation_nonfinite_recovered', 'NA'),
                  all_batch_losses_finite=status.get('all_batch_losses_finite', False),
                  val=status['val_evaluation'], test=status['test_evaluation'], test_attempts=1,
                  actual_args_yaml=str(Path(row['run_directory']) / 'args.yaml'),
                  git_clean=True, dataset_unchanged=True, registry_synchronized=True)
    lines = ['# Y11-BASE-001-R1 正式 YOLO11s baseline', '',
             '在训练协议完全不变的条件下修正 wrapper 监控策略后首次重跑；100 epochs；seed=42。',
             '训练期瞬时非有限 validation loss 只记录；train loss 和最终独立 val 指标仍必须有限。', '',
             '| split / class | Precision | Recall | mAP50 | mAP50-95 |', '|---|---:|---:|---:|---:|']
    for split in ('val', 'test'):
        groups = {'overall': result[split]['overall'], **result[split]['per_class']}
        for cls, values in groups.items():
            lines.append('| ' + split + ' / ' + cls + ' | ' + ' | '.join(str(values[k]) for k in
                         ('precision', 'recall', 'map50', 'map50_95')) + ' |')
    lines += ['', 'best_epoch 按原登记表 CSV 定义；native_best_checkpoint_epoch 记录官方实际保存 best.pt 的轮次，',
              '如因四舍五入或同分覆盖而不同，保留两者，不另选权重或重测。', '', '```json',
              json.dumps(result, ensure_ascii=False, indent=2), '```', '', '完成后停止，不自动继续其他实验。', '']
    outputs = {
        runner.WORKSPACE / 'analysis/formal_baseline_single_run_metrics.json': json.dumps(result, ensure_ascii=False, indent=2) + '\n',
        runner.WORKSPACE / 'notes/formal_baseline_single_run_report.md': '\n'.join(lines),
    }
    for path, text in outputs.items():
        runner.require(not path.exists() or path.read_text(encoding='utf-8') == text, f'已有不同内容，禁止覆盖：{path}')
    for path, text in outputs.items():
        if not path.exists():
            with path.open('x', encoding='utf-8') as stream:
                stream.write(text)
        print(path)
    return result


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        write_summary()
    except Exception as exc:
        print(f'停止：{exc}', file=sys.stderr)
        raise SystemExit(1)
