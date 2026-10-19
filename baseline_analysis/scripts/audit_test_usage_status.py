"""Read saved test evidence only; no model imports, inference, or training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'baseline_analysis'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


index_path = BASE / 'audit/run_status_index.json'
plan_path = BASE / 'audit/teacher_simple_plan.json'
index = read(index_path)
test_sources = next(s for s in read(plan_path)['sheets'] if s['split'] == 'test')['sources']
rows = []
for run in index['runs']:
    source = next(s for s in test_sources if s['experiment_id'] == run['experiment_id'])
    row = {key: run[key] for key in ('experiment_id', 'run_id', 'model', 'input_size',
                                    'train_images', 'selected_epoch', 'test_status', 'source')}
    row['evaluation_id'] = source['evaluation_id']
    metric_path = source['metric_source']
    row['metric_source'] = metric_path
    if metric_path:
        path = Path(metric_path)
        assert path.is_file(), path
        row['metric_sha256'] = sha(path)
        if path.suffix == '.json':
            metrics = read(path)
            row['saved_metric_keys'] = list(metrics)[:30]
        elif path.suffix in ('.txt', '.log'):
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
            row['saved_test_lines'] = [s for s in lines if any(
                word in s.lower() for word in ('all ', 'test', 'broken', 'loose'))][-16:]
    else:
        directory = Path(run['source'])
        assert directory.is_dir(), directory
        candidates = [str(p) for p in directory.rglob('*') if 'test' in p.name.lower()]
        logs = directory.parents[1] / 'logs'
        expected_step = 'step18' if run['model'] == 'YOLO11s' else 'step19'
        row['test_named_paths_in_run'] = candidates
        row['supplemental_runtime_logs'] = sorted(str(p) for p in logs.glob(expected_step + '*'))
        assert not candidates, f'Unexpected test evidence requires inspection: {candidates}'
        assert not row['supplemental_runtime_logs'], row['supplemental_runtime_logs']
    rows.append(row)

result = {
    'review_date_asia_shanghai': '2026-10-17',
    'scope': 'Nine recorded formal experiments; original test split: 23 images / 36 boxes',
    'completed_test_experiments': sum(bool(r['metric_source']) for r in rows),
    'not_tested_experiments': sum(not r['metric_source'] for r in rows),
    'diagnostic_experiments': index['diagnostics'],
    'sources': {str(p): sha(p) for p in (index_path, plan_path)},
    'runs': rows,
    'model_inference_started': False,
    'training_started': False,
}
out = BASE / 'audit/test_usage_review_20261017'
out.mkdir(exist_ok=True)
(out / 'verified_status.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
