r"""One fixed official YOLO11s run. --dry-run only inspects checkpoint metadata; no train/val/predict.

Usage: .venv\Scripts\python.exe -B workspace\scripts\run_formal_baseline.py
       --experiment-id Y11-BASE-001-R1 [--dry-run]
There are no seed/hyperparameter/override/resume CLI options.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import csv
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback
import yaml

from check_dataset import snapshot
from readonly_dataset import (
    BLOCKED_WRITES, DATASET, PROJECT, PROTECTED_ROOT, WORKSPACE,
    install_write_guard, redirect_label_caches,
)
from formal_baseline_registry import (
    sha256, exclusive_lock, read_registry, check_csv_pair, update_registry,
)

EXPERIMENTS = {'Y11-BASE-001-R1': 42}
FAILED_ATTEMPT_ID = 'Y11-BASE-001'
COMMIT = '9aa41d679d121afb8b4cc512d7a162f4f36886c3'
WEIGHTS_SHA256 = '85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5'
SOURCE = PROJECT / 'ultralytics_src'
REGISTRY = WORKSPACE / 'experiment_tracking/experiment_registry.xlsx'
PROTOCOL = WORKSPACE / 'configs/formal_baseline_protocol.yaml'
RUNS = WORKSPACE / 'runs'
RUN_LOCK = WORKSPACE / 'experiment_tracking/.formal_baseline.lock'
OUTPUT_FIELDS = {'project', 'name', 'save_dir'}
EXPECTED_COUNTS = {'train': 107, 'val': 23, 'test': 23}
EXPECTED_NAMES = {0: 'broken', 1: 'loose'}
METRICS = {
    'precision_at_best_epoch': 'metrics/precision(B)',
    'recall_at_best_epoch': 'metrics/recall(B)',
    'map50_at_best_epoch': 'metrics/mAP50(B)',
    'map50_95_at_best_epoch': 'metrics/mAP50-95(B)',
}
_test_model_reads_forbidden = False
_extra_guard_installed = False


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dump_json(path, data):
    # The caller exclusively owns its new audit directory; never a historical file.
    temporary = path.with_suffix('.pending.json')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def install_guards():
    global _extra_guard_installed
    install_write_guard()  # Existing protection of the ENTIRE central data project.
    if _extra_guard_installed:
        return

    def audit(event, args):
        paths, writing = [], False
        if event == 'open':
            path, mode, flags = args
            paths = [path]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        elif event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate'}:
            paths, writing = [args[0]], True
        elif event in {'os.rename', 'os.link', 'os.symlink'}:
            paths, writing = list(args[:2]), True
        for raw in paths:
            if not isinstance(raw, (str, bytes, os.PathLike)):
                continue
            path = Path(os.fsdecode(raw)).resolve()
            if writing and path.is_relative_to(SOURCE.resolve()):
                raise PermissionError(f'禁止写入官方源码：{path}')
            if event == 'open' and _test_model_reads_forbidden and any(
                path.is_relative_to((DATASET / kind / 'test').resolve()) for kind in ('images', 'labels')
            ):
                raise PermissionError(f'模型阶段禁止读取 test 文件：{path}')
    sys.addaudithook(audit)
    _extra_guard_installed = True


@contextmanager
def forbid_test_model_reads():
    global _test_model_reads_forbidden
    previous = _test_model_reads_forbidden
    _test_model_reads_forbidden = True
    try:
        yield
    finally:
        _test_model_reads_forbidden = previous


def git_state():
    def git(*args):
        return subprocess.check_output(
            ['git', '--no-optional-locks', '-C', str(SOURCE), *args],
            encoding='utf-8', errors='strict').strip()
    commit, status = git('rev-parse', 'HEAD'), git('status', '--porcelain=v1', '--untracked-files=all')
    require(commit == COMMIT, f'源码 commit 不匹配：{commit}')
    require(not status, f'官方源码不是 clean：\n{status}')
    return {'commit': commit, 'status': status}


def parameter_diff(reference, actual):
    return {key: {'reference': reference.get(key), 'actual': actual.get(key)}
            for key in sorted((set(reference) | set(actual)) - OUTPUT_FIELDS)
            if key not in reference or key not in actual or reference[key] != actual[key]}


def assert_protocol_diff(reference, actual, experiment_id):
    require(experiment_id in EXPERIMENTS, f'实验 ID 不在白名单：{experiment_id}')
    require(reference.get('seed') == 42, '本次 seed 必须为 42。')
    differences = parameter_diff(reference, actual)
    expected = {}
    require(differences == expected, '训练参数发生未经授权的漂移：\n' + json.dumps(differences, ensure_ascii=False, indent=2))
    return differences


def assert_effective_training_args(reference, actual, experiment_id):
    # The locked official build_optimizer(auto) sets warmup_bias_lr to 0.0
    # (trainer.py:1146), AFTER args.yaml is saved.
    # Keep the caller/protocol value 0.1 untouched; audit this derived runtime
    # state separately. This is not permission for caller-side parameter drift.
    require(reference['optimizer'] == 'auto', '锁定版本默认 optimizer 应为 auto。')
    expected_runtime = deepcopy(reference)
    expected_runtime['warmup_bias_lr'] = 0.0
    assert_protocol_diff(expected_runtime, actual, experiment_id)
    return {'warmup_bias_lr': {'frozen_input': reference['warmup_bias_lr'], 'official_runtime': 0.0,
                              'source': 'ultralytics/engine/trainer.py:1146; build_optimizer(auto)'}}


def read_results(path, expected_epochs=None):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = [{k.strip(): float(v) for k, v in raw.items()} for raw in csv.DictReader(stream)]
    require(bool(rows), f'结果 CSV 为空：{path}')
    require([r['epoch'] for r in rows] == list(range(1, len(rows) + 1)), 'epoch 必须连续且从 1 开始。')
    if expected_epochs is not None:
        require(len(rows) == expected_epochs, f'未完成约定轮数：{len(rows)}/{expected_epochs}')
    for row in rows:
        require(all(math.isfinite(row[key]) for key in METRICS.values()), 'P/R/AP 存在非有限值，不能选 best。')
    return rows


def result_summary(rows):
    best = max(rows, key=lambda r: (r['metrics/mAP50-95(B)'], r['metrics/mAP50(B)'], -r['epoch']))
    result = {'best_epoch': int(best['epoch']), 'epochs_completed': len(rows)}
    result.update({field: best[key] for field, key in METRICS.items()})
    for label in ('precision', 'recall'):
        key = f'metrics/{label}(B)'
        peak = max(rows, key=lambda r: (r[key], -r['epoch']))
        result[f'max_{label}_any_epoch'] = peak[key]
        result[f'epoch_of_max_{label}'] = int(peak['epoch'])
    return result


def explicit_training_args():
    # This is the entire user-authorized training override list. No DIAG args are read.
    return dict(model=str(WORKSPACE / 'weights/yolo11s.pt'),
                data=str(WORKSPACE / 'configs/conductor_bl_v1.yaml'),
                epochs=100, imgsz=1280, batch=4, workers=0, device='0', seed=42,
                deterministic=True, pretrained=True, cache=False, plots=True)


def load_protocol_context(protocol, registry=REGISTRY):
    wb, _, headers, records, setup, schema = read_registry(registry)
    wb.close()
    require(schema == protocol['evidence']['registry_schema_sha256'], 'Runs/FieldDefinitions schema 发生变化。')
    check_csv_pair(registry.with_suffix('.csv'), headers, records)
    require(protocol['protocol_version'] == 2 and protocol['experiments'] == EXPERIMENTS,
            '只允许监控策略修正后的 Y11-BASE-001-R1。')
    require(protocol['training_overrides'] == explicit_training_args(), '明确设置参数发生漂移，停止；不自动修复。')
    require(protocol['weights'] == {'path': str(WORKSPACE / 'weights/yolo11s.pt'), 'sha256': WEIGHTS_SHA256},
            '必须使用指定官方 COCO 预训练权重。')
    expected_val = dict(data=explicit_training_args()['data'], imgsz=1280, device='0')
    require(protocol['evaluation_overrides'] == expected_val, 'val/test 显式参数漂移。')
    files = {'defaults_sha256': SOURCE / 'ultralytics/cfg/default.yaml',
             'snapshot_sha256': WORKSPACE / 'logs/dataset_snapshot_before.json',
             'data_yaml_sha256': Path(explicit_training_args()['data']),
             'central_data_yaml_sha256': DATASET / 'data.yaml'}
    for key, path in files.items():
        require(sha256(path) == protocol['evidence'][key], f'锁定证据发生变化：{path}')
    return headers, records, setup


def load_failed_attempt_reference(protocol, records):
    """Load immutable actual args from the retained Failed attempt."""
    evidence = protocol['evidence']
    require(evidence['failed_attempt_id'] == FAILED_ATTEMPT_ID, '失败母记录 ID 发生变化。')
    matches = [r for _, r in records if r['experiment_id'] == FAILED_ATTEMPT_ID]
    require(len(matches) == 1, '无法唯一定位保留的 Y11-BASE-001 Failed 行。')
    row = matches[0]
    run = Path(evidence['failed_attempt_run_directory']).resolve()
    args_path = Path(evidence['failed_attempt_args_path']).resolve()
    require(row['status'] == 'Failed' and row['epochs_completed'] == 3 and row['test_used'] is False,
            '旧 Y11-BASE-001 必须保持 Failed / epochs_completed=3 / test_used=False。')
    require(Path(row['run_directory']).resolve() == run and args_path == run / 'args.yaml',
            '旧失败 run 或 args 路径发生变化。')
    require(sha256(args_path) == evidence['failed_attempt_args_sha256'], '旧失败 args.yaml 被修改。')
    return yaml.safe_load(args_path.read_text(encoding='utf-8-sig')), row


def final_arguments(protocol, experiment_id, run):
    require(experiment_id in EXPERIMENTS, f'非法 experiment_id：{experiment_id}')
    require(protocol['training_overrides'] == explicit_training_args(), '训练覆盖清单发生变化。')
    return {**protocol['training_overrides'], 'project': str(RUNS), 'name': run.name,
            'save_dir': str(run), 'exist_ok': False}


def resolved_training_arguments(params):
    from ultralytics.cfg import get_cfg
    # Audit defaults, but do NOT pass this expanded dictionary to model.train().
    return vars(get_cfg(overrides={**params, 'task': 'detect', 'mode': 'train'}))


def inspect_official_weights(weights):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    require(weights.resolve() == (WORKSPACE / 'weights/yolo11s.pt').resolve(), '预训练路径不正确。')
    require(sha256(weights) == WEIGHTS_SHA256, '官方 yolo11s.pt SHA256 不匹配。')
    # Trusted, hash-verified local checkpoint. CPU metadata only, no YOLO wrapper or inference.
    checkpoint = torch.load(weights, map_location='cpu', weights_only=False)
    model = checkpoint.get('ema') if checkpoint.get('ema') is not None else checkpoint.get('model')
    require(isinstance(model, DetectionModel), '预训练权重不是 DetectionModel。')
    validate_official_structure(model, 80)
    require(len(model.names) == 80, '预训练权重不是 COCO 80 类。')
    info = {'type': type(model).__name__, 'classes': len(model.names), 'scale': model.yaml.get('scale'),
            'strides': [int(x) for x in model.stride], 'official_yaml_matches': True,
            'inspection': 'CPU checkpoint metadata only; no forward/train/val/predict'}
    del checkpoint, model
    return info


def validate_official_structure(model, nc):
    head = model.model[-1]
    require(type(model).__name__ == 'DetectionModel' and type(head).__name__ == 'Detect'
            and head.nl == 3 and head.nc == nc and [int(x) for x in head.stride] == [8, 16, 32],
            '不是官方 YOLO11s P3/P4/P5 DetectionModel。')
    official = yaml.safe_load((SOURCE / 'ultralytics/cfg/models/11/yolo11.yaml').read_text(encoding='utf-8'))
    require(all(model.yaml[k] == official[k] for k in ('backbone', 'head'))
            and model.yaml.get('scale') == 's', '模型结构或规模不等于官方 YOLO11s。')


def environment_check(protocol, setup):
    import torch
    import ultralytics
    require(Path(sys.prefix).resolve() == (PROJECT / '.venv').resolve(), '必须使用项目 .venv 的 Python。')
    require(Path(ultralytics.__file__).resolve() == SOURCE / 'ultralytics/__init__.py', 'Ultralytics 导入路径错误。')
    direct = json.loads(importlib.metadata.distribution('ultralytics').read_text('direct_url.json') or '{}')
    require(direct.get('dir_info', {}).get('editable') is True, 'Ultralytics 不是 editable 安装。')
    require(torch.cuda.is_available(), 'CUDA 不可用，禁止改用 CPU。')
    actual = {'git_commit': git_state()['commit'], 'ultralytics': ultralytics.__version__,
              'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name(0)}
    mapping = {'git_commit': 'Ultralytics commit', 'ultralytics': 'Ultralytics version', 'python': 'Python',
               'torch': 'PyTorch', 'cuda': 'CUDA runtime', 'gpu': 'GPU'}
    require(actual == protocol['environment'], f'运行环境与冻结协议不一致：{actual}')
    require(all(str(actual[k]) == str(setup[v]) for k, v in mapping.items()), '运行环境与 ProjectSetup 不一致。')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()
    return {**actual, 'cuda_available': True, 'python_executable': sys.executable,
            'ultralytics_file': ultralytics.__file__, 'editable': True, 'nvidia_smi': gpu,
            'openpyxl': importlib.metadata.version('openpyxl')}


def verify_integrity():
    historical_path = WORKSPACE / 'logs/dataset_snapshot_before.json'
    baseline = json.loads(historical_path.read_text(encoding='utf-8'))
    current = snapshot()  # Existing function: metadata across central root, hashes in dataset.
    changes = [k for k in sorted(set(baseline) | set(current)) if baseline.get(k) != current.get(k)]
    require(not changes, f'中央数据与历史快照不一致，禁止训练：{changes[:20]}')
    return current, {'unchanged': True, 'changed_paths': [], 'entries_checked': len(current),
                     'sha256_scope': str(DATASET), 'metadata_scope': str(PROTECTED_ROOT),
                     'snapshot_reference_sha256': sha256(historical_path)}


def dataset_check(protocol, args):
    data_path = Path(args['data'])
    require(sha256(data_path) == protocol['evidence']['data_yaml_sha256'], '原 data 配置内容发生变化。')
    data = yaml.safe_load(data_path.read_text(encoding='utf-8-sig'))
    require(Path(data['path']).resolve() == DATASET.resolve(), 'data.path 未指向固定中央数据集。')
    require(data['names'] == EXPECTED_NAMES, '类别必须只有 0=broken、1=loose。')
    require(all(data[s] == f'images/{s}' for s in EXPECTED_COUNTS), 'split 路径发生变化。')
    central = yaml.safe_load((DATASET / 'data.yaml').read_text(encoding='utf-8-sig'))
    require(data == central, '中央 data.yaml 与项目 data YAML 不等价。')
    current, integrity = verify_integrity()
    with (DATASET / 'split_manifest.csv').open(encoding='utf-8-sig', newline='') as stream:
        manifest = list(csv.DictReader(stream))
    require(dict(Counter(r['split'] for r in manifest)) == EXPECTED_COUNTS, 'manifest 的 split 数量不匹配。')
    keys, groups, hashes = set(), defaultdict(set), defaultdict(set)
    counts = {}
    for row in manifest:
        key = (row['split'], row['final_name'])
        require(key not in keys, 'manifest 存在重复图片。')
        keys.add(key)
        groups[row['split_group_id']].add(row['split'])
        hashes[row['sha256']].add(row['split'])
        image = DATASET / 'images' / row['split'] / row['final_name']
        require(image.resolve().is_relative_to((DATASET / 'images' / row['split']).resolve()), 'manifest 路径越界。')
        entry = current.get(image.relative_to(PROTECTED_ROOT).as_posix(), {})
        require(entry.get('sha256') == row['sha256'], f'manifest 图片哈希不匹配：{image.name}')
    require(all(len(s) == 1 for s in groups.values()), '发现跨 split 同组。')
    require(all(len(s) == 1 for s in hashes.values()), '发现跨 split 同图。')
    for split, count in EXPECTED_COUNTS.items():
        images = {p.name for p in (DATASET / 'images' / split).iterdir() if p.is_file()}
        labels = {p.stem for p in (DATASET / 'labels' / split).glob('*.txt')}
        require(images == {name for s, name in keys if s == split}, f'{split} 文件列表与 manifest 不同。')
        require(len(images) == count and labels == {Path(n).stem for n in images}, f'{split} 图片/标签数量不正确。')
        counts[split] = len(images)
    return {**integrity, 'split_counts': counts, 'names': data['names'], 'data_yaml_equivalent': True,
            'central_data_yaml': str(DATASET / 'data.yaml'), 'project_data_yaml': str(data_path),
            'manifest_sha256': sha256(DATASET / 'split_manifest.csv'),
            'test_access': 'file metadata and SHA256 only; no decode/model inference/evaluation'}


def check_new_identity(experiment_id, run, records):
    require(not any(r['experiment_id'] == experiment_id for _, r in records), f'实验 ID 已登记，禁止重跑：{experiment_id}')
    require(not run.exists(), f'输出目录已存在，禁止覆盖：{run}')
    leftovers = list(RUNS.glob(experiment_id + '_*'))
    require(not leftovers, f'该实验有历史/中断 run 目录，禁止再次运行：{leftovers}')


def write_diff(path, reference, actual, experiment_id):
    diff = parameter_diff(reference, actual)
    lines = [f'Experiment: {experiment_id}',
             'Reference: retained Y11-BASE-001 Failed run actual args.yaml',
             'Ignored output identity fields only: project, name, save_dir',
             'Training protocol differences:', json.dumps(diff, ensure_ascii=False, indent=2)]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert_protocol_diff(reference, actual, experiment_id)
    return diff


def preflight(experiment_id, run, audit_dir):
    require(experiment_id in EXPERIMENTS, '实验 ID 不在白名单。')
    protocol = yaml.safe_load(PROTOCOL.read_text(encoding='utf-8-sig'))
    headers, records, setup = load_protocol_context(protocol)
    failed_args, failed_row = load_failed_attempt_reference(protocol, records)
    check_new_identity(experiment_id, run, records)
    environment = environment_check(protocol, setup)
    call_args = final_arguments(protocol, experiment_id, run)
    resolved = resolved_training_arguments(call_args)
    diff = write_diff(audit_dir / 'failed_attempt_args_diff.txt', failed_args, resolved, experiment_id)
    weights = Path(protocol['weights']['path'])
    structure = inspect_official_weights(weights)
    print('Preflight: read-only dataset / equivalent YAML / manifest / fixed split...', flush=True)
    integrity = dataset_check(protocol, resolved)
    from ultralytics.cfg import DEFAULT_CFG_DICT
    inherited = {k: v for k, v in resolved.items() if k not in call_args and k not in ('task', 'mode')}
    require(all(DEFAULT_CFG_DICT[k] == v for k, v in inherited.items()), '未显式设置参数没有沿用官方默认值。')
    report = {'passed': True, 'experiment_id': experiment_id, 'seed': 42,
              'parameter_source': 'locked official defaults; exact protocol checked against retained failed attempt',
              'parent_experiment': FAILED_ATTEMPT_ID, 'failed_attempt_run': failed_row['run_directory'],
              'failed_attempt_args': protocol['evidence']['failed_attempt_args_path'], 'training_diff': diff,
              'train_call_kwargs': call_args, 'resolved_training_args': resolved,
              'default_inherited_parameters': inherited, 'environment': environment,
              'pretrained_path': str(weights), 'pretrained_sha256': sha256(weights), 'structure': structure,
              'dataset_integrity': integrity, 'git': git_state(), 'run_directory': str(run),
              'protocol_sha256': sha256(PROTOCOL), 'test_used': False, 'model_train_called': False,
              'model_val_called': False, 'model_predict_called': False, 'registry_modified': False,
              'registered_experiments': len(records),
              'execution_code_sha256': {name: sha256(WORKSPACE / 'scripts' / name) for name in
                    ('run_formal_baseline.py', 'formal_baseline_registry.py', 'summarize_formal_baseline.py')},
              'evaluation_overrides': protocol['evaluation_overrides'],
              'test_gate': '100 epochs completed -> fix best.pt SHA256 -> save val -> consume one test token'}
    dump_json(audit_dir / 'preflight.json', report)
    for name, values in [('train_call_kwargs.yaml', call_args), ('resolved_training_args.yaml', resolved)]:
        (audit_dir / name).write_text(yaml.safe_dump(values, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print('PREFLIGHT PASS: Y11-BASE-001-R1; exact training protocol match; no resume.', flush=True)
    return protocol, failed_args, headers, report


def initial_record(experiment_id, run, headers, params):
    record = {key: 'NA' for key in headers}
    for key in ('imgsz', 'batch', 'nbs', 'workers', 'seed', 'deterministic', 'warmup_epochs', 'mosaic',
                'close_mosaic', 'mixup', 'cutmix', 'fliplr', 'translate', 'scale', 'patience', 'amp'):
        record[key] = params[key]
    record.update(
        experiment_id=experiment_id, date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        stage='FormalBaseline', status='Running',
        purpose='Official YOLO11s single-run FormalBaseline using COCO-pretrained yolo11s.pt and the fixed broken/loose dataset.',
        parent_experiment=FAILED_ATTEMPT_ID,
        changed_variable=('Training protocol unchanged. Execution-monitoring policy changed only: transient non-finite '
                          'validation losses are logged instead of aborting training. Train-loss non-finite values remain fatal.'),
        model='YOLO11s', pretrained_weights='yolo11s.pt',
        dataset_version='02_yolo11_dataset (split_manifest controlled)', dataset_path=str(DATASET),
        train_images=107, val_images=23, test_images=23, classes='0=broken; 1=loose',
        git_commit=COMMIT, ultralytics_version='8.4.152', epochs_planned=100,
        epochs_completed=0, device='cuda:0', run_directory=str(run), best_pt=str(run / 'weights/best.pt'),
        results_csv=str(run / 'results.csv'), test_used=False, include_in_paper='Candidate',
        decision='First rerun after wrapper monitoring fix; no tuning, resume, second seed, or automatic retry.',
        notes='architecture_modified=False; training protocol unchanged; monitoring policy only; current run in progress.',
    )
    return record


def validate_trainer_start(trainer, reference, experiment_id, run):
    assert_effective_training_args(reference, vars(trainer.args), experiment_id)
    require(Path(trainer.save_dir).resolve() == run.resolve(), '实际输出目录与本次预留目录不同。')
    require(trainer.device.type == 'cuda' and trainer.batch_size == 4, '设备或 batch 被自动改变。')
    require(trainer.data['names'] == EXPECTED_NAMES, 'Trainer 类别不正确。')
    require(bool(trainer.amp), 'AMP 自检禁用了 AMP；停止，不自动改变冻结协议。')
    for loader, split in ((trainer.train_loader, 'train'), (trainer.test_loader, 'val')):
        require(len(loader.dataset) == EXPECTED_COUNTS[split], f'Trainer {split} 数量不正确。')
        require(all(Path(p).resolve().is_relative_to((DATASET / 'images' / split).resolve())
                    for p in loader.dataset.im_files), f'Trainer {split} 加载路径越界。')
    validate_official_structure(trainer.model, 2)


def assert_train_batch_finite(loss_items, total_loss):
    """Train loss non-finite values remain immediately fatal."""
    import torch
    values = {k: float(v.detach().float().cpu()) for k, v in loss_items.items()}
    finite = all(math.isfinite(v) for v in values.values()) and bool(torch.isfinite(total_loss).all())
    require(finite, '训练 batch 出现 NaN/Inf，停止，不自动修改协议。')
    return values


def check_final_validation_finite(validator):
    """Independent best.pt val/test metrics must be finite before completion."""
    metrics = getattr(getattr(validator, 'metrics', None), 'results_dict', {})
    require(all(math.isfinite(float(v)) for v in metrics.values()), '验证指标出现 NaN/Inf。')


def record_training_validation_nonfinite(validator, trainer, status, status_path, event_path):
    """Log transient training-time validation loss anomalies without altering official results."""
    import torch
    losses = getattr(validator, 'loss', None)
    if not isinstance(losses, dict):
        return None
    fields = {}
    for name, value in losses.items():
        tensor = value.detach()
        if bool(torch.isfinite(tensor).all()):
            continue
        kinds = []
        if bool(torch.isnan(tensor).any()):
            kinds.append('NaN')
        if bool(torch.isposinf(tensor).any()):
            kinds.append('+Inf')
        if bool(torch.isneginf(tensor).any()):
            kinds.append('-Inf')
        fields[name] = kinds or ['non-finite']
    if not fields:
        return None
    event = {
        'timestamp': datetime.now().isoformat(),
        'epoch': int(trainer.epoch + 1),
        'validation_batch_index': int(validator.batch_i + 1) if getattr(validator, 'batch_i', None) is not None else 'NA',
        'validation_batches': len(validator.dataloader) if getattr(validator, 'dataloader', None) is not None else 'NA',
        'nonfinite_fields': fields,
        'action': 'warning_only; official trainer continues; values are not replaced or recomputed',
    }
    status.setdefault('validation_nonfinite_events', []).append(event)
    status['validation_nonfinite_epochs'] = sorted({e['epoch'] for e in status['validation_nonfinite_events']})
    with event_path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
    dump_json(status_path, status)
    print('WARNING: transient non-finite training-time validation loss logged: '
          + json.dumps(event, ensure_ascii=False), flush=True)
    return event


def preserve_required_curves(run):
    # The locked release uses BoxPR_curve.png etc. Retain these originals and
    # provide the requested plain names as byte-identical copies, never redraw.
    for name in ('PR_curve.png', 'F1_curve.png', 'P_curve.png', 'R_curve.png'):
        target = run / name
        if not target.exists():
            original = run / ('Box' + name)
            require(original.is_file(), f'官方曲线缺失：{original.name}')
            with original.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        require(target.is_file(), f'训练曲线缺失：{name}')


def extract_evaluation_metrics(metrics):
    overall = {label: float(metrics.results_dict[key]) for label, key in
               [('precision', 'metrics/precision(B)'), ('recall', 'metrics/recall(B)'),
                ('map50', 'metrics/mAP50(B)'), ('map50_95', 'metrics/mAP50-95(B)')]}
    per_class = {}
    for row in metrics.summary(decimals=10):
        if row['Class'] not in EXPECTED_NAMES.values():
            continue
        per_class[row['Class']] = {
            'precision': float(row['Box-P']) if row.get('Box-P') is not None else 'NA',
            'recall': float(row['Box-R']) if row.get('Box-R') is not None else 'NA',
            'map50': float(row['mAP50']), 'map50_95': float(row['mAP50-95'])}
    require(set(per_class) == set(EXPECTED_NAMES.values()), '官方逐类别指标不完整。')
    require(all(v == 'NA' or math.isfinite(v) for group in [overall, *per_class.values()] for v in group.values()),
            '评价指标出现 NaN/Inf。')
    return {'overall': overall, 'per_class': per_class}


def evaluate_best(best, split, protocol, run, status):
    require(split in ('val', 'test'), '只允许固定 val 或一次最终 test。')
    path = run / 'weights/best.pt'
    require(sha256(path) == status['fixed_best_sha256'], '已固定 best.pt 发生变化。')
    validate_official_structure(best.model, 2)
    if split == 'test':
        require(status.get('val_completed') is True and status.get('test_attempts') == 1
                and status.get('test_state') == 'Started', 'test 执行凭据未消耗，禁止读取。')
    args = {**protocol['evaluation_overrides'], 'split': split,
            'project': str(run), 'name': f'final_{split}', 'exist_ok': False}
    require(not (run / args['name']).exists(), f'{split} 目录已存在，禁止覆盖或重复评价。')
    # Freshly loaded best.pt only carries the framework's minimal checkpoint
    # overrides. Check the expanded args, but pass only the minimal kwargs above.
    from ultralytics.cfg import get_cfg
    expected = vars(get_cfg(overrides={**args, 'model': str(path), 'task': 'detect', 'mode': 'val', 'rect': True}))
    actual_input = vars(get_cfg(overrides={**best.overrides, 'rect': True, **args, 'mode': 'val'}))
    require(not parameter_diff(expected, actual_input), 'best.pt 隐含参数偏离标准验证默认值。')
    expected['conf'] = 0.001  # BaseValidator resolves conf=None to the detect default.
    if split == 'test':
        prior, current = dict(status['val_evaluation_args']), dict(expected)
        prior.pop('split'); current.pop('split')
        require(not parameter_diff(prior, current), 'test 参数必须在调用前与 val 一致。')
    observed = {}

    def on_start(validator):
        require(validator.args.split == split, '评价 split 发生变化。')
        ds = validator.dataloader.dataset
        require(len(ds) == EXPECTED_COUNTS[split] and all(
            Path(p).resolve().is_relative_to((DATASET / 'images' / split).resolve()) for p in ds.im_files),
            '评价数据路径或数量不符。')
        current = vars(validator.args).copy()
        require(not parameter_diff(expected, current), '运行时验证参数偏离标准默认值。')
        observed.update(current)

    best.reset_callbacks()
    best.add_callback('on_val_start', on_start)
    best.add_callback('on_val_end', check_final_validation_finite)
    evaluation = extract_evaluation_metrics(best.val(**args))
    require(observed, '未得到实际验证参数。')
    if split == 'test':
        prior = dict(status['val_evaluation_args'])
        current = dict(observed)
        prior.pop('split'); current.pop('split')
        require(not parameter_diff(prior, current), 'val/test 参数不一致。')
    status[f'{split}_evaluation_args'] = observed
    require(sha256(path) == status['fixed_best_sha256'], '评价期间 best.pt 发生变化。')
    return evaluation


def consume_test_token(run, audit_dir, status):
    require(status.get('training_completed') is True and status.get('epochs_completed') == 100,
            '完整 100 epoch 训练尚未完成，禁止 test。')
    require(status.get('val_completed') is True and status.get('val_evaluation'), 'val 尚未记录，禁止 test。')
    require(status.get('test_attempts', 0) == 0 and not status.get('test_used'), '最终 test 已尝试，禁止重试。')
    read_results(run / 'results.csv', 100)
    require(sha256(run / 'weights/best.pt') == status['fixed_best_sha256'], 'best.pt 未固定或已变化。')
    val_path = audit_dir / 'val_metrics.json'
    require(sha256(val_path) == status['val_metrics_sha256']
            and json.loads(val_path.read_text(encoding='utf-8')) == status['val_evaluation'], 'val 记录未持久保存。')
    token = audit_dir / 'final_test_started.json'
    # Consume BEFORE any test model operation. A crash leaves evidence and cannot retry.
    with token.open('x', encoding='utf-8') as stream:
        json.dump({'best_sha256': status['fixed_best_sha256'], 'val_metrics_sha256': status['val_metrics_sha256'],
                   'started': datetime.now().isoformat(), 'maximum_test_attempts': 1}, stream, indent=2)
    status.update(test_attempts=1, test_used=True, test_state='Started')
    dump_json(audit_dir / 'status.json', status)


def run_final_test(best, protocol, run, audit_dir, status):
    consume_test_token(run, audit_dir, status)
    evaluation = evaluate_best(best, 'test', protocol, run, status)
    dump_json(audit_dir / 'test_metrics.json', evaluation)
    status.update(test_evaluation=evaluation, test_state='Completed', test_completed=True,
                  test_metrics_path=str(audit_dir / 'test_metrics.json'),
                  test_metrics_sha256=sha256(audit_dir / 'test_metrics.json'))
    dump_json(audit_dir / 'status.json', status)


def execute_training(experiment_id, run, audit_dir, prepared, status):
    """Called only from the non-dry-run branch, never by preparation or summary."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import model_info

    protocol, reference, headers, pre = prepared
    params = deepcopy(pre['train_call_kwargs'])
    status_path = audit_dir / 'status.json'
    # Recheck files after preflight and before reserving a new run/registry row.
    require(sha256(PROTOCOL) == pre['protocol_sha256'], 'Preflight 后协议文件变化。')
    _, records, _ = load_protocol_context(protocol)
    check_new_identity(experiment_id, run, records)
    run.mkdir(exist_ok=False)  # Atomically reserve an entirely new directory.
    (run / 'explicit_training_overrides.yaml').write_bytes((audit_dir / 'train_call_kwargs.yaml').read_bytes())
    record = initial_record(experiment_id, run, headers, reference)
    registered = False
    started = time.monotonic()
    warm_accumulates = set()
    status.update(state='Running', training_completed=False, test_used=False, batches_checked=0,
                  all_batch_losses_finite=True, audit_directory=str(audit_dir), run_directory=str(run),
                  preflight=pre, optimizer_actual=None, steady_accumulate='NA',
                  validation_nonfinite_events=[], validation_nonfinite_epochs=[],
                  validation_nonfinite_log=str(audit_dir / 'validation_nonfinite_events.jsonl'))
    dump_json(status_path, status)
    try:
        status['registry_backups_start'] = update_registry(
            REGISTRY, record, append=True, expected_schema=protocol['evidence']['registry_schema_sha256'])
        registered = True
        dump_json(status_path, status)
        redirect_label_caches()  # Reuse the existing adapter, only after data protection is installed.
        weights = Path(protocol['weights']['path'])
        require(sha256(weights) == pre['pretrained_sha256'], '进入模型阶段前权重发生变化。')
        with forbid_test_model_reads():
            model = YOLO(str(weights))
            validate_official_structure(model.model, 80)
            require(model.task == 'detect' and len(model.names) == 80, '必须从原始官方 COCO 80 类权重开始。')

            def verify_before_dataset(trainer):
                assert_protocol_diff(reference, vars(trainer.args), experiment_id)
                require(Path(trainer.save_dir).resolve() == run.resolve(), 'Trainer 输出路径漂移。')
                require(trainer.args.split == 'val', 'Trainer split 必须为 val。')

            def on_start(trainer):
                validate_trainer_start(trainer, reference, experiment_id, run)
                status['official_runtime_adjustments'] = assert_effective_training_args(reference, vars(trainer.args), experiment_id)
                defaults = trainer.optimizer.defaults
                groups = [{
                    'initial_lr': float(g.get('initial_lr', g['lr'])),
                    'momentum': float(g['betas'][0] if 'betas' in g else g.get('momentum', 0.0)),
                    'weight_decay': float(g.get('weight_decay', 0.0)),
                } for g in trainer.optimizer.param_groups]
                lrs = {g['initial_lr'] for g in groups}
                momenta = {g['momentum'] for g in groups}
                decays = {g['weight_decay'] for g in groups if g['weight_decay'] != 0}
                status['optimizer_actual'] = {
                    'name': type(trainer.optimizer).__name__, 'parameter_groups': groups,
                    'lr0': next(iter(lrs)) if len(lrs) == 1 else 'NA',
                    'momentum': next(iter(momenta)) if len(momenta) == 1 else 'NA',
                    'weight_decay': next(iter(decays)) if len(decays) == 1 else ('NA' if decays else 0.0),
                    'betas': list(defaults['betas']) if 'betas' in defaults else None,
                }
                status['model_parameters'] = sum(p.numel() for p in trainer.model.parameters())
                status['actual_training_args'] = vars(trainer.args).copy()
                status['train_images_loaded'] = len(trainer.train_loader.dataset)
                status['val_images_loaded'] = len(trainer.test_loader.dataset)
                dump_json(status_path, status)

            def on_epoch_start(trainer):
                # Fail on OOM instead of official automatic batch reduction. No source edits.
                trainer._oom_retries = 3
                require(trainer.batch_size == 4 and trainer.args.batch == 4, 'batch 不允许自动下调。')

            def on_batch(trainer):
                require(trainer.batch_size == 4 and trainer.args.batch == 4, 'batch 发生变化。')
                values = assert_train_batch_finite(trainer.loss_items, trainer.loss)
                finite = True
                status['batches_checked'] += 1
                status['all_batch_losses_finite'] &= finite
                ni = status['batches_checked'] - 1
                if ni >= trainer._get_warmup_iterations(len(trainer.train_loader)):
                    warm_accumulates.add(int(trainer.accumulate))
                entry = {'epoch': trainer.epoch + 1, 'batch_index': status['batches_checked'],
                         'batch': trainer.batch_size, 'accumulate': int(trainer.accumulate),
                         'loss': {k: v if math.isfinite(v) else None for k, v in values.items()}, 'finite': finite}
                with (audit_dir / 'batches.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(entry, allow_nan=False) + '\n')

            def on_val_start(validator):
                require(validator.args.split == 'val', '禁止非 val 验证。')
                require(all(Path(p).resolve().is_relative_to((DATASET / 'images/val').resolve())
                            for p in validator.dataloader.dataset.im_files), '验证数据不是固定 val。')

            def on_training_val_batch_end(validator):
                record_training_validation_nonfinite(
                    validator, model.trainer, status, status_path, audit_dir / 'validation_nonfinite_events.jsonl')

            def on_model_save(trainer):
                if trainer.best_fitness == trainer.fitness:
                    status['native_best_checkpoint_epoch'] = trainer.epoch + 1
                    status['native_best_fitness'] = float(trainer.fitness)

            def on_epoch_end(trainer):
                status['last_finished_epoch'] = trainer.epoch + 1
                dump_json(status_path, status)

            model.add_callback('on_pretrain_routine_start', verify_before_dataset)
            model.add_callback('on_train_start', on_start)
            model.add_callback('on_train_epoch_start', on_epoch_start)
            model.add_callback('on_train_batch_end', on_batch)
            model.add_callback('on_val_start', on_val_start)
            model.add_callback('on_fit_epoch_end', on_epoch_end)
            model.add_callback('on_model_save', on_model_save)
            model.add_callback('on_val_batch_end', on_training_val_batch_end)
            # Only the explicit overrides are passed; full resolved defaults are audited separately.
            require(params == final_arguments(protocol, experiment_id, run), '实际 train kwargs 发生漂移。')
            write_diff(run / 'resolved_protocol_diff.txt', reference, resolved_training_arguments(params), experiment_id)
            torch.cuda.reset_peak_memory_stats(0)
            train_started = time.monotonic()
            status['model_train_called'] = True
            dump_json(status_path, status)
            model.train(**params)
            status['training_runtime_seconds'] = round(time.monotonic() - train_started, 4)
            status['peak_vram_mib'] = round(torch.cuda.max_memory_allocated(0) / 2**20, 2)
            status['peak_reserved_mib'] = round(torch.cuda.max_memory_reserved(0) / 2**20, 2)
            for name in ('weights/best.pt', 'weights/last.pt', 'results.csv', 'args.yaml', 'results.png', 'confusion_matrix.png'):
                require((run / name).is_file(), f'训练产物缺失：{name}')
            saved_args = yaml.safe_load((run / 'args.yaml').read_text(encoding='utf-8-sig'))
            assert_protocol_diff(reference, saved_args, experiment_id)
            rows = read_results(run / 'results.csv', int(params['epochs']))
            metrics = result_summary(rows)
            status['training_completed'] = True
            status['epochs_completed'] = len(rows)
            preserve_required_curves(run)
            status['best_epoch_metrics'] = metrics
            require(status['all_batch_losses_finite'], '训练 batch loss 曾出现 NaN/Inf。')
            require(all(math.isfinite(v) for r in rows for k, v in r.items() if k.startswith('train/') and 'loss' in k),
                    'results.csv 的训练 loss 出现 NaN/Inf。')
            csv_nonfinite = [
                {'epoch': int(r['epoch']),
                 'fields': [k for k, v in r.items() if k.startswith('val/') and 'loss' in k and not math.isfinite(v)]}
                for r in rows if any(k.startswith('val/') and 'loss' in k and not math.isfinite(v) for k, v in r.items())]
            status['validation_nonfinite_epochs_csv'] = csv_nonfinite
            status['validation_nonfinite_epochs'] = sorted(set(status['validation_nonfinite_epochs'])
                                                             | {e['epoch'] for e in csv_nonfinite})
            if csv_nonfinite:
                last_bad = max(e['epoch'] for e in csv_nonfinite)
                status['validation_nonfinite_recovered'] = any(
                    int(r['epoch']) > last_bad and all(math.isfinite(v) for k, v in r.items()
                                                      if k.startswith('val/') and 'loss' in k) for r in rows)
            else:
                status['validation_nonfinite_recovered'] = 'not_applicable'
            dump_json(status_path, status)
            status['steady_accumulate'] = next(iter(warm_accumulates)) if len(warm_accumulates) == 1 else 'NA'
            status['steady_accumulate_observed_values'] = sorted(warm_accumulates)
            best = YOLO(str(run / 'weights/best.pt'))
            require(best.names == EXPECTED_NAMES and [int(x) for x in best.model.stride] == [8, 16, 32], 'best.pt 结构或类别错误。')
            status['model_parameters'] = sum(p.numel() for p in best.model.parameters())
            # FLOPs is descriptive only and may legitimately be NA; no fabricated fallback.
            try:
                info = model_info(best.model, verbose=True, imgsz=1280)
                flops = float(info[3]) if info is not None else float('nan')
                status['gflops_1280'] = flops if math.isfinite(flops) and flops > 0 else 'NA'
            except Exception as exc:
                status['gflops_1280'] = 'NA'
                status['gflops_note'] = str(exc)
            status['fixed_best_sha256'] = sha256(run / 'weights/best.pt')
            status['fixed_best_at'] = datetime.now().isoformat()
            dump_json(status_path, status)
            # Test remains blocked through training and the fixed-best val pass.
            evaluation = evaluate_best(best, 'val', protocol, run, status)
            dump_json(audit_dir / 'val_metrics.json', evaluation)
            status['val_evaluation'] = evaluation
            status['val_metrics_path'] = str(audit_dir / 'val_metrics.json')
            status['val_metrics_sha256'] = sha256(audit_dir / 'val_metrics.json')
            status['val_completed'] = True
            dump_json(status_path, status)
        run_final_test(best, protocol, run, audit_dir, status)
        class_metrics = {f'{cls}_{key}': value for cls, values in status['val_evaluation']['per_class'].items()
                         for key, value in values.items() if key in ('map50', 'map50_95')}
        status['per_class_metrics'] = class_metrics
        status['artifact_sha256'] = {name: sha256(run / name) for name in
                                     ('args.yaml', 'results.csv', 'weights/best.pt', 'weights/last.pt')}
        status['state'] = 'Completed'
        record.update(metrics, **class_metrics, test_used=True, model_parameters=status['model_parameters'],
                      gflops_1280=status['gflops_1280'], steady_accumulate=status['steady_accumulate'],
                      peak_vram_mib=status['peak_vram_mib'])
    except BaseException as exc:
        status.update(state='Failed', error=traceback.format_exc(), interrupted=isinstance(exc, KeyboardInterrupt))
        print(status['error'], flush=True)
    finally:
        # Always recheck, including exceptions/KeyboardInterrupt; never erase artifacts.
        try:
            _, status['dataset_integrity_after'] = verify_integrity()
            status['git_after'] = git_state()
            require(not BLOCKED_WRITES, f'曾发生受保护目录写入尝试：{BLOCKED_WRITES}')
        except BaseException:
            status['state'] = 'Failed'
            status['postflight_error'] = traceback.format_exc()
        status['blocked_dataset_writes'] = list(BLOCKED_WRITES)
        status['runtime_seconds'] = round(time.monotonic() - started, 4)
        if torch.cuda.is_available():
            status.setdefault('peak_vram_mib', round(torch.cuda.max_memory_allocated(0) / 2**20, 2))
        optimizer = status.get('optimizer_actual') or {}
        record.update(status=status['state'], test_used=bool(status.get('test_used')), runtime_seconds=status['runtime_seconds'],
                      optimizer=(optimizer.get('name', 'NA') + ' (optimizer=auto)' if optimizer else 'NA'),
                      lr0_actual=optimizer.get('lr0', 'NA'), momentum_actual=optimizer.get('momentum', 'NA'),
                      weight_decay_actual=optimizer.get('weight_decay', 'NA'),
                      peak_vram_mib=status.get('peak_vram_mib', 'NA'),
                      notes=json.dumps({
                          'architecture_modified': False, 'status_json': str(status_path),
                          'args_yaml': str(run / 'args.yaml'), 'runtime_scope': 'run setup + training + val/test + postflight',
                          'best_epoch_definition': 'existing registry CSV max AP50-95, then AP50',
                          'native_best_checkpoint_epoch': status.get('native_best_checkpoint_epoch', 'NA'),
                          'validation_nonfinite_epochs': status.get('validation_nonfinite_epochs', []),
                          'validation_nonfinite_recovered': status.get('validation_nonfinite_recovered', 'NA'),
                          'validation_nonfinite_log': status.get('validation_nonfinite_log', 'NA'),
                          'val': status.get('val_evaluation', 'NA'), 'test': status.get('test_evaluation', 'NA'),
                          'test_attempts': status.get('test_attempts', 0),
                          'interrupted': status.get('interrupted', False),
                      }, ensure_ascii=False))
        if status['state'] != 'Completed':
            record['epochs_completed'] = status.get('last_finished_epoch', 0)
            record['decision'] = '运行或后处理未完整完成；保留 ID 和产物，禁止自动重试。详见 status JSON。'
        status['success'] = False
        dump_json(status_path, status)
        if registered:
            try:
                status['registry_backups_finish'] = update_registry(
                    REGISTRY, record, append=False, expected_schema=protocol['evidence']['registry_schema_sha256'])
                status['registry_synchronized'] = True
            except BaseException:
                status['registry_synchronized'] = False
                status['registry_error'] = traceback.format_exc()
                status['state'] = 'Failed'
        status['success'] = status['state'] == 'Completed' and status.get('registry_synchronized', False)
        dump_json(status_path, status)
    require(status['success'], f'实验未完整成功：{status_path}。不自动重跑或改变协议。')
    from summarize_formal_baseline import write_summary
    write_summary()  # Read saved evidence only; no extra model evaluation.


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log
    def write(self, value):
        self.log.write(value)
        self.log.flush()
        self.original.write(value)
        return len(value)
    def flush(self):
        self.log.flush()
        self.original.flush()
    def isatty(self):
        return False


def configure_process():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['YOLO_CONFIG_DIR'] = str(WORKSPACE / 'configs/ultralytics_settings')
    os.environ['MPLCONFIGDIR'] = str(WORKSPACE / 'cache/matplotlib')
    os.environ['TORCH_HOME'] = str(WORKSPACE / 'cache/torch')
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    install_guards()


def main(argv=None):
    parser = argparse.ArgumentParser(description='官方 YOLO11s 固定协议；每次只运行一个白名单实验。')
    parser.add_argument('--experiment-id', required=True, choices=list(EXPERIMENTS))
    parser.add_argument('--dry-run', action='store_true', help='只核验配置和权重元数据；不调用 train/val/predict，不写登记表。')
    args = parser.parse_args(argv)
    configure_process()
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run_name = f'{args.experiment_id}_{stamp}'
    run = RUNS / run_name
    audit_root = WORKSPACE / 'logs' / ('formal_baseline_preflight' if args.dry_run else '')
    audit_dir = audit_root / run_name
    audit_dir.mkdir(parents=True, exist_ok=False)
    status = {'experiment_id': args.experiment_id, 'seed': EXPERIMENTS[args.experiment_id],
              'dry_run': args.dry_run, 'state': 'Preflight', 'success': False,
              'model_train_called': False, 'test_used': False}
    status_path = audit_dir / 'status.json'
    with (audit_dir / f'{run_name}_console.log').open('x', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                with exclusive_lock(RUN_LOCK, {'experiment_id': args.experiment_id, 'dry_run': args.dry_run}):
                    prepared = preflight(args.experiment_id, run, audit_dir)
                    status['preflight'] = prepared[-1]
                    if args.dry_run:
                        status.update(state='PreflightPassed', success=True, registry_modified=False)
                        dump_json(status_path, status)
                        print(f'DRY-RUN ONLY: no model.train(), val() or predict(); checkpoint metadata inspection only. Report: {status_path}')
                    else:
                        execute_training(args.experiment_id, run, audit_dir, prepared, status)
                return 0
            except BaseException as exc:
                status.update(success=False, state='Failed', error=traceback.format_exc())
                dump_json(status_path, status)
                print(f'停止：{exc}\n状态文件：{status_path}', file=sys.stderr)
                return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == '__main__':
    raise SystemExit(main())
