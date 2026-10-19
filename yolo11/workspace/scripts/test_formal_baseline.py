"""Safety tests: temporary workbook copies / synthetic audit events, no training."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from copy import deepcopy
import os
import csv
import json
from types import SimpleNamespace
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import yaml

import run_formal_baseline as runner
import formal_baseline_registry as registry
import summarize_formal_baseline as summary


class FormalBaselineSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = yaml.safe_load(runner.PROTOCOL.read_text(encoding='utf-8'))
        runner.configure_process()
        cls.headers, cls.records, _ = runner.load_protocol_context(cls.protocol)
        cls.reference, cls.row = runner.load_failed_attempt_reference(cls.protocol, cls.records)

    def test_exact_seed_only_differences(self):
        for key, seed in runner.EXPERIMENTS.items():
            args = runner.final_arguments(self.protocol, key, runner.RUNS / f'{key}_TEST')
            diff = runner.assert_protocol_diff(self.reference, runner.resolved_training_arguments(args), key)
            self.assertEqual(set(diff), set() if seed == 42 else {'seed'})

    def test_training_drift_is_rejected_without_repair(self):
        for field, value in [('epochs', 30), ('nbs', 16), ('mosaic', 0), ('model', 'yolo11s_p2.yaml'),
                             ('optimizer', 'SGD'), ('imgsz', 640), ('lr0', 0.02), ('cls', 1.0), ('split', 'test')]:
            with self.subTest(field=field):
                args = deepcopy(self.reference)
                args[field] = value
                with self.assertRaises(RuntimeError):
                    runner.assert_protocol_diff(self.reference, args, 'Y11-BASE-001-R1')
                self.assertEqual(args[field], value)

    def test_missing_extra_and_wrong_seed_are_rejected(self):
        for mutate in (lambda a: a.pop('translate'), lambda a: a.update(extra_training_knob=1), lambda a: a.update(seed=1)):
            args = deepcopy(self.reference)
            mutate(args)
            with self.assertRaises(RuntimeError):
                runner.assert_protocol_diff(self.reference, args, 'Y11-BASE-001-R1')
        with self.assertRaises(RuntimeError):
            runner.final_arguments(self.protocol, 'Y11-BASE-004-S7', runner.RUNS / 'unused')

    def test_official_auto_runtime_adjustment_is_separate_from_frozen_input(self):
        actual = deepcopy(self.reference)
        actual['warmup_bias_lr'] = 0.0
        # Never accept it as a change to the caller's frozen protocol.
        with self.assertRaises(RuntimeError):
            runner.assert_protocol_diff(self.reference, actual, 'Y11-BASE-001-R1')
        derived = runner.assert_effective_training_args(self.reference, actual, 'Y11-BASE-001-R1')
        self.assertEqual(derived['warmup_bias_lr']['official_runtime'], 0.0)
        self.assertEqual(self.reference['warmup_bias_lr'], .1)
        actual['mosaic'] = 0.0
        with self.assertRaises(RuntimeError):
            runner.assert_effective_training_args(self.reference, actual, 'Y11-BASE-001-R1')

    def test_existing_id_and_directory_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='formal-baseline-identity-') as tmp:
            root = Path(tmp)
            key = 'Y11-BASE-001-R1'
            with patch.object(runner, 'RUNS', root):
                with self.assertRaises(RuntimeError):
                    runner.check_new_identity(key, root / 'new', [(1, {'experiment_id': key})])
                (root / f'{key}_old').mkdir()
                with self.assertRaises(RuntimeError):
                    runner.check_new_identity(key, root / 'new', [])
                with self.assertRaises(RuntimeError):
                    runner.check_new_identity(key, root / f'{key}_old', [])

    def test_best_epoch_tie_break_and_independent_peaks(self):
        rows = []
        for epoch, p, r, ap50, ap in [(1, .9, .1, .2, .1), (2, .2, .9, .3, .1), (3, .1, .2, .3, .1)]:
            rows.append({'epoch': epoch, 'metrics/precision(B)': p, 'metrics/recall(B)': r,
                         'metrics/mAP50(B)': ap50, 'metrics/mAP50-95(B)': ap})
        result = runner.result_summary(rows)
        self.assertEqual(result['best_epoch'], 2)
        self.assertEqual(result['precision_at_best_epoch'], .2)
        self.assertEqual(result['recall_at_best_epoch'], .9)
        self.assertEqual(result['epoch_of_max_precision'], 1)
        self.assertEqual(result['max_precision_any_epoch'], .9)

    def test_history_and_sheets_preserved_in_copy(self):
        with tempfile.TemporaryDirectory(prefix='formal-baseline-xlsx-') as tmp:
            path = Path(tmp) / 'experiment_registry.xlsx'
            shutil.copyfile(runner.REGISTRY, path)
            shutil.copyfile(runner.REGISTRY.with_suffix('.csv'), path.with_suffix('.csv'))
            old_hash = registry.sha256(path)
            wb, _, headers, old_rows, _, schema = registry.read_registry(path)
            target = max(n for n, _ in old_rows) + 1
            old_cells = registry.cell_snapshot(wb, target)
            old_tables = {name: wb['Runs'].tables[name].ref for name in wb['Runs'].tables}
            old_views = {ws.title: (ws.freeze_panes, str(ws.views), str(ws.sheet_format), str(ws.merged_cells)) for ws in wb}
            wb.close()
            key = 'Y11-BASE-001-R1'
            args = runner.final_arguments(self.protocol, key, Path(tmp) / key)
            record = runner.initial_record(key, Path(tmp) / key, headers, runner.resolved_training_arguments(args))
            backups = registry.update_registry(path, record, append=True, expected_schema=schema)
            self.assertEqual(registry.sha256(Path(backups[0])), old_hash)
            self.assertEqual(len(backups), 2)
            wb, _, headers, records, _, _ = registry.read_registry(path)
            self.assertEqual(len(records), len(old_rows) + 1)
            after = registry.cell_snapshot(wb)
            for coord, value in old_cells.items():
                self.assertEqual(after[coord], value, coord)
            for ws in wb:
                self.assertEqual(old_views[ws.title], (ws.freeze_panes, str(ws.views), str(ws.sheet_format), str(ws.merged_cells)))
            self.assertTrue(wb['Runs'].tables['RunsTable'].ref.endswith(str(target)))
            registry.check_csv_pair(path.with_suffix('.csv'), headers, records)
            wb.close()
            record.update(status='Completed', epochs_completed=100, best_epoch=28, test_used=True,
                          notes=json.dumps({'test_attempts': 1, 'val': {'synthetic': True}, 'test': {'synthetic': True}}),
                          precision_at_best_epoch=.00731, recall_at_best_epoch=.03571,
                          map50_at_best_epoch=.02141, map50_95_at_best_epoch=.00964)
            registry.update_registry(path, record, append=False, expected_schema=schema)
            wb, _, headers, records, _, _ = registry.read_registry(path)
            self.assertEqual(records[-1][1]['status'], 'Completed')
            self.assertEqual(len(records), len(old_rows) + 1)
            final_cells = registry.cell_snapshot(wb)
            self.assertTrue(all(final_cells[k] == v for k, v in old_cells.items()))
            registry.check_csv_pair(path.with_suffix('.csv'), headers, records)
            wb.close()
            with self.assertRaises(RuntimeError):
                registry.update_registry(path, record, append=True, expected_schema=schema)
            with self.assertRaises(RuntimeError):
                registry.update_registry(path, record, append=False, expected_schema=schema)

    def test_historical_row_update_is_refused(self):
        with tempfile.TemporaryDirectory(prefix='formal-baseline-history-') as tmp:
            path = Path(tmp) / 'experiment_registry.xlsx'
            shutil.copyfile(runner.REGISTRY, path)
            shutil.copyfile(runner.REGISTRY.with_suffix('.csv'), path.with_suffix('.csv'))
            before = registry.sha256(path)
            with self.assertRaises(RuntimeError):
                registry.update_registry(path, dict(self.row), append=False,
                                         expected_schema=self.protocol['evidence']['registry_schema_sha256'])
            self.assertEqual(registry.sha256(path), before)

    def test_pair_rolls_back_if_csv_replace_fails(self):
        with tempfile.TemporaryDirectory(prefix='formal-baseline-rollback-') as tmp:
            path = Path(tmp) / 'experiment_registry.xlsx'
            shutil.copyfile(runner.REGISTRY, path)
            shutil.copyfile(runner.REGISTRY.with_suffix('.csv'), path.with_suffix('.csv'))
            original = (registry.sha256(path), registry.sha256(path.with_suffix('.csv')))
            record = runner.initial_record('Y11-BASE-001-R1', Path(tmp) / 'run', self.headers, self.reference)
            replace = os.replace
            def fail_csv(src, dst):
                if Path(src).suffix == '.csv':
                    raise PermissionError('simulated locked CSV')
                return replace(src, dst)
            with patch.object(registry.os, 'replace', side_effect=fail_csv):
                with self.assertRaises(PermissionError):
                    registry.update_registry(path, record, append=True,
                                             expected_schema=self.protocol['evidence']['registry_schema_sha256'])
            self.assertEqual((registry.sha256(path), registry.sha256(path.with_suffix('.csv'))), original)

    def test_lock_refuses_existing_file_without_removal(self):
        with tempfile.TemporaryDirectory(prefix='formal-baseline-lock-') as tmp:
            path = Path(tmp) / 'active.lock'
            path.write_text('existing-owner', encoding='utf-8')
            with self.assertRaises(RuntimeError):
                with registry.exclusive_lock(path, {}):
                    self.fail('must not acquire existing lock')
            self.assertEqual(path.read_text(encoding='utf-8'), 'existing-owner')

    def test_main_dry_run_never_calls_training_entry(self):
        with tempfile.TemporaryDirectory(prefix='formal-baseline-dry-run-') as tmp:
            root = Path(tmp)
            lock = root / 'run.lock'
            with patch.object(runner, 'WORKSPACE', root), patch.object(runner, 'RUNS', root / 'runs'), \
                 patch.object(runner, 'RUN_LOCK', lock), patch.object(runner, 'configure_process'), \
                 patch.object(runner, 'preflight', return_value=(None, None, None, None, {'passed': True})), \
                 patch.object(runner, 'execute_training', side_effect=AssertionError('training entry must never be called')) as train:
                self.assertEqual(runner.main(['--experiment-id', 'Y11-BASE-001-R1', '--dry-run']), 0)
                train.assert_not_called()
            self.assertFalse((root / 'runs').exists())
            self.assertFalse(lock.exists())

    def test_only_explicit_training_overrides_and_current_id(self):
        args = runner.final_arguments(self.protocol, 'Y11-BASE-001-R1', runner.RUNS / 'new')
        self.assertEqual(set(args), set(runner.explicit_training_args()) | runner.OUTPUT_FIELDS | {'exist_ok'})
        for forbidden in ('nbs', 'mosaic', 'optimizer', 'lr0', 'momentum', 'warmup_epochs', 'box', 'cls', 'patience'):
            self.assertNotIn(forbidden, args)
        from ultralytics.cfg import DEFAULT_CFG_DICT
        resolved = runner.resolved_training_arguments(args)
        for field in set(DEFAULT_CFG_DICT) - set(args) - {'task', 'mode'}:
            self.assertEqual(resolved[field], DEFAULT_CFG_DICT[field])
        self.assertEqual(resolved['epochs'], 100)
        self.assertEqual(resolved['patience'], 100)
        self.assertEqual(resolved['save_period'], -1)
        self.assertEqual(runner.EXPERIMENTS, {'Y11-BASE-001-R1': 42})
        for retired in ('Y11-BASE-001', 'Y11-BASE-001-S42', 'Y11-BASE-002-S3407', 'Y11-BASE-003-S2026'):
            with self.assertRaises(RuntimeError):
                runner.final_arguments(self.protocol, retired, runner.RUNS / 'unused')

    def test_r1_args_exactly_match_retained_failed_attempt(self):
        actual = runner.resolved_training_arguments(
            runner.final_arguments(self.protocol, 'Y11-BASE-001-R1', runner.RUNS / 'Y11-BASE-001-R1_TEST'))
        self.assertEqual(runner.parameter_diff(self.reference, actual), {})
        self.assertEqual(self.row['experiment_id'], 'Y11-BASE-001')
        self.assertEqual(self.row['status'], 'Failed')
        self.assertEqual(self.row['epochs_completed'], 3)
        self.assertFalse(self.row['test_used'])
        self.assertEqual(runner.sha256(Path(self.protocol['evidence']['failed_attempt_args_path'])),
                         self.protocol['evidence']['failed_attempt_args_sha256'])

    def test_old_failed_run_is_preserved_and_r1_is_distinct(self):
        old = Path(self.row['run_directory'])
        before = {name: runner.sha256(old / name) for name in
                  ('args.yaml', 'results.csv', 'weights/best.pt', 'weights/last.pt')}
        candidate = runner.RUNS / 'Y11-BASE-001-R1_TEST_ONLY'
        runner.check_new_identity('Y11-BASE-001-R1', candidate, self.records)
        self.assertEqual(before, {name: runner.sha256(old / name) for name in before})

    def make_gate_fixture(self, root):
        run, audit = root / 'run', root / 'audit'
        (run / 'weights').mkdir(parents=True)
        audit.mkdir()
        (run / 'weights/best.pt').write_bytes(b'synthetic checkpoint; never loaded')
        with (run / 'results.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', *runner.METRICS.values()])
            writer.writeheader()
            for epoch in range(1, 101):
                writer.writerow({'epoch': epoch, **{k: 0.1 for k in runner.METRICS.values()}})
        val = {'overall': {'map50': 0.1}, 'per_class': {}}
        runner.dump_json(audit / 'val_metrics.json', val)
        status = dict(training_completed=True, epochs_completed=100, val_completed=True,
                      val_evaluation=val, fixed_best_sha256=runner.sha256(run / 'weights/best.pt'),
                      val_metrics_sha256=runner.sha256(audit / 'val_metrics.json'), test_attempts=0, test_used=False)
        return run, audit, status

    def test_test_gate_requires_completed_training_fixed_best_and_saved_val(self):
        with tempfile.TemporaryDirectory(prefix='formal-test-gate-') as tmp:
            run, audit, status = self.make_gate_fixture(Path(tmp))
            for change in ({'training_completed': False}, {'epochs_completed': 99}, {'val_completed': False},
                           {'test_attempts': 1}, {'fixed_best_sha256': 'bad'}, {'val_metrics_sha256': 'bad'}):
                bad = {**status, **change}
                with self.assertRaises(RuntimeError):
                    runner.consume_test_token(run, audit, bad)
                self.assertFalse((audit / 'final_test_started.json').exists())
            runner.consume_test_token(run, audit, status)
            self.assertEqual(status['test_attempts'], 1)
            self.assertTrue(status['test_used'])
            with self.assertRaises(RuntimeError):
                runner.consume_test_token(run, audit, status)
            # Even a stale in-memory status cannot bypass the durable consumed marker.
            stale = {**status, 'test_attempts': 0, 'test_used': False}
            with self.assertRaises(FileExistsError):
                runner.consume_test_token(run, audit, stale)

    def test_failed_test_is_not_retried(self):
        with tempfile.TemporaryDirectory(prefix='formal-test-failure-') as tmp:
            run, audit, status = self.make_gate_fixture(Path(tmp))
            with patch.object(runner, 'evaluate_best', side_effect=RuntimeError('synthetic CUDA failure')) as evaluate:
                with self.assertRaises(RuntimeError):
                    runner.run_final_test(None, self.protocol, run, audit, status)
                with self.assertRaises(RuntimeError):
                    runner.run_final_test(None, self.protocol, run, audit, status)
                self.assertEqual(evaluate.call_count, 1)
            self.assertEqual(status['test_attempts'], 1)
            self.assertFalse(status.get('test_completed', False))

    def test_official_per_class_precision_recall_or_na(self):
        metrics = SimpleNamespace(results_dict={key: .25 for key in runner.METRICS.values()},
            summary=lambda **kw: [{'Class': cls, 'mAP50': .2, 'mAP50-95': .1, **extra} for cls, extra in
                                 [('broken', {'Box-P': .3, 'Box-R': .4}), ('loose', {})]])
        result = runner.extract_evaluation_metrics(metrics)
        self.assertEqual(result['per_class']['broken']['precision'], .3)
        self.assertEqual(result['per_class']['loose']['precision'], 'NA')

    def test_train_batch_nan_remains_fatal(self):
        import torch
        with self.assertRaises(RuntimeError):
            runner.assert_train_batch_finite({'box_loss': torch.tensor(float('nan'))}, torch.tensor(1.0))
        with self.assertRaises(RuntimeError):
            runner.assert_train_batch_finite({'box_loss': torch.tensor(1.0)}, torch.tensor(float('inf')))

    def test_training_validation_nan_is_logged_without_stop(self):
        import torch
        with tempfile.TemporaryDirectory(prefix='formal-val-warning-') as tmp:
            root = Path(tmp)
            status = {}
            validator = SimpleNamespace(loss={'box_loss': torch.tensor(1.0),
                                              'cls_loss': torch.tensor(float('nan')),
                                              'dfl_loss': torch.tensor(float('inf'))},
                                        batch_i=1, dataloader=[None, None, None])
            event = runner.record_training_validation_nonfinite(
                validator, SimpleNamespace(epoch=3), status, root / 'status.json', root / 'events.jsonl')
            self.assertEqual(event['epoch'], 4)
            self.assertEqual(event['validation_batch_index'], 2)
            self.assertEqual(set(event['nonfinite_fields']), {'cls_loss', 'dfl_loss'})
            self.assertEqual(status['validation_nonfinite_epochs'], [4])
            self.assertEqual(len((root / 'events.jsonl').read_text(encoding='utf-8').splitlines()), 1)
            self.assertEqual(json.loads((root / 'status.json').read_text(encoding='utf-8'))
                             ['validation_nonfinite_events'][0]['action'], event['action'])

    def test_final_validation_metrics_must_be_finite_before_test(self):
        bad = SimpleNamespace(metrics=SimpleNamespace(results_dict={'metrics/mAP50(B)': float('nan')}))
        with self.assertRaises(RuntimeError):
            runner.check_final_validation_finite(bad)

    def test_final_val_test_use_same_official_defaults(self):
        from ultralytics.cfg import get_cfg
        with tempfile.TemporaryDirectory(prefix='formal-evaluation-defaults-') as tmp:
            run, _, status = self.make_gate_fixture(Path(tmp))
            class FakeBest:
                model = None
                overrides = {'model': str(run / 'weights/best.pt'), 'task': 'detect',
                             'data': runner.explicit_training_args()['data'], 'imgsz': 1280, 'single_cls': False}
                calls = []
                def reset_callbacks(self):
                    self.callbacks = {}
                def add_callback(self, event, fn):
                    self.callbacks[event] = fn
                def val(self, **kwargs):
                    self.calls.append(kwargs)
                    actual = get_cfg(overrides={**self.overrides, 'rect': True, **kwargs, 'mode': 'val'})
                    actual.conf = .001
                    files = [str(runner.DATASET / 'images' / kwargs['split'] / f'synthetic-{i}.jpg') for i in range(23)]
                    class FakeDataset:
                        im_files = files
                        def __len__(self): return 23
                    validator = SimpleNamespace(args=actual, dataloader=SimpleNamespace(dataset=FakeDataset()))
                    self.callbacks['on_val_start'](validator)
                    return None
            best = FakeBest()
            metrics = {'overall': {'map50': .1}, 'per_class': {}}
            with patch.object(runner, 'validate_official_structure'), patch.object(runner, 'extract_evaluation_metrics', return_value=metrics):
                runner.evaluate_best(best, 'val', self.protocol, run, status)
                status.update(test_attempts=1, test_state='Started')
                runner.evaluate_best(best, 'test', self.protocol, run, status)
                self.assertEqual(len(best.calls), 2)
                self.assertEqual(set(best.calls[0]), {'data', 'imgsz', 'device', 'split', 'project', 'name', 'exist_ok'})
                for field, expected in [('batch', 16), ('workers', 8), ('rect', True), ('conf', .001),
                                        ('iou', .7), ('max_det', 300), ('quantize', None)]:
                    self.assertEqual(status['val_evaluation_args'][field], expected)
                    self.assertEqual(status['test_evaluation_args'][field], expected)
                status['val_evaluation_args']['conf'] = .25
                with self.assertRaises(RuntimeError):
                    runner.evaluate_best(best, 'test', self.protocol, run, status)
                self.assertEqual(len(best.calls), 2, 'parameter mismatch must stop BEFORE evaluation')

    def test_completed_registry_requires_test_and_100_epochs(self):
        with tempfile.TemporaryDirectory(prefix='formal-completion-') as tmp:
            path = Path(tmp) / 'experiment_registry.xlsx'
            shutil.copyfile(runner.REGISTRY, path)
            shutil.copyfile(runner.REGISTRY.with_suffix('.csv'), path.with_suffix('.csv'))
            record = runner.initial_record('Y11-BASE-001-R1', Path(tmp) / 'run', self.headers, self.reference)
            schema = self.protocol['evidence']['registry_schema_sha256']
            registry.update_registry(path, record, append=True, expected_schema=schema)
            before = registry.sha256(path)
            record.update(status='Completed', epochs_completed=100, notes=json.dumps({'test': 'NA'}))
            with self.assertRaises(RuntimeError):
                registry.update_registry(path, record, append=False, expected_schema=schema)
            self.assertEqual(registry.sha256(path), before)

    def test_formal_summary_blocked_until_all_completed(self):
        with self.assertRaisesRegex(RuntimeError, '缺失或未 Completed'):
            summary.collect_completed(self.protocol)

    def test_synthetic_dataset_and_source_write_guards(self):
        runner.install_guards()
        for path in (runner.PROTECTED_ROOT / 'synthetic-no-file.txt', runner.SOURCE / 'synthetic-no-file.txt'):
            with self.assertRaises(PermissionError):
                sys.audit('open', str(path), 'w', os.O_WRONLY | os.O_CREAT)
        with self.assertRaises(PermissionError):
            sys.audit('os.rename', str(runner.PROTECTED_ROOT / 'synthetic-no-file.txt'), 'unused', -1, -1)

    def test_synthetic_test_reads_blocked_during_model_phase(self):
        runner.install_guards()
        path = runner.DATASET / 'images/test/synthetic-no-file.jpg'
        with runner.forbid_test_model_reads():
            with self.assertRaises(PermissionError):
                sys.audit('open', str(path), 'r', os.O_RDONLY)
            sys.audit('open', str(runner.DATASET / 'images/val/synthetic-no-file.jpg'), 'r', os.O_RDONLY)
        # Hash-only integrity access outside the model phase remains possible.
        sys.audit('open', str(path), 'r', os.O_RDONLY)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    unittest.main(verbosity=2)
