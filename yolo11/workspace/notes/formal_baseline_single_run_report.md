# Y11-BASE-001-R1 正式 YOLO11s baseline

在训练协议完全不变的条件下修正 wrapper 监控策略后首次重跑；100 epochs；seed=42。
训练期瞬时非有限 validation loss 只记录；train loss 和最终独立 val 指标仍必须有限。

| split / class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| val / overall | 0.012958261696168077 | 0.23412698412698413 | 0.02226210846221257 | 0.0043861621782815986 |
| val / broken | 0.0243902439 | 0.1111111111 | 0.0127777778 | 0.0025555556 |
| val / loose | 0.0015262795 | 0.3571428571 | 0.0317464391 | 0.0062167688 |
| test / overall | 0.016253338045868676 | 0.20535714285714285 | 0.008108350166140744 | 0.0022876577164153526 |
| test / broken | 0.03125 | 0.125 | 0.0089285714 | 0.0026785714 |
| test / loose | 0.0012566761 | 0.2857142857 | 0.0072881289 | 0.001896744 |

best_epoch 按原登记表 CSV 定义；native_best_checkpoint_epoch 记录官方实际保存 best.pt 的轮次，
如因四舍五入或同分覆盖而不同，保留两者，不另选权重或重测。

```json
{
  "experiment_id": "Y11-BASE-001-R1",
  "model": "YOLO11s",
  "pretrained_weights": "yolo11s.pt",
  "epochs_completed": 100,
  "seed": 42,
  "best_epoch": 15,
  "runtime_seconds": 1672.734,
  "peak_vram_mib": 4526.72,
  "model_parameters": 9428566,
  "best_pt": "D:\\小论文\\model_projects\\yolo11\\workspace\\runs\\Y11-BASE-001-R1_20260917_203044_883451\\weights\\best.pt",
  "run_directory": "D:\\小论文\\model_projects\\yolo11\\workspace\\runs\\Y11-BASE-001-R1_20260917_203044_883451",
  "native_best_checkpoint_epoch": 15,
  "validation_nonfinite_epochs": [
    4,
    5,
    9
  ],
  "validation_nonfinite_recovered": true,
  "all_batch_losses_finite": true,
  "val": {
    "overall": {
      "precision": 0.012958261696168077,
      "recall": 0.23412698412698413,
      "map50": 0.02226210846221257,
      "map50_95": 0.0043861621782815986
    },
    "per_class": {
      "broken": {
        "precision": 0.0243902439,
        "recall": 0.1111111111,
        "map50": 0.0127777778,
        "map50_95": 0.0025555556
      },
      "loose": {
        "precision": 0.0015262795,
        "recall": 0.3571428571,
        "map50": 0.0317464391,
        "map50_95": 0.0062167688
      }
    }
  },
  "test": {
    "overall": {
      "precision": 0.016253338045868676,
      "recall": 0.20535714285714285,
      "map50": 0.008108350166140744,
      "map50_95": 0.0022876577164153526
    },
    "per_class": {
      "broken": {
        "precision": 0.03125,
        "recall": 0.125,
        "map50": 0.0089285714,
        "map50_95": 0.0026785714
      },
      "loose": {
        "precision": 0.0012566761,
        "recall": 0.2857142857,
        "map50": 0.0072881289,
        "map50_95": 0.001896744
      }
    }
  },
  "test_attempts": 1,
  "actual_args_yaml": "D:\\小论文\\model_projects\\yolo11\\workspace\\runs\\Y11-BASE-001-R1_20260917_203044_883451\\args.yaml",
  "git_clean": true,
  "dataset_unchanged": true,
  "registry_synchronized": true
}
```

完成后停止，不自动继续其他实验。
