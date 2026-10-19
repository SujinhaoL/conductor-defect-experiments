# RT-DETR single-run report

- Experiment: RTD-BASE-001-R1
- Stage: FormalBaseline
- Model: RT-DETR-L (official Ultralytics implementation)
- Pretrained weights: D:\小论文\model_projects\RT-DETR\workspace\weights\rtdetr-l.pt
- Pretrained weights SHA256: 6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f
- Network structure modified: No
- Epochs: 100
- Image size: 1280
- Batch: 4
- Seed: 42
- Deterministic: False
- Best epoch: 71
- Training runtime (wall seconds): 2593.096
- Total runtime (wall seconds): 2613.508
- Peak CUDA reserved VRAM (GiB): 11.279
- Peak CUDA allocated VRAM (GiB): 10.300
- Parameters: 32810186
- GFLOPs at imgsz=1280: 426.771824
- best.pt: D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\weights\best.pt
- best.pt SHA256: 15ba1e62e1170d21e0628753be3eef0a710e06300643b3f9dd421f2c02d53ccd
- Run directory: D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120
- Test executions: 1
- Experiment registry appended: True (RTD-BASE-001-R1)
- Git tracked source status clean after experiment: True
- Formal images and .txt labels unchanged: True
- Generated label caches: labels/test.cache, labels/train.cache, labels/val.cache
- Generated label caches cleaned: True

| split/class | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| val overall | 0.508274 | 0.017857 | 0.018199 | 0.005388 |
| val broken | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| val loose | 0.016548 | 0.035714 | 0.036397 | 0.010775 |
| test overall | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| test broken | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| test loose | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
