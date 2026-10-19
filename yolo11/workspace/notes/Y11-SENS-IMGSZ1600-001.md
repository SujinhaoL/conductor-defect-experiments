# Y11-SENS-IMGSZ1600-001 sensitivity record

## Experiment definition

- Experiment ID: `Y11-SENS-IMGSZ1600-001`
- Status: `Completed`
- Purpose: Input-resolution sensitivity experiment for small conductor defects.
- Baseline reference: `Y11-BASE-001-R1`
- Experiment type: `Sensitivity / Diagnostic`
- Model: official YOLO11s
- Pretrained weights: `D:\小论文\model_projects\yolo11\workspace\weights\yolo11s.pt`
- Pretrained weights SHA256: `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`
- Dataset: `D:\小论文\conductor_defect_dataset\02_yolo11_dataset`
- Dataset YAML: `D:\小论文\model_projects\yolo11\workspace\configs\conductor_bl_v1.yaml`
- Classes: `0=broken; 1=loose`
- Split counts: `train=107; val=23; test=23`
- Epochs planned: `100`
- Image size: `1600`
- Batch: `4`
- Workers: `0`
- Device: `0`
- Seed: `42`
- Deterministic: `True`
- Cache: `False`
- AMP: `True`
- Optimizer: `auto`
- Network structure modified: `NO`
- Only planned training-variable change from baseline: `imgsz 1280 -> 1600`
- Expected run directory: `D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001`
- Test executed: `NO (prohibited for this sensitivity experiment)`

## Planned command

```cmd
yolo detect train model="D:\小论文\model_projects\yolo11\workspace\weights\yolo11s.pt" data="D:\小论文\model_projects\yolo11\workspace\configs\conductor_bl_v1.yaml" epochs=100 imgsz=1600 batch=4 workers=0 device=0 seed=42 deterministic=True pretrained=True cache=False amp=True plots=True optimizer=auto patience=100 nbs=64 mosaic=1.0 close_mosaic=10 mixup=0.0 cutmix=0.0 fliplr=0.5 translate=0.1 scale=0.5 val=True split=val project="D:\小论文\model_projects\yolo11\workspace\runs" name=Y11-SENS-IMGSZ1600-001 exist_ok=False
```

## Results

- Completed all 100 epochs: `YES`
- Epochs completed: `100`
- Best epoch: `54` (selected by the highest `metrics/mAP50-95(B)`; `metrics/mAP50(B)` is the tie-breaker)
- Training runtime: `1839.24 seconds` (about `30 min 39 sec`; terminal summary: `0.511 hours`)
- Peak VRAM: `7.61 GiB` (maximum `GPU_mem` value displayed by Ultralytics during training; not an independently sampled hardware peak)
- Val Precision: `0.17132`
- Val Recall: `0.05556`
- Val mAP50: `0.03002`
- Val mAP50-95: `0.01483`
- Broken AP50: `0.0575`
- Broken AP50-95: `0.0287`
- Loose AP50: `0.00252`
- Loose AP50-95: `0.000907`
- Actual run directory: `D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001`
- best.pt path: `D:\小论文\model_projects\yolo11\workspace\runs\Y11-SENS-IMGSZ1600-001\weights\best.pt`
- best.pt SHA256: `2cc998a09da3eb36d717df79dfa28adf2a41cebb1121ecd43d984ff32ea4c92e`
- OOM: `NO`
- NaN/Inf: Training loss values remained finite. Validation loss fields were `NaN` at epochs `4` and `6`, then recovered; the final `best.pt` validation metrics were finite.
- Test executed: `NO`
- Notes: Overall validation mAP improved relative to the 1280 baseline, driven mainly by the `broken` class. Recall decreased and the `loose` class AP deteriorated substantially, so increasing `imgsz` to 1600 did not produce a uniform improvement. Metrics were volatile; the recorded validation result is from `best.pt` at epoch 54, not the final epoch. Ultralytics generated `train.cache` and `val.cache`; no `test.cache` was observed.

## Comparison with Y11-BASE-001-R1 validation

| Metric | Baseline (`imgsz=1280`) | Sensitivity (`imgsz=1600`) | Result |
|---|---:|---:|---|
| Precision | 0.01296 | 0.17132 | Increased |
| Recall | 0.23413 | 0.05556 | Decreased |
| mAP50 | 0.02226 | 0.03002 | Increased by about 34.9% |
| mAP50-95 | 0.00439 | 0.01483 | Increased by about 238% |
| Broken AP50 | 0.01278 | 0.05750 | Increased |
| Broken AP50-95 | 0.00256 | 0.02870 | Increased |
| Loose AP50 | 0.03175 | 0.00252 | Decreased by about 92% |
| Loose AP50-95 | 0.00622 | 0.000907 | Decreased by about 85% |

## Stop rule

If `CUDA out of memory` occurs at `imgsz=1600, batch=4`, mark the run incomplete and stop. Do not automatically reduce batch size, change image size, disable AMP, modify the model, resume, or try another setting.
