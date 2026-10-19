# RTD-SENS-IMGSZ1600-001 sensitivity record

## Experiment definition

- Experiment ID: `RTD-SENS-IMGSZ1600-001`
- Status: `Completed`
- Purpose: Input-resolution sensitivity experiment for small conductor defects.
- Baseline reference: `RTD-BASE-001-R1`
- Experiment type: `Sensitivity / Diagnostic`
- Model: official Ultralytics RT-DETR-L
- Pretrained weights: `D:\小论文\model_projects\RT-DETR\workspace\weights\rtdetr-l.pt`
- Pretrained weights SHA256: `6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f`
- Dataset: `D:\小论文\conductor_defect_dataset\02_yolo11_dataset`
- Dataset YAML: `D:\小论文\model_projects\RT-DETR\workspace\configs\conductor_bl_v1.yaml`
- Classes: `0=broken; 1=loose`
- Split counts: `train=107; val=23; test=23`
- Epochs planned: `100`
- Image size: `1600`
- Batch: `4`
- Workers: `0`
- Device: `0`
- Seed: `42`
- Deterministic: `False`
- Cache: `False`
- AMP: `True`
- Optimizer: `auto`
- Network structure modified: `NO`
- Only planned training-variable change from baseline: `imgsz 1280 -> 1600`
- Expected run directory: `D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001`
- Test executed: `NO (prohibited for this sensitivity experiment)`

## Planned command

```cmd
yolo detect train model="D:\小论文\model_projects\RT-DETR\workspace\weights\rtdetr-l.pt" data="D:\小论文\model_projects\RT-DETR\workspace\configs\conductor_bl_v1.yaml" epochs=100 imgsz=1600 batch=4 workers=0 device=0 seed=42 deterministic=False pretrained=True cache=False amp=True plots=True optimizer=auto patience=100 nbs=64 mosaic=1.0 close_mosaic=10 mixup=0.0 cutmix=0.0 fliplr=0.5 translate=0.1 scale=0.5 val=True split=val project="D:\小论文\model_projects\RT-DETR\workspace\runs" name=RTD-SENS-IMGSZ1600-001 exist_ok=False
```

## Results

- Completed all 100 epochs: `YES`
- Epochs completed: `100`
- Best epoch: `42` (highest `metrics/mAP50-95(B)` in `results.csv`; the same metrics are stored in `best.pt`)
- Training runtime: `35422.4 seconds` (about `9 h 50 min 22 sec`; terminal summary: `9.840 hours`)
- Peak VRAM: `18.2 G` (maximum `GPU_mem` value displayed by Ultralytics; not an independently sampled hardware peak)
- Val Precision: `0.533` (automatic post-training revalidation of `best.pt`)
- Val Recall: `0.0357` (automatic post-training revalidation of `best.pt`)
- Val mAP50: `0.0211` (automatic post-training revalidation of `best.pt`)
- Val mAP50-95: `0.00998` (automatic post-training revalidation of `best.pt`)
- Broken AP50: `0.0000`
- Broken AP50-95: `0.0000`
- Loose AP50: `0.0422`
- Loose AP50-95: `0.0200`
- Actual run directory: `D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001`
- best.pt path: `D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-SENS-IMGSZ1600-001\weights\best.pt`
- best.pt SHA256: `29d23c8bb93bd650800ea439c9f43755b81f1763d0550ce4e14119dee4121a13`
- OOM: `NO`
- NaN/Inf: Training loss values remained finite. Validation loss fields were `NaN` at epoch `7`, then recovered; the final `best.pt` validation metrics were finite.
- Test executed: `NO`
- Notes: The epoch-42 selection metrics in `results.csv` and the stripped `best.pt` metadata are Precision=`0.53552`, Recall=`0.03571`, mAP50=`0.02292`, and mAP50-95=`0.01197`. Ultralytics then automatically revalidated `best.pt` and reported the primary values recorded above. The small difference is retained rather than selecting only the higher result. Relative to the 1280 RT-DETR baseline, overall validation AP improved, but the gain came entirely from `loose`; `broken` AP remained zero. The normalized confusion matrix at its plotting threshold placed both classes in background, indicating severe missed detection. Runtime and displayed memory pressure increased sharply, so 1600 is not an efficient default setting on the current hardware.

## Comparison with RTD-BASE-001-R1 validation

| Metric | Baseline (`imgsz=1280`) | Sensitivity (`imgsz=1600`) | Result |
|---|---:|---:|---|
| Precision | 0.508274 | 0.533 | Increased slightly |
| Recall | 0.017857 | 0.0357 | Approximately doubled, but still very low |
| mAP50 | 0.018199 | 0.0211 | Increased by about 15.9% |
| mAP50-95 | 0.005388 | 0.00998 | Increased by about 85.2% |
| Broken AP50 | 0.000000 | 0.000000 | No improvement |
| Broken AP50-95 | 0.000000 | 0.000000 | No improvement |
| Loose AP50 | 0.036397 | 0.0422 | Increased by about 15.9% |
| Loose AP50-95 | 0.010775 | 0.0200 | Increased by about 85.6% |

## Stop rule

If `CUDA out of memory` occurs at `imgsz=1600, batch=4`, mark the run incomplete and stop. Do not automatically reduce batch size, change image size, disable AMP, modify the model, resume, or try another setting.
