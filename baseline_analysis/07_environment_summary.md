# FormalBaseline 环境摘要

以下信息来自两个成功 FormalBaseline 的现有 status/metrics 文件和锁定环境记录。未安装或升级任何软件。

| Experiment | Y11-BASE-001-R1 | RTD-BASE-001-R1 |
| --- | --- | --- |
| Model | YOLO11s | RT-DETR-L |
| Python | 3.12.13 | 3.12.13 |
| PyTorch | 2.8.0+cu128 | 2.8.0+cu128 |
| CUDA | 12.8 | 12.8 |
| GPU | NVIDIA GeForce RTX 5070 Ti | NVIDIA GeForce RTX 5070 Ti |
| Ultralytics | 8.4.152 | 8.4.152 |
| git commit | 9aa41d679d121afb8b4cc512d7a162f4f36886c3 | 9aa41d679d121afb8b4cc512d7a162f4f36886c3 |
| Official pretrained weight | yolo11s.pt | rtdetr-l.pt |
| Pretrained SHA256 | 85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5 | 6de60b10d4bc566f00cda0f5b4d64afe4b66d48dc9695d2171effb7859d8e73f |
| Network modified | No | No |

## 说明

- 两个实验复用了同一 Ultralytics 8.4.152 editable 源码和同一 git commit。
- YOLO11s 使用官方 `yolo11s.pt`；RT-DETR-L 使用官方 `rtdetr-l.pt`。
- 两个模型均未修改网络结构。
- RT-DETR 的 deterministic=False 是已记录的模型运行约束；YOLO11 为 deterministic=True。
