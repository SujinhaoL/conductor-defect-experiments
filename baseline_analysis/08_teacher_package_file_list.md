# 导师材料人工压缩清单

不要自动复制或打包。建议人工建立以下压缩包内分类：

- `00_总说明`
- `01_数据集与划分`
- `02_YOLO11_FormalBaseline`
- `03_RT-DETR_FormalBaseline`
- `04_横向分析与bbox统计`
- `05_实验登记表`

`best.pt` 默认不放入压缩包。报告中保留路径和 SHA256；只有导师明确要求模型复现时再单独提供权重。不要发送整个图片数据集。

## 文件清单

| 压缩包分类 | 来源部分 | 文件名 | 绝对路径 | 为什么需要 | 必选/可选 | 已定位 |
| --- | --- | --- | --- | --- | --- | --- |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 01_baseline_comparison_report.md | D:\小论文\model_projects\baseline_analysis\01_baseline_comparison_report.md | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 02_baseline_metrics_comparison.csv | D:\小论文\model_projects\baseline_analysis\02_baseline_metrics_comparison.csv | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 03_bbox_size_summary.md | D:\小论文\model_projects\baseline_analysis\03_bbox_size_summary.md | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 04_bbox_object_statistics.csv | D:\小论文\model_projects\baseline_analysis\04_bbox_object_statistics.csv | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 05_bbox_size_distribution.png | D:\小论文\model_projects\baseline_analysis\05_bbox_size_distribution.png | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 06_small_target_examples.png | D:\小论文\model_projects\baseline_analysis\06_small_target_examples.png | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 07_environment_summary.md | D:\小论文\model_projects\baseline_analysis\07_environment_summary.md | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 04_横向分析与bbox统计 | A. 本次横向分析 | 08_teacher_package_file_list.md | D:\小论文\model_projects\baseline_analysis\08_teacher_package_file_list.md | 本次生成的对比、bbox 统计、图表或环境/材料说明。 | 必选 | 是 |
| 01_数据集与划分 | B. 数据集和划分 | data.yaml | D:\小论文\conductor_defect_dataset\02_yolo11_dataset\data.yaml | 正式数据路径和 0=broken、1=loose 类别定义。 | 必选 | 是 |
| 01_数据集与划分 | B. 数据集和划分 | split_manifest.csv | D:\小论文\conductor_defect_dataset\02_yolo11_dataset\split_manifest.csv | 固定 train/val/test 划分清单。 | 必选 | 是 |
| 01_数据集与划分 | B. 数据集和划分 | dataset_report.txt | D:\小论文\conductor_defect_dataset\02_yolo11_dataset\dataset_report.txt | 数据集数量与生成摘要。 | 必选 | 是 |
| 01_数据集与划分 | B. 数据集和划分 | metadata_final.csv | D:\小论文\conductor_defect_dataset\01_master_pool\metadata_final.csv | 主池元数据；仅在导师需要追溯原始样本元信息时提供，不能替代正式 split manifest。 | 可选 | 是 |
| 01_数据集与划分 | B. 数据集和划分 | split_report.txt | D:\小论文\conductor_defect_dataset\01_master_pool\split_report.txt | 固定划分形成过程的历史说明。 | 可选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | run_formal_baseline.py | D:\小论文\model_projects\yolo11\workspace\scripts\run_formal_baseline.py | 正式训练/评估编排脚本。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | conductor_bl_v1.yaml | D:\小论文\model_projects\yolo11\workspace\configs\conductor_bl_v1.yaml | FormalBaseline 数据集配置。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | args.yaml | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\args.yaml | 实际训练参数。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | results.csv | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\results.csv | 100 epochs 逐轮训练结果。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | results.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\results.png | 训练曲线总览。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | PR_curve.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\PR_curve.png | 训练结束时 PR 曲线。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | P_curve.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\P_curve.png | 训练结束时 Precision 曲线。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | R_curve.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\R_curve.png | 训练结束时 Recall 曲线。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | F1_curve.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\F1_curve.png | 训练结束时 F1 曲线。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | confusion_matrix.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\confusion_matrix.png | 训练结束时混淆矩阵。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | val_metrics.json | D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\val_metrics.json | 固定 best.pt 的正式 val 指标。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | test_metrics.json | D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\test_metrics.json | 唯一一次正式 test 指标。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | formal_baseline_single_run_report.md | D:\小论文\model_projects\yolo11\workspace\notes\formal_baseline_single_run_report.md | YOLO FormalBaseline 最终报告。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | formal_baseline_single_run_metrics.json | D:\小论文\model_projects\yolo11\workspace\analysis\formal_baseline_single_run_metrics.json | YOLO FormalBaseline 汇总 metrics JSON。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | status.json | D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\status.json | 成功状态、环境、完整参数和完整性审计。 | 必选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | Y11-BASE-001-R1_20260917_203044_883451_console.log | D:\小论文\model_projects\yolo11\workspace\logs\Y11-BASE-001-R1_20260917_203044_883451\Y11-BASE-001-R1_20260917_203044_883451_console.log | 必要时追溯训练过程。 | 可选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | BoxPR_curve.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\final_val\BoxPR_curve.png | 正式 val PR 曲线。 | 可选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | confusion_matrix.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\final_val\confusion_matrix.png | 正式 val 混淆矩阵。 | 可选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | BoxPR_curve.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\final_test\BoxPR_curve.png | 正式 test PR 曲线。 | 可选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | confusion_matrix.png | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\final_test\confusion_matrix.png | 正式 test 混淆矩阵。 | 可选 | 是 |
| 02_YOLO11_FormalBaseline | C. YOLO11 FormalBaseline | best.pt | D:\小论文\model_projects\yolo11\workspace\runs\Y11-BASE-001-R1_20260917_203044_883451\weights\best.pt | 模型权重；默认不打包。路径与 SHA256=01f6246e45a0b42fe93762120c8784d8a885dcba21567ab5906d94af2edb1cc9 已记录，导师明确要求复现时再提供。 | 可选（默认不打包） | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | train_rtdetr_formal_baseline.py | D:\小论文\model_projects\RT-DETR\workspace\scripts\train_rtdetr_formal_baseline.py | 成功 R1 实际使用的 RT-DETR 训练/评估脚本。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | conductor_bl_v1.yaml | D:\小论文\model_projects\RT-DETR\workspace\configs\conductor_bl_v1.yaml | RT-DETR 数据集配置。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | args.yaml | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\args.yaml | 实际训练参数。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | results.csv | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\results.csv | 100 epochs 逐轮训练结果。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | results.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\results.png | 训练曲线总览。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | BoxPR_curve.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\BoxPR_curve.png | 训练结束时 PR 曲线。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | BoxP_curve.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\BoxP_curve.png | 训练结束时 Precision 曲线。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | BoxR_curve.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\BoxR_curve.png | 训练结束时 Recall 曲线。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | BoxF1_curve.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\BoxF1_curve.png | 训练结束时 F1 曲线。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | confusion_matrix.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\confusion_matrix.png | 训练结束时混淆矩阵。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | formal_val_metrics.json | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_val_metrics.json | 固定 best.pt 的正式 val 指标。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | formal_test_metrics.json | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_test_metrics.json | 唯一一次正式 test 指标。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | rtdetr_single_run_report.md | D:\小论文\model_projects\RT-DETR\workspace\notes\rtdetr_single_run_report.md | RT-DETR FormalBaseline 最终报告。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | rtdetr_single_run_metrics.json | D:\小论文\model_projects\RT-DETR\workspace\analysis\rtdetr_single_run_metrics.json | RT-DETR 汇总 metrics JSON。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | RTD-BASE-001-R1_status.json | D:\小论文\model_projects\RT-DETR\workspace\analysis\RTD-BASE-001-R1_status.json | 成功状态与 test 次数记录。 | 必选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | RTD-BASE-001-R1_20260918_172120.log | D:\小论文\model_projects\RT-DETR\workspace\logs\RTD-BASE-001-R1_20260918_172120.log | 必要时追溯训练过程。 | 可选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | BoxPR_curve.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_val\BoxPR_curve.png | 正式 val PR 曲线。 | 可选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | confusion_matrix.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_val\confusion_matrix.png | 正式 val 混淆矩阵。 | 可选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | BoxPR_curve.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_test\BoxPR_curve.png | 正式 test PR 曲线。 | 可选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | confusion_matrix.png | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\formal_test\confusion_matrix.png | 正式 test 混淆矩阵。 | 可选 | 是 |
| 03_RT-DETR_FormalBaseline | D. RT-DETR FormalBaseline | best.pt | D:\小论文\model_projects\RT-DETR\workspace\runs\RTD-BASE-001-R1_20260918_172120\weights\best.pt | 模型权重；默认不打包。路径与 SHA256=15ba1e62e1170d21e0628753be3eef0a710e06300643b3f9dd421f2c02d53ccd 已记录，导师明确要求复现时再提供。 | 可选（默认不打包） | 是 |
| 05_实验登记表 | E. 实验总登记表 | experiment_registry.xlsx | D:\小论文\model_projects\yolo11\workspace\experiment_tracking\experiment_registry.xlsx | 论文唯一实验总账；包含且仅各一条成功 FormalBaseline R1 记录。 | 必选 | 是 |
| 05_实验登记表 | E. 实验总登记表 | experiment_registry.csv | D:\小论文\model_projects\yolo11\workspace\experiment_tracking\experiment_registry.csv | 实验总账的可读 CSV 版本。 | 必选 | 是 |

## 登记表核验

- `Y11-BASE-001-R1`：FormalBaseline / Completed，唯一一条。
- `RTD-BASE-001-R1`：FormalBaseline / Completed，唯一一条。
- 本次分析没有修改登记表。
