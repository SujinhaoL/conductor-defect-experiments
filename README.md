# 输电导线缺陷检测实验

用于管理本项目的自定义代码、配置、实验说明和指标来源。GitHub仓库保持私有：https://github.com/SujinhaoL/conductor-defect-experiments 。首次版本记录从2026-10-19开始，既往实验依靠原始日志、冻结配置和文件哈希追溯，不补造提交历史。

本机项目根目录：`D:\小论文\model_projects`。请先阅读 [AGENTS.md](AGENTS.md) 的最新约定，再阅读 [版本管理入门](docs/版本管理入门.md)。

## 当前进度

已有9次正式实验；7次已有test结果，YOLO11s 1600和RT-DETR-L 1600尚未test，用户已暂不继续补测。当前围绕Faster R-CNN 1280默认anchor的一个既有运行排查。DIAG01保存预测分析已通过用户核验；下一步是输入数据管线检查，不是追加训练或修改尺寸。状态以AGENTS及最新实际记录为准。

正式运行索引：[run_status_index.json](baseline_analysis/audit/run_status_index.json)。该索引保留原状态字面值；未测项目“暂停”的决定见AGENTS及补测进度记录。

## 从哪里看

- [最新教师讨论报告](baseline_analysis/11_all_experiments_summary_for_teacher.md) 与 [统一指标CSV](baseline_analysis/10_all_experiments_metrics.csv)：区分val/test；比例值为0到1，空值不等于0。
- [Faster R-CNN排查路线](Faster%20R-CNN/workspace/notes/FRCNN1280_排查路线.md) 和 [第一步代码导读](Faster%20R-CNN/workspace/notes/DIAG01_低分预测排查与代码导读.md)。
- `Faster R-CNN/workspace/scripts`、`configs`、`notes`：项目侧工具、配置和实验说明；YOLO11及RT-DETR各自目录结构相同。
- [官方代码与本地依赖来源](provenance/local_dependencies.json)：官方源码commit、COCO初始化来源和环境位置；仓库不打包这些依赖。
- [首次纳入文件哈希清单](provenance/initial_snapshot_manifest.json)：记录建仓时纳入的现有文件，可核对具体版本。

## 保存范围

跟踪自定义代码/配置/说明、已生成的汇总CSV、小型运行参数、冻结代码及checkpoint来源/哈希记录。旧STEP脚本保留用于追溯，不要求用户依次重跑。新增同类实验优先复用已有工具、另建小配置和独立结果目录。

原图/标签、虚拟环境、官方源码克隆、权重、完整日志/预测/可视化、安装器及Excel文件留在本机，不上传。GitHub是代码与记录的副本，**不是完整实验备份**。恢复整个实验仍需另行保存这些本地大文件；此处不自动打包或迁移它们。

仓库保留既有Windows绝对路径，不承诺下载后直接运行。某些配置继承了旧运行内的小配置，因此这些依赖文件也按原路径纳入。官方代码保持独立仓库，版本固定，不作为子模块上传。

## 运行原则

用户在VS Code的Windows cmd终端启动训练/测试；助手准备、检查和讲解，禁止自行启动。已有结果足以核查时不重新推理。模型环境保持原对应关系，Faster使用 `D:\小论文\model_projects\Faster R-CNN\.venv\Scripts\python.exe`。不要因为新建Git仓库重装环境。

原始153张图的train107/val23/test23划分不变。已有test已在项目中查看，后续开发依据train/val，保留test使用历史，不声称严格盲测。Git目录隔离不能消除这段历史。
