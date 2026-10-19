"""Record the human-visible audit of STEP11; no detector operations."""
from datetime import datetime
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/'workspace/audit/step11_review'
BASE=ROOT.parent/'baseline_analysis'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
a=read(AUDIT/'acceptance.json'); run=Path(a['run_dir'])
assert a['status']=='PASS' and a['selected_epoch']==5
backup=AUDIT/'records_before_update'
backup.mkdir(exist_ok=True)
for src in [BASE/'10_all_experiments_metrics.csv',BASE/'outputs/模型实验结果总表.xlsx',BASE/'11_all_experiments_summary_for_teacher.md',ROOT.parent/'AGENTS.md']:
    target=backup/src.name
    if not target.exists(): shutil.copy2(src,target)
a.update(visual_review_pending=False,visual_review_completed=True,
    visual_review_note='Assistant inspected the new-vs-baseline trajectory plot, epoch100 fixed-train8 contact sheet and selected-val GT/display overlay. Selected-val display has zero predictions above0.30; finite low-threshold predictions and exact COCO checked separately.')
(AUDIT/'acceptance.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
m,b=a['selected_metrics'],a['baseline_selected_metrics']; p=read(run/'pretraining_record.json'); s=read(run/'selected_checkpoint.json')
lines=['# FRCNN-R50FPN-SCALE1280-VAL-001','',
'状态：训练及验证结果已复核；本次test未运行。原始运行文件保持原样，其REVIEW_PENDING是运行时记录，最新状态见本说明及audit/step11_review/acceptance.json。','',
f'运行编号：`{run.name}`。原始目录：`{run}`。','',
'## 目的与实际设置','',
'检验Faster R-CNN仅提高整图输入分辨率的影响。对照VAL-001，原train107/168框（broken39、loose129）、val23/37框（9、28），test23/36框未参与本轮。类别YOLO0/1→COCO1/2→模型0/1，num_classes=2。','',
'实际训练、验证及固定训练图快照均Resize(2133,1280)、keep_ratio=True；batch2、FP32、workers0、seed42、100epoch、5400更新。SGD lr0.0025/momentum0.9/weight_decay0.0001；warmup500，milestones67/92、gamma0.1；flip0.5。配置还原尺寸后与冻结原基线一致，5400条实际学习率轨迹也一致。','',
f'同一官方完整COCO检测器重新初始化：`{p["checkpoint"]["path"]}`；来源：{p["checkpoint"]["url"]}；SHA256：`{p["checkpoint"]["sha256"]}`。344项兼容权重逐值加载；仅ROI分类/回归输出的weight/bias四项因80→2类别重新初始化，不是只加载ImageNet骨干。','',
f'耗时（含val、保存和固定图快照）{a["wall_seconds"]:.3f}秒，约{a["wall_seconds"]/60:.2f}分钟；峰值CUDA allocated {a["peak_cuda_allocated_mib"]:.3f}MiB（约{a["peak_cuda_allocated_mib"]/1024:.2f}GiB），不是设备总占用。原基线对应2193.604秒、3769.939MiB。','',
'## 模型选择与原始结果','',
f'按每轮原生val coco/bbox_mAP（三位小数）最大值、同值保留首次，选定第{s["selected_epoch"]}轮。不得按mAP50或完整精度复算另挑轮次。最佳权重：`{s["best_checkpoint"]}`；SHA256：`{s["best_checkpoint_sha256"]}`。末轮权重：`{s["last_checkpoint"]}`；SHA256：`{s["last_checkpoint_sha256"]}`。','',
'|指标（23张val）|原800基线，第66轮|新1280，第5轮|','|---|---:|---:|']
for k,name in [('mAP50','mAP50'),('mAP50_95','mAP50-95'),('AR100','COCO AR100')]: lines.append(f'|{name}|{b[k]*100:.6f}%|{m[k]*100:.6f}%|')
for c in ['broken','loose']:
    for k in ['AP50','AP50_95']:lines.append(f'|{c} {k}|{b["per_class"][c][k]*100:.6f}%|{m["per_class"][c][k]*100:.6f}%|')
lines += ['', '原生mAP50=0.032、mAP50-95=0.015；上表为相同保存预测的精确COCO复算。普通Precision/Recall未计算，AR100单独报告；test未评估，留空，不填0。','',
'## 训练过程与可视化发现','',
f'5400条损失及分项全部有限。首100次总损失均值{a["loss_first100"]:.8f}，末100次{a["loss_last100"]:.8f}。固定train8/15GT末轮mAP50-95：原800为{a["baseline_train8_final"]["mAP50_95"]*100:.6f}%，1280为{a["train8_final"]["mAP50_95"]*100:.6f}%。本次broken训练AP为100%，loose约60.396%；11/15GT有同类候选IoU≥0.50，另外4个loose仍未覆盖。该覆盖数不等于普通Recall，也不是RPN proposal recall。','',
f'第100轮val精确mAP50={a["final_val_metrics"]["mAP50"]*100:.6f}%、mAP50-95={a["final_val_metrics"]["mAP50_95"]*100:.6f}%，低于选定第5轮。训练拟合增强而验证仍低且后期下降，是泛化问题/过拟合的迹象，不能单凭此确定唯一机制。','',
'选定第5轮保存2300个预测，23张各100个，均未超过显示阈值0.30；AP过滤阈值是0.001。图上无红框不代表零预测。检查了原图坐标范围、类别映射、全100轮文件和固定train8详细预测导出。loss与预测图已目视检查，但没有全量逐层特征/RPN/ROI内部采样检查。','',
'## 结论与下一步边界','',
f'本次完成了预定尺寸敏感性对照；mAP50-95变化{(m["mAP50_95"]-b["mAP50_95"])*100:+.6f}个百分点，mAP50变化{(m["mAP50"]-b["mAP50"])*100:+.6f}个百分点。不能说放大尺寸已解决问题、也不能证明尺寸完全无影响。单seed、23张val不足以宣称显著性；本轮同时提高训练和验证尺度，不区分二者贡献。','',
'停止机械递增尺寸或立即跑test。下一阶段建议仅在固定原train8上做已训练模型的RPN→ROI诊断，定位4个未覆盖框；不优化权重，不增加训练，不拿test调参。旧test已被查看，不再称完全未接触盲测。','',
'## 证据完整路径','']
for label,path in [('运行配置',run/'resolved_scale1280_val_001.py'),('依赖',run/'requirements_freeze.txt'),('初始化/数据/归档哈希',run/'pretraining_record.json'),('实际加载',run/'artifacts/actual_training_weight_load.json'),('损失与学习率',run/'artifacts/loss_curves.png'),('固定训练图末轮',run/'artifacts/epoch_100/predictions_contact_sheet.jpg'),('选定验证预测',Path(s['validation_predictions'])),('精确指标及checkpoint',run/'selected_checkpoint.json'),('完整日志',ROOT/'workspace/logs'/run.name/'step11_console.txt'),('独立复核',AUDIT/'acceptance.json'),('训练验证对照曲线',AUDIT/'resolution_training_validation_comparison.png'),('验证GT与显示预测',AUDIT/'selected_val_gt_green_pred_red.jpg'),('逐GT诊断',AUDIT/'train8_epoch100_per_gt.csv')]:
    assert path.exists(),path
    lines.append(f'- {label}：`{path}`')
note=ROOT/'workspace/notes/FRCNN-R50FPN-SCALE1280-VAL-001.md'
note.write_text('\n'.join(lines)+'\n',encoding='utf-8')
(ROOT/'workspace/notes/STEP11_result_review.txt').write_text(f'STEP11_RESULT_REVIEW_PASS\nRun: {run}\nSelected epoch: 5\nVal mAP50: {m["mAP50"]}\nVal mAP50-95: {m["mAP50_95"]}\nFull note: {note}\nNo training or model inference performed during review. Test not run.\n',encoding='utf-8')
print('STEP11_NOTES_AND_VISUAL_REVIEW_RECORDED')
