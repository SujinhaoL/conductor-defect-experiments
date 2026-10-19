"""Finalize visually reviewed saved results and preserve pre-registration files. No inference."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent/'baseline_analysis'
RUN = ROOT/'workspace/runs/FRCNN-R50FPN-SCALE1280-VAL-001/step11_31112_14115'
DEST = RUN/'test_selected_epoch5/step15_23467_7311'
AUDIT = ROOT/'workspace/audit/step15_review'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


review = json.loads((AUDIT/'acceptance.json').read_text(encoding='utf-8'))
assert review['status'] == 'PASS' and review['results_recovered']
review.update(visual_review_pending=False, visual_review_completed=True,
    visual_review_note='Assistant inspected the contact sheet of all23 saved test visualizations. No visible boxes matches all2300 scores<=0.30. File decoding/dimensions and original coordinates were checked separately; no inference repeated.')
save(AUDIT/'acceptance.json', review)
note_path = ROOT/'workspace/notes/FRCNN-R50FPN-SCALE1280-VAL-001.md'
note = note_path.read_text(encoding='utf-8')
note = note.replace('状态：训练及验证结果已复核；本次test未运行。', '状态：训练、验证及固定第5轮test结果均已复核；STEP15末尾归档错误已通过保存结果恢复，未重跑推理。')
note = note.replace('普通Precision/Recall未计算，AR100单独报告；test未评估，留空，不填0。', '普通Precision/Recall未计算，AR100单独报告。补测test的结果与异常记录见末尾STEP15部分。')
note = note.replace('停止机械递增尺寸或立即跑test。下一阶段建议仅在固定原train8上做已训练模型的RPN→ROI诊断，定位4个未覆盖框；不优化权重，不增加训练，不拿test调参。旧test已被查看，不再称完全未接触盲测。', '训练阶段结束时没有运行test；后续RPN→ROI诊断、anchor对照与最后一次1600尺寸对照已分别归档。用户于2026-10-14指定固定已有权重补齐最终test，第5轮补测已核查。没有根据此次test改选epoch或调整预测阈值。旧test已被查看，不再称完全未接触盲测。')
if '## STEP15 固定第5轮最终test' not in note:
    note += f'''
## STEP15 固定第5轮最终test

用户于2026-10-14启动原test23图/36框（broken8、loose28）评价。test运行编号step15_23467_7311。保持Resize(2133,1280)、keep_ratio=True、默认anchor8、FP32、score_thr0.001、ROI NMS0.5、max100。选模仍为此前val确定的第5轮，348项权重严格加载核查通过。

|指标|val：37框|test：36框|
|---|---:|---:|
|mAP50|3.189325%|0.819432%|
|mAP50-95|1.487789%|0.286644%|
|COCO AR100|11.607143%|6.250000%|
|broken AP50|0.000000%|0.323806%|
|broken AP50-95|0.000000%|0.032381%|
|loose AP50|6.378650%|1.315058%|
|loose AP50-95|2.975578%|0.540908%|

上表均为已保存预测的精确COCO指标。原生test mAP50=0.008、mAP50-95=0.003，类别AP50-95舍入为broken0.000、loose0.005；舍入零不表示精确零。普通P/R未计算，不填0。COCO AR100不是普通Recall。

保存2300个低阈值预测，每图100个；均不超过显示阈值0.30，所以23张预测图没有可见框。已核对类别、原图坐标、COCO导出一致性、23图完整解码和哈希，独立COCO复算与原保存指标一致。助手已查看保存图总览；无可见框不等于模型无预测，也不宣称全链路无错。

异常：模型完成23图推理和指标计算后，最终哈希函数接收字符串而非Path对象，导致AttributeError。保留原日志/manifest中的FAIL，不改写历史。已修复当前脚本，并仅根据保存预测完成CPU复核、补齐test_summary与证据manifest，没有重新模型推理或训练。原始运行未及时落盘的推理耗时和峰值显存留空，不估算或填0；训练耗时/显存记录完整保留。

结果显示该固定模型的test AP仍很低，不能据此确定唯一原因。与val数值不同是不同集合的观测，不能混成同一评价结果；单seed、小test、旧test已有使用的限制保留。本阶段的目的已达成：补齐固定权重最终评价并留存证据。按原先指定顺序下一项是1280 anchor4第15轮，不根据本次分数改选。

- 完整test结果：`{DEST}`
- 精确指标：`{DEST/'test_metrics.json'}`
- 原生指标：`{DEST/'native_coco_metrics.json'}`
- 全部预测：`{DEST/'predictions.bbox.json'}`
- 坐标/类别/分数：`{DEST/'predictions_details.json'}`
- 23张预测图：`{DEST/'predicted_images'}`
- 恢复记录：`{DEST/'postprocessing_recovery.json'}`
- 原完整终端日志：`{ROOT/'workspace/logs/step15_23467_7311/step15_console.txt'}`
- 核查：`{AUDIT/'acceptance.json'}`
- 独立复算：`{AUDIT/'independent_coco_check.txt'}`
- 保存图总览：`{AUDIT/'saved_test_visualizations.jpg'}`
'''
note_path.write_text(note, encoding='utf-8')
backup = BASE/'audit/step15_registration_before'
backup.mkdir(exist_ok=True)
manifest = {}
for source in [BASE/'10_all_experiments_metrics.csv', BASE/'outputs/模型实验结果总表.xlsx',
               BASE/'11_all_experiments_summary_for_teacher.md', BASE/'audit/workbook_plan.json']:
    target = backup/source.name
    if not target.exists():
        shutil.copy2(source, target)
    manifest[str(source)] = dict(backup=str(target), sha256=sha(target))
save(backup/'backup_manifest.json', manifest)
progress = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
    fixed_queue=str(BASE/'audit/test_completion_review_20261014/fixed_test_queue.json'),
    completed=[dict(order=1, experiment_id='FRCNN-R50FPN-SCALE1280-VAL-001', epoch=5,
                    result_dir=str(DEST), status='RESULT_REVIEW_PASS_REGISTRATION_PENDING')],
    next_order=2, training_started=False, assistant_model_inference_performed=False)
save(BASE/'audit/test_completion_review_20261014/test_completion_progress.json', progress)
print('STEP15_RESULT_REVIEW_FINALIZED; originals preserved; registration backup saved; no inference/training')
