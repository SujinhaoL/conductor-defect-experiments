"""Record the completed saved-output diagnosis, leaving formal CSV/XLSX unchanged."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT.parent / 'baseline_analysis'
AUDIT = ROOT / 'workspace/audit/step12_review'


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    a = read(AUDIT / 'acceptance.json')
    assert a['status'] == 'PASS' and a['visual_review_completed']
    run = Path(a['run_dir'])
    backup = AUDIT / 'records_before_update'
    backup.mkdir(exist_ok=True)
    receipt_path = BASE / 'audit/registration_acceptance.json'
    index_path = BASE / 'audit/run_status_index.json'
    report_path = BASE / '11_all_experiments_summary_for_teacher.md'
    csv_path = BASE / '10_all_experiments_metrics.csv'
    book_path = BASE / 'outputs/模型实验结果总表.xlsx'
    for p in (receipt_path, index_path, report_path, ROOT.parent / 'AGENTS.md'):
        if not (backup / p.name).exists():
            shutil.copy2(p, backup / p.name)
    receipt = read(receipt_path)
    assert receipt['status'] == 'PASS'
    assert sha(csv_path) == receipt['csv_sha256'] and sha(book_path) == receipt['excel_sha256']
    before_hashes = {str(p): sha(p) for p in (csv_path, book_path)}
    rows = list(csv.DictReader((AUDIT / 'anchor_proposal_roi_comparison.csv').open(encoding='utf-8-sig', newline='')))
    missing = [r for r in rows if r['model'] == 'scale1280' and r['diagnostic_flag'] == 'NO_RPN_IOU50']
    assert {int(r['annotation_id']) for r in missing} == {87, 123, 124, 160}
    lines = ['# FRCNN-DIAG-RPNROI-TRAIN8', '',
        '状态：STEP12用户运行成功；保存结果独立复核PASS。属于训练集流程诊断，不是正式泛化实验。原始summary.json中的REVIEW_PENDING保留为运行时历史状态，当前以本说明和独立acceptance.json为准。', '',
        f'运行编号：`{run.name}`。原始目录：`{run}`。完整终端日志：`{ROOT / "workspace/logs" / run.name / "step12_console.txt"}`。', '',
        '## 目的和范围', '',
        '定位800→1280后固定训练图仍有4个目标没有足够重合的最终同类框，查看RPN输出、ROI类别回归及过滤环节。采用原107张训练运行的epoch100权重（VAL-001和SCALE1280-VAL-001），同一原train8/15GT，各推理8张。不是8图重新训练，不做优化器更新，不使用val/test。原val最佳epoch66/5未改变。', '',
        '类别COCO1=broken/2=loose，模型0/1，ROI第3个概率为背景。显示阈值0.30与评价过滤阈值0.001分开；局部诊断图的红框是最终返回框中与该GT同类IoU最高的框，不另用0.30筛选。', '',
        '## 已核验的事实', '',
        '两份checkpoint各348项严格加载，哈希/配置对应。16份NPZ保存数组均独立复算：RPN坐标N×4，ROI类别相关回归N×2×4，分类概率N×3；类别、有限值、原图坐标边界、FPN层尺寸通过。诊断最终预测与既有epoch100快照逐项比较通过（原图框绝对容差0.1像素，score绝对容差1e-4）。没有为复核再次运行模型。', '',
        '|模型尺寸|GT总数|RPN候选覆盖|ROI同类回归、过滤前覆盖|最终同类框覆盖|',
        '|---|---:|---:|---:|---:|']
    for m in a['models']:
        lines.append(f'|{m["label"]}|15|{m["rpn_covered"]}|{m["roi_before_score_covered"]}|{m["final_covered"]}|')
    lines += ['', '覆盖定义为每个GT是否存在IoU≥0.50候选；不做一对一检测匹配，不考虑完整精确率曲线。因此不是普通Recall或COCO AR100，也不是验证/测试AP。800→1280恢复了GT6/18/122，其余原8个覆盖目标保留。', '',
        '|1280未覆盖GT|图片|输入中的框宽×高（像素）|默认anchor最大IoU|RPN最大IoU|ROI同类过滤前最大IoU|',
        '|---|---|---:|---:|---:|---:|']
    for r in missing:
        lines.append(f'|{r["annotation_id"]}（loose）|{r["file_name"]}|{float(r["resized_width"]):.2f}×{float(r["resized_height"]):.2f}|{float(r["default_anchor_max_iou"]):.4f}|{float(r["rpn_max_iou"]):.4f}|{float(r["roi_same_class_max_iou_before_filter"]):.4f}|')
    lines += ['',
        '4个目标在RPN传给ROI的候选中均无IoU≥0.50框，ROI同类回归在过滤前最大IoU仍小于0.50。因此不能只靠降低显示阈值或更改最后NMS把这些当前预测变成IoU≥0.50命中。4个目标恰好对应默认anchor几何检查中无正样本的目标，支持进一步检验anchor尺寸匹配。', '',
        '## 尚未证实的解释与可视化边界', '',
        '这不是证实MMDetection实现有bug，也不是所有val/test低分由anchor造成的证明。800下GT88没有正anchor却仍被检测到，说明零正anchor不是数学上的绝对不可检测。ROI训练配置会加入GT作为候选，不能说RPN零正anchor意味着整个模型从未学习该目标。', '',
        '保存的RPN候选已经过RPN自身NMS及top1000筛选，尚未区分原始候选生成、评分、排序与RPN NMS的各自贡献。已保存RPN/ROI中间框数组、每GT局部对照图和FPN形状；没有恢复历史训练的随机ROI采样或保存所有层的完整特征图。', '',
        '已目视检查4个1280漏检局部图及800/1280并排图。绿色GT通常比候选更紧、更小；这说明定位重合不足，不代表标注一定错误。缺陷语义及标注边界标准需领域判断，不凭本次图像改标。', '',
        '## 本阶段结论与下一阶段', '',
        '目的达成：诊断将这4个训练目标的问题定位到RPN交接候选覆盖不足、ROI现有回归没有修复，排除了仅由显示阈值造成的解释。它直接服务于小目标低分问题，没有因低分盲目加尺寸。诊断范围有限，不宣称全链路全部无错。', '',
        '随后仅用原train107/168框做CPU匹配核查：固定1280，anchor scale8→4使无正anchor目标33→3，翻转前后结论一致，原正样本目标无新增遗漏。下一次只准备固定1280、anchor scale4的单因素对照，从同一COCO权重重训107/23，100epoch5400更新，val选模，不做test。具体设置和停止条件见STEP13_anchor4_plan.md；尚未启动，不能记成成绩。', '',
        '## 证据完整路径', '']
    for name, p in [('原始诊断摘要', run / 'summary.json'), ('原800逐GT', run / 'baseline800/per_gt.csv'),
                    ('新1280逐GT', run / 'scale1280/per_gt.csv'), ('独立复算验收', AUDIT / 'acceptance.json'),
                    ('anchor与RPN/ROI联合表', AUDIT / 'anchor_proposal_roi_comparison.csv'),
                    ('四个目标800/1280对照图', AUDIT / 'four_uncovered_targets_comparison.jpg'),
                    ('复算脚本', ROOT / 'workspace/scripts/step12_review_results.py'),
                    ('后续CPU几何核查', ROOT / 'workspace/audit/step13_anchor_geometry/anchor_geometry.json'),
                    ('下一阶段方案', ROOT / 'workspace/notes/STEP13_anchor4_plan.md')]:
        assert p.is_file(), p
        lines.append(f'- {name}：`{p}`')
    note = ROOT / 'workspace/notes/FRCNN-DIAG-RPNROI-TRAIN8.md'
    note.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    helpers = runpy.run_path(str(BASE / 'scripts/refresh_experiment_records.py'))
    index = read(index_path)
    index['diagnostics'] = helpers['diagnostic_entries']()
    save(index_path, index)
    report = report_path.read_text(encoding='utf-8')
    start = '## 固定训练图RPN与ROI诊断（不列入泛化成绩）'
    if start in report:
        left = report.index(start); right = report.index('## 原始证据索引', left)
        report = report[:left] + report[right:]
    report = report.replace('下一步建议针对固定train8检查RPN候选区域及ROI阶段，不机械追加尺寸训练。',
        '固定train8的RPN与ROI后续诊断状态见单独说明，不机械追加尺寸训练。')
    report = report.replace('## 原始证据索引', '\n'.join(helpers['diagnostic_report_lines']()) + '\n## 原始证据索引', 1)
    report_path.write_text(report, encoding='utf-8')
    for p in (csv_path, book_path):
        assert sha(p) == before_hashes[str(p)], 'Formal metric tables must not change for diagnosis'
    # Keep the previous cell-verification timestamp. Only the report hash changes.
    receipt['report_sha256'] = sha(report_path)
    receipt['diagnostic_report_update'] = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        experiment='FRCNN-DIAG-RPNROI-TRAIN8', formal_csv_and_excel_unchanged=True,
        previous_cell_verification_still_applies_to_identical_files=True, acceptance=str(AUDIT / 'acceptance.json'))
    save(receipt_path, receipt)
    save(AUDIT / 'recording_receipt.json', dict(status='PASS', diagnostic_run=str(run),
        note=str(note), note_sha256=sha(note), report_sha256=sha(report_path),
        unchanged_formal_tables_sha256=before_hashes, formal_evaluation_rows=receipt['evaluation_rows'],
        new_formal_evaluation_rows=0, training_started=False, detector_forward_performed=False))
    (ROOT / 'workspace/notes/STEP12_result_review.txt').write_text(
        f'STEP12_RESULT_REVIEW_PASS\nRUN_DIR={run}\nNOTE={note}\nDiagnostic only; formal 16 evaluation rows unchanged.\n', encoding='utf-8')
    print('STEP12_DIAGNOSTIC_RECORDS_PASS; note/report/index updated; formal CSV/Excel hashes unchanged')


if __name__ == '__main__':
    main()
