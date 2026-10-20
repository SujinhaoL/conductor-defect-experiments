"""Trace a fixed validation diagnostic through official MMDetection inference.

No training entry point, optimizer, test-set evaluation, or parameter search.
--check-only never builds a detector. Only the user's --run invokes inference.
"""
import argparse
import builtins
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = ROOT / "workspace/configs/diag04_trace1280_val3.json"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, obj):
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def check(plan_path):
    """Identity, immutable inputs, config and scope only; no detector forward."""
    from mmengine.config import Config

    require(Path(sys.executable).resolve() == (ROOT / ".venv/Scripts/python.exe").resolve(),
            "Use the Faster R-CNN .venv Python")
    plan = read(plan_path)
    require(plan["experiment"] == "FRCNN-DIAG-TRACE1280-001", "Unexpected diagnostic")
    require(plan["epoch"] == 5 and plan["target_gt_ids"] == [172, 194, 171],
            "This phase is fixed to epoch5 and the three reviewed targets")
    require(plan["image_ids"] == [110, 124, 109], "Do not expand this phase")
    for filename, expected in plan["source_sha256"].items():
        require(sha(filename) == expected, "Source changed: " + filename)
    versions = {name: importlib.metadata.version(name) for name in plan["versions"]}
    require(versions == plan["versions"], "Package versions changed")
    source = ROOT / "mmdetection"
    git = ["git", "-c", "safe.directory=" + str(source)]
    require(subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=source,
                                    text=True).strip() == plan["mmdet_commit"],
            "Official MMDetection commit changed")
    require(not subprocess.check_output(git + ["diff", "HEAD", "--name-only"], cwd=source),
            "Official tracked source was modified")
    selected = read(plan["selected_checkpoint_record"])
    require(selected["selected_epoch"] == 5, "Selected epoch changed")
    require(Path(selected["best_checkpoint"]).resolve() == Path(plan["checkpoint"]).resolve(),
            "Checkpoint path is not the selected epoch5 checkpoint")
    require(selected["best_checkpoint_sha256"] == plan["source_sha256"][plan["checkpoint"]],
            "Checkpoint hash differs from selection record")
    cfg = Config.fromfile(plan["config"], import_custom_modules=False)
    require(cfg.experiment_id == "FRCNN-R50FPN-SCALE1280-VAL-001", "Wrong run config")
    require(cfg.model.backbone.init_cfg is None and not cfg.resume, "Unexpected initialization")
    require(cfg.model.roi_head.bbox_head.num_classes == 2, "Wrong class count")
    require(list(cfg.metainfo.classes) == ["broken", "loose"], "Wrong class order")
    require(list(cfg.model.rpn_head.anchor_generator.scales) == [8], "Wrong anchors")
    require(dict(cfg.model.test_cfg.rcnn) == dict(
        max_per_img=100, nms=dict(iou_threshold=0.5, type="nms"), score_thr=0.001),
        "Historical ROI filters changed")
    require(tuple(cfg.val_dataloader.dataset.pipeline[1].scale) == (2133, 1280),
            "Wrong validation resize")
    require(cfg.val_dataloader.batch_size == 1 and cfg.val_dataloader.dataset.test_mode,
            "Validation must use batch1 and test_mode (inference behavior, NOT test split)")
    ann = Path(cfg.val_dataloader.dataset.data_root) / cfg.val_dataloader.dataset.ann_file
    require(ann.resolve() == Path(plan["annotation"]).resolve(), "Wrong annotation split")
    truth = read(ann)
    require(len(truth["images"]) == 23 and len(truth["annotations"]) == 37,
            "Original validation set changed")
    images = {im["id"]: im for im in truth["images"]}
    for image_id in plan["image_ids"]:
        require(images[image_id]["file_name"].startswith("val/"), "Only val images allowed")
        file_path = Path(cfg.data_root) / "images" / images[image_id]["file_name"]
        require(str(file_path).replace("\\", "/") in plan["source_sha256"],
                "Unpinned image")
    targets = {a["id"]: a for a in truth["annotations"]}
    require([targets[i]["image_id"] for i in plan["target_gt_ids"]] == plan["image_ids"],
            "Target/image identity changed")
    saved = read(plan["saved_predictions"])
    require(len(saved) == 2300, "Historical prediction count changed")
    for image_id in plan["image_ids"]:
        require(sum(p["image_id"] == image_id for p in saved) == 100,
                "Historical image prediction count changed")
    print("DIAG04_PREPARATION_PASS: epoch5; 3 val images; no model built by checks",
          flush=True)
    return plan, cfg, truth, saved


def load_checkpoint(path):
    """Use the same constrained loading pattern as STEP12; epoch is now five."""
    import numpy as np
    import torch
    from mmengine.logging import HistoryBuffer

    known = {
        "mmengine.logging.history_buffer.HistoryBuffer": HistoryBuffer,
        "numpy.core.multiarray._reconstruct": np.core.multiarray._reconstruct,
        "numpy.core.multiarray.scalar": np.core.multiarray.scalar,
        "numpy.ndarray": np.ndarray, "numpy.dtype": np.dtype,
        "builtins.getattr": builtins.getattr,
    }
    present = torch.serialization.get_unsafe_globals_in_checkpoint(path)
    require(set(present) <= set(known), "Unexpected checkpoint globals: " + str(present))
    dtype_types = [type(np.dtype(k)) for k in ["float64", "float32", "int64", "int32", "uint8", "bool"]]
    with torch.serialization.safe_globals([known[k] for k in present] + dtype_types):
        data = torch.load(path, map_location="cpu", weights_only=True)
    require(data["meta"]["epoch"] == 5 and data["meta"]["iter"] == 270,
            "Checkpoint metadata is not epoch5 / iteration270")
    require(tuple(data["meta"]["dataset_meta"]["classes"]) == ("broken", "loose"),
            "Checkpoint classes changed")
    require(len(data["state_dict"]) == 348, "Unexpected state dictionary")
    return data


def ious(boxes, gt):
    import numpy as np
    boxes = np.asarray(boxes).reshape(-1, 4)
    wh = np.maximum(0, np.minimum(boxes[:, 2:], gt[2:]) - np.maximum(boxes[:, :2], gt[:2]))
    intersection = wh.prod(axis=1)
    areas = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    return intersection / np.maximum(areas + (gt[2] - gt[0]) * (gt[3] - gt[1]) - intersection, 1e-12)


def describe(boxes, gt):
    values = ious(boxes, gt)
    return dict(count=len(values), max_iou=float(values.max()) if len(values) else 0.0,
                covered_iou50=int((values >= 0.5).sum()))


def target_rows(annotations, arrays, target_ids, threshold):
    """Maxima describe sets; same-proposal lineage is recorded separately."""
    import numpy as np
    rows = []
    roi = arrays["roi_boxes_xyxy_original"]
    scores = arrays["roi_class_scores"]
    nms_ids = arrays["nms_candidate_ids"]
    final_ids = arrays["final_candidate_ids"]
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        gt = np.array([x, y, x + w, y + h])
        cls = annotation["category_id"] - 1
        nms_rows = nms_ids[nms_ids % 2 == cls] // 2
        final_rows = final_ids[final_ids % 2 == cls] // 2
        stages = {
            "rpn_handoff": describe(arrays["rpn_boxes_xyxy_original"], gt),
            "roi_same_class_before_filter": describe(roi[:, cls], gt),
            "after_score": describe(roi[scores[:, cls] > threshold, cls], gt),
            "after_nms_before_top100": describe(roi[nms_rows, cls], gt),
            "final_top100": describe(roi[final_rows, cls], gt),
        }
        rpn_iou = ious(arrays["rpn_boxes_xyxy_original"], gt)
        best = int(rpn_iou.argmax()) if len(rpn_iou) else None
        # Keep candidate identity, so we never compare two maxima as one box.
        lineage = None if best is None else dict(
            proposal_row=best, rpn_iou=float(rpn_iou[best]),
            roi_broken_iou=float(ious(roi[best:best+1, 0], gt)[0]),
            roi_loose_iou=float(ious(roi[best:best+1, 1], gt)[0]),
            scores_broken_loose_background=scores[best].tolist(),
            same_class_survives_nms=bool(best * 2 + cls in nms_ids),
            same_class_survives_top100=bool(best * 2 + cls in final_ids),
        )
        qualified = np.flatnonzero(ious(roi[:, cls], gt) >= 0.5)
        losses = {
            "below_score": int((scores[qualified, cls] <= threshold).sum()),
            "removed_by_nms": sum(int(scores[k, cls] > threshold and k * 2 + cls not in nms_ids)
                                  for k in qualified),
            "removed_by_top100": sum(int(k * 2 + cls in nms_ids and k * 2 + cls not in final_ids)
                                     for k in qualified),
            "retained": sum(int(k * 2 + cls in final_ids) for k in qualified),
        }
        require(sum(losses.values()) == len(qualified), "Candidate lineage partition failed")
        other_rows = final_ids[final_ids % 2 != cls] // 2
        other_values = ious(roi[other_rows, 1-cls], gt)
        other = None
        if len(other_values) and other_values.max() > 0:
            proposal = int(other_rows[int(other_values.argmax())])
            other = dict(
                proposal_row=proposal, class_name=("broken", "loose")[1-cls],
                iou=float(other_values.max()), score=float(scores[proposal, 1-cls]),
                same_proposal_true_class_iou=float(ious(roi[proposal:proposal+1, cls], gt)[0]),
                same_proposal_true_class_score=float(scores[proposal, cls]),
                same_proposal_true_class_retained=bool(proposal * 2 + cls in final_ids))
        rows.append(dict(gt_id=annotation["id"], image_id=annotation["image_id"],
                         class_name=("broken", "loose")[cls],
                         primary_target=annotation["id"] in target_ids,
                         bbox_xywh=annotation["bbox"], stages=stages,
                         best_rpn_proposal_lineage=lineage,
                         best_overlapping_other_class_final=other,
                         qualified_roi_candidate_fates=losses))
    return rows


def trace(plan, cfg, truth, saved, destination):
    """Exactly one official model.test_step per selected image; no evaluator."""
    import numpy as np
    import torch
    from mmengine.dataset import pseudo_collate
    from mmengine.runner import set_random_seed
    from mmdet.models.layers import multiclass_nms
    from mmdet.registry import DATASETS, MODELS
    from mmdet.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    set_random_seed(42, deterministic=True)
    torch.backends.cudnn.benchmark = False
    require(torch.cuda.is_available(), "CUDA unavailable; do not silently switch device")
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    require(dataset.cat2label == {1: 0, 2: 1}, "Dataset category mapping changed")
    indices = {dataset.get_data_info(i)["img_id"]: i for i in range(len(dataset))}
    data = load_checkpoint(plan["checkpoint"])
    model = MODELS.build(cfg.model).cpu()
    model.load_state_dict(data["state_dict"], strict=True)
    require(all(torch.equal(model.state_dict()[k], v) for k, v in data["state_dict"].items()),
            "Loaded parameters differ")
    save(destination / "weight_load.json", dict(status="PASS", epoch=5, matched=348,
         checkpoint=plan["checkpoint"], sha256=plan["source_sha256"][plan["checkpoint"]]))
    del data
    model.eval().requires_grad_(False).to("cuda:0")
    require(not any(m.training for m in model.modules()), "Model is not in evaluation mode")
    print("DIAG04_CHECKPOINT_LOAD_PASS: epoch5 matched=348; eval; gradients disabled", flush=True)
    captured = {}

    def roi_input(module, args):
        require("rois" not in captured, "Unexpected second ROI invocation")
        captured["rois"] = args[1].detach().clone()

    def head_output(module, args, output):
        captured["cls"], captured["reg"] = [v.detach().clone() for v in output]

    handles = [model.roi_head.bbox_roi_extractor.register_forward_pre_hook(roi_input),
               model.roi_head.bbox_head.register_forward_hook(head_output)]
    image_results, rows = [], []
    try:
        for index, image_id in enumerate(plan["image_ids"], 1):
            captured.clear()
            # Exact frozen validation dataset, loader and preprocessor, batch1.
            batch = pseudo_collate([dataset[indices[image_id]]])
            with torch.inference_mode():
                result = model.test_step(batch)[0]
                require(set(captured) == {"rois", "cls", "reg"}, "Missing diagnostic hooks")
                pre = model.roi_head.bbox_head.predict_by_feat(
                    rois=(captured["rois"],), cls_scores=(captured["cls"],),
                    bbox_preds=(captured["reg"],), batch_img_metas=[result.metainfo],
                    rcnn_test_cfg=None, rescale=True)[0]
                # Decode captured tensors; no second neural forward. Preserve
                # native score/NMS settings; omit only the final limit in this
                # auxiliary trace so its effect can be measured separately.
                filt = cfg.model.test_cfg.rcnn
                dets, labels, candidate_ids = multiclass_nms(
                    pre.bboxes, pre.scores, filt.score_thr, filt.nms,
                    max_num=-1, return_inds=True)
            limit = filt.max_per_img
            final = result.pred_instances
            require(torch.equal(labels[:limit], final.labels), "Replay final labels differ")
            require(torch.allclose(dets[:limit, :4], final.bboxes, rtol=0, atol=1e-4),
                    "Replay final coordinates differ")
            require(torch.allclose(dets[:limit, 4], final.scores, rtol=0, atol=1e-6),
                    "Replay final scores differ")
            final = final.cpu()
            original = [p for p in saved if p["image_id"] == image_id]
            old_boxes = np.array([p["bbox"] for p in original], dtype=float)
            old_boxes[:, 2:] += old_boxes[:, :2]
            boxes, scores = final.bboxes.numpy(), final.scores.numpy()
            require(final.labels.tolist() == [p["category_id"] - 1 for p in original],
                    "Historical prediction labels/count/order differ; stop for review")
            require(np.allclose(boxes, old_boxes, rtol=0, atol=0.1),
                    "Historical boxes differ by >0.1 original pixels; stop for review")
            require(np.allclose(scores, [p["score"] for p in original], rtol=0, atol=1e-4),
                    "Historical scores differ by >1e-4; stop for review")
            scale = np.array(list(result.metainfo["scale_factor"]) * 2)
            arrays = dict(
                rpn_boxes_xyxy_original=captured["rois"][:, 1:].cpu().numpy() / scale,
                roi_boxes_xyxy_original=pre.bboxes.cpu().numpy().reshape(-1, 2, 4),
                roi_class_scores=pre.scores.cpu().numpy(),
                roi_class_logits=captured["cls"].cpu().numpy(),
                roi_bbox_deltas=captured["reg"].cpu().numpy(),
                nms_candidate_ids=candidate_ids.cpu().numpy(),
                final_candidate_ids=candidate_ids[:limit].cpu().numpy(),
                final_boxes_xyxy_original=boxes, final_scores=scores,
                final_labels=final.labels.numpy(),
            )
            require(all(np.isfinite(a).all() for a in arrays.values()), "Non-finite arrays")
            require(arrays["roi_class_scores"].shape[1] == 3, "Expected two classes plus background")
            require(np.allclose(arrays["roi_class_scores"].sum(axis=1), 1, atol=1e-5),
                    "Invalid ROI probabilities")
            im = next(im for im in truth["images"] if im["id"] == image_id)
            stem = Path(im["file_name"]).stem
            np.savez_compressed(destination / (stem + "_intermediates.npz"), **arrays)
            rows.extend(target_rows([a for a in truth["annotations"] if a["image_id"] == image_id],
                                    arrays, plan["target_gt_ids"], filt.score_thr))
            image_results.append(dict(
                image_id=image_id, file_name=im["file_name"], img_shape=list(result.img_shape),
                scale_factor=list(result.scale_factor), rpn_count=len(arrays["rpn_boxes_xyxy_original"]),
                after_nms_count=len(candidate_ids), final_count=len(scores),
                saved_predictions_match=True,
                max_original_box_error_px=float(np.abs(boxes - old_boxes).max()),
                max_score_error=float(np.abs(scores - np.array([p["score"] for p in original])).max()),
            ))
            print(f"DIAG04_IMAGE_PASS {index}/3 {im['file_name']} saved_match=True", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    require(len(image_results) == 3 and len(rows) == 4, "Expected three images / four GT")
    save(destination / "per_gt.json", rows)
    save(destination / "images.json", image_results)
    return rows, image_results


def report(destination, rows, image_results, plan):
    """Simple Chinese report; independent stage maxima are not a causal proof."""
    import numpy as np
    from PIL import Image, ImageDraw
    labels = [("rpn_handoff", "RPN交接候选"),
              ("roi_same_class_before_filter", "ROI同类回归框，过滤前"),
              ("after_score", "分数>0.001之后"),
              ("after_nms_before_top100", "NMS去重后，尚未限100框"),
              ("final_top100", "最终最多100框")]
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>三图候选过程·讨论版</title>',
             '<style>body{font:16px/1.7 "Microsoft YaHei",sans-serif;max-width:1080px;margin:30px auto;padding:16px;color:#243746}'
             'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd7df;padding:9px}th{background:#edf4fa}'
             'img{max-width:100%}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}'
             'figure{margin:0;padding:8px;border:1px solid #ccd7df}section{margin:30px 0}pre{white-space:pre-wrap}</style>',
             '<h1>第5轮验证集三图候选过程 · 讨论版</h1>'
             '<p>新推理仅涉及3张val图，含4个GT；主要观察GT172、194、171，GT195是同图附带记录。'
             '没有训练、调参、新AP或test。与历史输出一致性已检查。</p>'
             '<p>RPN＝候选区域生成网络；ROI＝对候选区域分类并调整位置的阶段；NMS＝按重叠和分数去重。'
             '表中每项独立求最大IoU，不表示同一个框从左到右变化；同一候选的沿途记录单独列出。</p>'
             '<p>RPN交接候选已通过它自身的筛选和最多1000框限制，这里不恢复RPN更早全部候选。'
             '低于旧显示阈值0.30的框也参与本报告。局部图只为观看，不改变模型输入。</p>']
    for row in sorted(rows, key=lambda r: (not r["primary_target"], plan["target_gt_ids"].index(r["gt_id"]) if r["primary_target"] else r["gt_id"])):
        im = next(im for im in image_results if im["image_id"] == row["image_id"])
        x, y, w, h = row["bbox_xywh"]
        with Image.open(Path(plan["image_root"]) / im["file_name"]) as original:
            rgb = original.convert("RGB")
        pad = max(60, w * 0.7, h * 0.7)
        crop = (max(0, int(x-pad)), max(0, int(y-pad)),
                min(rgb.width, int(x+w+pad)), min(rgb.height, int(y+h+pad)))
        canvas = rgb.crop(crop)
        ImageDraw.Draw(canvas).rectangle((x-crop[0], y-crop[1], x+w-crop[0], y+h-crop[1]),
                                        outline="#00994e", width=2)
        crop_name = f"gt_{row['gt_id']}_reference.png"
        canvas.save(destination / crop_name)
        parts.append(f'<section><h2>GT{row["gt_id"]} · {html.escape(row["class_name"])}'
                     + ('（主要案例）' if row["primary_target"] else '（同图附带案例）') + '</h2>'
                     + f'<p>{html.escape(im["file_name"])}</p><img src="{crop_name}" alt="原图局部与人工标注">'
                     + '<table><tr><th>阶段</th><th>候选数</th><th>最大IoU</th><th>IoU≥0.50候选数</th></tr>')
        for key, label in labels:
            item = row["stages"][key]
            parts.append(f'<tr><td>{label}</td><td>{item["count"]}</td>'
                         f'<td>{item["max_iou"]:.6f}</td><td>{item["covered_iou50"]}</td></tr>')
        parts.append('</table><p>绿色＝人工标注；蓝色＝RPN候选；橙色＝同类ROI候选。'
                     '每格只画该阶段与GT最重合的一框；零重合时不画远处无关框。'
                     '局部窗口相同，预测框可能延伸出窗口。每格最大值不一定来自同一个候选。</p><div class="grid">')
        with np.load(destination / (Path(im["file_name"]).stem + "_intermediates.npz"),
                     allow_pickle=False) as arrays:
            cls = 0 if row["class_name"] == "broken" else 1
            roi = arrays["roi_boxes_xyxy_original"]
            nms_rows = arrays["nms_candidate_ids"]
            nms_rows = nms_rows[nms_rows % 2 == cls] // 2
            final_rows = arrays["final_candidate_ids"]
            final_rows = final_rows[final_rows % 2 == cls] // 2
            sets = [arrays["rpn_boxes_xyxy_original"], roi[:, cls],
                    roi[arrays["roi_class_scores"][:, cls] > 0.001, cls],
                    roi[nms_rows, cls], roi[final_rows, cls]]
            for (key, label), boxes in zip(labels, sets):
                panel = canvas.copy()
                values = ious(boxes, np.array([x, y, x+w, y+h]))
                if len(values) and values.max() > 0:
                    box = boxes[int(values.argmax())]
                    ImageDraw.Draw(panel).rectangle(
                        tuple(box - np.array([crop[0], crop[1], crop[0], crop[1]])),
                        outline="#2356bf" if key == "rpn_handoff" else "#e0790a", width=2)
                name = f"gt_{row['gt_id']}_{key}.png"
                panel.save(destination / name)
                parts.append(f'<figure><figcaption>{label} · 最大IoU={row["stages"][key]["max_iou"]:.6f}'
                             f'</figcaption><img src="{name}" alt="{label}局部对照"></figure>')
        parts.append('</div>')
        lineage = row["best_rpn_proposal_lineage"]
        fates = row["qualified_roi_candidate_fates"]
        parts.append('<p>ROI中位置合格的同类候选，后续去向（候选数，不是正确检出数）：'
                     f'分数筛掉{fates["below_score"]}；NMS去重移除{fates["removed_by_nms"]}；'
                     f'未进前100框{fates["removed_by_top100"]}；最终保留{fates["retained"]}。</p>')
        if lineage:
            p = lineage["scores_broken_loose_background"]
            parts.append(f'<p>同一个最佳RPN候选（数组第{lineage["proposal_row"]}行，从0编号）：'
                         f'原IoU={lineage["rpn_iou"]:.6f}；断股回归IoU={lineage["roi_broken_iou"]:.6f}；'
                         f'松股回归IoU={lineage["roi_loose_iou"]:.6f}。'
                         f'断股/松股/背景概率分别为{p[0]:.6f} / {p[1]:.6f} / {p[2]:.6f}。</p>')
        other = row["best_overlapping_other_class_final"]
        if other:
            parts.append(f'<p>最终错类框中最重合者：{html.escape(other["class_name"])}，'
                         f'IoU={other["iou"]:.6f}，分数={other["score"]:.6f}；'
                         f'来自第{other["proposal_row"]}行候选。该同一候选的正确类别回归框'
                         f'IoU={other["same_proposal_true_class_iou"]:.6f}，'
                         f'正确类别分数={other["same_proposal_true_class_score"]:.6f}。</p>')
        parts.append('</section>')
    parts.append('<p>这里追踪候选身份，不等于预测实例的一对一匹配。</p>'
                 '<h2>下一步交接</h2><p>请先发送终端完整输出、RESULT_DIR和LOG_DIR。'
                 '助手复核保存数组后解释环节差异；不要求你判断程序正确，也不需要重新确认缺陷类别。</p>'
                 '<p>详细数据：<a href="per_gt.json">逐目标记录</a> · <a href="images.json">逐图一致性</a>'
                 ' · <a href="summary.json">运行状态</a></p></html>')
    (destination / "index.html").write_text("".join(parts), encoding="utf-8")


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file

    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = ("diag04_preflight_" if args.check_only else "diag04_") + stamp
    logs = ROOT / "workspace/logs" / name
    base = ROOT / "workspace/audit/FRCNN-DIAG-TRACE1280-001"
    destination = base / name
    logs.mkdir(parents=True, exist_ok=False)
    destination.mkdir(parents=True, exist_ok=False)
    code = 1
    with (logs / "console.txt").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print("DIAG04_CHECK_ONLY" if args.check_only else "DIAG04_USER_VAL3_INFERENCE", flush=True)
            print("RESULT_DIR=" + str(destination), flush=True)
            print("LOG_DIR=" + str(logs), flush=True)
            try:
                plan, cfg, truth, saved = check(args.plan)
                save(destination / "preparation.json", dict(
                    status="PASS", plan=plan, plan_sha256=sha(args.plan),
                    script_sha256=sha(__file__), detector_built=False, model_forward=False))
                if args.check_only:
                    print("DIAG04_CHECK_ONLY_PASS; no detector built; no inference", flush=True)
                else:
                    (destination / "code").mkdir()
                    for src in [Path(__file__), args.plan, Path(plan["config"])]:
                        shutil.copy2(src, destination / "code" / src.name)
                    rows, images = trace(plan, cfg, truth, saved, destination)
                    for src, expected in plan["source_sha256"].items():
                        require(sha(src) == expected, "Source changed during diagnosis: " + src)
                    report(destination, rows, images, plan)
                    save(destination / "summary.json", dict(
                        status="DIAG04_TRACE_PASS_REVIEW_PENDING", experiment=plan["experiment"],
                        epoch=5, images=3, gt=4, primary_gt_ids=plan["target_gt_ids"],
                        image_results=images, training=False, inference=True, neural_forwards=3,
                        test=False, metric_changes=False, annotation_changes=False,
                        limitation="RPN handoff only; not RPN pre-filter output or historical training replay."))
                    outputs = [p for p in destination.rglob("*") if p.is_file()]
                    save(destination / "manifest.json", dict(
                        source_sha256=plan["source_sha256"],
                        output_sha256={str(p): sha(p) for p in outputs},
                        user_launch=sys.argv, interpreter=sys.executable))
                    (base / "latest.html").write_text(
                        '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='
                        + name + '/index.html"><a href="' + name + '/index.html">打开本次报告</a>',
                        encoding="utf-8")
                    print("DIAG04_TRACE_PASS: 3 val images; historical outputs matched; review pending", flush=True)
                    print("REPORT=" + str(destination / "index.html"), flush=True)
                code = 0
            except Exception:
                traceback.print_exc()
                save(destination / "failure.json", dict(status="FAIL", traceback=traceback.format_exc()))
                print("DIAG04_FAIL; send full output; do not retry or change settings", flush=True)
            print("DIAG04_EXIT_CODE=" + str(code), flush=True)
            print("RESULT_DIR=" + str(destination), flush=True)
            print("LOG_DIR=" + str(logs), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
