# STEP12 固定训练8图的RPN到ROI诊断

状态：准备阶段，助手仅做静态检查，尚未运行模型推理。用户在cmd启动。

目的：STEP11末轮固定train8中仍有4个loose目标没有同类预测IoU达到0.50。对原800和新1280两次107张训练的第100轮做固定8图诊断，查看RPN候选区域、ROI分类和回归、score过滤及NMS/最多100框的影响。

不是新训练，不创建优化器，不计算梯度、不更新参数；不调用train.py，不读取test图片或评价test。也不是恢复训练时的随机ROI采样。第100轮只是末期拟合诊断，绝不替换原val选择的第66/5轮。此诊断不能独自解释整个验证集泛化低的原因。

输入：原先固定的instances_debug8.json（原train8、15框），两个既有运行的epoch_100.pth；各自使用训练时的Resize，其他推理设置不变。完整来源及哈希保存到preparation.json。官方预测流程上安装只读hook捕获ROI输入及输出，ROI分支额外解码已有输出以观察NMS前结果，不增加第二次神经网络前向。

每图最终预测与历史末轮快照比较：类别/数量相同，原图框绝对误差不超过0.1像素，分数误差不超过1e-4。不一致即停止，先查诊断实现，不据不一致结果调整模型。严格加载348项权重，运行前后校验checkpoint哈希。

保存每个目标的RPN最大IoU、同类回归框过滤前后IoU、ROI目标类/背景分数，以及绿色GT、蓝色最佳RPN、红色最佳同类最终预测的局部图；红框不受0.30展示阈值隐藏，具体分数见保存数组与CSV。RPN使用推理期NMS后的最多1000个候选区域，不是全部原始anchor，也不是训练期的采样结果。per_gt的分类标记是描述性定位，不能当作唯一原因的证明。

完整中间数组存npz：rpn_boxes_xyxy_original、roi_boxes_xyxy_original、roi_class_scores（broken、loose、background）、final_boxes_xyxy_original、final_labels、final_scores。额外记录实际缩放尺寸及FPN各层shape，不保存海量逐层特征或宣称特征图已全面检查。

输出目录：`D:\小论文\model_projects\Faster R-CNN\workspace\runs\FRCNN-DIAG-RPNROI-TRAIN8\step12_<唯一编号>`。
日志目录：`D:\小论文\model_projects\Faster R-CNN\workspace\logs\step12_<同一编号>`。

```cmd
cd /d "D:\小论文\model_projects\Faster R-CNN"
call "D:\小论文\model_projects\Faster R-CNN\workspace\scripts\run_step12_diagnose_rpn_roi.cmd"
```

正常：STEP12_PREPARATION_PASS，分别两组DIAG_IMAGE_PASS 1/8到8/8，STEP12_DIAGNOSTIC_PASS，STEP12_EXIT_CODE=0。
反馈完整短输出和RUN_DIR、LOG_DIR。发生错误保留现场，不自动重试，不改模型参数。

停止条件：完成这16次固定图推理并验收后再决定是否值得调整任何训练因素；不自动追加训练、加大尺寸或跑test。指标不进入正式泛化排名。原始最佳checkpoint选择保持不变。
