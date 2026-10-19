import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const project = "D:/小论文/model_projects/yolo11";
const workspace = `${project}/workspace`;
const outputDir = `${workspace}/experiment_tracking`;
const xlsxPath = `${outputDir}/experiment_registry.xlsx`;
const csvPath = `${outputDir}/experiment_registry.csv`;
const mode = process.argv[2] || "initial";

const columns = [
  "experiment_id", "date", "stage", "status", "purpose", "parent_experiment", "changed_variable",
  "model", "pretrained_weights", "dataset_version", "dataset_path", "train_images", "val_images", "test_images", "classes",
  "git_commit", "ultralytics_version", "imgsz", "batch", "nbs", "steady_accumulate", "epochs_planned", "epochs_completed",
  "workers", "device", "seed", "deterministic", "optimizer", "lr0_actual", "momentum_actual", "weight_decay_actual",
  "warmup_epochs", "mosaic", "close_mosaic", "mixup", "cutmix", "fliplr", "translate", "scale", "patience", "amp",
  "best_epoch", "precision_at_best_epoch", "recall_at_best_epoch", "map50_at_best_epoch", "map50_95_at_best_epoch",
  "max_precision_any_epoch", "epoch_of_max_precision", "max_recall_any_epoch", "epoch_of_max_recall",
  "broken_map50", "broken_map50_95", "loose_map50", "loose_map50_95", "model_parameters", "gflops_1280", "peak_vram_mib", "runtime_seconds",
  "run_directory", "best_pt", "results_csv",
  "test_used", "include_in_paper", "decision", "notes",
];

function parseCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const headers = lines[0].split(",").map((s) => s.trim());
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = line.split(",");
    return Object.fromEntries(headers.map((header, i) => [header, values[i]?.trim() ?? ""]));
  });
}

function num(value) {
  if (value === undefined || value === null || value === "") return "NA";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : "NA";
}

function bestBy(rows, primary, secondary = null) {
  return [...rows].sort((a, b) => {
    const delta = num(b[primary]) - num(a[primary]);
    if (delta !== 0) return delta;
    if (secondary) {
      const second = num(b[secondary]) - num(a[secondary]);
      if (second !== 0) return second;
    }
    return num(a.epoch) - num(b.epoch);
  })[0];
}

function epochMetrics(rows) {
  const best = bestBy(rows, "metrics/mAP50-95(B)", "metrics/mAP50(B)");
  const maxPrecision = bestBy(rows, "metrics/precision(B)");
  const maxRecall = bestBy(rows, "metrics/recall(B)");
  return {
    best_epoch: num(best.epoch),
    precision_at_best_epoch: num(best["metrics/precision(B)"]),
    recall_at_best_epoch: num(best["metrics/recall(B)"]),
    map50_at_best_epoch: num(best["metrics/mAP50(B)"]),
    map50_95_at_best_epoch: num(best["metrics/mAP50-95(B)"]),
    max_precision_any_epoch: num(maxPrecision["metrics/precision(B)"]),
    epoch_of_max_precision: num(maxPrecision.epoch),
    max_recall_any_epoch: num(maxRecall["metrics/recall(B)"]),
    epoch_of_max_recall: num(maxRecall.epoch),
  };
}

function csvEscape(value) {
  let text = value instanceof Date ? value.toISOString().slice(0, 10) : String(value);
  if (/[",\r\n]/.test(text)) text = `"${text.replaceAll('"', '""')}"`;
  return text;
}

function recordToArray(record, forXlsx = false) {
  return columns.map((column) => {
    const value = record[column] ?? "NA";
    if (forXlsx && column === "date" && /^\d{4}-\d{2}-\d{2}$/.test(String(value))) {
      return new Date(`${value}T00:00:00`);
    }
    return value;
  });
}

async function historicalRecords() {
  const smokeRun = `${workspace}/runs/yolo11n_smoke_20260915_163741`;
  const diagRun = `${workspace}/runs/yolo11s_diag_1280_e30_20260916_104603`;
  const smokeStatus = JSON.parse(await fs.readFile(`${workspace}/logs/yolo11n_smoke_20260915_163741_status.json`, "utf8"));
  const diagStatus = JSON.parse(await fs.readFile(`${workspace}/logs/latest_yolo11s_diagnostic_status.json`, "utf8"));
  const smokeRows = parseCsv(await fs.readFile(`${smokeRun}/results.csv`, "utf8"));
  const diagRows = parseCsv(await fs.readFile(`${diagRun}/results.csv`, "utf8"));
  const diagClass = parseCsv(await fs.readFile(`${workspace}/analysis/yolo11s_best_val_per_class.csv`, "utf8"));
  const broken = diagClass.find((row) => row.Class === "broken");
  const loose = diagClass.find((row) => row.Class === "loose");
  const smokeMetrics = epochMetrics(smokeRows);
  const diagMetrics = epochMetrics(diagRows);
  const common = {
    dataset_version: "02_yolo11_dataset (split_manifest controlled)",
    dataset_path: "D:/小论文/conductor_defect_dataset/02_yolo11_dataset",
    train_images: 107,
    val_images: 23,
    test_images: 23,
    classes: "0=broken; 1=loose",
    git_commit: "9aa41d679d121afb8b4cc512d7a162f4f36886c3",
    ultralytics_version: "8.4.152",
    batch: 4,
    nbs: 64,
    workers: 0,
    device: "cuda:0",
    seed: 42,
    deterministic: true,
    optimizer: "AdamW (optimizer=auto)",
    lr0_actual: 0.001667,
    momentum_actual: 0.9,
    weight_decay_actual: 0.0005,
    warmup_epochs: 3,
    mosaic: 1,
    close_mosaic: 10,
    mixup: 0,
    cutmix: 0,
    fliplr: 0.5,
    translate: 0.1,
    scale: 0.5,
    amp: true,
    test_used: false,
    include_in_paper: "No",
  };
  return [
    {
      ...common,
      experiment_id: "Y11-SMOKE-001",
      date: "2026-09-15",
      stage: "Smoke",
      status: "Completed",
      purpose: "验证 YOLO11 完整训练管线是否正常。",
      parent_experiment: "NA",
      changed_variable: "NA",
      model: "YOLO11n",
      pretrained_weights: "yolo11n.pt",
      imgsz: 640,
      steady_accumulate: "NA",
      epochs_planned: 5,
      epochs_completed: smokeRows.length,
      patience: 100,
      ...smokeMetrics,
      broken_map50: "NA",
      broken_map50_95: "NA",
      loose_map50: "NA",
      loose_map50_95: "NA",
      peak_vram_mib: smokeStatus.peak_allocated_MiB,
      runtime_seconds: smokeStatus.elapsed_seconds,
      run_directory: smokeRun,
      best_pt: `${smokeRun}/weights/best.pt`,
      results_csv: `${smokeRun}/results.csv`,
      decision: "仅证明训练管线跑通，不作为正式论文检测结果。",
      notes: "历史运行未记录 steady_accumulate，也未生成逐类别 AP；按要求填 NA。",
    },
    {
      ...common,
      experiment_id: "Y11-DIAG-001",
      date: "2026-09-16",
      stage: "Diagnostic",
      status: "Completed",
      purpose: "验证 YOLO11s + 1280 输入下当前极小目标数据集是否具备可学习性。",
      parent_experiment: "Y11-SMOKE-001",
      changed_variable: "model YOLO11n -> YOLO11s; imgsz 640 -> 1280; epochs 5 -> 30",
      model: "YOLO11s",
      pretrained_weights: "yolo11s.pt",
      imgsz: 1280,
      steady_accumulate: "NA",
      epochs_planned: 30,
      epochs_completed: diagRows.length,
      patience: 0,
      ...diagMetrics,
      broken_map50: num(broken.mAP50),
      broken_map50_95: num(broken["mAP50-95"]),
      loose_map50: num(loose.mAP50),
      loose_map50_95: num(loose["mAP50-95"]),
      peak_vram_mib: diagStatus.peak_allocated_MiB,
      runtime_seconds: diagStatus.elapsed_seconds,
      run_directory: diagRun,
      best_pt: `${diagRun}/weights/best.pt`,
      results_csv: `${diagRun}/results.csv`,
      decision: "确认模型开始学习但指标仍低；用于选择后续训练设置，不是正式论文 baseline。",
      notes: "历史运行未记录 trainer.accumulate；steady_accumulate 按要求填 NA。epoch 4 的 val/cls_loss 为 NaN。",
    },
  ];
}

async function currentRecord() {
  const status = JSON.parse(await fs.readFile(`${workspace}/logs/latest_yolo11s_nbs16_status.json`, "utf8"));
  const run = status.save_dir.replaceAll("\\", "/");
  const rows = parseCsv(await fs.readFile(`${run}/results.csv`, "utf8"));
  const classes = parseCsv(await fs.readFile(`${workspace}/analysis/yolo11s_nbs16_best_val_per_class.csv`, "utf8"));
  const broken = classes.find((row) => row.Class === "broken");
  const loose = classes.find((row) => row.Class === "loose");
  const metrics = epochMetrics(rows);
  return {
    experiment_id: "Y11-DIAG-002",
    date: "2026-09-16",
    stage: "Diagnostic",
    status: status.success ? "Completed" : "Failed",
    purpose: "在其他主要训练条件不变时，检查较小 nominal batch size / gradient accumulation 是否改善小数据集学习效率。",
    parent_experiment: "Y11-DIAG-001",
    changed_variable: "nbs: 64 -> 16",
    model: "YOLO11s",
    pretrained_weights: "yolo11s.pt",
    dataset_version: "02_yolo11_dataset (split_manifest controlled)",
    dataset_path: "D:/小论文/conductor_defect_dataset/02_yolo11_dataset",
    train_images: 107,
    val_images: 23,
    test_images: 23,
    classes: "0=broken; 1=loose",
    git_commit: status.git_commit,
    ultralytics_version: status.ultralytics,
    imgsz: 1280,
    batch: status.actual_batch,
    nbs: status.actual_nbs,
    steady_accumulate: status.steady_accumulate,
    epochs_planned: 30,
    epochs_completed: rows.length,
    workers: 0,
    device: status.training_device,
    seed: 42,
    deterministic: true,
    optimizer: status.optimizer_actual.name,
    lr0_actual: status.optimizer_actual.lr0,
    momentum_actual: status.optimizer_actual.momentum,
    weight_decay_actual: status.optimizer_actual.weight_decay,
    warmup_epochs: 3,
    mosaic: 1,
    close_mosaic: 10,
    mixup: 0,
    cutmix: 0,
    fliplr: 0.5,
    translate: 0.1,
    scale: 0.5,
    patience: 0,
    amp: true,
    ...metrics,
    broken_map50: num(broken.mAP50),
    broken_map50_95: num(broken["mAP50-95"]),
    loose_map50: num(loose.mAP50),
    loose_map50_95: num(loose["mAP50-95"]),
    peak_vram_mib: status.peak_allocated_MiB,
    runtime_seconds: status.elapsed_seconds,
    run_directory: run,
    best_pt: status.best_pt.replaceAll("\\", "/"),
    results_csv: `${run}/results.csv`,
    test_used: false,
    include_in_paper: "No",
    decision: status.decision,
    notes: status.registry_notes,
  };
}

async function tuneRecord() {
  const statusPath = `${workspace}/logs/yolo11s_tune_1280_e30_mosaic0_20260916_170351_status.json`;
  const status = JSON.parse(await fs.readFile(statusPath, "utf8"));
  const run = status.save_dir.replaceAll("\\", "/");
  const rows = parseCsv(await fs.readFile(`${run}/results.csv`, "utf8"));
  const classes = parseCsv(await fs.readFile(`${workspace}/analysis/mosaic1_vs_mosaic0_per_class.csv`, "utf8"))
    .filter((row) => row.experiment_id === "Y11-TUNE-001");
  const broken = classes.find((row) => row.class_name === "broken");
  const loose = classes.find((row) => row.class_name === "loose");
  const metrics = epochMetrics(rows);
  return {
    experiment_id: "Y11-TUNE-001",
    date: "2026-09-16",
    stage: "Tuning",
    status: status.success ? "Completed" : "Failed",
    purpose: "严格单变量验证：Mosaic 的数据多样性收益能否抵消其对极小目标尺寸的不利影响。",
    parent_experiment: "Y11-DIAG-001",
    changed_variable: "mosaic: 1.0 -> 0.0",
    model: "YOLO11s",
    pretrained_weights: "yolo11s.pt",
    dataset_version: "02_yolo11_dataset (split_manifest controlled)",
    dataset_path: "D:/小论文/conductor_defect_dataset/02_yolo11_dataset",
    train_images: 107,
    val_images: 23,
    test_images: 23,
    classes: "0=broken; 1=loose",
    git_commit: status.source_commit,
    ultralytics_version: "8.4.152",
    imgsz: 1280,
    batch: status.parameters.batch,
    nbs: status.parameters.nbs,
    steady_accumulate: "NA",
    epochs_planned: status.parameters.epochs,
    epochs_completed: rows.length,
    workers: status.parameters.workers,
    device: "cuda:0",
    seed: status.parameters.seed,
    deterministic: status.parameters.deterministic,
    optimizer: "AdamW (optimizer=auto)",
    lr0_actual: 0.001667,
    momentum_actual: 0.9,
    weight_decay_actual: 0.0005,
    warmup_epochs: status.parameters.warmup_epochs,
    mosaic: status.parameters.mosaic,
    close_mosaic: status.parameters.close_mosaic,
    mixup: status.parameters.mixup,
    cutmix: status.parameters.cutmix,
    fliplr: status.parameters.fliplr,
    translate: status.parameters.translate,
    scale: status.parameters.scale,
    patience: status.parameters.patience,
    amp: status.parameters.amp,
    ...metrics,
    broken_map50: num(broken.map50),
    broken_map50_95: num(broken.map50_95),
    loose_map50: num(loose.map50),
    loose_map50_95: num(loose.map50_95),
    peak_vram_mib: status.peak_allocated_MiB,
    runtime_seconds: status.elapsed_seconds,
    run_directory: run,
    best_pt: status.best_pt.replaceAll("\\", "/"),
    results_csv: status.results_csv.replaceAll("\\", "/"),
    test_used: false,
    include_in_paper: "No",
    decision: "Mosaic=0 明显更差；保留官方 Mosaic 策略作为后续候选。",
    notes: "严格单变量对照；第 5 轮 val loss 为 NaN，epoch>=10 无非有限值，810 个训练 batch loss 均有限；test 未使用；steady_accumulate 未实测，填 NA。",
  };
}

async function tune2Record() {
  const statusPath = `${workspace}/logs/yolo11s_tune_1280_e100_20260916_191542_status.json`;
  const status = JSON.parse(await fs.readFile(statusPath, "utf8"));
  const run = status.save_dir.replaceAll("\\", "/");
  const rows = parseCsv(await fs.readFile(`${run}/results.csv`, "utf8"));
  const classes = parseCsv(await fs.readFile(`${workspace}/analysis/Y11-TUNE-002_per_class.csv`, "utf8"));
  const broken = classes.find((row) => row.class_name === "broken");
  const loose = classes.find((row) => row.class_name === "loose");
  const metrics = epochMetrics(rows);
  return {
    experiment_id: "Y11-TUNE-002",
    date: "2026-09-16",
    stage: "Tuning",
    status: status.success ? "Completed" : "Failed",
    purpose: "在当前最佳已测试设置下检查 30 epoch 是否训练不足，并分析 100 epoch 收敛及 Mosaic 关闭前后变化。",
    parent_experiment: "Y11-DIAG-001",
    changed_variable: "epochs: 30 -> 100",
    model: "YOLO11s",
    pretrained_weights: "yolo11s.pt",
    dataset_version: "02_yolo11_dataset (split_manifest controlled)",
    dataset_path: "D:/小论文/conductor_defect_dataset/02_yolo11_dataset",
    train_images: 107,
    val_images: 23,
    test_images: 23,
    classes: "0=broken; 1=loose",
    git_commit: status.source_commit,
    ultralytics_version: "8.4.152",
    imgsz: status.parameters.imgsz,
    batch: status.parameters.batch,
    nbs: status.parameters.nbs,
    steady_accumulate: "NA",
    epochs_planned: status.parameters.epochs,
    epochs_completed: rows.length,
    workers: status.parameters.workers,
    device: "cuda:0",
    seed: status.parameters.seed,
    deterministic: status.parameters.deterministic,
    optimizer: "AdamW (optimizer=auto)",
    lr0_actual: 0.001667,
    momentum_actual: 0.9,
    weight_decay_actual: 0.0005,
    warmup_epochs: status.parameters.warmup_epochs,
    mosaic: status.parameters.mosaic,
    close_mosaic: status.parameters.close_mosaic,
    mixup: status.parameters.mixup,
    cutmix: status.parameters.cutmix,
    fliplr: status.parameters.fliplr,
    translate: status.parameters.translate,
    scale: status.parameters.scale,
    patience: status.parameters.patience,
    amp: status.parameters.amp,
    ...metrics,
    broken_map50: num(broken.map50),
    broken_map50_95: num(broken.map50_95),
    loose_map50: num(loose.map50),
    loose_map50_95: num(loose.map50_95),
    peak_vram_mib: status.peak_allocated_MiB,
    runtime_seconds: status.elapsed_seconds,
    run_directory: run,
    best_pt: status.best_pt.replaceAll("\\", "/"),
    results_csv: status.results_csv.replaceAll("\\", "/"),
    test_used: false,
    include_in_paper: "No",
    decision: "100 epoch 未获得明确 AP 改善；减少训练超参数试探，转向极小目标结构问题。",
    notes: "100epoch convergence diagnostic; mosaic=1.0; close_mosaic=10; test not used. best epoch=15; epoch 4/5/9 val loss NaN, epoch>=10 none; 2700 train batches finite.",
  };
}

async function arch1Record() {
  const statusPath = `${workspace}/logs/yolo11s_p2_diag_1280_e30_20260916_202431_status.json`;
  const status = JSON.parse(await fs.readFile(statusPath, "utf8"));
  const run = status.save_dir.replaceAll("\\", "/");
  const rows = parseCsv(await fs.readFile(`${run}/results.csv`, "utf8"));
  const classes = parseCsv(await fs.readFile(`${workspace}/analysis/Y11-ARCH-001_per_class.csv`, "utf8"))
    .filter((row) => row.experiment_id === "Y11-ARCH-001");
  const broken = classes.find((row) => row.class_name === "broken");
  const loose = classes.find((row) => row.class_name === "loose");
  const metrics = epochMetrics(rows);
  const info = status.post_analysis.structure.p2_model_info;
  return {
    experiment_id: "Y11-ARCH-001",
    date: "2026-09-16",
    stage: "ArchitectureDiagnostic",
    status: status.success && status.post_analysis.success ? "Completed" : "Failed",
    purpose: "在完全相同训练协议下，验证加入 P2/stride=4 高分辨率检测分支能否改善超小缺陷检测。",
    parent_experiment: "Y11-DIAG-001",
    changed_variable: "detection_scales: P3/P4/P5 -> P2/P3/P4/P5",
    model: "YOLO11s-P2",
    pretrained_weights: "yolo11s.pt",
    dataset_version: "02_yolo11_dataset (split_manifest controlled)",
    dataset_path: "D:/小论文/conductor_defect_dataset/02_yolo11_dataset",
    train_images: 107,
    val_images: 23,
    test_images: 23,
    classes: "0=broken; 1=loose",
    git_commit: status.source_commit,
    ultralytics_version: "8.4.152",
    imgsz: status.parameters.imgsz,
    batch: status.parameters.batch,
    nbs: status.parameters.nbs,
    steady_accumulate: "NA",
    epochs_planned: status.parameters.epochs,
    epochs_completed: rows.length,
    workers: status.parameters.workers,
    device: "cuda:0",
    seed: status.parameters.seed,
    deterministic: status.parameters.deterministic,
    optimizer: `${status.optimizer_actual.name} (optimizer=auto)`,
    lr0_actual: status.optimizer_actual.lr0,
    momentum_actual: status.optimizer_actual.momentum,
    weight_decay_actual: status.parameters.weight_decay,
    warmup_epochs: status.parameters.warmup_epochs,
    mosaic: status.parameters.mosaic,
    close_mosaic: status.parameters.close_mosaic,
    mixup: status.parameters.mixup,
    cutmix: status.parameters.cutmix,
    fliplr: status.parameters.fliplr,
    translate: status.parameters.translate,
    scale: status.parameters.scale,
    patience: status.parameters.patience,
    amp: status.parameters.amp,
    ...metrics,
    broken_map50: num(broken.map50),
    broken_map50_95: num(broken.map50_95),
    loose_map50: num(loose.map50),
    loose_map50_95: num(loose.map50_95),
    model_parameters: info.parameters,
    gflops_1280: info.gflops_1280,
    peak_vram_mib: status.peak_allocated_MiB,
    runtime_seconds: status.elapsed_seconds,
    run_directory: run,
    best_pt: status.best_pt.replaceAll("\\", "/"),
    results_csv: status.results_csv.replaceAll("\\", "/"),
    test_used: false,
    include_in_paper: "No",
    decision: "情况C：整体 mAP50-95 明显下降；broken 未检出，loose 的 mAP50 上升但 mAP50-95 下降。仅增加 P2 不足以解决问题。",
    notes: "Detect strides=4/8/16/32；迁移 448/563 state items，非预期缺失=0；AdamW lr=0.001667 momentum=0.9；810 个训练 batch loss 均有限；test 未使用；数据集不变。",
  };
}

function styleRuns(sheet, rowCount) {
  const end = rowCount + 4;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(3);
  sheet.getRange(`A2:BM2`).format.font = { name: "Arial", size: 14, bold: true, color: "#17365D" };
  sheet.getRange(`A4:BM${end}`).format.font = { name: "Arial", size: 9, color: "#1F2937" };
  sheet.getRange("A4:BM4").format = {
    fill: "#1F4E78",
    font: { name: "Arial", size: 9, bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`B5:B${end}`).format.numberFormat = "yyyy-mm-dd";
  for (const range of ["AC5:AE100", "AQ5:AU100", "AW5:AW100", "AY5:BB100"]) sheet.getRange(range).format.numberFormat = "0.000000";
  sheet.getRange("BC5:BC100").format.numberFormat = "#,##0";
  sheet.getRange("BD5:BD100").format.numberFormat = "0.000";
  sheet.getRange("BE5:BF100").format.numberFormat = "0.00";
  sheet.getRange(`A4:BM${end}`).format.verticalAlignment = "center";
  sheet.getRange(`E5:G${end}`).format.wrapText = true;
  sheet.getRange(`BL5:BM${end}`).format.wrapText = true;
  sheet.getRange("A:BM").format.columnWidth = 12;
  for (const col of ["A", "C", "D", "H", "I", "P", "Q", "Y", "AB", "BK"]) sheet.getRange(`${col}:${col}`).format.columnWidth = 16;
  for (const col of ["E", "F", "G", "J", "K", "O", "BG", "BH", "BI", "BL", "BM"]) sheet.getRange(`${col}:${col}`).format.columnWidth = 28;
  sheet.getRange("A4:BM4").format.rowHeight = 42;
  sheet.getRange(`A5:BM${end}`).format.rowHeight = 42;
  sheet.tabColor = "#1F4E78";
}

async function createInitial(records) {
  const workbook = Workbook.create();
  const runs = workbook.worksheets.add("Runs");
  const setup = workbook.worksheets.add("ProjectSetup");
  const definitions = workbook.worksheets.add("FieldDefinitions");

  runs.getRange("A2").values = [["YOLO11 实验总表"]];
  runs.getRange("A3").values = [["一行对应一次实际训练；test_used=False 的诊断结果不得当作最终测试结果。"]];
  runs.getRange("A4:BM4").values = [columns];
  runs.getRange(`A5:BM${records.length + 4}`).values = records.map((record) => recordToArray(record, true));
  const table = runs.tables.add(`A4:BM${records.length + 4}`, true, "RunsTable");
  table.style = "TableStyleMedium2";
  styleRuns(runs, records.length);

  const setupRows = [
    ["Field", "Value", "Source / evidence"],
    ["Project root", project, "Current workspace"],
    ["Dataset root", "D:/小论文/conductor_defect_dataset/02_yolo11_dataset", "workspace/configs/conductor_bl_v1.yaml"],
    ["Classes", "0=broken; 1=loose", "data.yaml and dataset_check.json"],
    ["Train", "107 images; broken=39; loose=129", "dataset_check.json"],
    ["Val", "23 images; broken=9; loose=28", "dataset_check.json"],
    ["Test", "23 images; broken=8; loose=28", "dataset_check / split_manifest evidence; test not used for tuning"],
    ["Split method", "Grouped by split_group_id; no group or identical image crosses splits", "dataset_check.json"],
    ["Split manifest", "D:/小论文/conductor_defect_dataset/02_yolo11_dataset/split_manifest.csv", "Manifest contains SHA256, dimensions, box counts and split"],
    ["Dataset integrity", "Passed; 1749 entries unchanged after Y11-ARCH-001", "yolo11s_p2_diag_1280_e30_20260916_202431_status.json"],
    ["Ultralytics commit", "9aa41d679d121afb8b4cc512d7a162f4f36886c3", "git rev-parse HEAD"],
    ["Ultralytics version", "8.4.152", "Runtime status JSON"],
    ["Python", "3.12.13", "environment_torch_verified.txt"],
    ["PyTorch", "2.8.0+cu128", "environment_torch_verified.txt"],
    ["CUDA runtime", "12.8", "torch.version.cuda"],
    ["GPU", "NVIDIA GeForce RTX 5070 Ti", "torch.cuda.get_device_name(0)"],
    ["GPU memory", "16303 MiB", "nvidia-smi"],
  ];
  setup.getRange(`A1:C${setupRows.length}`).values = setupRows;
  setup.tables.add(`A1:C${setupRows.length}`, true, "ProjectSetupTable").style = "TableStyleMedium2";
  setup.showGridLines = false;
  setup.freezePanes.freezeRows(1);
  setup.getRange("A1:C1").format = { fill: "#1F4E78", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, wrapText: true };
  setup.getRange(`A2:C${setupRows.length}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
  setup.getRange(`A1:C${setupRows.length}`).format.verticalAlignment = "center";
  setup.getRange("A:A").format.columnWidth = 24;
  setup.getRange("B:B").format.columnWidth = 58;
  setup.getRange("C:C").format.columnWidth = 54;
  setup.getRange(`A2:C${setupRows.length}`).format.wrapText = true;
  setup.tabColor = "#5B9BD5";

  const definitionRows = [
    ["Field", "Allowed values / form", "Definition"],
    ["stage", "Smoke; Diagnostic; Tuning; ArchitectureDiagnostic; FormalBaseline; Ablation; Comparison; FinalTest", "实验在研究流程中的阶段。"],
    ["include_in_paper", "No; Candidate; Yes", "该运行是否进入论文结果候选或最终表格。"],
    ["test_used", "True; False", "是否使用 test 做模型评估。调参与诊断必须为 False。"],
    ["Precision", "0-1", "预测为目标的框中，与标注正确匹配的比例。"],
    ["Recall", "0-1", "标注目标中被模型正确检出的比例。"],
    ["mAP50", "0-1", "IoU=0.50 时各类别 AP 的平均值。"],
    ["mAP50-95", "0-1", "IoU 0.50 到 0.95、步长 0.05 的 COCO 风格 mAP。"],
    ["nbs", "positive integer", "Ultralytics nominal batch size，用于权重衰减缩放和梯度累积目标。"],
    ["batch", "positive integer", "每个训练迭代实际送入模型的图像数量。"],
    ["accumulate", "positive integer", "若干训练 batch 后执行一次 optimizer update；warmup 中可动态变化。"],
    ["steady_accumulate", "integer or NA", "warmup 结束后的稳定 accumulate；只记录实测或可核验值。"],
    ["best_epoch", "1..epochs_completed", "按最高 mAP50-95，若并列再按 mAP50 选出的 epoch。"],
    ["precision_at_best_epoch", "0-1", "best_epoch 对应同一行的 Precision。"],
    ["recall_at_best_epoch", "0-1", "best_epoch 对应同一行的 Recall。"],
    ["map50_at_best_epoch", "0-1", "best_epoch 对应同一行的 mAP50。"],
    ["map50_95_at_best_epoch", "0-1", "best_epoch 对应同一行的 mAP50-95，也是 best_epoch 的首要选择依据。"],
    ["max_precision_any_epoch", "0-1", "整个训练过程中独立出现的最高 Precision，不代表 best_epoch。"],
    ["epoch_of_max_precision", "1..epochs_completed", "max_precision_any_epoch 出现的 epoch；并列时记录最早 epoch。"],
    ["max_recall_any_epoch", "0-1", "整个训练过程中独立出现的最高 Recall，不代表 best_epoch。"],
    ["epoch_of_max_recall", "1..epochs_completed", "max_recall_any_epoch 出现的 epoch；并列时记录最早 epoch。"],
    ["changed_variable", "text", "相对 parent_experiment 有意改变的变量；单变量实验只列一项。"],
    ["parent_experiment", "experiment_id or NA", "作为直接对照基准的实验编号。"],
    ["model_parameters", "nonnegative integer or NA", "按本实验实际类别数构建的模型参数量。"],
    ["gflops_1280", "nonnegative number or NA", "1280×1280 输入下的模型计算量，单位 GFLOPs。"],
    ["NA", "text marker", "原始证据中没有可靠值；不得用猜测替代。"],
  ];
  definitions.getRange(`A1:C${definitionRows.length}`).values = definitionRows;
  definitions.tables.add(`A1:C${definitionRows.length}`, true, "FieldDefinitionsTable").style = "TableStyleMedium2";
  definitions.showGridLines = false;
  definitions.freezePanes.freezeRows(1);
  definitions.getRange("A1:C1").format = { fill: "#1F4E78", font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" }, wrapText: true };
  definitions.getRange(`A2:C${definitionRows.length}`).format.font = { name: "Arial", size: 10, color: "#1F2937" };
  definitions.getRange(`A1:C${definitionRows.length}`).format.verticalAlignment = "center";
  definitions.getRange("A:A").format.columnWidth = 24;
  definitions.getRange("B:B").format.columnWidth = 52;
  definitions.getRange("C:C").format.columnWidth = 72;
  definitions.getRange(`A2:C${definitionRows.length}`).format.wrapText = true;
  definitions.tabColor = "#A5A5A5";
  return workbook;
}

async function writeCsv(records) {
  const lines = [columns.map(csvEscape).join(","), ...records.map((record) => recordToArray(record).map(csvEscape).join(","))];
  await fs.writeFile(csvPath, `\uFEFF${lines.join("\r\n")}\r\n`, "utf8");
}

await fs.mkdir(outputDir, { recursive: true });
if (mode === "inspect") {
  const input = await FileBlob.load(xlsxPath);
  const existing = await SpreadsheetFile.importXlsx(input);
  const summary = await existing.inspect({ kind: "workbook,sheet,table", maxChars: 12000, tableMaxRows: 10, tableMaxCols: 12 });
  const preview = await existing.render({ sheetName: "Runs", range: "A1:O9", scale: 1, format: "png" });
  const metricsPreview = await existing.render({ sheetName: "Runs", range: "AY4:BM10", scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/preview_before_arch1.png`, new Uint8Array(await preview.arrayBuffer()));
  await fs.writeFile(`${outputDir}/preview_arch1_metrics.png`, new Uint8Array(await metricsPreview.arrayBuffer()));
  console.log(summary.ndjson);
  process.exit(0);
}
const historical = await historicalRecords();
let workbook;
let allRecords;
if (mode === "initial") {
  workbook = await createInitial(historical);
  allRecords = historical;
} else if (mode === "final") {
  const input = await FileBlob.load(xlsxPath);
  workbook = await SpreadsheetFile.importXlsx(input);
  const runs = workbook.worksheets.getItem("Runs");
  const used = runs.getUsedRange().values;
  if (used.flat().includes("Y11-DIAG-002")) throw new Error("Y11-DIAG-002 already exists; refusing duplicate registry row.");
  const current = await currentRecord();
  runs.tables.getItem("RunsTable").rows.add(null, [recordToArray(current, true)]);
  styleRuns(runs, historical.length + 1);
  allRecords = [...historical, current];
} else if (mode === "repair") {
  const input = await FileBlob.load(xlsxPath);
  workbook = await SpreadsheetFile.importXlsx(input);
  const setup = workbook.worksheets.getItem("ProjectSetup");
  setup.getRange("C10").values = [["dataset_integrity_after_yolo11s_nbs16.json"]];
  allRecords = [...historical, await currentRecord()];
} else if (mode === "semantics") {
  allRecords = [...historical, await currentRecord()];
  workbook = await createInitial(allRecords);
} else if (mode === "tune") {
  allRecords = [...historical, await currentRecord(), await tuneRecord()];
  workbook = await createInitial(allRecords);
} else if (mode === "tune2") {
  allRecords = [...historical, await currentRecord(), await tuneRecord(), await tune2Record()];
  workbook = await createInitial(allRecords);
} else if (mode === "arch1") {
  allRecords = [...historical, await currentRecord(), await tuneRecord(), await tune2Record(), await arch1Record()];
  workbook = await createInitial(allRecords);
} else {
  throw new Error(`Unknown mode: ${mode}`);
}

workbook.recalculate();
const inspectRuns = await workbook.inspect({ kind: "table", range: `Runs!A1:BM${allRecords.length + 4}`, include: "values,formulas", tableMaxRows: 12, tableMaxCols: 65, maxChars: 30000 });
const inspectSetup = await workbook.inspect({ kind: "table", range: "ProjectSetup!A1:C17", include: "values,formulas", tableMaxRows: 20, tableMaxCols: 3, maxChars: 10000 });
const inspectDefs = await workbook.inspect({ kind: "table", range: "FieldDefinitions!A1:C26", include: "values,formulas", tableMaxRows: 30, tableMaxCols: 3, maxChars: 14000 });
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan" });
await fs.writeFile(`${outputDir}/registry_inspection.ndjson`, [inspectRuns.ndjson, inspectSetup.ndjson, inspectDefs.ndjson, errors.ndjson].join("\n"), "utf8");

for (const [sheetName, range, filename] of [
  ["Runs", `A1:O${allRecords.length + 4}`, "preview_runs.png"],
  ["ProjectSetup", "A1:C17", "preview_project_setup.png"],
  ["FieldDefinitions", "A1:C26", "preview_field_definitions.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${outputDir}/${filename}`, new Uint8Array(await preview.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(xlsxPath);
await writeCsv(allRecords);
console.log(JSON.stringify({ mode, xlsxPath, csvPath, experimentIds: allRecords.map((row) => row.experiment_id) }, null, 2));
