import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const experimentId = "RTD-BASE-001-R1";
const registryXlsx = "D:/小论文/model_projects/yolo11/workspace/experiment_tracking/experiment_registry.xlsx";
const registryCsv = "D:/小论文/model_projects/yolo11/workspace/experiment_tracking/experiment_registry.csv";
const metricsPath = "D:/小论文/model_projects/RT-DETR/workspace/analysis/rtdetr_single_run_metrics.json";
const resultsPath = "D:/小论文/model_projects/RT-DETR/workspace/runs/RTD-BASE-001-R1_20260918_172120/results.csv";
const afterPreview = "D:/小论文/model_projects/RT-DETR/workspace/analysis/experiment_registry_runs_after.png";
const tempXlsx = `${registryXlsx}.${experimentId}.tmp.xlsx`;
const tempCsv = `${registryCsv}.${experimentId}.tmp.csv`;
const backupXlsx = `${registryXlsx}.${experimentId}.bak`;
const backupCsv = `${registryCsv}.${experimentId}.bak`;

function parseSimpleCsv(text) {
  const lines = text.replace(/^\uFEFF/, "").trim().split(/\r?\n/);
  const headers = lines[0].split(",").map((value) => value.trim());
  return lines.slice(1).map((line) => {
    const values = line.split(",").map((value) => value.trim());
    return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
  });
}

function csvEscape(value) {
  const text = value === true ? "True" : value === false ? "False" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

async function safeUnlink(path) {
  try {
    await fs.unlink(path);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
}

const metrics = JSON.parse(await fs.readFile(metricsPath, "utf8"));
if (metrics.experiment_id !== experimentId || metrics.status !== "success") {
  throw new Error("RT-DETR success metrics are missing or mismatched.");
}
if (metrics.epochs_completed !== 100 || metrics.test_execution_count !== 1 || !metrics.dataset_unchanged) {
  throw new Error("Completion, test-count, or dataset-integrity gate failed.");
}

const resultRows = parseSimpleCsv(await fs.readFile(resultsPath, "utf8"));
if (resultRows.length !== 100) throw new Error(`Expected 100 result rows, found ${resultRows.length}.`);
const numberAt = (row, key) => Number(row[key]);
const bestEpoch = metrics.best_epoch.best_epoch;
const bestRow = resultRows.find((row) => Number(row.epoch) === bestEpoch);
if (!bestRow) throw new Error(`Best epoch ${bestEpoch} is absent from results.csv.`);
const maxPrecisionRow = resultRows.reduce((best, row) => numberAt(row, "metrics/precision(B)") > numberAt(best, "metrics/precision(B)") ? row : best);
const maxRecallRow = resultRows.reduce((best, row) => numberAt(row, "metrics/recall(B)") > numberAt(best, "metrics/recall(B)") ? row : best);

const val = metrics.formal_evaluations.val;
const test = metrics.formal_evaluations.test;
const notes = JSON.stringify({
  architecture_modified: false,
  official_pretrained_weight_sha256: metrics.pretrained_weight_sha256,
  status_json: "D:\\小论文\\model_projects\\RT-DETR\\workspace\\analysis\\RTD-BASE-001-R1_status.json",
  args_yaml: `${metrics.run_dir}\\args.yaml`,
  runtime_scope: "training wall time; val/test reported separately in metrics JSON",
  best_epoch_definition: metrics.best_epoch.selection_basis,
  val: { overall: val.overall, per_class: val.classes },
  test: { overall: test.overall, per_class: test.classes },
  test_attempts: metrics.test_execution_count,
  dataset_manifest_sha256: metrics.dataset_manifest_after_sha256,
  dataset_unchanged: metrics.dataset_unchanged,
  generated_label_caches_cleaned: Object.keys(metrics.label_cache_audit.remaining_after_cleanup).length === 0,
  git_tracked_status_clean: metrics.source_after.git_tracked_status_clean,
  oom: false,
  nonfinite_failure: false,
  no_tuning_or_retry: true,
});

const row = [
  experimentId, 46283, "FormalBaseline", "Completed",
  "Official Ultralytics RT-DETR-L single-run FormalBaseline using COCO-pretrained rtdetr-l.pt and the fixed broken/loose dataset.",
  "RTD-BASE-001",
  "Training protocol unchanged. R1 wrapper integrity rule excludes Ultralytics-generated label cache files from formal image/.txt label hashing; fresh start from official weights.",
  "RT-DETR-L", "rtdetr-l.pt", "02_yolo11_dataset (fixed 107/23/23 split)",
  "D:\\小论文\\conductor_defect_dataset\\02_yolo11_dataset", 107, 23, 23, "0=broken; 1=loose",
  metrics.environment.git_commit, metrics.environment.ultralytics, 1280, 4, 64, 16, 100, 100, 0, "cuda:0", 42, false,
  "AdamW (optimizer=auto)", 0.001667, 0.9, 0.0005, 3, 1, 10, 0, 0, 0.5, 0.1, 0.5, 100, true,
  bestEpoch,
  numberAt(bestRow, "metrics/precision(B)"), numberAt(bestRow, "metrics/recall(B)"),
  numberAt(bestRow, "metrics/mAP50(B)"), numberAt(bestRow, "metrics/mAP50-95(B)"),
  numberAt(maxPrecisionRow, "metrics/precision(B)"), Number(maxPrecisionRow.epoch),
  numberAt(maxRecallRow, "metrics/recall(B)"), Number(maxRecallRow.epoch),
  val.classes.broken.map50, val.classes.broken.map50_95, val.classes.loose.map50, val.classes.loose.map50_95,
  metrics.model_profile.parameters, metrics.model_profile.gflops_at_1280,
  metrics.peak_vram.max_reserved_bytes / 1048576, metrics.runtime.training_wall_seconds,
  metrics.run_dir, metrics.best_pt.path, resultsPath, true, "Candidate",
  "Official RT-DETR-L comparison baseline completed; retain the one-run val/test results without tuning or test-driven changes.", notes,
];
if (row.length !== 65) throw new Error(`Registry row has ${row.length} columns instead of 65.`);

const existingCsv = await fs.readFile(registryCsv, "utf8");
if (existingCsv.includes(`\n${experimentId},`) || existingCsv.startsWith(`${experimentId},`)) {
  throw new Error(`${experimentId} already exists in registry CSV.`);
}
const csvHeader = existingCsv.replace(/^\uFEFF/, "").split(/\r?\n/, 1)[0].split(",");
if (csvHeader.length !== 65) throw new Error(`Registry CSV has ${csvHeader.length} columns instead of 65.`);
const csvRow = row.map((value, index) => index === 1 ? "2026-09-18" : value).map(csvEscape).join(",");
const csvOutput = `${existingCsv.replace(/\r?\n?$/, "\r\n")}${csvRow}\r\n`;

const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(registryXlsx));
const sheet = workbook.worksheets.getItem("Runs");
const table = sheet.tables.getItem("RunsTable");
const existingIds = sheet.getRange("A5:A200").values.flat().filter(Boolean);
if (existingIds.includes(experimentId)) throw new Error(`${experimentId} already exists in workbook.`);
table.rows.add(null, [row]);
await workbook.recalculate();

const afterInspect = await workbook.inspect({
  kind: "table",
  sheetId: "Runs",
  range: "A4:BM13",
  maxChars: 30000,
  tableMaxRows: 20,
  tableMaxCols: 65,
  tableMaxCellChars: 220,
});
const errorInspect = await workbook.inspect({
  kind: "match",
  sheetId: "Runs",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  maxChars: 10000,
});
const preview = await workbook.render({ sheetName: "Runs", range: "A1:BM13", scale: 1, format: "png" });
await fs.writeFile(afterPreview, new Uint8Array(await preview.arrayBuffer()));

await safeUnlink(tempXlsx);
await safeUnlink(tempCsv);
await safeUnlink(backupXlsx);
await safeUnlink(backupCsv);
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(tempXlsx);
await fs.writeFile(tempCsv, csvOutput, "utf8");

let xlsxBackedUp = false;
let csvBackedUp = false;
let xlsxInstalled = false;
let csvInstalled = false;
try {
  await fs.rename(registryXlsx, backupXlsx);
  xlsxBackedUp = true;
  await fs.rename(registryCsv, backupCsv);
  csvBackedUp = true;
  await fs.rename(tempXlsx, registryXlsx);
  xlsxInstalled = true;
  await fs.rename(tempCsv, registryCsv);
  csvInstalled = true;

  const verifyWorkbook = await SpreadsheetFile.importXlsx(await FileBlob.load(registryXlsx));
  const verifyValues = verifyWorkbook.worksheets.getItem("Runs").getRange("A13:BM13").values[0];
  if (verifyValues[0] !== experimentId || verifyValues.length !== 65) throw new Error("Workbook verification failed after installation.");
  const verifyCsv = await fs.readFile(registryCsv, "utf8");
  if (!verifyCsv.includes(`\n${experimentId},`) && !verifyCsv.includes(`\r\n${experimentId},`)) throw new Error("CSV verification failed after installation.");
} catch (error) {
  if (csvInstalled) await safeUnlink(registryCsv);
  if (xlsxInstalled) await safeUnlink(registryXlsx);
  if (csvBackedUp) await fs.rename(backupCsv, registryCsv);
  if (xlsxBackedUp) await fs.rename(backupXlsx, registryXlsx);
  throw error;
}

await safeUnlink(backupXlsx);
await safeUnlink(backupCsv);
console.log(JSON.stringify({
  updated: true,
  experimentId,
  workbook: registryXlsx,
  csv: registryCsv,
  preview: afterPreview,
  rowCount: 65,
  resultsRows: resultRows.length,
  inspect: afterInspect.ndjson,
  formulaErrors: errorInspect.ndjson,
}));
