import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const registryPath = "D:/小论文/model_projects/yolo11/workspace/experiment_tracking/experiment_registry.xlsx";
const previewPath = "D:/小论文/model_projects/RT-DETR/workspace/analysis/experiment_registry_runs_before.png";

const input = await FileBlob.load(registryPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const overview = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 10000,
  tableMaxRows: 12,
  tableMaxCols: 60,
  tableMaxCellChars: 120,
});
console.log(overview.ndjson);

const runs = await workbook.inspect({
  kind: "table",
  sheetId: "Runs",
  maxChars: 30000,
  tableMaxRows: 100,
  tableMaxCols: 80,
  tableMaxCellChars: 200,
});
console.log(runs.ndjson);

const formulas = await workbook.inspect({
  kind: "formula",
  sheetId: "Runs",
  maxChars: 10000,
  options: { maxResults: 200 },
});
console.log(formulas.ndjson);

const preview = await workbook.render({
  sheetName: "Runs",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
console.log(JSON.stringify({ previewPath }));
