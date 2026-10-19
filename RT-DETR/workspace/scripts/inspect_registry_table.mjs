import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const registryPath = "D:/小论文/model_projects/yolo11/workspace/experiment_tracking/experiment_registry.xlsx";
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(registryPath));
const sheet = workbook.worksheets.getItem("Runs");
console.log(JSON.stringify({
  sheets: workbook.worksheets.items.map((item) => item.name),
  tables: sheet.tables.items.map((item) => ({ name: item.name, range: item.getRange().address })),
}));
