// One workbook derived from the canonical CSV projections prepared by Python.
// Uses the bundled artifact-tool only. Never imports training frameworks.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const base = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const plan = JSON.parse(await fs.readFile(path.join(base, 'audit/workbook_plan.json'), 'utf8'));
const workbook = Workbook.create();
const col = n => { let s=''; while(n){const r=(n-1)%26;s=String.fromCharCode(65+r)+s;n=Math.floor((n-1)/26);} return s; };
await fs.mkdir(plan.preview_dir, { recursive: true });
for (const spec of plan.sheets) {
  const sheet = workbook.worksheets.add(spec.name);
  sheet.showGridLines = false;
  const endCol = col(spec.headers.length);
  const endRow = 4 + spec.rows.length;
  const used = sheet.getRange(`A1:${endCol}${endRow}`);
  used.format.font = { name: 'Microsoft YaHei', size: 10, color: '#263647' };
  used.format.verticalAlignment = 'center';
  used.format.rowHeight = 32;
  sheet.getRange('A1').values = [[spec.name]];
  sheet.getRange(`A1:${endCol}1`).format.rowHeight = 30;
  sheet.getRange('A1').format.font = { name:'Microsoft YaHei', size:16, bold:true, color:'#153B56' };
  sheet.getRange('A2').values = [[spec.subtitle]];
  // Subtitle intentionally spans a merged display row; the actual data table
  // has a single unmerged header row and one typed observation per row.
  const subtitleCol=col(Math.min(spec.headers.length,spec.name==='证据索引'?4:12));
  sheet.mergeCells(`A2:${subtitleCol}2`);
  sheet.getRange(`A2:${subtitleCol}2`).format = { wrapText:true, rowHeight:40, fill:'#EDF4F8', font:{name:'Microsoft YaHei',size:10,color:'#344F64'} };
  sheet.getRange(`A4:${endCol}4`).values = [spec.headers];
  sheet.getRange(`A4:${endCol}4`).format = {fill:'#153B56',font:{name:'Microsoft YaHei',size:10,bold:true,color:'#FFFFFF'},horizontalAlignment:'center',wrapText:true,rowHeight:44};
  if(spec.rows.length) {
    sheet.getRange(`A5:${endCol}${endRow}`).values = spec.rows;
    sheet.getRange(`A5:${endCol}${endRow}`).format.wrapText = true;
    for(let r=5;r<=endRow;r++) {
      if(r%2===1) sheet.getRange(`A${r}:${endCol}${r}`).format.fill='#F2F6FA';
      const textLengths=spec.rows[r-5].map((v,i)=>typeof v==='string'?Math.ceil(v.length/(spec.widths[i]*0.65)):1);
      const lines=Math.max(2,...textLengths);
      sheet.getRange(`A${r}:${endCol}${r}`).format.rowHeight=Math.max(38,Math.min(170,lines*17+8));
    }
    for(const c of spec.ratio_columns) {
      sheet.getRange(`${col(c+1)}5:${col(c+1)}${endRow}`).setNumberFormat('0.00%');
    }
  }
  for(let c=0;c<spec.headers.length;c++) {
    const range=sheet.getRange(`${col(c+1)}1:${col(c+1)}${endRow}`);
    range.format.columnWidth=spec.widths[c];
    if(spec.rows.some(r=>typeof r[c]==='number')) {
      sheet.getRange(`${col(c+1)}5:${col(c+1)}${endRow}`).format.horizontalAlignment='right';
    }
  }
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(1);
}
workbook.recalculate();
const inspection=[];
for (const spec of plan.sheets) {
  const check=await workbook.inspect({kind:'table',range:`'${spec.name}'!A4:${col(Math.min(12,spec.headers.length))}${Math.min(10,4+spec.rows.length)}`,include:'values,formulas',tableMaxRows:7,tableMaxCols:12,maxChars:2000});
  inspection.push({sheet:spec.name,inspection:check.ndjson});
  const last=4+spec.rows.length;
  const preview=await workbook.render({sheetName:spec.name,range:`A1:${col(Math.min(spec.headers.length,spec.name==='证据索引'?4:12))}${last}`,scale:1,format:'png'});
  await fs.writeFile(path.join(plan.preview_dir,`${spec.name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3000});
await fs.writeFile(path.join(base,'audit/workbook_inspection.json'),JSON.stringify({inspection,errors:errors.ndjson},null,2));
const result=await SpreadsheetFile.exportXlsx(workbook);
await result.save(plan.output_path);
console.log(`WORKBOOK_EXPORTED=${plan.output_path}`);
