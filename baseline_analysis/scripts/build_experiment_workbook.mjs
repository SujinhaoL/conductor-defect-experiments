// One workbook derived from the canonical CSV projections prepared by Python.
// Uses the bundled artifact-tool only. Never imports training frameworks.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile, FileBlob } from '@oai/artifact-tool';

const base = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const plan = JSON.parse(await fs.readFile(path.join(base, 'audit/workbook_plan.json'), 'utf8'));
if (process.argv.includes('--preview-before-step17') || process.argv.includes('--verify-existing-step17')) {
  const existing=await SpreadsheetFile.importXlsx(await FileBlob.load(plan.output_path));
  const after=process.argv.includes('--verify-existing-step17');
  for(const [sheetName,range,name] of (after ? [
    ['测试集结果','A11:M16','step17_test_detail.png'],['分类别结果','A49:K56','step17_classes_detail.png'],
    ['登记说明','A14:C17','step17_status_detail.png']
  ] : [['测试集结果','A1:M14','workbook_before_step17.png']])) {
    const preview=await existing.render({sheetName,range,scale:1,format:'png'});
    await fs.writeFile(path.join(base,'audit',name),new Uint8Array(await preview.arrayBuffer()));
  }
  if(after) {
    const values=existing.worksheets.getItem('测试集结果').getRange('I16:M16').values[0];
    if(values[0]!==0.05520830195125764 || values[1]!==0.018482112673997353 || values[4]!==0.14821428571428572 ||
       values[2]!==null || values[3]!==null) throw new Error('STEP17 exact metrics or missing P/R changed');
    console.log('SAVED_STEP17_EXACT_TEST_VALUES='+JSON.stringify(values));
  }
  process.exit(0);
}
if (process.argv.includes('--verify-existing-step16')) {
  const existing=await SpreadsheetFile.importXlsx(await FileBlob.load(plan.output_path));
  for(const [sheetName,range,name] of [
    ['分类别结果','A41:K48','step16_classes_detail.png'],
    ['证据索引','A25:D26','step16_evidence_detail.png'],
    ['登记说明','A14:C17','step16_status_detail.png'],
  ]) {
    const preview=await existing.render({sheetName,range,scale:1,format:'png'});
    await fs.writeFile(path.join(base,'audit',name),new Uint8Array(await preview.arrayBuffer()));
  }
  const values=existing.worksheets.getItem('测试集结果').getRange('I14:M14').values[0];
  if(values[0]!==0.019089244127215983 || values[1]!==0.004246304645777617 || values[4]!==0.08660714285714285 ||
     values[2]!==null || values[3]!==null) throw new Error('STEP16 exact test values or missing P/R changed');
  const model=existing.worksheets.getItem('测试集结果').getRange('B14').values[0][0];
  if(!model.includes('anchor4')) throw new Error('Anchor4 result is not clearly labeled');
  console.log('SAVED_STEP16_EXACT_TEST_VALUES='+JSON.stringify(values));
  process.exit(0);
}
if (process.argv.includes('--preview-before-step16')) {
  const existing=await SpreadsheetFile.importXlsx(await FileBlob.load(plan.output_path));
  const preview=await existing.render({sheetName:'测试集结果',range:'A1:M12',scale:1,format:'png'});
  await fs.writeFile(path.join(base,'audit/workbook_before_step16.png'),new Uint8Array(await preview.arrayBuffer()));
  console.log((await existing.inspect({kind:'sheet',include:'id,name',maxChars:1200})).ndjson);
  process.exit(0);
}
if (process.argv.includes('--verify-existing-step15')) {
  const existing=await SpreadsheetFile.importXlsx(await FileBlob.load(plan.output_path));
  for(const [sheetName,range,name] of [
    ['分类别结果','A33:K40','step15_classes_detail.png'],
    ['证据索引','A21:D22','step15_evidence_detail.png'],
    ['实验设置','A9:O11','step15_settings_detail.png'],
  ]) {
    const preview=await existing.render({sheetName,range,scale:1,format:'png'});
    await fs.writeFile(path.join(base,'audit',name),new Uint8Array(await preview.arrayBuffer()));
  }
  const values=existing.worksheets.getItem('测试集结果').getRange('I12:M12').values[0];
  if(values[0]!==0.008194318011112834 || values[1]!==0.002866443023255048 || values[4]!==0.0625 ||
     values[2]!==null || values[3]!==null) throw new Error('New exact test values or missing P/R changed');
  console.log('SAVED_STEP15_EXACT_TEST_VALUES='+JSON.stringify(values));
  process.exit(0);
}
if (process.argv.includes('--preview-before-step15')) {
  const existing = await SpreadsheetFile.importXlsx(await FileBlob.load(plan.output_path));
  const preview=await existing.render({sheetName:'测试集结果',range:'A1:M10',scale:1,format:'png'});
  await fs.writeFile(path.join(base,'audit/workbook_before_step15.png'),new Uint8Array(await preview.arrayBuffer()));
  console.log((await existing.inspect({kind:'table',range:'测试集结果!A4:M10',include:'values,formulas',tableMaxRows:7,tableMaxCols:13,maxChars:1600})).ndjson);
  process.exit(0);
}
if (process.argv.includes('--preview-existing')) {
  const existing = await SpreadsheetFile.importXlsx(await FileBlob.load(plan.output_path));
  const image = await existing.render({sheetName:'验证集结果',range:'A1:M14',scale:1,format:'png'});
  await fs.writeFile(path.join(base,'audit/workbook_before_step14.png'),new Uint8Array(await image.arrayBuffer()));
  const info = await existing.inspect({kind:'sheet',include:'id,name',maxChars:1500});
  console.log(info.ndjson);
  process.exit(0);
}
if (process.argv.includes('--verify-existing-step14')) {
  const existing = await SpreadsheetFile.importXlsx(await FileBlob.load(plan.output_path));
  for (const [name, range, output] of [
    ['分类别结果','A37:K44','step14_classes_detail.png'],
    ['证据索引','A21:D24','step14_evidence_detail.png'],
  ]) {
    const image=await existing.render({sheetName:name,range,scale:1,format:'png'});
    await fs.writeFile(path.join(base,'audit',output),new Uint8Array(await image.arrayBuffer()));
  }
  const broken=existing.worksheets.getItem('分类别结果').getRange('G43:H43').values[0];
  if (!(broken[0]>0 && broken[1]>0)) throw new Error('Exact broken AP was rounded to zero in stored workbook values');
  console.log('SAVED_WORKBOOK_EXACT_BROKEN_AP='+JSON.stringify(broken));
  process.exit(0);
}
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
  const subtitleCol=col(Math.min(spec.headers.length,spec.name==='证据索引'?4:13));
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
  const inputColumn=spec.headers.indexOf('输入');
  if(inputColumn>=0 && spec.rows.length) sheet.getRange(`${col(inputColumn+1)}5:${col(inputColumn+1)}${endRow}`).format.horizontalAlignment='center';
  sheet.freezePanes.freezeRows(4);
  sheet.freezePanes.freezeColumns(1);
}
workbook.recalculate();
const inspection=[];
for (const spec of plan.sheets) {
  const check=await workbook.inspect({kind:'table',range:`'${spec.name}'!A4:${col(Math.min(12,spec.headers.length))}${Math.min(10,4+spec.rows.length)}`,include:'values,formulas',tableMaxRows:7,tableMaxCols:12,maxChars:2000});
  inspection.push({sheet:spec.name,inspection:check.ndjson});
  const last=4+spec.rows.length;
  const preview=await workbook.render({sheetName:spec.name,range:`A1:${col(Math.min(spec.headers.length,spec.name==='证据索引'?4:13))}${last}`,scale:1,format:'png'});
  await fs.writeFile(path.join(plan.preview_dir,`${spec.name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:3000});
await fs.writeFile(path.join(base,'audit/workbook_inspection.json'),JSON.stringify({inspection,errors:errors.ndjson},null,2));
const result=await SpreadsheetFile.exportXlsx(workbook);
await result.save(plan.output_path);
console.log(`WORKBOOK_EXPORTED=${plan.output_path}`);
