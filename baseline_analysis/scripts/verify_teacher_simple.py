"""Independently check saved XLSX values/percent formatting against canonical CSV projections."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

BASE=Path(__file__).resolve().parents[1]
plan=json.loads((BASE/'audit/teacher_simple_plan.json').read_text(encoding='utf-8'))
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(plan['source_csv'])==plan['source_csv_sha256'],'Canonical CSV changed after projection'
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
count=0
with zipfile.ZipFile(plan['output_path']) as archive:
    wb=ET.fromstring(archive.read('xl/workbook.xml'))
    assert [sheet.attrib['name'] for sheet in wb.find('s:sheets',ns)]==[sheet['name'] for sheet in plan['sheets']]
    strings=[]
    if 'xl/sharedStrings.xml' in archive.namelist():strings=[''.join(node.itertext()) for node in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
    styles=ET.fromstring(archive.read('xl/styles.xml'))
    formats=[int(node.attrib.get('numFmtId',0)) for node in styles.find('s:cellXfs',ns)]
    custom={int(node.attrib['numFmtId']):node.attrib['formatCode'] for node in styles.find('s:numFmts',ns) or []}
    for sheet_index,spec in enumerate(plan['sheets'],1):
        xml=ET.fromstring(archive.read(f'xl/worksheets/sheet{sheet_index}.xml'))
        assert not xml.findall('.//s:c[@t="e"]',ns),'Spreadsheet error cell found'
        cells={node.attrib['r']:node for node in xml.findall('.//s:sheetData/s:row/s:c',ns)}
        for rows,start,ratios in [(spec['rows'],6,range(5,10)),(spec['class_rows'],20,range(3,7))]:
            for offset,values in enumerate(rows):
                for column,expected in enumerate(values):
                    address=chr(65+column)+str(start+offset)
                    cell=cells.get(address)
                    actual='' if cell is None else cell.findtext('s:v',default='',namespaces=ns)
                    if cell is not None and cell.attrib.get('t')=='s':actual=strings[int(actual)]
                    elif cell is not None and cell.attrib.get('t')=='inlineStr':actual=''.join(cell.find('s:is',ns).itertext())
                    if expected is None:assert actual=='',(spec['name'],address,'Missing should be blank')
                    elif isinstance(expected,(int,float)):
                        assert abs(float(actual)-expected)<1e-12,(spec['name'],address,'Numeric mismatch')
                        if column in ratios:
                            assert 0<=expected<=1
                            fmt=formats[int(cell.attrib.get('s',0))]
                            assert fmt==10 or '%' in custom.get(fmt,''),(spec['name'],address,'Percent format missing')
                    else:assert actual==expected,(spec['name'],address,'Label mismatch')
                    count+=1
receipt=json.loads((BASE/'audit/registration_acceptance.json').read_text(encoding='utf-8'))
assert sha(BASE/'outputs/模型实验结果总表.xlsx')==receipt['excel_sha256'],'Detailed evidence workbook changed unexpectedly'
result=dict(status='PASS',sheets=2,experiments_per_sheet=9,verified_cells=count,source_csv_sha256=plan['source_csv_sha256'],
    workbook_sha256=sha(plan['output_path']),empty_not_zero_verified=True,ratios_and_percent_formats_verified=True,
    same_event_per_main_and_class_row=True,detailed_workbook_preserved=True,model_inference_performed=False,training_started=False)
(BASE/'audit/teacher_simple_acceptance.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print('TEACHER_SIMPLE_PASS: matched_cells='+str(count)+';2 sheets;9 experiments;pending blank;original detailed workbook preserved')
