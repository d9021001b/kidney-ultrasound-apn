import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.argv[2];
if (!root) throw new Error("Workspace root argument is required");
const sourcePath = path.join(root, "02_experiments", "NCKUH_DMSA_label_audit_20261102", "patient_label_linkage_private.csv");
const outputDir = path.join(root, "02_experiments", "NCKUH_clinical_data_request_20261107");
const outputPath = path.join(outputDir, "NCKUH_210_clinical_data_request_blinded_20261107.xlsx");

const csvText = await fs.readFile(sourcePath, "utf8");
const sourceBook = await Workbook.fromCSV(csvText, { sheetName: "Source" });
const sourceValues = sourceBook.worksheets.getItem("Source").getUsedRange().values;
const sourceHeader = sourceValues[0].map((value) => String(value).replace(/^\uFEFF/, "").trim());
const at = Object.fromEntries(sourceHeader.map((name, index) => [name, index]));
const eligible = sourceValues.slice(1).filter((row) => ["N", "L", "R", "B"].includes(String(row[at.DMSA_APN] || "").toUpperCase()));
if (eligible.length !== 210) throw new Error(`Expected 210 labeled patients, found ${eligible.length}`);

const workbook = Workbook.create();
const entry = workbook.worksheets.add("資料回填");
const codebook = workbook.worksheets.add("欄位定義");
entry.showGridLines = false;
codebook.showGridLines = false;
entry.tabColor = "#146C60";
codebook.tabColor = "#556B7A";

const columns = [
  "病歷號", "研究代碼", "腎臟超音波日期", "回填狀態",
  "最高體溫_C", "發燒天數", "CRP_mg_L", "WBC_10e9_L", "嗜中性球_%",
  "尿液WBC_HPF", "白血球酯酶", "亞硝酸鹽", "尿液培養陽性", "尿液病原菌", "血液培養陽性",
  "超音波前抗生素_小時", "檢驗距超音波_小時",
  "左腎_報告疑似APN", "右腎_報告疑似APN",
  "左腎_局部實質異常", "右腎_局部實質異常",
  "左腎_腎盂擴張", "右腎_腎盂擴張",
  "左腎_Doppler灌流缺損", "右腎_Doppler灌流缺損",
  "左側VUR分級", "右側VUR分級", "備註",
];
const rows = eligible.map((row) => [
  String(row[at.patient_id] || ""),
  String(row[at.patient_alias] || ""),
  String(row[at.image_exam_date] || ""),
  "待回填",
  ...Array(columns.length - 4).fill(null),
]);

entry.getRange("A1:AB1").values = [["成大 APN 多模態資料回填表（DMSA 結果盲化）", ...Array(27).fill(null)]];
entry.getRange("A2:AB2").values = [[
  "請以腎臟超音波日期附近的臨床事件為準；不得查閱或抄入 DMSA_APN、DMSA_RF、DMSA_LF。未知請留白，不以 0 代替。",
  ...Array(27).fill(null),
]];
entry.mergeCells("A1:J1");
entry.mergeCells("A2:J2");
entry.getRange("A3:AB3").values = [[...columns]];
entry.getRangeByIndexes(3, 0, rows.length, columns.length).values = rows;
entry.freezePanes.freezeRows(3);
entry.freezePanes.freezeColumns(3);

const titleFormat = {
  fill: "#146C60", font: { bold: true, color: "#FFFFFF", size: 16 },
  verticalAlignment: "center",
};
entry.getRange("A1:AB1").format = titleFormat;
entry.getRange("A1:AB1").format.rowHeight = 28;
entry.getRange("A2:AB2").format = {
  fill: "#E8F3F1", font: { color: "#143A36", italic: true, size: 10 }, wrapText: true,
};
entry.getRange("A2:AB2").format.rowHeight = 34;
entry.getRange("A3:AB3").format = {
  fill: "#284B63", font: { bold: true, color: "#FFFFFF", size: 10 },
  wrapText: true, horizontalAlignment: "center", verticalAlignment: "center",
};
entry.getRange("A3:AB3").format.rowHeight = 42;
entry.getRange(`A4:AB${rows.length + 3}`).format = {
  font: { size: 10, color: "#202A30" }, verticalAlignment: "center",
  borders: { bottom: { color: "#D7DEE2", style: "thin" } },
};
entry.getRange(`D4:AB${rows.length + 3}`).format.fill = "#FFFDF4";
entry.getRange(`A4:C${rows.length + 3}`).format.fill = "#F2F6F7";
entry.getRange(`A4:A${rows.length + 3}`).setNumberFormat("@");
entry.getRange(`C4:C${rows.length + 3}`).setNumberFormat("yyyy-mm-dd");

const widths = [14, 12, 16, 12, 12, 10, 12, 12, 12, 12, 12, 10, 14, 18, 14, 18, 18, 17, 17, 18, 18, 16, 16, 20, 20, 12, 12, 24];
for (let col = 0; col < widths.length; col += 1) {
  entry.getRangeByIndexes(0, col, rows.length + 3, 1).format.columnWidth = widths[col];
}

const definitions = [
  ["欄位", "定義／時間窗", "單位或允許值", "用途與注意事項"],
  ["病歷號", "成大醫院病歷號", "文字", "僅供院內回溯；不可放入公開資料。"],
  ["研究代碼", "本研究去識別化病人代碼", "P001-P210", "模型串接鍵，不含診斷資訊。"],
  ["腎臟超音波日期", "目前研究影像的檢查日期", "yyyy-mm-dd", "所有臨床欄位應盡量對齊此急性感染事件。"],
  ["回填狀態", "該列資料抽取進度", "待回填／完成／無病歷／時間窗不符", "完成不代表每欄都有值；未知保留空白。"],
  ["最高體溫_C", "超音波前後 48 小時內最高體溫", "攝氏度", "病人層級感染嚴重度。"],
  ["發燒天數", "至超音波當日累積發燒日數", "天", "若僅有範圍，於備註記錄。"],
  ["CRP_mg_L", "最接近超音波、且在前後 48 小時內的 CRP", "mg/L", "不得使用 DMSA 衍生數值。"],
  ["WBC_10e9_L", "最接近超音波、且在前後 48 小時內的周邊血白血球", "10^9/L", "保留原始單位；若來源不同請換算。"],
  ["嗜中性球_%", "與 WBC 同次檢驗之 neutrophil fraction", "%", "0-100。"],
  ["尿液WBC_HPF", "超音波前後 48 小時內最接近的尿沉渣 WBC", "cells/HPF", "若報告為區間，填上限並於備註保留原文。"],
  ["白血球酯酶", "同一急性事件尿液 leukocyte esterase", "陰性／trace／1+／2+／3+", "不得將缺失填成陰性。"],
  ["亞硝酸鹽", "同一急性事件尿液 nitrite", "陰性／陽性", "不得將缺失填成陰性。"],
  ["尿液培養陽性", "超音波前後 7 天內尿液培養", "否／是／污染／未做", "若陽性，於尿液病原菌填菌名。"],
  ["尿液病原菌", "尿液培養主要病原菌", "文字", "多菌種以分號分隔；污染不列病原。"],
  ["血液培養陽性", "超音波前後 7 天內血液培養", "否／是／污染／未做", "病人層級感染嚴重度。"],
  ["超音波前抗生素_小時", "首次有效抗生素至超音波的時間；超音波後給藥填負值", "小時", "治療可能改變影像可見度，必須保留時間方向。"],
  ["檢驗距超音波_小時", "採取 CRP/WBC 檢體時間減超音波時間", "小時", "負值表示檢驗早於超音波。"],
  ["左／右腎_報告疑似APN", "原始超音波報告是否對該側明確描述 APN 或腎實質感染", "否／是／不確定／未描述", "必須分側；抽取者不得參考 DMSA 結果。"],
  ["左／右腎_局部實質異常", "報告記載局部高低回音、腫大、楔形或皮髓質分化異常", "否／是／不確定／未描述", "保留報告文字於備註。"],
  ["左／右腎_腎盂擴張", "報告記載該側 renal pelvic dilatation／hydronephrosis", "否／是／不確定／未描述", "屬輔助徵象，非 APN 真值。"],
  ["左／右腎_Doppler灌流缺損", "Color/Power Doppler 報告記載該側灌流下降或缺損", "否／是／不確定／未做", "B-mode 未做 Doppler 者填未做，不填否。"],
  ["左／右側VUR分級", "急性事件前已知或同次評估之 vesicoureteral reflux grade", "0-5／未知／未做", "記錄檢查日期於備註；事件後才取得者需註明。"],
  ["備註", "原文、單位轉換、時間窗例外與資料衝突", "文字", "不要抄入 DMSA APN、RF 或 LF。"],
];
codebook.getRange("A1:D1").values = [["成大 APN 臨床資料回填欄位定義", null, null, null]];
codebook.getRange("A2:D2").values = [[
  "目的：新增與 DMSA 結果獨立的感染與分側資訊。DMSA_APN、DMSA_RF、DMSA_LF 均不得作為模型輸入。",
  null, null, null,
]];
codebook.mergeCells("A1:D1");
codebook.mergeCells("A2:D2");
codebook.getRangeByIndexes(2, 0, definitions.length, 4).values = definitions;
codebook.getRange("A1:D1").format = titleFormat;
codebook.getRange("A1:D1").format.rowHeight = 28;
codebook.getRange("A2:D2").format = { fill: "#E8F3F1", font: { italic: true, color: "#143A36" }, wrapText: true };
codebook.getRange("A2:D2").format.rowHeight = 34;
codebook.getRange("A3:D3").format = {
  fill: "#284B63", font: { bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center", verticalAlignment: "center", wrapText: true,
};
codebook.getRange(`A4:D${definitions.length + 2}`).format = {
  font: { size: 10, color: "#202A30" }, wrapText: true, verticalAlignment: "top",
  borders: { bottom: { color: "#D7DEE2", style: "thin" } },
};
codebook.getRange(`A4:A${definitions.length + 2}`).format.fill = "#F2F6F7";
codebook.getRange(`A1:A${definitions.length + 2}`).format.columnWidth = 25;
codebook.getRange(`B1:B${definitions.length + 2}`).format.columnWidth = 45;
codebook.getRange(`C1:C${definitions.length + 2}`).format.columnWidth = 26;
codebook.getRange(`D1:D${definitions.length + 2}`).format.columnWidth = 44;
codebook.getRange(`A3:D${definitions.length + 2}`).format.autofitRows();
codebook.freezePanes.freezeRows(3);

workbook.recalculate();
await fs.mkdir(outputDir, { recursive: true });
const inspection = await workbook.inspect({
  kind: "table", sheetId: "資料回填", range: "A1:H10", include: "values,formulas",
  tableMaxRows: 10, tableMaxCols: 8, maxChars: 6000,
});
console.log(inspection.ndjson);
const errors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 100 }, summary: "final formula error scan",
});
console.log(errors.ndjson);
for (const [sheetName, range, fileName] of [
  ["資料回填", "A1:J18", "preview_data_entry.png"],
  ["欄位定義", "A1:D24", "preview_codebook.png"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1.4, format: "png" });
  await fs.writeFile(path.join(outputDir, fileName), new Uint8Array(await preview.arrayBuffer()));
}
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, patients: eligible.length, columns: columns.length }));
