import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "C:/Users/udoo_w2/Desktop/work_kidney_inflammation";
const PACKET = path.join(
  ROOT,
  "02_experiments/NCKUH_unilateral_visibility_annotation_20261103/blinded_packet",
);
const SOURCE = path.join(PACKET, "annotation_form.csv");
const OUTPUT = path.join(PACKET, "annotation_form_clinician_20261105.xlsx");
const PREVIEW = path.join(PACKET, "annotation_form_clinician_20261105_preview.png");

const csv = await fs.readFile(SOURCE, "utf8");
const lines = csv.trim().split(/\r?\n/);
const sourceHeaders = lines[0].split(",");
const caseIds = lines.slice(1).map((line) => line.split(",")[0]);
if (caseIds.length !== 83 || new Set(caseIds).size !== 83) {
  throw new Error(`Expected 83 unique blinded cases, found ${caseIds.length}`);
}

const expectedHeaders = [
  "case_id",
  "left_laterality_confirmed",
  "right_laterality_confirmed",
  "left_visible_abnormality",
  "right_visible_abnormality",
  "left_finding_type",
  "right_finding_type",
  "left_region",
  "right_region",
  "left_confidence_1_to_5",
  "right_confidence_1_to_5",
  "image_quality_exclusion",
  "reviewer_notes",
];
if (JSON.stringify(sourceHeaders) !== JSON.stringify(expectedHeaders)) {
  throw new Error("The source annotation schema changed; review before building the workbook.");
}

const displayHeaders = [
  "case_id\n盲化案例編號",
  "left_laterality_confirmed\n左腎影像側別確認",
  "right_laterality_confirmed\n右腎影像側別確認",
  "left_visible_abnormality\n左腎可見異常",
  "right_visible_abnormality\n右腎可見異常",
  "left_finding_type\n左腎主要徵象",
  "right_finding_type\n右腎主要徵象",
  "left_region\n左腎異常區域",
  "right_region\n右腎異常區域",
  "left_confidence_1_to_5\n左腎信心1至5",
  "right_confidence_1_to_5\n右腎信心1至5",
  "image_quality_exclusion\n影像品質需排除",
  "reviewer_notes\n判讀備註",
];

const workbook = Workbook.create();
const review = workbook.worksheets.add("標註表");
const instructions = workbook.worksheets.add("填寫說明");

review.showGridLines = false;
review.getRange("A1:M1").merge();
review.getRange("A1").values = [["成大單側APN腎臟超音波盲化可見性標註"]];
review.getRange("A1:M1").format = {
  fill: "#1F4E5F",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
review.getRange("A1:M1").format.rowHeight = 30;

review.getRange("A2:F2").values = [[
  "判讀者代碼",
  "",
  "判讀日期",
  "",
  "請依同名PNG影像判讀",
  "不可查看private_key或DMSA結果",
]];
review.getRange("A2:F2").format = {
  fill: "#EAF2F4",
  font: { bold: true, color: "#1F2937" },
  verticalAlignment: "center",
};
review.getRange("B2").format.fill = "#FFF2CC";
review.getRange("D2").format.fill = "#FFF2CC";
review.getRange("D2").setNumberFormat("yyyy-mm-dd");

review.getRange("A3:M3").merge();
review.getRange("A3").values = [[
  "側別確認僅核對左／右腎影像是否正確；可見異常是超音波影像判讀，不等同DMSA診斷。無明確異常時請選 no，不要依模型分數推測。",
]];
review.getRange("A3:M3").format = {
  fill: "#F7F7F7",
  font: { color: "#3F3F46" },
  wrapText: true,
  verticalAlignment: "center",
};
review.getRange("A3:M3").format.rowHeight = 34;

review.getRange("A4:M4").values = [displayHeaders];
review.getRange("A4:M4").format = {
  fill: "#2F6F73",
  font: { bold: true, color: "#FFFFFF", size: 10 },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { preset: "all", style: "thin", color: "#D1D5DB" },
};
review.getRange("A4:M4").format.rowHeight = 48;

const rows = caseIds.map((caseId) => [caseId, "", "", "", "", "", "", "", "", "", "", "", ""]);
review.getRange(`A5:M${caseIds.length + 4}`).values = rows;
review.getRange(`A5:M${caseIds.length + 4}`).format = {
  borders: { preset: "all", style: "thin", color: "#D9E1E3" },
  verticalAlignment: "center",
  wrapText: true,
};
review.getRange(`A5:A${caseIds.length + 4}`).format = {
  fill: "#EAF2F4",
  font: { bold: true, color: "#1F4E5F" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { preset: "all", style: "thin", color: "#D9E1E3" },
};
review.getRange(`B5:M${caseIds.length + 4}`).format.fill = "#FFFDF2";

const lastRow = caseIds.length + 4;
review.getRange(`B5:E${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain"] },
};
review.getRange(`F5:G${lastRow}`).dataValidation = {
  rule: {
    type: "list",
    values: [
      "none",
      "focal hyperechoic",
      "focal hypoechoic",
      "swelling",
      "reduced corticomedullary differentiation",
      "perfusion abnormality",
      "collecting-system abnormality",
      "other",
      "uncertain",
    ],
  },
};
review.getRange(`H5:I${lastRow}`).dataValidation = {
  rule: {
    type: "list",
    values: ["upper pole", "middle", "lower pole", "diffuse", "none", "uncertain"],
  },
};
review.getRange(`J5:K${lastRow}`).dataValidation = {
  rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
};
review.getRange(`L5:L${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["no", "yes", "uncertain"] },
};

review.freezePanes.freezeRows(4);
review.freezePanes.freezeColumns(1);
const table = review.tables.add(`A4:M${lastRow}`, true, "UnilateralVisibilityReview");
table.style = "TableStyleMedium2";
table.showFilterButton = true;
table.showBandedRows = true;

const widths = [14, 19, 19, 18, 18, 27, 27, 18, 18, 16, 16, 18, 34];
widths.forEach((width, index) => {
  review.getRangeByIndexes(0, index, lastRow, 1).format.columnWidth = width;
});
review.getRange(`A5:M${lastRow}`).format.rowHeight = 24;

instructions.showGridLines = false;
instructions.getRange("A1:D1").merge();
instructions.getRange("A1").values = [["填寫說明與允許值"]];
instructions.getRange("A1:D1").format = {
  fill: "#1F4E5F",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
instructions.getRange("A3:D3").values = [["步驟", "欄位", "允許值", "判讀原則"]];
instructions.getRange("A3:D3").format = {
  fill: "#2F6F73",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D1D5DB" },
};
const instructionRows = [
  [1, "case_id", "U001至U083", "開啟blinded_packet資料夾內同名PNG；不可查看private_key。"],
  [2, "laterality_confirmed", "yes / no / uncertain", "只確認頁面上的左、右腎影像側別是否可信。"],
  [3, "visible_abnormality", "yes / no / uncertain", "只依B-mode超音波可見內容判讀，不以DMSA或模型分數回填。"],
  [4, "finding_type", "下拉選單", "選擇最主要的一項；沒有異常選none，不確定選uncertain。"],
  [5, "region", "upper pole / middle / lower pole / diffuse / none / uncertain", "記錄主要異常所在區域。"],
  [6, "confidence_1_to_5", "1至5", "1代表很不確定，5代表高度確定。"],
  [7, "image_quality_exclusion", "no / yes / uncertain", "若影像品質不足以判讀，選yes並在備註說明。"],
  [8, "reviewer_notes", "自由文字", "可記錄人工標記、量測游標、視野不足或其他限制。"],
];
instructions.getRange("A4:D11").values = instructionRows;
instructions.getRange("A4:D11").format = {
  borders: { preset: "all", style: "thin", color: "#D9E1E3" },
  wrapText: true,
  verticalAlignment: "top",
};
instructions.getRange("A13:D13").merge();
instructions.getRange("A13").values = [[
  "完成後請直接回傳此xlsx。研究端會先做盲化一致性與可判讀性分析，再開啟獨立DMSA對照鍵。",
]];
instructions.getRange("A13:D13").format = {
  fill: "#FFF2CC",
  font: { bold: true, color: "#7F6000" },
  wrapText: true,
};
instructions.getRange("A:A").format.columnWidth = 8;
instructions.getRange("B:B").format.columnWidth = 27;
instructions.getRange("C:C").format.columnWidth = 42;
instructions.getRange("D:D").format.columnWidth = 58;
instructions.getRange("A3:D13").format.rowHeight = 34;
instructions.freezePanes.freezeRows(3);

await workbook.recalculate();
const inspection = await workbook.inspect({
  kind: "sheet,table,region",
  maxChars: 5000,
  tableMaxRows: 8,
  tableMaxCols: 13,
});
console.log(inspection.ndjson);

const preview = await workbook.render({
  sheetName: "填寫說明",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(PREVIEW, new Uint8Array(await preview.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(OUTPUT);
console.log(JSON.stringify({ output: OUTPUT, preview: PREVIEW, cases: caseIds.length }));
