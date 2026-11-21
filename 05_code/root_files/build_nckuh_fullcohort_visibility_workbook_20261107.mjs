import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = "C:/Users/udoo_w2/Desktop/work_kidney_inflammation";
const PACKET = path.join(
  ROOT,
  "02_experiments/NCKUH_fullcohort_visibility_annotation_20261103/blinded_packet",
);
const MANIFEST = path.join(PACKET, "page_manifest.csv");
const OUTPUT = path.join(PACKET, "annotation_form_clinician_fullcohort_20261107.xlsx");
const PREVIEW = path.join(PACKET, "annotation_form_clinician_fullcohort_20261107_preview.png");

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (quoted && line[index + 1] === '"') {
        value += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}

const manifestText = (await fs.readFile(MANIFEST, "utf8")).replace(/^\uFEFF/, "").trim();
const manifestLines = manifestText.split(/\r?\n/);
const manifestHeaders = parseCsvLine(manifestLines[0]);
const expectedManifestHeaders = ["case_id", "side", "view_type", "page_file", "image_slot"];
if (JSON.stringify(manifestHeaders) !== JSON.stringify(expectedManifestHeaders)) {
  throw new Error("Unexpected page manifest schema.");
}

const viewsByCaseSide = new Map();
for (const line of manifestLines.slice(1)) {
  const [caseId, side, viewType] = parseCsvLine(line);
  const key = `${caseId}|${side}`;
  if (!viewsByCaseSide.has(key)) viewsByCaseSide.set(key, new Set());
  viewsByCaseSide.get(key).add(viewType);
}

const caseIds = [...new Set([...viewsByCaseSide.keys()].map((key) => key.split("|")[0]))].sort();
if (caseIds.length !== 210 || caseIds[0] !== "V001" || caseIds.at(-1) !== "V210") {
  throw new Error(`Expected V001-V210, found ${caseIds.length} cases.`);
}

const displayHeaders = [
  "case_id\n盲化案例編號",
  "kidney_side\n腎臟側別",
  "available_views\n可用切面",
  "image_readable\n影像可判讀",
  "laterality_confirmed\n側別確認",
  "apn_compatible_finding\nAPN相容可見異常",
  "primary_echo_pattern\n主要回音型態",
  "abnormal_region\n異常區域",
  "wedge_shaped_focus\n楔形局部異常",
  "renal_enlargement\n腎臟腫大",
  "poor_corticomedullary_diff\n皮髓質分界不清",
  "pelvic_dilation\n腎盂擴張",
  "perfusion_defect\n灌流缺損",
  "artifact_obscuration\n標記或偽影遮蔽",
  "confidence_1_to_5\n判讀信心1至5",
  "reviewer_notes\n判讀備註",
];

const rows = [];
for (const caseId of caseIds) {
  for (const side of ["left", "right"]) {
    const views = [...(viewsByCaseSide.get(`${caseId}|${side}`) ?? new Set())].sort();
    rows.push([
      caseId,
      side,
      views.length ? views.join("+") : "none",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
      "",
    ]);
  }
}

const workbook = Workbook.create();
const review = workbook.worksheets.add("腎臟標註");
const guide = workbook.worksheets.add("填寫說明");

review.showGridLines = false;
review.getRange("A1:P1").merge();
review.getRange("A1").values = [["成大全隊列腎臟超音波盲化可見性標註"]];
review.getRange("A1:P1").format = {
  fill: "#174A5B",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
review.getRange("A1:P1").format.rowHeight = 30;

review.getRange("A2:H2").values = [[
  "判讀者代碼",
  "",
  "判讀日期",
  "",
  "每個case對應同名PNG",
  "左右腎分列填寫",
  "不可查看DMSA對照鍵",
  "不可依模型分數推測",
]];
review.getRange("A2:H2").format = {
  fill: "#E6F0F2",
  font: { bold: true, color: "#1F2937" },
  verticalAlignment: "center",
};
review.getRange("B2").format.fill = "#FFF2CC";
review.getRange("D2").format.fill = "#FFF2CC";
review.getRange("D2").setNumberFormat("yyyy-mm-dd");

review.getRange("A3:P3").merge();
review.getRange("A3").values = [[
  "請只依頁面所示B-mode或彩色都卜勒影像判讀。APN相容異常不等同DMSA診斷；未顯示都卜勒時，灌流缺損請選not shown。",
]];
review.getRange("A3:P3").format = {
  fill: "#F7F7F7",
  font: { color: "#3F3F46" },
  wrapText: true,
  verticalAlignment: "center",
};
review.getRange("A3:P3").format.rowHeight = 34;

review.getRange("A4:P4").values = [displayHeaders];
review.getRange("A4:P4").format = {
  fill: "#2F6F73",
  font: { bold: true, color: "#FFFFFF", size: 9 },
  wrapText: true,
  horizontalAlignment: "center",
  verticalAlignment: "center",
  borders: { preset: "all", style: "thin", color: "#D1D5DB" },
};
review.getRange("A4:P4").format.rowHeight = 52;

const lastRow = rows.length + 4;
review.getRange(`A5:P${lastRow}`).values = rows;
review.getRange(`A5:P${lastRow}`).format = {
  borders: { preset: "all", style: "thin", color: "#D9E1E3" },
  verticalAlignment: "center",
  wrapText: true,
};
review.getRange(`A5:C${lastRow}`).format.fill = "#EAF2F4";
review.getRange(`A5:B${lastRow}`).format.font = { bold: true, color: "#174A5B" };
review.getRange(`D5:P${lastRow}`).format.fill = "#FFFDF2";
review.getRange(`A5:P${lastRow}`).format.rowHeight = 24;

review.getRange(`D5:D${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain", "not imaged"] },
};
review.getRange(`E5:E${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain", "not imaged"] },
};
review.getRange(`F5:F${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain", "not assessable"] },
};
review.getRange(`G5:G${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["none", "hyperechoic", "hypoechoic", "mixed", "heterogeneous", "uncertain", "not assessable"] },
};
review.getRange(`H5:H${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["upper pole", "middle", "lower pole", "multifocal", "diffuse", "none", "uncertain", "not assessable"] },
};
for (const column of ["I", "J", "K", "L", "N"]) {
  review.getRange(`${column}5:${column}${lastRow}`).dataValidation = {
    rule: { type: "list", values: ["yes", "no", "uncertain", "not assessable"] },
  };
}
review.getRange(`M5:M${lastRow}`).dataValidation = {
  rule: { type: "list", values: ["yes", "no", "uncertain", "not shown", "not assessable"] },
};
review.getRange(`O5:O${lastRow}`).dataValidation = {
  rule: { type: "whole", operator: "between", formula1: 1, formula2: 5 },
};

review.freezePanes.freezeRows(4);
review.freezePanes.freezeColumns(3);
const table = review.tables.add(`A4:P${lastRow}`, true, "FullCohortKidneyVisibilityReview");
table.style = "TableStyleMedium2";
table.showFilterButton = true;
table.showBandedRows = true;

const widths = [13, 12, 14, 15, 17, 22, 20, 18, 18, 16, 23, 16, 16, 19, 17, 35];
widths.forEach((width, index) => {
  review.getRangeByIndexes(0, index, lastRow, 1).format.columnWidth = width;
});

guide.showGridLines = false;
guide.getRange("A1:D1").merge();
guide.getRange("A1").values = [["全隊列腎臟超音波盲化判讀說明"]];
guide.getRange("A1:D1").format = {
  fill: "#174A5B",
  font: { bold: true, color: "#FFFFFF", size: 16 },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
guide.getRange("A3:D3").values = [["欄位", "允許值", "判讀原則", "建模用途"]];
guide.getRange("A3:D3").format = {
  fill: "#2F6F73",
  font: { bold: true, color: "#FFFFFF" },
  wrapText: true,
  borders: { preset: "all", style: "thin", color: "#D1D5DB" },
};
const guideRows = [
  ["case_id / kidney_side", "V001-V210 / left-right", "依同名PNG判讀，每位病人的左右腎各一列。", "腎臟層級連結"],
  ["available_views", "L、T、L+T、none", "由盲化影像清單預填，不需修改。", "控制切面缺失"],
  ["image_readable", "yes / no / uncertain / not imaged", "影像足以進行實質判讀才選yes。", "品質排除"],
  ["laterality_confirmed", "yes / no / uncertain / not imaged", "只確認頁面標示的左右側是否可信。", "側別品質控制"],
  ["apn_compatible_finding", "yes / no / uncertain / not assessable", "記錄B-mode或彩色都卜勒上是否有APN相容異常；不等同DMSA診斷。", "主要可見性特徵"],
  ["primary_echo_pattern", "none / hyperechoic / hypoechoic / mixed / heterogeneous / uncertain / not assessable", "選最主要的回音型態。", "可解釋影像特徵"],
  ["abnormal_region", "upper pole / middle / lower pole / multifocal / diffuse / none / uncertain / not assessable", "記錄主要異常區域。", "局部病灶位置"],
  ["wedge_shaped_focus", "yes / no / uncertain / not assessable", "是否有楔形或三角形局部異常。", "局部實質異常"],
  ["renal_enlargement", "yes / no / uncertain / not assessable", "依整體腎臟外觀判讀是否腫大。", "瀰漫性變化"],
  ["poor_corticomedullary_diff", "yes / no / uncertain / not assessable", "皮質與髓質界線是否不清。", "實質組織變化"],
  ["pelvic_dilation", "yes / no / uncertain / not assessable", "是否有腎盂或集尿系統擴張。", "泌尿道共變項"],
  ["perfusion_defect", "yes / no / uncertain / not shown / not assessable", "只有頁面有彩色都卜勒資訊時才判讀；未顯示時選not shown。", "灌流資訊"],
  ["artifact_obscuration", "yes / no / uncertain / not assessable", "量測游標、文字、陰影或其他偽影是否遮蔽腎實質。", "偽影控制"],
  ["confidence_1_to_5", "1-5", "1為很不確定，5為高度確定。", "信心加權或敏感度分析"],
  ["reviewer_notes", "自由文字", "記錄主要限制或其他可見徵象。", "質性補充"],
];
guide.getRange(`A4:D${guideRows.length + 3}`).values = guideRows;
guide.getRange(`A4:D${guideRows.length + 3}`).format = {
  borders: { preset: "all", style: "thin", color: "#D9E1E3" },
  wrapText: true,
  verticalAlignment: "top",
};
guide.getRange("A20:D20").merge();
guide.getRange("A20").values = [[
  "完成後只需回傳此XLSX。研究端會先完成盲化品質與一致性檢查，再由獨立程式連結DMSA對照鍵。",
]];
guide.getRange("A20:D20").format = {
  fill: "#FFF2CC",
  font: { bold: true, color: "#7F6000" },
  wrapText: true,
};
guide.getRange("A:A").format.columnWidth = 31;
guide.getRange("B:B").format.columnWidth = 48;
guide.getRange("C:C").format.columnWidth = 62;
guide.getRange("D:D").format.columnWidth = 28;
guide.getRange("A3:D20").format.rowHeight = 38;
guide.freezePanes.freezeRows(3);

await workbook.recalculate();
const inspection = await workbook.inspect({
  kind: "sheet,table,region,formula",
  maxChars: 8000,
  tableMaxRows: 8,
  tableMaxCols: 16,
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
console.log(JSON.stringify({ output: OUTPUT, preview: PREVIEW, cases: caseIds.length, kidneyRows: rows.length }));
