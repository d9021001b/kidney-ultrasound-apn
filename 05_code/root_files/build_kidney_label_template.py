"""
Build the per-kidney DMSA label request for the clinical team.

Why.  Every label used so far is patient-constant: all views of a child with acute
pyelonephritis are labelled positive, including views of a kidney that DMSA may have read
as normal.  The clinical lead confirmed on 2026-09-14 that DMSA diagnoses exist separately
for the left and right kidney.  Nothing in the repository or in ki-data/ stores them -- the
-L / -R token on many ki-data filenames marks where the kidney sits in the frame, not which
side is affected -- so they have to come from the clinical records.

What this writes.  One row per ultrasound EXAMINATION, covering
  * every exam already analysed (362 patients),
  * other exam dates found in ki-data/ for those same patients,
  * every exam of the patients who have images in ki-data/ but no label anywhere.
The clinician fills the DMSA date and a 0 / 1 / NA result for each kidney.  The stored
patient-level label is deliberately NOT shown, so reading the DMSA report is not anchored to
it; apply_kidney_labels.py compares the two afterwards and lists every disagreement.

Usage
    python build_kidney_label_template.py
    -> kidney_label_request/kidney_label_request.xlsx  (plus a UTF-8 .csv copy)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
COHORT_CSV = ROOT / "patient_characteristics" / "used_ultrasound_image_filenames.csv"
KI_IMAGES_CSV = ROOT / "scratch_kidata" / "ki_images_parsed.csv"
KI_MASTER_CSV = ROOT / "scratch_kidata" / "master_patient_table.csv"

VIEWS = ["LL", "LT", "RL", "RT"]

# Header cell = "<key> <Chinese>".  apply_kidney_labels.py keys on the text before the first
# space, so the Chinese half can be reworded without breaking the round trip.
COLS = [
    ("row", "序號"),
    ("patient_id", "病歷號"),
    ("us_exam_date", "超音波檢查日"),
    ("site", "院所"),
    ("cohort_status", "資料狀態"),
    ("views_on_file", "已有影像視角"),
    ("n_images", "影像張數"),
    ("sex", "性別"),
    ("age_years", "年齡(歲)"),
    ("dmsa_date", "DMSA檢查日"),
    ("left_dmsa", "左腎DMSA"),
    ("right_dmsa", "右腎DMSA"),
    ("left_finding", "左腎描述"),
    ("right_finding", "右腎描述"),
    ("notes", "備註"),
]
KEYS = [k for k, _ in COLS]
FILL_KEYS = {"dmsa_date", "left_dmsa", "right_dmsa", "left_finding", "right_finding", "notes"}
WIDTHS = {"row": 6, "patient_id": 12, "us_exam_date": 13, "site": 9, "cohort_status": 34,
          "views_on_file": 14, "n_images": 8, "sex": 8, "age_years": 9, "dmsa_date": 13,
          "left_dmsa": 11, "right_dmsa": 11, "left_finding": 24, "right_finding": 24,
          "notes": 36}

STATUS_ZH = {
    "analysed": "已分析",
    "analysed_patient_extra_exam": "已分析病人的另一次檢查",
    "new_patient": "新病人(目前無標籤)",
}
STATUS_ORDER = {"analysed": 0, "analysed_patient_extra_exam": 1, "new_patient": 2}


def pid8(value) -> str | None:
    m = re.match(r"\s*0*(\d+)", str(value))
    return m.group(1).zfill(8) if m else None


def norm_date(raw) -> tuple[str | None, str]:
    """(YYYYMMDD, note).  The analysed cohort carries one malformed 9-digit date, 201208087,
    which ki-data records as 20120808 for the same patient."""
    s = re.sub(r"\D", "", str(raw))
    if len(s) == 8 and pd.notna(pd.to_datetime(s, format="%Y%m%d", errors="coerce")):
        return s, ""
    if len(s) == 9 and pd.notna(pd.to_datetime(s[:8], format="%Y%m%d", errors="coerce")):
        return s[:8], f"source filename date {s} read as {s[:8]}; please confirm"
    return None, f"unreadable exam date {raw!r}"


def views_on_file(series: pd.Series) -> str:
    present = set(series.astype(str))
    return " ".join(v for v in VIEWS if v in present)


def analysed_exams() -> pd.DataFrame:
    df = pd.read_csv(COHORT_CSV, encoding="utf-8-sig", dtype=str)
    df["pid8"] = df["patient_id"].map(pid8)
    parsed = df["exam_date_raw"].map(norm_date)
    df["date"] = parsed.map(lambda t: t[0])
    df["date_note"] = parsed.map(lambda t: t[1])
    out = (df.groupby(["pid8", "date"], dropna=False)
             .agg(site=("site", "first"), views_on_file=("view", views_on_file),
                  n_images=("filename", "size"), sex=("gender", "first"),
                  age_years=("age_years", "first"), notes=("date_note", "first"))
             .reset_index())
    out["cohort_status"] = "analysed"
    return out


def kidata_exams(analysed: pd.DataFrame) -> pd.DataFrame:
    ki = pd.read_csv(KI_IMAGES_CSV, dtype=str)
    ki["pid8"] = ki["pid8"].map(pid8)
    ki = ki[ki["pid8"].notna()
            & ki["date"].fillna("").str.fullmatch(r"\d{8}")
            & ki["view"].isin(VIEWS)]

    master = pd.read_csv(KI_MASTER_CSV, dtype=str)
    master["pid8"] = master["pid8"].map(pid8)
    in_cych3 = set(master.loc[master["f_CYCH3"].astype(int) > 0, "pid8"])
    in_nckuh3 = set(master.loc[master["f_NCKUH3"].astype(int) > 0, "pid8"])

    cohort_site = analysed.groupby("pid8")["site"].first().to_dict()
    cohort_sex = analysed.groupby("pid8")["sex"].first().to_dict()
    known = set(zip(analysed["pid8"], analysed["date"]))

    g = (ki.groupby(["pid8", "date"])
           .agg(views_on_file=("view", views_on_file),
                n_images=("name", "nunique"),
                folders=("folder", lambda s: "|".join(sorted(set(s)))))
           .reset_index())
    g = g[[(p, d) not in known for p, d in zip(g["pid8"], g["date"])]].copy()

    def site_of(p: str) -> str:
        if p in cohort_site:
            return cohort_site[p]
        if p in in_cych3:
            return "CYCH"
        if p in in_nckuh3:
            return "NCKUH"
        return ""

    g["site"] = g["pid8"].map(site_of)
    g["cohort_status"] = ["analysed_patient_extra_exam" if p in cohort_site else "new_patient"
                          for p in g["pid8"]]
    g["sex"] = [cohort_sex.get(p, "") if s == "analysed_patient_extra_exam" else ""
                for p, s in zip(g["pid8"], g["cohort_status"])]
    g["age_years"] = ""
    g["notes"] = ["ki-data folder: " + f + ("" if s else "; site unknown, please fill")
                  for f, s in zip(g["folders"], g["site"])]
    return g.drop(columns="folders")


def build() -> tuple[pd.DataFrame, dict]:
    a = analysed_exams()
    k = kidata_exams(a)
    rows = pd.concat([a, k], ignore_index=True)
    rows["_o"] = rows["cohort_status"].map(STATUS_ORDER)
    rows = rows.sort_values(["_o", "site", "pid8", "date"], na_position="last").reset_index(drop=True)

    table = pd.DataFrame({
        "row": range(1, len(rows) + 1),
        "patient_id": rows["pid8"],
        "us_exam_date": rows["date"].fillna(""),
        "site": rows["site"].fillna(""),
        "cohort_status": [f"{s} {STATUS_ZH[s]}" for s in rows["cohort_status"]],
        "views_on_file": rows["views_on_file"],
        "n_images": rows["n_images"],
        "sex": rows["sex"].fillna(""),
        "age_years": rows["age_years"].fillna(""),
        "dmsa_date": "", "left_dmsa": "", "right_dmsa": "",
        "left_finding": "", "right_finding": "",
        "notes": rows["notes"].fillna(""),
    })[KEYS]

    counts = {}
    for status in STATUS_ORDER:
        sub = rows[rows["cohort_status"] == status]
        counts[status] = {"exams": int(len(sub)), "patients": int(sub["pid8"].nunique())}
    counts["total_rows"] = int(len(rows))
    counts["new_patient_site_unknown"] = int(((rows["cohort_status"] == "new_patient")
                                             & (rows["site"].fillna("") == "")).sum())
    return table, counts


def instructions(c: dict) -> list[str]:
    a, x, n = c["analysed"], c["analysed_patient_extra_exam"], c["new_patient"]
    return [
        f"本表共 {c['total_rows']} 列，每一列是一次超音波檢查："
        f"已分析 {a['exams']} 列（{a['patients']} 位病人）、"
        f"已分析病人的另一次檢查 {x['exams']} 列（{x['patients']} 位）、"
        f"新病人 {n['exams']} 列（{n['patients']} 位）。",
        "黃色欄位是要填的，其他欄位請勿修改。",
        "左腎DMSA、右腎DMSA：0 = 正常；1 = 急性腎盂腎炎；NA = 該側未評估或報告未提及。"
        "儲存格有下拉選單，只接受這三種值。無法判定時請填 NA，並在備註說明原因。",
        "DMSA檢查日：請填 YYYYMMDD。若同一次發燒做了不只一次 DMSA，"
        "請填最接近超音波檢查日的那一次，並在備註註明。",
        "左腎描述、右腎描述（選填）：例如「上極局部攝取缺損」、「瀰漫性攝取下降」。"
        "有這些描述，之後才能分析病灶範圍與超音波表現的關係。",
        "若某側 DMSA 顯示的是舊疤痕而非急性病灶，或該腎有先天異常（如發育不全、重複集尿系統），"
        "請在備註註明，這類腎臟會另行處理，不會直接當作陽性或陰性。",
        "本表刻意不顯示目前使用的病人層級標籤，避免判讀受其影響。回填後程式會自動比對："
        "若某次檢查左右皆為 0 但原標籤是陽性，或反之，會列出來請您再確認一次。",
        "資料狀態「analysed 已分析」：目前分析中的病人，請優先填寫這一批。",
        "資料狀態「analysed_patient_extra_exam」：同一批病人在 ki-data 裡另一個日期的檢查，"
        "若該次也有 DMSA 請一併填寫。",
        "資料狀態「new_patient 新病人」：ki-data 裡有影像、但專案中找不到任何標籤的病人。"
        "填上 DMSA 結果後即可納入分析。這些病人目前也缺院所、性別、年齡，請一併補上；"
        f"其中 {c['new_patient_site_unknown']} 列的院所完全無從判斷，院所欄可下拉選 CYCH 或 NCKUH。",
        "病歷號與日期欄是文字格式，請保留前導 0。",
    ]


def write_excel(table: pd.DataFrame, counts: dict, path: Path) -> None:
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    headers = [f"{k} {zh}" for k, zh in COLS]
    sheet = table.copy()
    sheet.columns = headers

    with pd.ExcelWriter(path, engine="openpyxl") as w:
        pd.DataFrame({"填寫說明": instructions(counts)}).to_excel(w, sheet_name="填寫說明", index=False)
        sheet.to_excel(w, sheet_name="待填表", index=False)

        guide = w.sheets["填寫說明"]
        guide.column_dimensions["A"].width = 120
        for (cell,) in guide.iter_rows(min_row=2):
            cell.alignment = Alignment(wrap_text=True, vertical="top")

        ws = w.sheets["待填表"]
        last = len(sheet) + 1
        head = PatternFill("solid", fgColor="DCE6F1")
        todo = PatternFill("solid", fgColor="FFF2CC")
        for j, key in enumerate(KEYS, start=1):
            letter = get_column_letter(j)
            ws.column_dimensions[letter].width = WIDTHS.get(key, 14)
            top = ws.cell(row=1, column=j)
            top.font = Font(bold=True)
            top.fill = head
            top.alignment = Alignment(wrap_text=True, vertical="center")
            for i in range(2, last + 1):
                cell = ws.cell(row=i, column=j)
                if key in ("patient_id", "us_exam_date", "dmsa_date"):
                    cell.number_format = "@"
                if key in FILL_KEYS:
                    cell.fill = todo

        def rng(key: str) -> str:
            letter = get_column_letter(KEYS.index(key) + 1)
            return f"{letter}2:{letter}{last}"

        dmsa = DataValidation(type="list", formula1='"0,1,NA"', allow_blank=True,
                              showErrorMessage=True, errorTitle="只能填 0、1 或 NA",
                              error="0 = 正常，1 = 急性腎盂腎炎，NA = 該側未評估")
        site = DataValidation(type="list", formula1='"CYCH,NCKUH"', allow_blank=True)
        ws.add_data_validation(dmsa)
        ws.add_data_validation(site)
        dmsa.add(rng("left_dmsa"))
        dmsa.add(rng("right_dmsa"))
        site.add(rng("site"))
        ws.freeze_panes = "D2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(KEYS))}{last}"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default="kidney_label_request")
    args = p.parse_args(argv)

    out = ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    table, counts = build()

    xlsx = out / "kidney_label_request.xlsx"
    write_excel(table, counts, xlsx)
    table.to_csv(out / "kidney_label_request.csv", index=False, encoding="utf-8-sig")

    print(f"rows written: {counts['total_rows']}")
    for status in STATUS_ORDER:
        print(f"  {status:<30} exams {counts[status]['exams']:>4}   "
              f"patients {counts[status]['patients']:>4}")
    print(f"  new-patient rows with no site evidence: {counts['new_patient_site_unknown']}")
    if counts["analysed"]["patients"] != 362:
        print(f"  WARNING: expected 362 analysed patients, found {counts['analysed']['patients']}")
    print(f"-> {xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
