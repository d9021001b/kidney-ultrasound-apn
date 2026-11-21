"""Package the 28 new, unlabelled CYCH patients found in ki-data for clinical labelling.

Inputs : 04_quality_archive/scratch_kidata/new_cych28_files_20261003.csv (rebuilt 2026-10-03
         from the current ki-data files: CYCH3 patients not in the analysed CYCH or NCKUH sets)
Output : 03_data_models/CYCH_new28_unlabelled_20261003/ and the matching .zip
"""
import hashlib
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[2]
NAME = "CYCH_new28_unlabelled_20261003"
OUT = ROOT / "03_data_models" / NAME
LIST = ROOT / "04_quality_archive" / "scratch_kidata" / "new_cych28_files_20261003.csv"
LABELME = ROOT / "03_data_models" / "ki-data" / "before-labeled"
VIEWS = ["LL", "LT", "RL", "RT"]


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    s = pd.read_csv(LIST, dtype={"pid": str, "date": str}).fillna({"date": "", "rest": ""})
    pids = set(s.pid)
    extra = []
    for f in sorted(LABELME.glob("*.json")):
        m = re.match(r"^0*(\d+)(?:_(\d{8}))?_?(.*)$", f.stem)
        if m and m.group(1) in pids:
            extra.append(dict(folder="before-labeled", path=str(f), name=f.name, pid=m.group(1),
                              date=m.group(2) or "", rest=m.group(3), ext=".json"))
    files = pd.concat([s, pd.DataFrame(extra)], ignore_index=True).drop_duplicates("path")
    assert files.pid.nunique() == 28, files.pid.nunique()

    rows = []
    for r in files.itertuples():
        pid8 = str(r.pid).zfill(8)
        dst = OUT / "images" / pid8 / r.folder / r.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(r.path, dst)
        view = re.match(r"(LL|LT|RL|RT)", str(r.rest))
        rows.append(dict(patient_id=pid8, exam_date=str(r.date), view=view.group(1) if view else "",
                         source_folder=r.folder, file=r.name,
                         package_path=dst.relative_to(OUT).as_posix(), bytes=dst.stat().st_size,
                         sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
    index = pd.DataFrame(rows).sort_values(["patient_id", "exam_date", "source_folder", "file"])
    index.to_csv(OUT / "file_index.csv", index=False, encoding="utf-8-sig")

    img = index[index.file.str.lower().str.endswith((".png", ".tif")) & index.exam_date.ne("")]
    ex = (img.groupby(["patient_id", "exam_date"])
             .agg(views=("view", lambda v: " ".join(x for x in VIEWS if x in set(v))),
                  files=("file", "size"),
                  folders=("source_folder", lambda v: " | ".join(sorted(set(v)))))
             .reset_index())
    ex.insert(0, "row", range(1, len(ex) + 1))
    for c in ("dmsa_date", "dmsa_label", "sex", "age_years", "notes"):
        ex[c] = ""
    heads = {"row": "序號", "patient_id": "病歷號", "exam_date": "超音波檢查日", "views": "已有視角",
             "files": "檔案數", "folders": "來源資料夾", "dmsa_date": "DMSA檢查日", "dmsa_label": "DMSA結果",
             "sex": "性別", "age_years": "年齡(歲)", "notes": "備註"}
    keys = list(ex.columns)
    ex.columns = [f"{c} {heads[c]}" for c in keys]
    two = ex.groupby(ex.columns[1]).size()
    multi = ", ".join(f"{p}" for p in two[two > 1].index)
    guide = [
        f"本表每一列是一次超音波檢查，共 {len(ex)} 列、28 位病人。這些病人有影像，但專案中沒有任何標籤。",
        "黃色欄位請填寫，其他欄位請勿修改。病歷號與日期是文字格式，請保留前導 0。",
        "DMSA結果：0 = 正常；1 = 急性腎盂腎炎；NA = 未做或無法判定。儲存格有下拉選單。左右腎結果一致，每列填一個即可。",
        "DMSA檢查日：請填 YYYYMMDD。若同次發燒做了不只一次，請填最接近超音波檢查日的那一次。",
        "性別、年齡為選填，只用於病人特徵表，不進入模型。",
        f"有兩次檢查的病人：{multi}。請分別填寫；分析時預設只用第一次檢查。" if multi else "每位病人只有一次檢查。",
        "注意：影像畫面頂端燒錄有病人姓名、病歷號與醫院名稱，未去識別化，請勿離開院內或研究環境。",
    ]
    xlsx = OUT / "label_request_CYCH_new28.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        pd.DataFrame({"填寫說明": guide}).to_excel(w, sheet_name="填寫說明", index=False)
        ex.to_excel(w, sheet_name="待填表", index=False)
        g = w.sheets["填寫說明"]
        g.column_dimensions["A"].width = 110
        for (cell,) in g.iter_rows(min_row=2):
            cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws = w.sheets["待填表"]
        last = len(ex) + 1
        widths = dict(row=6, patient_id=12, exam_date=13, views=14, files=7, folders=26,
                      dmsa_date=13, dmsa_label=11, sex=8, age_years=9, notes=30)
        for j, k in enumerate(keys, 1):
            col = get_column_letter(j)
            ws.column_dimensions[col].width = widths[k]
            ws.cell(1, j).font = Font(bold=True)
            ws.cell(1, j).fill = PatternFill("solid", fgColor="DCE6F1")
            for i in range(2, last + 1):
                if k in ("patient_id", "exam_date", "dmsa_date"):
                    ws.cell(i, j).number_format = "@"
                if k in ("dmsa_date", "dmsa_label", "sex", "age_years", "notes"):
                    ws.cell(i, j).fill = PatternFill("solid", fgColor="FFF2CC")
        dv = DataValidation(type="list", formula1='"0,1,NA"', allow_blank=True,
                            showErrorMessage=True, errorTitle="只能填 0、1 或 NA")
        ws.add_data_validation(dv)
        col = get_column_letter(keys.index("dmsa_label") + 1)
        dv.add(f"{col}2:{col}{last}")
        ws.freeze_panes = "C2"

    counts = index.source_folder.value_counts().to_dict()
    readme = f"""嘉基新病人資料包（28 位，尚無標籤）
建立日期：2026-10-03

內容
  label_request_CYCH_new28.xlsx  給醫師填寫的標籤表，每列一次檢查，共 {len(ex)} 列
  file_index.csv                 每個檔案的病歷號、檢查日、視角、來源資料夾與 SHA-256
  images/<病歷號>/<來源資料夾>/   原始檔案，複製自 03_data_models/ki-data，未做任何修改

來源資料夾說明
  CYCH3           1600x600 PNG，左半是原始超音波畫面，右半是腎臟二值遮罩（{counts.get('CYCH3', 0)} 個）
  2               800x600 TIF，同一畫面加上紅色腎臟輪廓（{counts.get('2', 0)} 個）
  before-labeled  TIF 與 LabelMe JSON 腎臟多邊形（{counts.get('before-labeled', 0)} 個）
  test_predict    1600x600 PNG（{counts.get('test_predict', 0)} 個）

統計：28 位病人、{len(ex)} 次檢查、{len(index)} 個檔案。
這 28 人不在目前分析的 98 位嘉基病人中，也不在成大名單中。

填寫方式：見 Excel 的「填寫說明」分頁。左右腎結果一致，每次檢查填一個 DMSA 結果即可。

隱私提醒
  影像畫面頂端燒錄有病人英文姓名、病歷號、年齡、性別與醫院名稱，未去識別化。
  本資料包只能留在院內或核准的研究環境，不可上傳、寄送或放入投稿檔案。
"""
    (OUT / "README_請先讀.txt").write_text(readme, encoding="utf-8-sig")

    zp = OUT.with_suffix(".zip")
    if zp.exists():
        zp.unlink()
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(OUT.parent).as_posix())
    with zipfile.ZipFile(zp) as z:
        bad, n = z.testzip(), len(z.namelist())
    print("folder:", OUT)
    print("zip   :", zp, f"{zp.stat().st_size / 1e6:.1f} MB, {n} entries, integrity", "OK" if bad is None else bad)
    print("patients", index.patient_id.nunique(), "exams", len(ex), "files", len(index), counts)
    print("patients with two exams:", multi or "none")


if __name__ == "__main__":
    main()
