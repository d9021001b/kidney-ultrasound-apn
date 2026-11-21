"""Validate and package the voting extension without modifying earlier artifacts."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
from PIL import Image

import renal_voting_validation as V


def package():
    tests=subprocess.run([sys.executable,"-m","unittest","test_renal_voting_validation","-v"],
                         cwd=V.ROOT,capture_output=True,text=True)
    (V.OUT/"validation_test_log.txt").write_text(tests.stdout+tests.stderr,encoding="utf-8")
    if tests.returncode or "skipped" in tests.stderr:
        raise RuntimeError(tests.stdout+tests.stderr)
    s=pd.read_csv(V.OUT/"summary.csv").set_index("model")
    provenance=json.loads((V.OUT/"model_provenance.json").read_text())
    reversed_count=sum(v["calibrated_direction_reversed"] for v in provenance)
    pairs=pd.read_csv(V.OUT/"paired_vs_selected_single.csv").set_index("model")
    names=[("validation_selected_single","驗證組選出的單一模型"),("all_hard_raw","原始硬投票"),
           ("all_hard_calibrated","校正後硬投票"),("all_soft","全部模型等權軟投票"),
           ("family_balanced_soft","三類特徵平衡軟投票"),("validation_weighted_soft","驗證組加權軟投票"),
           (V.PRIMARY,"加權軟投票，加上固定權重收縮")]
    rows=["| 方法 | 平均 AUROC | 平均 AP | 同次兩者 >0.8 |","|---|---:|---:|---:|"]
    for model,label in names:
        r=s.loc[model]
        rows.append(f"| {label} | {r.auroc_mean:.4f} | {r.average_precision_mean:.4f} | {int(r.joint_target_draws)}/10 |")
    p=pairs.loc[V.PRIMARY]
    text="\n".join([
        "# 嘉基腎臟影像多模型投票實驗", "",
        "這是實際完成的延伸實驗，不是只提出建議。沿用原先分割後的腎臟影像特徵與既有醫學預訓練權重；原稿、上一輪資料夾和 ZIP 不覆寫。", "",
        "## 本輪補上什麼", "",
        "- 五組特徵：原圖與腎臟 ROI P1-P99 正規化的 PyRadiomics、原圖與同樣正規化的 RadImageNet DenseNet-121，以及原圖 Green Learning 特徵。",
        "- 每組各用支援向量機（SVM）、L2 邏輯斯迴歸（LR）、隨機森林（RF），共 15 個成員。模型名稱及投票定義附官方文件或原始論文。",
        "- 比較原始硬投票、校正後硬投票、等權軟投票、特徵類別平衡、驗證組選權重、權重收縮，以及三類特徵各自投票和移除一類的消融分析。",
        "- 98 位病人、349 張影像，保留相同十次隨機抽樣的 58 位訓練與 20 位測試病人。原 20 位驗證病人分成 10 位校正、10 位選投票權重，同一次四組病人互斥。",
        "- 不重訓分割模型或影像編碼器。舊分類頭曾以完整 20 人驗證組調參，因此本輪只重新擬合固定設定的分類頭，避免假裝校正組從未參與舊分類頭的選擇。",
        "- 不直接平均不同 SVM 的原始分數；以獨立 10 人校正組擬合一維正則化 sigmoid 機率映射。沒有額外的折分驗證、生成式影像或額外旋轉增廣。", "",
        "## 十次抽樣的完整比較", "",*rows,"",
        f"本輪事前指定的收縮加權流程，相對同流程選出的單一模型，平均 AUROC 差值 {p.delta_auroc_mean:+.4f}、AP 差值 {p.delta_ap_mean:+.4f}；兩者同時改善 {int(p.joint_improved_draws)}/10 次。", "",
        f"150 個成員校正器中，{reversed_count} 個 sigmoid 斜率為負，代表小校正組出現分數排序方向不穩定；已保留紀錄，不用測試結果偷偷改方向。請同時查看未校正的單一模型分數。",
        "這是同一批已探索病人的進一步內部實驗，不能因拆出新的校正組就稱為未碰過的外部驗證。十次抽樣的病人重複出現，標準差不是獨立樣本的信賴區間。過去最佳模型成績不可與本輪不同分類器設定及驗證配置混成同一公平比較。", "",
        "AP（average precision）和梯形積分 PR 面積分開提供。硬投票的 AUROC/AP 使用正票比例，不是最終二元類別；最後分類以 >0.5 判陽性，同票判陰性。硬投票比例不視為校準後臨床機率。", "",
        "## 下一步的重點", "",
        "不以增加相似弱模型數量當作提升保證。先用本輪三類特徵投票與移除一類的結果，檢查弱訊號是否稀釋有效訊號；再評估有獨立資訊的不同醫學 backbone 或原生尺度局部區塊特徵。",
        "本輪未新增 stacking、多實例學習（MIL）、新影像編碼器或 GAN。這些需要各自的訓練與選擇紀錄，不能寫成已完成。若小樣本權重與校正持續不穩，應擴大可用開發樣本、簡化固定投票規則，並規劃真正獨立的後續驗證；不能反覆挑測試組直到達標。",
        "遮罩內的量測虛線與十字仍是輸入風險。影像視覺觀察不等同病灶真值，不能憑模型輸出新建疤痕、發炎或腫脹標籤。", "",
        "## 檔案", "",
        "- `RESULTS.md`：英文完整方法、結果、限制及參考文獻。",
        "- `summary.csv`：28 個流程各十次的彙總；`metrics_per_draw.csv` 共 280 筆。",
        "- `patient_test_predictions.csv`：5,600 筆病人測試預測。",
        "- `member_predictions_by_role.csv`：6,000 筆成員的校正、權重選擇及測試分數。",
        "- `ensemble_selection.json`：每次所有候選權重與選擇依據；`vote_selection_diversity.csv`：1,050 組成員相關性、決策差異與共同錯誤率。",
        "- `raw_member_metrics.csv`：150 組未校正成員的排序結果；`paired_vs_selected_single.csv`：同次配對差值。",
        "- `image_manifest.csv`：原始腎臟影像檔名、病人與 label 對照。",
        "- `features/`：可直接重跑分類與投票的病人層級特徵。",
        "- `models/`：150 個分類頭與各自校正器；`model_provenance.json`：訓練名單、參數、特徵選擇與雜湊。",
        "- `figures_600dpi/`：600 dpi PNG 與 TIFF；`qa/`：檢視預覽及紀錄。",
        "- `source/`：程式、測試、環境版本與交接。",
        "- `probability_control/`：看到主要結果後另行鎖定規則的機率對照；額外 6 個流程、60 筆指標、1,200 筆病人預測。不是事前主分析。", "",
        "## 重跑", "",
        "使用包內病人特徵重跑 evaluate/report 不需要重新抽取影像特徵；prepare 或完整來源雜湊測試仍需兩個前次實驗資料夾與原 GreenExtractor。", "",
        "```powershell",
        f"$env:CYCH_PROJECT_ROOT = '{V.ROOT}'",
        f"$env:CYCH_VOTING_DIR = '{V.OUT}'",
        f"& '{sys.executable}' '{V.OUT/'source/renal_voting_validation.py'}' evaluate",
        f"& '{sys.executable}' '{V.OUT/'source/renal_voting_validation.py'}' report",
        "```", "",
        "套件包含病人識別碼與影像檔名，只供本機研究交付，請勿直接公開。", "",
    ])
    control=pd.read_csv(V.OUT/"probability_control/summary.csv").set_index("model")
    followup=["## 後續機率對照（主結果之後提出）", "",
        "固定相同 LR/RF 成員，比較原生機率與額外 sigmoid 校正。沒有新增訓練、權重搜尋或更換測試病人。SVM 在兩側都排除，因為原始分數不是機率。", "",
        "| 固定投票規則 | 原生 AUROC / AP | 校正後 AUROC / AP |", "|---|---:|---:|"]
    for rule,label in (("all_members","10 個 LR/RF 成員"),("family_balanced","三類特徵平衡"),("without_green","放射組學與 RadImageNet 的 8 個 LR/RF 成員")):
        a,b=control.loc["native__"+rule],control.loc["calibrated__"+rule]
        followup.append(f"| {label} | {a.auroc_mean:.4f} / {a.average_precision_mean:.4f} | {b.auroc_mean:.4f} / {b.average_precision_mean:.4f} |")
    raw=pd.read_csv(V.OUT/"raw_member_metrics.csv").groupby("member")[["auroc","average_precision"]].mean()
    deep=raw.loc["radimagenet_percentile__lr"]
    radiomics=raw.loc["radiomics_percentile__lr"]
    followup.extend(["", "三組固定投票在本次對照中都以原生機率較好，支持檢查小校正組造成的退步。最佳探索性平均仍未達 AUROC/AP >0.8；不能與舊最佳融合模型當作只差投票的一對一比較，也不能把 LR/RF 原生機率當作已校準的臨床風險。", "",
        f"重要的單模型對照：未校正的 ROI 正規化 RadImageNet＋LR 平均 AUROC/AP 為 {deep.auroc:.4f}/{deep.average_precision:.4f}，ROI 正規化 PyRadiomics＋LR 為 {radiomics.auroc:.4f}/{radiomics.average_precision:.4f}。因此投票尚未證明優於較強的單模型。這是完整結果中的描述性比較，不是獨立資料確認的模型選擇。", ""])
    text=text.replace("## 下一步的重點","\n".join(followup)+"\n## 下一步的重點")
    (V.OUT/"README.md").write_text(text,encoding="utf-8")
    for name in ("renal_voting_validation.py","renal_voting_probability_control.py","test_renal_voting_validation.py","package_renal_voting.py",
                 "renal_random_validation.py","renal_enhancement_validation.py","radimagenet_backbones.py",
                 "green_learning_cv.py","single_center_cv.py","recrop_with_yolo.py","HANDOFF_CYCH_RANDOM_VALIDATION.md"):
        shutil.copy2(V.ROOT/name,V.OUT/"source"/name)
    versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","opencv-python",
        "scikit-learn","torch","torchvision","pyradiomics","SimpleITK","PyWavelets","Pillow","matplotlib","joblib","threadpoolctl","tqdm")}
    V.R.write_json(V.OUT/"source/environment.json",dict(python=sys.version,executable=sys.executable,versions=versions))
    figures=[]
    for path in sorted((V.OUT/"figures_600dpi").glob("*")):
        with Image.open(path) as im:
            dpi=tuple(float(v) for v in im.info["dpi"])
            assert all(abs(v-600)<.01 for v in dpi)
            figures.append(dict(file=path.name,width=im.width,height=im.height,dpi=dpi))
    assert len(figures)>=4
    V.R.write_json(V.OUT/"figure_resolution_audit.json",figures)
    original=V.ROOT/"yolo_aug_triEMR_PCA_SVM_renal_2022_v2.docx"
    digest=V.R.sha256(original)
    assert digest=="2b6275278f38072b3f722c43bad067e6d408c992a453a7f08f050108c6b47a75", "Manuscript changed: do not restore or overwrite; inspect first"
    V.R.write_json(V.OUT/"original_manuscript_preservation.json",dict(path=str(original),sha256=digest,modified_by_voting_extension=False))
    files=[p for p in V.OUT.rglob("*") if p.is_file() and p.name!="artifact_manifest.csv" and "__pycache__" not in p.parts]
    pd.DataFrame([dict(path=str(p.relative_to(V.OUT)),bytes=p.stat().st_size,sha256=V.R.sha256(p)) for p in sorted(files)]).to_csv(V.OUT/"artifact_manifest.csv",index=False)
    zip_path=V.ROOT/(V.OUT.name+".zip")
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files+[V.OUT/"artifact_manifest.csv"]):
            z.write(path,arcname=str(Path(V.OUT.name)/path.relative_to(V.OUT)))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        count=len(z.infolist())
    print(json.dumps(dict(zip=str(zip_path),files=count,bytes=zip_path.stat().st_size,sha256=V.R.sha256(zip_path)),indent=2))


if __name__=="__main__":
    package()
