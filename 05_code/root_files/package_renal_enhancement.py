"""Verify and package the completed enhancement extension without replacing v2."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pandas as pd
from PIL import Image
import renal_enhancement_validation as E


def package():
    tests = subprocess.run([sys.executable,"-m","unittest","test_renal_enhancement_validation",
                            "test_renal_random_validation","-v"],cwd=E.ROOT,capture_output=True,text=True)
    (E.OUT/"validation_test_log.txt").write_text(tests.stdout+tests.stderr,encoding="utf-8")
    if tests.returncode:
        raise RuntimeError(tests.stdout+tests.stderr)
    s=pd.read_csv(E.OUT/"summary.csv").set_index("model")
    base=s.loc["original__fusion_aug"]
    primary=s.loc["validation_selected_fusion_aug"]
    fixed=s.loc[[f"{mode}__fusion_aug" for mode in E.MODES]]
    best=fixed.sort_values("auroc_mean",ascending=False).iloc[0]
    best_mode=best.name.split("__")[0]
    text="\n".join([
        "# 腎臟輪廓內影像強化實驗", "",
        "本資料夾是嘉基隨機抽樣驗證的延伸實驗，不取代上一輪論文或原始影像。", "",
        "## 已完成", "",
        "- 349 張影像、98 位病人；沿用相同十次病人層級 58/20/20 訓練、驗證、測試分組。",
        "- 七組處理：原圖、腎臟內百分位強度正規化、CLAHE、雙邊濾波、雙邊濾波後 CLAHE、非局部均值去噪、非局部均值後 CLAHE。",
        "- 原遮罩完全不變，輸出腎臟外像素全為零。統計與邊界填值不讀取原圖腎臟外灰階。",
        "- 確實重抽真實 PyRadiomics 與既有 RadImageNet DenseNet-121 特徵；模型權重與 BatchNorm 固定。",
        "- 沿用每次原訓練組擬合的 GreenExtractor，測試強化後輸入；沒有把所有病人拿去重新擬合。此部分是固定特徵轉換器敏感度分析，不等同強化影像重訓 Green Learning。",
        "- 35 種固定設定各跑十次，另保留 6 種只由驗證集選擇處理方式的流程，共 410 筆結果、8,200 筆病人預測紀錄。", "",
        "## 同一融合模型的結果", "",
        f"- 原圖：平均 AUROC {base.auroc_mean:.4f}、AP {base.average_precision_mean:.4f}。",
        f"- 固定處理中，依本輪測試平均 AUROC 排序最高的是 {E.LABELS[best_mode]}：AUROC {best.auroc_mean:.4f}、AP {best.average_precision_mean:.4f}。這是事後探索排名，不是獨立證實。",
        f"- 事前指定的主要流程，僅由各次驗證組選強化方式：平均 AUROC {primary.auroc_mean:.4f}、AP {primary.average_precision_mean:.4f}。", "",
        f"- 放射組學單獨使用 ROI 強度正規化：AUROC {s.loc['percentile__radiomics','auroc_mean']:.4f}、AP {s.loc['percentile__radiomics','average_precision_mean']:.4f}；原圖則為 {s.loc['original__radiomics','auroc_mean']:.4f}、{s.loc['original__radiomics','average_precision_mean']:.4f}。",
        f"- 固定 RadImageNet 特徵單獨使用 ROI 強度正規化：AUROC {s.loc['percentile__radimagenet','auroc_mean']:.4f}、AP {s.loc['percentile__radimagenet','average_precision_mean']:.4f}；原圖則為 {s.loc['original__radimagenet','auroc_mean']:.4f}、{s.loc['original__radimagenet','average_precision_mean']:.4f}。", "",
        "不能只挑最高一次抽樣或最高測試組宣稱達標。AP 是 average precision，與梯形積分的 PR 面積分開提供；重複抽樣之病人不可視為互相獨立的新樣本。", "",
        "## 視覺檢視與限制", "",
        "CLAHE 在檢視範例中使部分明暗結構更突出，但也強化量測虛線與十字標記；去噪使部分低解析影像的細紋減少。這些是影像觀察，不是疤痕、腫脹或發炎診斷。六張範例按前輪固定的系統性清單選取，未依本輪分類成績挑圖。",
        "OpenCV 的 CLAHE 使用腎臟內像素延伸邊界後再處理，延伸像素會參與局部直方圖，並非另外實作遮罩加權 CLAHE。",
        "一般 NLM 的高斯噪聲假設尚未在這批 B-mode 影像驗證；不是已證實的超音波散斑物理復原。未使用 GAN、生成式補圖、或自動病灶標註。",
        "目前仍須處理遮罩內量測標記干擾，並由醫師建立皮質/腎竇及疑似異常區域標註。沒有毫米尺度及參考標註，不把亮區直接當腎竇切掉，也不把輪廓凹陷直接叫疤痕。", "",
        "## 檔案", "",
        "- `RESULTS.md`：完整英文方法、比較結果、限制與原始文獻。",
        "- `summary.csv`：全部 41 個流程的十次彙總。",
        "- `paired_enhancement_comparisons.csv`：同一切分、同一模型的增益比較。",
        "- `metrics_per_draw.csv`、`patient_test_predictions.csv`：所有結果及預測。",
        "- `validation_selection.json`、`validation_tuning.json`：驗證集選擇依據與完整候選分數。",
        "- `image_manifest.csv`：原始影像檔名與病人/label 對應。",
        "- `images/`、`masks/`：七組實際影像與固定遮罩。",
        "- `features/`、`models/`：特徵與分類器。",
        "- `figures_600dpi/`：真正輸出的 600 dpi PNG/TIFF；提高輸出 dpi 不增加原超音波影像的解剖解析度。",
        "- `source/`：程式、環境版本與外部依賴路徑。", "",
        "## 重跑", "",
        "此為延伸包；完整重跑仍需本機上一輪 `CYCH_random_validation_20260913` 資料夾及記錄的既有權重，沒有將大型外部預訓練權重重複塞入 ZIP。", "",
        "```powershell",
        f"$env:CYCH_PROJECT_ROOT = '{E.ROOT}'",
        f"$env:CYCH_RUN_DIR = '{E.BASE}'",
        f"$env:CYCH_ENHANCEMENT_DIR = '{E.OUT}'",
        f"& '{sys.executable}' '{E.OUT / 'source/renal_enhancement_validation.py'}' all",
        "```", "",
        "內含病人識別碼與完整檔名，請勿直接公開。原始論文及前一 ZIP 保留不動。", "",
    ])
    (E.OUT/"README.md").write_text(text,encoding="utf-8")
    source_names=["renal_enhancement_validation.py","renal_random_validation.py","radimagenet_backbones.py",
                  "green_learning_cv.py","single_center_cv.py","recrop_with_yolo.py","test_renal_enhancement_validation.py",
                  "test_renal_random_validation.py","package_renal_enhancement.py","HANDOFF_CYCH_RANDOM_VALIDATION.md"]
    for name in source_names:
        shutil.copy2(E.ROOT/name,E.OUT/"source"/name)
    versions={}
    for name in ["numpy","scipy","pandas","opencv-python","scikit-learn","torch","torchvision",
                 "pyradiomics","SimpleITK","PyWavelets","Pillow","matplotlib","joblib","threadpoolctl","tqdm"]:
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]="not found under this distribution name"
    E.R.write_json(E.OUT/"source/environment.json",dict(python=sys.version,executable=sys.executable,versions=versions))
    external=[E.BASE/"features/radiomics.csv",E.BASE/"features/masked.npy",
              *[E.BASE/"features"/f"green_{seed}.npy" for seed in E.R.SEEDS],
              *[E.BASE/"models"/f"green_extractor_{seed}.joblib" for seed in E.R.SEEDS],
              Path(json.loads((E.OUT/"features/pretrained_provenance.json").read_text())["path"])]
    E.R.write_json(E.OUT/"source/external_dependencies.json",[
        dict(path=str(p),sha256=E.R.sha256(p),bytes=p.stat().st_size) for p in external])
    figure_audit=[]
    for path in sorted((E.OUT/"figures_600dpi").glob("*")):
        with Image.open(path) as im:
            dpi=tuple(float(v) for v in im.info["dpi"])
            assert all(abs(v-600)<.01 for v in dpi)
            figure_audit.append(dict(file=path.name,width=im.width,height=im.height,dpi=dpi))
    E.R.write_json(E.OUT/"figure_resolution_audit.json",figure_audit)
    files=[p for p in E.OUT.rglob("*") if p.is_file() and p.name!="artifact_manifest.csv" and "__pycache__" not in p.parts]
    pd.DataFrame([dict(path=str(p.relative_to(E.OUT)),bytes=p.stat().st_size,sha256=E.R.sha256(p))
                  for p in sorted(files)]).to_csv(E.OUT/"artifact_manifest.csv",index=False)
    zip_path=E.ROOT/"CYCH_roi_enhancement_20260913.zip"
    with zipfile.ZipFile(zip_path,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(files+[E.OUT/"artifact_manifest.csv"]):
            z.write(path,arcname=str(Path(E.OUT.name)/path.relative_to(E.OUT)))
    with zipfile.ZipFile(zip_path) as z:
        assert z.testzip() is None
        count=len(z.infolist())
    print(json.dumps(dict(zip=str(zip_path),files=count,bytes=zip_path.stat().st_size,
                         sha256=E.R.sha256(zip_path)),indent=2))


if __name__=="__main__":
    package()
