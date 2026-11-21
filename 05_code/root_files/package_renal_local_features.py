"""Verify and package the completed local-feature extension."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import renal_local_features as L


def package():
    tests=subprocess.run([sys.executable,"-m","unittest","test_renal_local_features","-v"],cwd=L.ROOT,capture_output=True,text=True)
    (L.OUT/"validation_test_log.txt").write_text(tests.stdout+tests.stderr,encoding="utf-8")
    if tests.returncode or "skipped" in tests.stderr:
        raise RuntimeError(tests.stdout+tests.stderr)
    audit=[]
    for item in json.loads((L.OUT/"head_provenance.json").read_text()):
        head=joblib.load(L.OUT/item["file"])
        scores=head[2].scores_
        selected=np.isfinite(scores[item["selected_indices"]]).all()
        assert selected
        audit.append(dict(seed=item["seed"],group=item["group"],undefined_anova_scores=int((~np.isfinite(scores)).sum()),
            selected_all_finite=bool(selected),scaler_training_patients=int(head[1].n_samples_seen_)))
    pd.DataFrame(audit).to_csv(L.OUT/"qa/numerical_feature_selection_audit.csv",index=False)
    sources=("renal_local_features.py","test_renal_local_features.py","report_renal_local_features.py","package_renal_local_features.py",
             "renal_random_validation.py","renal_enhancement_validation.py","renal_voting_validation.py","radimagenet_backbones.py",
             "green_learning_cv.py","single_center_cv.py","recrop_with_yolo.py","HANDOFF_CYCH_RANDOM_VALIDATION.md")
    for name in sources:
        shutil.copy2(L.ROOT/name,L.OUT/"source"/name)
    shutil.copy2(L.V.OUT/"voting_patient_assignments.csv",L.OUT/"prior_voting_patient_assignments.csv")
    versions={name:importlib.metadata.version(name) for name in ("numpy","scipy","pandas","opencv-python","scikit-learn",
        "torch","torchvision","pyradiomics","SimpleITK","PyWavelets","Pillow","matplotlib","joblib","threadpoolctl","tqdm")}
    L.R.write_json(L.OUT/"source/environment.json",dict(python=sys.version,executable=sys.executable,versions=versions))
    instructions="\n".join([
        "# Reproduction and Dependency Notes", "",
        "The packaged patient-level features, original random assignments, and prior calibration assignments are sufficient to rerun classification, fixed voting and temperature fitting with the listed Python environment. Source inspection and full provenance tests additionally access the original baseline, enhancement and voting directories at their recorded paths.", "",
        "```powershell",
        f"$env:CYCH_PROJECT_ROOT = '{L.ROOT}'",
        f"$env:CYCH_LOCAL_DIR = '{L.OUT}'",
        f"& '{sys.executable}' '{L.OUT/'source/renal_local_features.py'}' evaluate",
        f"& '{sys.executable}' '{L.OUT/'source/report_renal_local_features.py'}'",
        "```", "",
        "The green stage additionally requires the original train-fitted GreenExtractor models. The encoders stage requires the existing RadImageNet/PanEcho checkpoints recorded in protocol.json and radimagenet_backbones.py, plus a CUDA-capable installation. No large external medical weights are copied into this extension. The prepare stage requires the previous experiment directories. The full current run was executed on the local machine, with existing weights and no new downloads.", "",
        "Undefined univariate ANOVA F scores occurred in constant inherited PyRadiomics/DenseNet columns. The audit verifies that all32 selected features in every one of70 fitted heads had finite F scores. Those warnings did not introduce nonfinite classifier inputs or predictions. The original helper's handling of constant columns was retained for exact baseline comparability.", "",
        "Outputs include patient IDs and source filenames; this is a private research package, not a public deidentified release.", "",
    ])
    (L.OUT/"source/REPRODUCE.md").write_text(instructions,encoding="utf-8")
    figure_audit=[]
    for path in sorted((L.OUT/"figures_600dpi").glob("*")):
        with Image.open(path) as im:
            dpi=tuple(float(v) for v in im.info["dpi"])
            assert all(abs(v-600)<.01 for v in dpi)
            figure_audit.append(dict(file=path.name,width=im.width,height=im.height,dpi=dpi))
    assert len(figure_audit)==6
    L.R.write_json(L.OUT/"figure_resolution_audit.json",figure_audit)
    preservation=json.loads((L.OUT/"original_manuscript_preservation.json").read_text())
    assert L.R.sha256(preservation["path"])==preservation["sha256"], "Original manuscript changed; do not overwrite or restore"
    files=[p for p in L.OUT.rglob("*") if p.is_file() and p.name!="artifact_manifest.csv" and "__pycache__" not in p.parts]
    pd.DataFrame([dict(path=str(p.relative_to(L.OUT)),bytes=p.stat().st_size,sha256=L.R.sha256(p)) for p in sorted(files)]).to_csv(L.OUT/"artifact_manifest.csv",index=False)
    target=L.ROOT/(L.OUT.name+".zip")
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(files+[L.OUT/"artifact_manifest.csv"]):
            archive.write(p,arcname=str(Path(L.OUT.name)/p.relative_to(L.OUT)))
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        count=len(archive.infolist())
    print(json.dumps(dict(zip=str(target),files=count,bytes=target.stat().st_size,sha256=L.R.sha256(target)),indent=2))


if __name__=="__main__":
    package()
