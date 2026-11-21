"""Separately registered follow-up: native versus calibrated LR/RF probabilities."""
from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd

import renal_voting_validation as V

OUT=V.OUT/"probability_control"
MEMBERS=tuple(m for m in V.MEMBERS if m.endswith(("__lr","__rf")))


def combine(p):
    if p.shape[1]!=10 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Expected ten LR/RF probability columns")
    f=np.column_stack([p[:,[m.startswith(family+"_") for m in MEMBERS]].mean(axis=1) for family in V.FAMILIES])
    return dict(all_members=p.mean(axis=1),family_balanced=f.mean(axis=1),without_green=f[:,:2].mean(axis=1))


def run():
    OUT.mkdir(exist_ok=True)
    protocol=dict(
        status="Post-primary exploratory follow-up, registered after the primary results were seen; not independent confirmation.",
        rationale="Check whether the ten-patient calibration step degrades the same LR/RF member ensemble, without adding models, tuning weights, or changing training/test patients.",
        members=list(MEMBERS),rules=["all_members","family_balanced","without_green"],
        arms=["native","calibrated"],
        native="Existing LR/RF head.predict_proba, without the additional ten-patient sigmoid map. Balanced class weights mean these are not necessarily calibrated clinical probabilities.",
        calibrated="Same ten existing LR/RF heads, with their previously fitted ten-patient sigmoid maps.",
        comparison="Paired native versus calibrated probabilities, with identical members and fixed rules. SVM excluded from both arms because its uncalibrated margin is not a probability.",
        inference="All six workflows reported. No new fit, learned weights, selection, success claim from an isolated seed, or correction of original primary results.",
        parent_protocol_sha256=V.R.sha256(V.OUT/"protocol.json"),
        source_model_provenance_sha256=V.R.sha256(V.OUT/"model_provenance.json"),
        parent_member_predictions_sha256=V.R.sha256(V.OUT/"member_predictions_by_role.csv"))
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError("Follow-up protocol changed")
    V.R.write_json(path,protocol)
    if not (OUT/"protocol_locked.json").exists():
        V.R.write_json(OUT/"protocol_locked.json",dict(utc=datetime.now(timezone.utc).isoformat(),sha256=V.R.sha256(path)))
    patients=V.frame("patients.csv")
    splits=V.frame("voting_patient_assignments.csv")
    member_predictions,results,predictions=[],[],[]
    for seed in V.R.SEEDS:
        arrays=V.load_features(seed)
        ids=splits.loc[splits.seed.eq(seed)&splits.role.eq("test"),"patient_id"]
        ix=np.flatnonzero(patients.patient_id.isin(ids))
        y=patients.label.iloc[ix].to_numpy()
        native,calibrated=np.zeros((20,10)),np.zeros((20,10))
        for j,name in enumerate(MEMBERS):
            fitted=joblib.load(V.OUT/"models"/f"{name}_{seed}.joblib")
            head,cal=fitted["head"],fitted["calibrator"]
            x=arrays[name.split("__")[0]][ix]
            native[:,j]=head.predict_proba(x)[:,list(head.classes_).index(1)]
            calibrated[:,j]=cal.predict_proba(V.raw_score(head,x)[:,None])[:,1]
            for k,i in enumerate(ix):
                member_predictions.append(dict(seed=seed,patient_id=patients.patient_id.iloc[i],member=name,label=int(y[k]),
                    native_probability=float(native[k,j]),calibrated_probability=float(calibrated[k,j])))
        for arm,p in (("native",native),("calibrated",calibrated)):
            for rule,scores in combine(p).items():
                name=arm+"__"+rule
                results.append(dict(seed=seed,model=name,**V.score_metrics(y,scores)))
                for i,score in zip(ix,scores):
                    predictions.append(dict(seed=seed,model=name,patient_id=patients.patient_id.iloc[i],label=int(patients.label.iloc[i]),score=float(score)))
    pd.DataFrame(member_predictions).to_csv(OUT/"member_test_probabilities.csv",index=False)
    pd.DataFrame(predictions).to_csv(OUT/"patient_test_predictions.csv",index=False)
    m=pd.DataFrame(results)
    m.to_csv(OUT/"metrics_per_draw.csv",index=False)
    rows=[]
    for name,g in m.groupby("model",sort=False):
        row=dict(model=name,draws=len(g),joint_target_draws=int(((g.auroc>.8)&(g.average_precision>.8)).sum()))
        for field in ("auroc","average_precision","auprc_trapezoid","brier","log_loss"):
            for stat in ("mean","std","min","max"):
                row[f"{field}_{stat}"]=float(getattr(g[field],stat)())
        rows.append(row)
    s=pd.DataFrame(rows)
    s.to_csv(OUT/"summary.csv",index=False)
    pairs=[]
    for rule in protocol["rules"]:
        a=m[m.model.eq("native__"+rule)].set_index("seed")
        b=m[m.model.eq("calibrated__"+rule)].set_index("seed").loc[a.index]
        pairs.append(dict(rule=rule,native_minus_calibrated_auroc=float((a.auroc-b.auroc).mean()),
            native_minus_calibrated_ap=float((a.average_precision-b.average_precision).mean()),
            joint_improved_draws=int(((a.auroc>b.auroc)&(a.average_precision>b.average_precision)).sum())))
    pd.DataFrame(pairs).to_csv(OUT/"paired_native_minus_calibrated.csv",index=False)
    print(s[["model","auroc_mean","average_precision_mean","joint_target_draws"]].to_string(index=False),flush=True)
    report()


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m=pd.read_csv(OUT/"metrics_per_draw.csv")
    s=pd.read_csv(OUT/"summary.csv")
    rows=["| Probability / fixed rule | Mean AUROC | Mean AP | Both >0.80 draws |","|---|---:|---:|---:|"]
    for r in s.itertuples():
        rows.append(f"| {r.model} | {r.auroc_mean:.4f} | {r.average_precision_mean:.4f} | {r.joint_target_draws}/10 |")
    text="\n".join([
        "# Native-probability voting control", "",
        "This additional exploratory comparison was specified AFTER observing the primary voting results. It does not replace the primary analysis or provide independent confirmation.", "",
        "The same ten existing LR/RF heads were reused, with and without their ten-patient sigmoid calibration maps. No classifier was retrained, no ensemble weight was learned, and no test patients were changed. SVM was excluded from both arms because its native margin is not a probability. All-member soft voting averages ten members; family-balanced voting first averages the 4/4/2 radiomics/RadImageNet/Green members and gives the three families equal weight; without-Green voting gives the remaining two families equal weight.", "",
        *rows, "",
        "For context, the raw percentile-RadImageNet LR member averaged AUROC/AP 0.676768/0.746167, and the raw percentile-PyRadiomics LR member averaged 0.662626/0.763389. These descriptive fixed-member results do not establish an independently selected best model, but they prevent interpreting the ensemble as superior to strong individual members.", "",
        "Native LR/RF predict_proba values are not automatically clinically calibrated probabilities, particularly after balanced-class-weight training. This comparison isolates the additional small-sample calibration step, not improved image representation. These are post-primary, reused-cohort exploratory results. A high isolated draw does not establish stable AUROC/AP >0.80. See the parent report for methods, provenance and references.", "",
    ])
    (OUT/"RESULTS.md").write_text(text,encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(10,4.9))
    names=[("all_members","All ten LR/RF members"),("family_balanced","Three-family balance"),("without_green","Radiomics + RadImageNet")]
    for ax,field,title in zip(axes,["auroc","average_precision"],["AUROC","Average precision (AP)"]):
        for j,(name,label) in enumerate(names):
            for arm,offset,color in (("native",-.12,"#167c71"),("calibrated",.12,"#b64368")):
                a=m.loc[m.model.eq(arm+"__"+name),field].to_numpy()
                ax.scatter(a,j+offset+np.linspace(-.035,.035,10),s=16,alpha=.6,color=color)
                ax.scatter(a.mean(),j+offset,marker="D",s=34,color=color,label=arm.capitalize() if j==0 else None)
        ax.set_yticks(range(3),[n[1] for n in names],fontsize=9)
        ax.set_xlim(0,1)
        ax.set_ylim(2.55,-.55)
        ax.axvline(.8,color="#666666",lw=1,ls="--")
        ax.set_title(title,fontsize=12)
        ax.grid(axis="x",alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    axes[1].legend(loc="lower left",fontsize=8,frameon=False)
    fig.suptitle("Same LR/RF members: native versus calibrated soft voting",fontsize=12)
    fig.subplots_adjust(left=.23,right=.99,bottom=.22,top=.86,wspace=1.1)
    fig.text(.04,.06,"Post-primary exploratory control, with no new fit or weight search. Dots: ten draws; diamonds: means.\nNative model probabilities are not a claim of calibrated clinical risk.",fontsize=9)
    V.export_figure(fig,"Figure_V3_probability_control")
    plt.close(fig)


if __name__=="__main__":
    run()
