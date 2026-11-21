"""Patient-random ensemble ablation with separated calibration and vote selection.

Existing image encoders and feature caches are reused. Only small classification
heads and one-dimensional regularized sigmoid maps are fitted in this extension.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

import renal_random_validation as R
import renal_enhancement_validation as E

ROOT=R.ROOT
OUT=Path(os.environ.get("CYCH_VOTING_DIR",str(ROOT/"CYCH_voting_20260913")))
FAMILIES=("radiomics","radimagenet","green")
GROUPS=("radiomics_original","radiomics_percentile","radimagenet_original","radimagenet_percentile","green_original")
ALGORITHMS=("svm","lr","rf")
MEMBERS=tuple(f"{group}__{algorithm}" for group in GROUPS for algorithm in ALGORITHMS)
PRIMARY="validation_weighted_shrunk_soft"
ENSEMBLES=("all_hard_raw","all_hard_calibrated","all_soft","family_balanced_soft",
           "validation_weighted_soft",PRIMARY,"validation_selected_single",
           *[f"family_{family}_soft" for family in FAMILIES],
           *[f"without_{family}_soft" for family in FAMILIES])
REFERENCES=[
    dict(citation="scikit-learn 1.8. VotingClassifier: majority rule and weighted soft voting.",
         url="https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.VotingClassifier.html",
         use="Defines standard hard and soft voting. This implementation combines already fitted per-feature-space models explicitly; VotingClassifier.fit would clone and refit them."),
    dict(citation="scikit-learn. Probability calibration.",url="https://scikit-learn.org/stable/modules/calibration.html",
         use="Calibration data must be disjoint from classifier fitting. Regularized one-dimensional logistic calibration is used here, not a claim of clinically validated probabilities."),
    dict(citation="Breiman L. Random Forests. Machine Learning. 2001;45:5-32.",
         url="https://doi.org/10.1023/A:1010933404324",use="Random forest classification head."),
    dict(citation="Cortes C, Vapnik V. Support-vector networks. Machine Learning. 1995;20:273-297.",
         url="https://doi.org/10.1007/BF00994018",use="Radial-basis-function support vector machine (SVM)."),
    dict(citation="scikit-learn 1.8. LogisticRegression.",
         url="https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html",
         use="L2-regularized logistic regression (LR), both as a classification head and as a separate sigmoid score map."),
    dict(citation="Caruana R, Niculescu-Mizil A, Crew G, Ksikes A. Ensemble selection from libraries of models. ICML. 2004.",
         url="https://doi.org/10.1145/1015330.1015432",
         use="Motivation for held-out ensemble selection. The present 15-point family-weight grid is NOT their forward-selection algorithm."),
]


def frame(name):
    return pd.read_csv(OUT/name,dtype={"patient_id":str})


def family_indices():
    return {family:np.array([i for i,name in enumerate(MEMBERS) if name.startswith(family+"_")]) for family in FAMILIES}


def weight_grid(shrink=0.0):
    points=[np.array([i,j,4-i-j],dtype=float)/4 for i in range(5) for j in range(5-i)]
    return [(1-shrink)*w+shrink*np.ones(3)/3 for w in points]


def prepare():
    OUT.mkdir(exist_ok=True)
    for name in ("features","models","figures_600dpi","qa","source"):
        (OUT/name).mkdir(exist_ok=True)
    for name in ("image_manifest.csv","patients.csv","random_patient_assignments.csv"):
        dest=OUT/name
        if dest.exists() and R.sha256(dest)!=R.sha256(R.OUT/name):
            raise RuntimeError(f"Cohort drift: {name}")
        shutil.copy2(R.OUT/name,dest)
    old=frame("random_patient_assignments.csv")
    assignments=[]
    for seed in R.SEEDS:
        per=old[old.seed.eq(seed)].copy()
        val=per[per.role.eq("validation")].sort_values("patient_id").reset_index(drop=True)
        cal,vote=next(StratifiedShuffleSplit(n_splits=1,test_size=.5,random_state=seed+2000).split(val,val.label))
        roles={**{p:"calibration" for p in val.patient_id.iloc[cal]},
               **{p:"vote_selection" for p in val.patient_id.iloc[vote]}}
        per["role"]=per.apply(lambda row:roles.get(row.patient_id,row.role),axis=1)
        assert per.role.value_counts().to_dict()==dict(train=58,test=20,calibration=10,vote_selection=10)
        assignments.append(per)
    splits=pd.concat(assignments,ignore_index=True)
    splits.to_csv(OUT/"voting_patient_assignments.csv",index=False)
    protocol=dict(
        design="Further retrospective internal patient-random validation on the already explored CYCH cohort. No external or untouched-cohort claim.",
        base_run=str(R.OUT),enhancement_run=str(E.OUT),seeds=R.SEEDS,
        roles=dict(train=58,calibration=10,vote_selection=10,test=20),
        split="Original train/test patients unchanged; original 20 validation patients split once, stratified, seed+2000, into 10 calibration and 10 vote-selection patients.",
        members=list(MEMBERS),families=list(FAMILIES),ensembles=list(ENSEMBLES),primary=PRIMARY,
        features="Patient mean and population standard deviation. Existing original and ROI P1-P99 PyRadiomics/RadImageNet caches, plus each draw's original training-fitted Green cache. No clinical covariates or additional image generation.",
        head_preprocessing="Train-only median imputation, standardization, and ANOVA selection of 32 features.",
        classifiers=dict(svm=dict(C=1.0,kernel="rbf",gamma="scale",class_weight="balanced",probability=False),
                         lr=dict(C=.1,solver="liblinear",class_weight="balanced",max_iter=2000),
                         rf=dict(n_estimators=256,max_depth=4,min_samples_leaf=3,max_features="sqrt",class_weight="balanced",n_jobs=1)),
        classification_heads="Fixed settings declared before this run; fitted only on 58 training patients. Previous heads selected on all 20 validation patients are not reused as if they had independent calibration.",
        calibration="StandardScaler plus L2 LogisticRegression(C=1,lbfgs,max_iter=2000), fitted to one-dimensional raw scores on 10 calibration patients only. Regularized sigmoid mapping, not exact unregularized Platt fitting. No hidden SVC probability fitting, model selection, or fold routine.",
        augmentation="No additional augmentation. Angle-zero features only, to isolate feature/classifier diversity and score aggregation.",
        soft="Mean calibrated positive-class probabilities; family-balanced means give radiomics/RadImageNet/Green one third each despite 6/6/3 members.",
        hard="Raw hard votes use native classifiers' labels; calibrated hard votes use p>0.5. AUROC/AP for hard votes use fraction of positive votes, not collapsed final labels. Ties at 0.5 are negative.",
        weight_selection="15 nonnegative family-weight triples at step .25 summing to one; select by vote-selection mean(AUROC,AP). Tie: closest to uniform, then grid order. Shrunk variant transforms each grid point halfway toward uniform before selection; all effective weights >=1/6.",
        single_selection="Select one calibrated member using the same vote-selection mean(AUROC,AP), protocol order for ties.",
        ablations="Unweighted soft, family-balanced soft, raw/calibrated hard votes, selected weights with/without shrinkage, three single-family soft votes, and three leave-one-family-out soft votes.",
        diversity="Pairwise probability correlation, threshold disagreement and double-fault rates on vote-selection patients only; descriptive, not independent evidence and not an additional selection rule.",
        caution="Only 10 calibration and 10 vote-selection patients per draw: probabilities, weights and selections may be unstable. No success guarantee or confidence interval from dependent repeats.",
        cohort_hashes={name:R.sha256(OUT/name) for name in ("image_manifest.csv","patients.csv","random_patient_assignments.csv","voting_patient_assignments.csv")})
    path=OUT/"protocol.json"
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError("Protocol changed; use a new output directory")
    R.write_json(path,protocol)
    if not (OUT/"protocol_locked.json").exists():
        R.write_json(OUT/"protocol_locked.json",dict(utc=datetime.now(timezone.utc).isoformat(),sha256=R.sha256(path)))
    R.write_json(OUT/"references.json",REFERENCES)
    df=frame("image_manifest.csv")
    patients=frame("patients.csv")
    source_manifest=pd.read_csv(E.OUT/"image_manifest.csv",dtype={"patient_id":str})
    pd.testing.assert_frame_equal(df[["image_id","patient_id","label"]],source_manifest[["image_id","patient_id","label"]])
    provenance=[]
    for family in ("radiomics","radimagenet"):
        for mode in ("original","percentile"):
            if family=="radiomics":
                src=E.OUT/"features"/f"radiomics_{mode}.csv"
                a=pd.read_csv(src).set_index("image_id").loc[df.image_id]
                a=a.loc[:,~a.columns.str.endswith(("_Energy","_TotalEnergy"))].to_numpy()
                assert a.shape==(349,457)
            else:
                src=E.OUT/"features"/f"radimagenet_{mode}.npy"
                a=np.load(src)[0]
                assert a.shape==(349,1024)
            dest=OUT/"features"/f"{family}_{mode}.npy"
            np.save(dest,R.aggregate(a,df,patients))
            provenance.append(dict(source=str(src),source_sha256=R.sha256(src),output=str(dest.relative_to(OUT)),output_sha256=R.sha256(dest)))
    for seed in R.SEEDS:
        src=R.OUT/"features"/f"green_{seed}.npy"
        model_path=R.OUT/"models"/f"green_extractor_{seed}.joblib"
        extractor=joblib.load(model_path)
        train=splits.loc[splits.seed.eq(seed)&splits.role.eq("train"),"patient_id"]
        expected=set(df.loc[df.patient_id.isin(train),"image_id"])
        assert expected==extractor.fit_paths
        dest=OUT/"features"/f"green_original_{seed}.npy"
        np.save(dest,R.aggregate(np.load(src),df,patients))
        provenance.append(dict(source=str(src),source_sha256=R.sha256(src),output=str(dest.relative_to(OUT)),output_sha256=R.sha256(dest),
            seed=seed,extractor_path=str(model_path),extractor_sha256=R.sha256(model_path),fit_images=sorted(extractor.fit_paths),train_patients=sorted(train)))
    R.write_json(OUT/"features/provenance.json",provenance)
    shutil.copy2(E.OUT/"features/pretrained_provenance.json",OUT/"features/pretrained_provenance.json")
    print("Locked 15-member protocol; existing 98-patient feature caches ready",flush=True)


def load_features(seed):
    values={}
    for group in GROUPS:
        name=f"{group}_{seed}.npy" if group.startswith("green") else f"{group}.npy"
        values[group]=np.load(OUT/"features"/name)
        assert len(values[group])==98 and np.isfinite(values[group]).all()
    return values


def make_head(algorithm,seed,dimension):
    if algorithm=="svm":
        head=SVC(C=1.,kernel="rbf",gamma="scale",class_weight="balanced",probability=False)
    elif algorithm=="lr":
        head=LogisticRegression(C=.1,solver="liblinear",class_weight="balanced",max_iter=2000,random_state=seed)
    elif algorithm=="rf":
        head=RandomForestClassifier(n_estimators=256,max_depth=4,min_samples_leaf=3,max_features="sqrt",
                                    class_weight="balanced",n_jobs=1,random_state=seed)
    else:
        raise ValueError(algorithm)
    return make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),SelectKBest(f_classif,k=min(32,dimension)),head)


def raw_score(model,x):
    if hasattr(model,"decision_function"):
        return model.decision_function(x)
    return model.predict_proba(x)[:,list(model.classes_).index(1)]


def fit_member(x,y,train,calibration,algorithm,seed):
    if set(train)&set(calibration):
        raise ValueError("Classifier and calibration patient sets must be disjoint")
    head=make_head(algorithm,seed,x.shape[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        warnings.simplefilter("ignore",UserWarning)
        head.fit(x[train],y[train])
    digest=joblib.hash(head)
    calibrator=make_pipeline(StandardScaler(),LogisticRegression(C=1.,solver="lbfgs",max_iter=2000))
    calibrator.fit(raw_score(head,x[calibration]).reshape(-1,1),y[calibration])
    assert digest==joblib.hash(head)
    assert head[1].n_samples_seen_==len(train)
    assert calibrator[0].n_samples_seen_==len(calibration)
    return head,calibrator


def family_probabilities(probabilities):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=len(MEMBERS) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError("Expected finite patient-by-member probabilities in [0,1]")
    return np.column_stack([p[:,ids].mean(axis=1) for ids in family_indices().values()])


def quality(y,score):
    m=R.metrics(y,score)
    return (m["auroc"]+m["average_precision"])/2


def select_rules(probabilities,y):
    """Only the dedicated vote-selection patients enter this function."""
    p=np.asarray(probabilities)
    if len(p)!=len(y):
        raise ValueError("Vote selection score/label length mismatch")
    f=family_probabilities(p)
    member_scores=[quality(y,p[:,j]) for j in range(len(MEMBERS))]
    selected=int(np.argmax(member_scores))
    rules=dict(selected_member=MEMBERS[selected],member_validation_scores=dict(zip(MEMBERS,member_scores)))
    for shrink,name in ((0.,"validation_weighted_soft"),(.5,PRIMARY)):
        candidates=[]
        for w in weight_grid(shrink):
            candidates.append(dict(weights=w.tolist(),validation_score=quality(y,f@w),distance_to_uniform=float(np.sum((w-1/3)**2))))
        best=max(candidates,key=lambda v:(round(v["validation_score"],12),-v["distance_to_uniform"]))
        rules[name]=dict(weights=best["weights"],validation_score=best["validation_score"],candidates=candidates)
    return rules


def combine_votes(probabilities,raw_labels,rules):
    p=np.asarray(probabilities,dtype=float)
    f=family_probabilities(p)
    labels=np.asarray(raw_labels)
    if labels.shape!=p.shape or not np.isin(labels,[0,1]).all():
        raise ValueError("Binary native labels must match probability matrix")
    out=dict(all_hard_raw=labels.mean(axis=1),all_hard_calibrated=(p>.5).mean(axis=1),
             all_soft=p.mean(axis=1),family_balanced_soft=f.mean(axis=1),
             validation_selected_single=p[:,MEMBERS.index(rules["selected_member"])])
    for name in ("validation_weighted_soft",PRIMARY):
        w=np.asarray(rules[name]["weights"])
        if w.shape!=(3,) or np.any(w<0) or not np.isclose(w.sum(),1):
            raise ValueError("Family weights must be nonnegative and sum to one")
        out[name]=f@w
    for j,family in enumerate(FAMILIES):
        out[f"family_{family}_soft"]=f[:,j]
        out[f"without_{family}_soft"]=f[:,np.arange(3)!=j].mean(axis=1)
    return out


def score_metrics(y,score,is_probability=True):
    y=np.asarray(y)
    predicted=np.asarray(score)>.5
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return dict(**R.metrics(y,score),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),
        accuracy=float((tp+tn)/len(y)),brier=float(brier_score_loss(y,score)) if is_probability else None,
        log_loss=float(log_loss(y,np.clip(score,1e-7,1-1e-7),labels=[0,1])) if is_probability else None)


def evaluate():
    assert json.loads((OUT/"protocol_locked.json").read_text())["sha256"]==R.sha256(OUT/"protocol.json")
    patients=frame("patients.csv")
    splits=frame("voting_patient_assignments.csv")
    y=patients.label.to_numpy()
    results,predictions,member_rows,raw_metrics,rules_all,diversity,provenance=[],[],[],[],[],[],[]
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            per=splits[splits.seed.eq(seed)]
            ix={role:np.flatnonzero(patients.patient_id.isin(per.loc[per.role.eq(role),"patient_id"]))
                for role in ("train","calibration","vote_selection","test")}
            arrays=load_features(seed)
            probs=np.zeros((98,len(MEMBERS)))
            hard=np.zeros_like(probs,dtype=np.int8)
            for j,member in enumerate(MEMBERS):
                group,algorithm=member.split("__")
                x=arrays[group]
                head,cal=fit_member(x,y,ix["train"],ix["calibration"],algorithm,seed)
                scores=raw_score(head,x)
                probs[:,j]=cal.predict_proba(scores[:,None])[:,list(cal.classes_).index(1)]
                hard[:,j]=head.predict(x)
                model_path=OUT/"models"/f"{member}_{seed}.joblib"
                joblib.dump(dict(head=head,calibrator=cal),model_path,compress=3)
                provenance.append(dict(seed=seed,member=member,model_file=str(model_path.relative_to(OUT)),sha256=R.sha256(model_path),
                    feature_group=group,feature_dimension=x.shape[1],selected_indices=head[2].get_support(indices=True).tolist(),
                    train_patients=patients.patient_id.iloc[ix["train"]].tolist(),
                    calibration_patients=patients.patient_id.iloc[ix["calibration"]].tolist(),
                    sigmoid_slope=float(cal[-1].coef_[0,0]),sigmoid_intercept=float(cal[-1].intercept_[0]),
                    calibrated_direction_reversed=bool(cal[-1].coef_[0,0]<0)))
                raw_metrics.append(dict(seed=seed,member=member,**R.metrics(y[ix["test"]],scores[ix["test"]])))
                for role in ("calibration","vote_selection","test"):
                    for i in ix[role]:
                        member_rows.append(dict(seed=seed,role=role,patient_id=patients.patient_id.iloc[i],member=member,
                            label=int(y[i]),raw_score=float(scores[i]),raw_label=int(hard[i,j]),probability=float(probs[i,j])))
            vote=ix["vote_selection"]
            rules=select_rules(probs[vote],y[vote])
            rules_all.append(dict(seed=seed,vote_selection_patients=patients.patient_id.iloc[vote].tolist(),**rules))
            test=ix["test"]
            all_scores={f"member__{name}":probs[test,j] for j,name in enumerate(MEMBERS)}
            all_scores.update(combine_votes(probs[test],hard[test],rules))
            for model,score in all_scores.items():
                results.append(dict(seed=seed,model=model,**score_metrics(y[test],score,not model.startswith("all_hard"))))
                for i,value in zip(test,score):
                    predictions.append(dict(seed=seed,model=model,patient_id=patients.patient_id.iloc[i],label=int(y[i]),
                                            score=float(value),prediction=int(value>.5)))
            for a in range(len(MEMBERS)):
                for b in range(a+1,len(MEMBERS)):
                    pa,pb=probs[vote,a],probs[vote,b]
                    correlation=float(np.corrcoef(pa,pb)[0,1]) if pa.std()>1e-12 and pb.std()>1e-12 else None
                    ea,eb=(pa>.5)!=y[vote],(pb>.5)!=y[vote]
                    diversity.append(dict(seed=seed,member_a=MEMBERS[a],member_b=MEMBERS[b],role="vote_selection",n=10,
                        probability_correlation=correlation,disagreement=float(np.mean((pa>.5)!=(pb>.5))),
                        double_fault=float(np.mean(ea&eb))))
            pd.DataFrame(results).to_csv(OUT/"metrics_per_draw.csv",index=False)
            pd.DataFrame(predictions).to_csv(OUT/"patient_test_predictions.csv",index=False)
            pd.DataFrame(member_rows).to_csv(OUT/"member_predictions_by_role.csv",index=False)
            pd.DataFrame(raw_metrics).to_csv(OUT/"raw_member_metrics.csv",index=False)
            pd.DataFrame(diversity).to_csv(OUT/"vote_selection_diversity.csv",index=False)
            R.write_json(OUT/"ensemble_selection.json",rules_all)
            R.write_json(OUT/"model_provenance.json",provenance)
            item=next(v for v in reversed(results) if v["model"]==PRIMARY)
            print(f"{seed}: 15 heads + held-out calibrators; weighted vote ROC={item['auroc']:.4f} AP={item['average_precision']:.4f}",flush=True)
    summarize()


def summarize():
    m=pd.read_csv(OUT/"metrics_per_draw.csv")
    rows=[]
    for name,g in m.groupby("model",sort=False):
        row=dict(model=name,draws=len(g),joint_target_draws=int(((g.auroc>.8)&(g.average_precision>.8)).sum()))
        for field in ("auroc","average_precision","auprc_trapezoid","sensitivity","specificity","accuracy","brier","log_loss"):
            for stat in ("mean","std","min","max"):
                row[f"{field}_{stat}"]=float(getattr(g[field],stat)())
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/"summary.csv",index=False)
    baseline=m[m.model.eq("validation_selected_single")].set_index("seed")
    paired=[]
    for name in ENSEMBLES:
        g=m[m.model.eq(name)].set_index("seed").loc[baseline.index]
        a,b=g.auroc-baseline.auroc,g.average_precision-baseline.average_precision
        paired.append(dict(model=name,delta_auroc_mean=float(a.mean()),delta_ap_mean=float(b.mean()),
                           joint_improved_draws=int(((a>0)&(b>0)).sum()),draws=len(g)))
    pd.DataFrame(paired).to_csv(OUT/"paired_vs_selected_single.csv",index=False)
    print(summary.loc[summary.model.isin(ENSEMBLES),["model","auroc_mean","average_precision_mean","joint_target_draws"]].to_string(index=False),flush=True)


def export_figure(fig,name):
    from PIL import Image
    path=OUT/"figures_600dpi"/f"{name}.png"
    fig.savefig(path,dpi=600,facecolor="white")
    with Image.open(path) as im:
        im.save(path.with_suffix(".tiff"),compression="tiff_lzw",dpi=(600,600))
    fig.savefig(OUT/"qa"/f"{name}_preview.png",dpi=140)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s=pd.read_csv(OUT/"summary.csv").set_index("model")
    m=pd.read_csv(OUT/"metrics_per_draw.csv")
    items=[("validation_selected_single","Validation-selected single"),("all_hard_raw","Raw hard vote"),
           ("all_hard_calibrated","Calibrated hard vote"),("all_soft","All-model soft vote"),
           ("family_balanced_soft","Family-balanced soft vote"),("validation_weighted_soft","Validation-weighted soft vote"),
           (PRIMARY,"Shrunk weighted soft vote")]
    fig,axes=plt.subplots(1,2,figsize=(11,5.7))
    for ax,field,title in zip(axes,["auroc","average_precision"],["AUROC","Average precision (AP)"]):
        for i,(model,label) in enumerate(items):
            values=m.loc[m.model.eq(model),field].to_numpy()
            ax.scatter(values,i+np.linspace(-.12,.12,len(values)),color="#267c70" if i else "#777777",s=19,alpha=.7)
            ax.scatter(values.mean(),i,color="#111111",marker="D",s=26)
        ax.axvline(.8,ls="--",lw=1,color="#b64163")
        ax.set_xlim(0,1)
        ax.set_ylim(6.6,-.6)
        ax.set_yticks(range(7),[v[1] for v in items],fontsize=9)
        ax.set_title(title,fontsize=12)
        ax.grid(axis="x",alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Patient-level voting: identical test patients, separated calibration and selection",fontsize=12)
    fig.subplots_adjust(left=.21,right=.99,top=.87,bottom=.19,wspace=.95)
    fig.text(.04,.055,"Dots: all ten patient-random draws. Diamonds: means. Dashed line: 0.80. Repeated patients are not independent cohorts.\nHard-vote ranking metrics use the positive-vote fraction. No clinical probability or external-validation claim.",fontsize=9)
    export_figure(fig,"Figure_V1_voting_results")
    plt.close(fig)
    selections=json.loads((OUT/"ensemble_selection.json").read_text())
    weights=np.array([item[PRIMARY]["weights"] for item in selections])
    fig,ax=plt.subplots(figsize=(7.4,5.8))
    cmap=ax.imshow(weights,vmin=0,vmax=1,cmap="cividis",aspect="auto")
    ax.set_xticks(range(3),["Radiomics","RadImageNet","Green Learning"],fontsize=10)
    ax.set_yticks(range(10),[str(v["seed"]) for v in selections],fontsize=9)
    for i in range(10):
        for j in range(3):
            ax.text(j,i,f"{weights[i,j]:.3f}",ha="center",va="center",color="white" if weights[i,j]<.45 else "black",fontsize=10)
    fig.colorbar(cmap,ax=ax,label="Effective vote weight")
    ax.set_title("Validation-selected family weights, with fixed shrinkage",fontsize=12,pad=14)
    fig.subplots_adjust(left=.18,right=.93,top=.88,bottom=.15)
    fig.text(.05,.05,"Weights selected using ten vote-selection patients only. Each row sums to one.\nFixed 50% shrinkage toward equal family weights; minimum family weight is 1/6.",fontsize=9)
    export_figure(fig,"Figure_V2_family_weights")
    plt.close(fig)
    allrows=["| Workflow | Mean AUROC | Mean AP | Both >0.80 draws |","|---|---:|---:|---:|"]
    for model,label in items:
        row=s.loc[model]
        allrows.append(f"| {label} | {row.auroc_mean:.4f} | {row.average_precision_mean:.4f} | {int(row.joint_target_draws)}/10 |")
    prov=json.loads((OUT/"model_provenance.json").read_text())
    reversed_count=sum(p["calibrated_direction_reversed"] for p in prov)
    if (OUT/"inherited_method_references.json").exists():
        inherited=json.loads((OUT/"inherited_method_references.json").read_text())
    else:
        inherited=json.loads((R.OUT/"references.json").read_text())
        inherited=[r for r in inherited if 4<=r["number"]<=9]
    R.write_json(OUT/"inherited_method_references.json",inherited)
    text="\n".join([
        "# Multi-model voting: internal patient-random experiment", "",
        "## Design", "",
        "The original 98 patients, 349 images and ten original train/test allocations were retained. Within each draw, the original 20 validation patients were split into ten probability-calibration patients and ten vote-selection patients. The original 58 training patients and 20 test patients were unchanged. No patient occurs in multiple roles within a draw.", "",
        "This is further retrospective exploration of an already explored cohort. Reusing the same patients does not constitute untouched or external validation. All ten draws are reported; repeated appearances are not independent patients.", "",
        "## Base models and reuse", "",
        "Fifteen models combine five existing patient feature groups with three established classification heads. Feature groups are original/ROI-percentile PyRadiomics, original/ROI-percentile frozen RadImageNet DenseNet-121, and the original training-only fitted GreenExtractor. Each group uses SVM, L2 logistic regression, and random forest. Patient features are image mean and population standard deviation. No image-generation model, new segmentation fit, image-encoder retraining, clinical covariate, or extra rotation was used.", "",
        "Each classifier head is fitted only on the same 58 training patients, with train-only median imputation, standardization and ANOVA selection of 32 features. Settings are fixed, not selected on the calibration patients: SVM uses RBF/C=1/gamma=scale; LR uses C=0.1/liblinear; RF uses 256 trees, maximum depth 4, minimum leaf size 3, and square-root feature sampling. All heads use balanced class weights. The previously selected SVM heads were not silently reused, because their hyperparameters had already been selected on all 20 previous validation patients.", "",
        "## Calibration and votes", "",
        "A one-dimensional StandardScaler plus L2 logistic regression (C=1, lbfgs) maps each member's scores to an estimated positive-class probability, fitted only on the ten calibration patients. This is regularized sigmoid calibration, not an exact unregularized Platt implementation and not a claim of well-calibrated clinical probabilities. No implicit probability-fitting split or model-refitting wrapper is used.", "",
        "Raw hard voting averages native binary decisions. Calibrated hard voting averages p>0.5 decisions. Their AUROC/AP use the fraction of positive votes, whereas the final class uses a >0.5 threshold; ties are negative. Hard-vote fractions are not assigned probability-calibration metrics. Soft voting averages estimated probabilities. Family-balanced soft voting first averages within radiomics/RadImageNet/Green families, then equally across the three families, preventing the 6/6/3 member counts from imposing unintended family weights.", "",
        "Weighted voting searches 15 nonnegative family-weight triples at step 0.25, selecting only on the ten vote-selection patients using mean(AUROC,AP). Ties prefer weights closer to uniform. The protocol-designated primary variant shrinks every candidate halfway toward uniform before selection. The selected-single comparator is selected on the identical vote-selection patients. Three single-family and three leave-one-family-out soft votes are reported as ablations. This is not a replication of Caruana et al.'s forward-selection algorithm.", "",
        "## Results", "", *allrows, "",
        f"Across 150 member calibrators, {reversed_count} had a negative fitted sigmoid slope. This indicates calibration-direction instability on ten-patient samples; it is explicitly retained and audited rather than corrected using test outcomes. Compare raw_member_metrics.csv with calibrated member results before interpreting any gain as improved image representation.", "",
        "All 28 workflows, individual predictions, raw-member scores, per-role member probabilities, weights, hyperparameters, input hashes and fitted models are preserved. Pairwise probability correlation, decision disagreement and double-fault rates use vote-selection patients only. Correlated models can share errors; a larger number of votes is not evidence of greater diagnostic information.", "",
        "## Interpretation", "",
        "Voting is evaluated against a same-protocol selected-single classifier, not just a historically selected best result. Comparisons with earlier manuscripts are contextual because classifier settings and validation-role allocation changed. Ten calibration and ten vote-selection patients can produce unstable probability maps and weights. Mean/SD/range across dependent draws are descriptive, not confidence intervals. No model with an isolated good draw is presented as reliably exceeding the target. AP and trapezoidal PR area are separately retained.", "",
        "If votes do not improve both discrimination metrics consistently, the next priority is more informative, less artifact-dependent input and local/patient-bag representations, not enlarging a library of correlated weak predictions. Multiple backbones or a stacking meta-model would require separately documented training and selection; neither is claimed here.", "",
        "## Post-primary follow-up", "",
        "A separately registered native-versus-calibrated LR/RF probability control was initiated after these primary results were observed. Its six fixed workflows and all ten draws are preserved in probability_control/RESULTS.md, with a separate protocol lock. It does not modify or replace the primary results above.", "",
        "## References", "", *[f"- {r['citation']} {r['url']} Scope: {r['use']}" for r in REFERENCES], "",
        "## Inherited feature methods", "", *[f"- {r['citation']} {r['url']}" for r in inherited], "",
    ])
    (OUT/"RESULTS.md").write_text(text,encoding="utf-8")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["prepare","evaluate","report","all"])
    args=parser.parse_args()
    for stage in (["prepare","evaluate","report"] if args.stage=="all" else [args.stage]):
        globals()[stage]()
