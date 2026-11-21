"""Matched mean-pooling and gated-attention MIL on frozen DenseNet image features."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from torch import nn

import renal_dinov3_native_probe as N
import renal_random_validation as R
import renal_voting_validation as V

OUT=R.ROOT/"CYCH_patient_bag_probe_20260913"
OPT_SEEDS=(11,22,33)


class BagClassifier(nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.classifier=nn.Linear(16,1)
        self.attention=attention
        if attention:
            self.v=nn.Linear(16,16)
            self.u=nn.Linear(16,16)
            self.w=nn.Linear(16,1,bias=False)

    def forward(self,x,mask):
        if self.attention:
            logits=self.w(torch.tanh(self.v(x))*torch.sigmoid(self.u(x))).squeeze(-1)
            weights=torch.softmax(logits.masked_fill(~mask,-torch.inf),dim=1)
        else:
            weights=mask.float()/mask.sum(1,keepdim=True)
        pooled=(x*weights.unsqueeze(-1)).sum(1)
        return self.classifier(pooled).squeeze(-1),weights


def read(name):
    return pd.read_csv(OUT/name,dtype={"patient_id":str})


def prepare():
    native=pd.read_csv(N.OUT/"summary_by_role.csv")
    dense=native[(native.role=="development_validation")&native.group.isin(["DN-A","DN-B","DN-C"])].copy()
    dense["selection_score"]=(dense.auroc_mean+dense.average_precision_mean)/2
    selected=dense.sort_values(["selection_score","group"],ascending=[False,True]).iloc[0].group
    feature="dense_tight_image.npy" if selected=="DN-C" else "dense_canvas_image.npy"
    OUT.mkdir(exist_ok=True)
    for folder in ("source","models","figures_600dpi","qa","features"):
        (OUT/folder).mkdir(exist_ok=True)
    for name in ("patients.csv","image_manifest.csv","random_patient_assignments.csv"):
        shutil.copy2(N.OUT/name,OUT/name)
    shutil.copy2(N.OUT/"features"/feature,OUT/"features"/feature)
    protocol=dict(
        design="Ten previously defined patient-random draws; all98patients previously explored; no k-fold/external confirmation.",
        encoder="Frozen existing RadImageNet DenseNet121; native DINOv3 failed its standalone gate. Select DenseNet input with highest development mean(AUROC+AP)/2, simpler ID tie. No test-driven choice.",
        native_selected_configuration=selected, image_features=feature, image_feature_sha256=R.sha256(OUT/"features"/feature),
        source_decision_sha256=R.sha256(N.OUT/"development_decision.json"),
        projection="Image embeddings -> train-only StandardScaler -> PCA16 full solver -> StandardScaler. For epoch selection fit only46 inner-training patients; refit on58 outer-training patients for final models.",
        bags="One patient per bag, all1-4 original images; padded instances masked. Patient-level class-balanced BCE; no image-count feature.",
        models="Same16-dimensional linear classifier: mean pooling vs Ilse2018 gated attention tanh(Vx)*sigmoid(Ux),hidden16,one head. No dropout. Encoder remains frozen.",
        training=dict(optimizer="Adam",lr=.001,weight_decay=.001,full_batch=True,max_epochs=200,optimizer_seeds=OPT_SEEDS,
            epoch_selection="StratifiedShuffleSplit(test_size=12,random_state=outer seed) within58. Lowest inner12 loss over200epochs; ties earliest. Retrain new model on58 for selected epoch. Outer20 development never used for early stopping."),
        comparison="Retain all3 optimization seeds and use their fixed probability mean per method as primary. No best-seed selection. Attention must exceed matched mean pooling in development mean AUROC>=.03/AP>=.02 and joint improvements>=6/10, plus both metrics >= original DenseNet-.02, before one fixed RAD+DN+attention vote.",
        vote_gate="Development vs original fixed pair: AUROC delta>=.03,AP delta>=.02,joint improvements>=6/10; otherwise retain pair. No test tuning.",
        attribution="Attention is relative model weighting, not inflammation, fibrosis, scar, or swelling localization. Display only image aliases.",
        reference=dict(citation="Ilse M, Tomczak JM, Welling M. Attention-based Deep Multiple Instance Learning. ICML2018;80:2127-2136.",url="https://proceedings.mlr.press/v80/ilse18a.html"))
    protocol=json.loads(json.dumps(protocol))
    path=OUT/"protocol.json"
    if path.exists():
        assert json.loads(path.read_text())==protocol
    else:
        R.write_json(path,protocol)
    lock=OUT/"protocol_locked.json"
    if not lock.exists():
        R.write_json(lock,dict(utc=datetime.now(timezone.utc).isoformat(),sha256=R.sha256(path),source_sha256=R.sha256(Path(__file__))))
    shutil.copy2(__file__,OUT/"source"/Path(__file__).name)
    print(f"Locked patient-bag experiment; selected {selected} using development only",flush=True)


def tensor_bags(image_x,df,patients):
    bags=np.zeros((len(patients),4,16),np.float32)
    masks=np.zeros((len(patients),4),bool)
    for i,p in enumerate(patients.patient_id):
        values=image_x[df.patient_id.eq(p)]
        assert 1<=len(values)<=4
        bags[i,:len(values)]=values
        masks[i,:len(values)]=True
    return torch.from_numpy(bags),torch.from_numpy(masks)


def project(features,df,patients,train):
    train_images=df.patient_id.isin(patients.patient_id.iloc[train]).to_numpy()
    projector=make_pipeline(StandardScaler(),PCA(n_components=16,svd_solver="full"),StandardScaler())
    projector.fit(features[train_images])
    x,mask=tensor_bags(projector.transform(features).astype(np.float32),df,patients)
    return projector,x,mask


def fit(x,mask,y,train,attention,optimizer_seed,epochs,validation=None):
    torch.manual_seed(optimizer_seed)
    model=BagClassifier(attention)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.001)
    weight=float((1-y[train]).sum()/y[train].sum())
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    best_epoch,best_loss=1,float("inf")
    losses=[]
    for epoch in range(1,epochs+1):
        optimizer.zero_grad()
        logits,_=model(x[train],mask[train])
        loss=criterion(logits,y[train])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        if validation is not None:
            with torch.inference_mode():
                value=float(criterion(model(x[validation],mask[validation])[0],y[validation]))
            losses.append(dict(epoch=epoch,inner_loss=value))
            if value<best_loss:
                best_loss,best_epoch=value,epoch
    return model.eval(),best_epoch,losses


def evaluate():
    lock=json.loads((OUT/"protocol_locked.json").read_text())
    assert lock["sha256"]==R.sha256(OUT/"protocol.json") and lock["source_sha256"]==R.sha256(Path(__file__))
    protocol=json.loads((OUT/"protocol.json").read_text())
    features=np.load(OUT/"features"/protocol["image_features"])
    assert R.sha256(OUT/"features"/protocol["image_features"])==protocol["image_feature_sha256"]
    patients,df,split=read("patients.csv"),read("image_manifest.csv"),read("random_patient_assignments.csv")
    labels=patients.label.to_numpy(int)
    y=torch.from_numpy(labels.astype(np.float32))
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    scores,records,predictions,attributions,audit,history={},{},[],[],[],[]
    records=[]
    with threadpool_limits(limits=4):
        for seed in R.SEEDS:
            indices=N.indices_for(patients,split,seed)
            tr=indices["train"]
            inner=StratifiedShuffleSplit(n_splits=1,test_size=12,random_state=seed)
            itr,iv=next(inner.split(np.zeros(len(tr)),labels[tr]))
            itr,iv=tr[itr],tr[iv]
            pi,xi,mi=project(features,df,patients,itr)
            pf,xf,mf=project(features,df,patients,tr)
            joblib.dump(dict(inner=pi,final=pf),OUT/"models"/f"projection_{seed}.joblib",compress=3)
            scores[seed]={}
            for name,attention in (("MIL-mean",False),("MIL-attention",True)):
                all_scores=[]
                for opt_seed in OPT_SEEDS:
                    _,epoch,losses=fit(xi,mi,y,itr,attention,opt_seed,200,iv)
                    model,_,_=fit(xf,mf,y,tr,attention,opt_seed,epoch)
                    with torch.inference_mode():
                        logits,weights=model(xf,mf)
                        score=torch.sigmoid(logits).numpy()
                        reversed_logits=model(xf.flip(1),mf.flip(1))[0]
                        perm_error=float((logits-reversed_logits).abs().max())
                        perturbed=xf.clone()
                        perturbed[~mf]=999
                        padding_error=float((logits-model(perturbed,mf)[0]).abs().max())
                    assert perm_error<1e-5 and padding_error<1e-5
                    path=OUT/"models"/f"{name}_{seed}_{opt_seed}.pt"
                    torch.save(model.state_dict(),path)
                    restored=BagClassifier(attention).eval()
                    restored.load_state_dict(torch.load(path,weights_only=True))
                    with torch.inference_mode():
                        assert np.allclose(torch.sigmoid(restored(xf,mf)[0]).numpy(),score,atol=1e-7)
                    scores[seed][f"{name}-seed{opt_seed}"]=score
                    all_scores.append(score)
                    audit.append(dict(seed=seed,group=name,optimizer_seed=opt_seed,selected_epoch=epoch,
                        train_patients=patients.patient_alias.iloc[tr].tolist(),inner_train=patients.patient_alias.iloc[itr].tolist(),
                        inner_validation=patients.patient_alias.iloc[iv].tolist(),parameters=sum(p.numel() for p in model.parameters()),
                        permutation_maxabs=perm_error,padding_maxabs=padding_error,model_sha256=R.sha256(path)))
                    history.extend(dict(seed=seed,group=name,optimizer_seed=opt_seed,**l) for l in losses)
                    if attention:
                        for i in np.concatenate([indices["validation"],indices["test"]]):
                            imgs=df.loc[df.patient_id.eq(patients.patient_id.iloc[i]),"image_id"].tolist()
                            for j,image_id in enumerate(imgs):
                                attributions.append(dict(seed=seed,optimizer_seed=opt_seed,patient_alias=patients.patient_alias.iloc[i],
                                    image_id=image_id,attention=float(weights[i,j]),role="development_validation" if i in indices["validation"] else "exploratory_test"))
                scores[seed][name]=np.mean(all_scores,axis=0)
            for group,score in scores[seed].items():
                ix=indices["validation"]
                records.append(dict(seed=seed,group=group,role="development_validation",**V.score_metrics(labels[ix],score[ix])))
            print(f"MIL mean/attention, three seeds each: {seed}",flush=True)
    dev=pd.DataFrame(records).set_index(["group","seed"])
    delta=dev.loc["MIL-attention",["auroc","average_precision"]]-dev.loc["MIL-mean",["auroc","average_precision"]]
    avg=dev.loc["MIL-attention",["auroc","average_precision"]].mean()
    wins=int(delta.gt(0).all(axis=1).sum())
    advance=bool(delta.auroc.mean()>=.03-1e-12 and delta.average_precision.mean()>=.02-1e-12 and wins>=6 and avg.auroc>=.68 and avg.average_precision>=.7530878609850607)
    decision=dict(selection_role="development_validation",matched_delta_auroc=float(delta.auroc.mean()),matched_delta_ap=float(delta.average_precision.mean()),joint_wins=wins,advance_to_vote=advance)
    baseline=pd.read_csv(N.OUT/"patient_predictions_by_role.csv")
    for seed in R.SEEDS:
        for name in ("PAIR","RAD","DN-A"):
            score=np.full(98,np.nan)
            per=baseline[(baseline.seed==seed)&(baseline.group==name)].set_index("patient_alias")
            for i,p in enumerate(patients.patient_alias):
                if p in per.index:
                    score[i]=per.loc[p,"score"]
            scores[seed][name]=score
        if advance:
            scores[seed]["VOTE"]=(scores[seed]["RAD"]+scores[seed]["DN-A"]+scores[seed]["MIL-attention"])/3
    if advance:
        deltas=[]
        for seed in R.SEEDS:
            ix=N.indices_for(patients,split,seed)["validation"]
            mv=V.score_metrics(labels[ix],scores[seed]["VOTE"][ix])
            mb=V.score_metrics(labels[ix],scores[seed]["PAIR"][ix])
            deltas.append([mv["auroc"]-mb["auroc"],mv["average_precision"]-mb["average_precision"]])
        delta=np.array(deltas)
        decision["vote"]=dict(delta_auroc=float(delta[:,0].mean()),delta_ap=float(delta[:,1].mean()),joint_wins=int((delta>0).all(1).sum()),
            promoted=bool(delta[:,0].mean()>=.03 and delta[:,1].mean()>=.02 and (delta>0).all(1).sum()>=6))
    R.write_json(OUT/"development_decision.json",decision)
    for seed in R.SEEDS:
        ix=N.indices_for(patients,split,seed)
        for group,score in scores[seed].items():
            for role,label in (("validation","development_validation"),("test","exploratory_test")):
                assert np.isfinite(score[ix[role]]).all()
                if role=="test" or group in ("PAIR","RAD","DN-A","VOTE"):
                    records.append(dict(seed=seed,group=group,role=label,**V.score_metrics(labels[ix[role]],score[ix[role]])))
                for i in ix[role]:
                    predictions.append(dict(seed=seed,group=group,role=label,patient_alias=patients.patient_alias.iloc[i],label=int(labels[i]),score=float(score[i])))
    metrics=pd.DataFrame(records)
    metrics.to_csv(OUT/"metrics_by_role_per_draw.csv",index=False)
    N.summarize(metrics).to_csv(OUT/"summary_by_role.csv",index=False)
    pd.DataFrame(predictions).to_csv(OUT/"patient_predictions_by_role.csv",index=False)
    pd.DataFrame(attributions).to_csv(OUT/"image_attention_weights.csv",index=False)
    pd.DataFrame(history).to_csv(OUT/"inner_validation_loss.csv",index=False)
    R.write_json(OUT/"fit_audit.json",audit)
    print(N.summarize(metrics).query("group in ['MIL-mean','MIL-attention','PAIR']")[["role","group","auroc_mean","average_precision_mean"]].to_string(index=False),flush=True)
    print(json.dumps(decision),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("prepare","evaluate","all"))
    args=parser.parse_args()
    for stage in (("prepare","evaluate") if args.stage=="all" else (args.stage,)):
        globals()[stage]()
