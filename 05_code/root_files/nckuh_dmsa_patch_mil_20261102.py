"""Ilse-style gated MIL of frozen in-mask patch tokens, patient-grouped kidney endpoint."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from nckuh_dmsa_kidney_baseline_20261102 import ROOT,AUDIT,OUT as BASE,SOURCE,SEEDS,load_reference,verify_lock
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_contralateral_probe_20261102 import rank_metrics
from nckuh_only_study_20261102 import digest,write_json

OUT=ROOT/'02_experiments/NCKUH_DMSA_patch_mil_20261102'
CACHE=ROOT/'02_experiments/NCKUH_normal_patch_probe_20261102'
PILOT_SEEDS=SEEDS[:3]
ENCODERS=['dinov3','usf_mae']
MODES=['mean','gated']
REFERENCE='Summary three source fusion'


class PatchBagModel(nn.Module):
    def __init__(self,mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode=mode
        self.projection=nn.Linear(32,32)
        self.dropout=nn.Dropout(.25)
        if mode=='mean':
            self.second=nn.Linear(32,32)
        else:
            self.v=nn.Linear(32,16)
            self.u=nn.Linear(32,16)
            self.w=nn.Linear(16,1,bias=False)
        self.classifier=nn.Linear(32,1)

    def forward(self,x,mask):
        hidden=self.dropout(torch.relu(self.projection(x)))
        if self.mode=='mean':
            hidden=torch.relu(self.second(hidden))
            weights=mask.float()/mask.sum(1,keepdim=True)
        else:
            attention=self.w(torch.tanh(self.v(hidden))*torch.sigmoid(self.u(hidden))).squeeze(-1)
            weights=torch.softmax(attention.masked_fill(~mask,-torch.inf),dim=1)
        pooled=(weights.unsqueeze(-1)*hidden).sum(1)
        return self.classifier(pooled).squeeze(-1),weights


def balanced_patch_sample(patch_aliases,train_aliases,seed):
    rng=np.random.default_rng(seed+7000)
    indices=[]
    for alias in sorted(train_aliases):
        available=np.flatnonzero(patch_aliases==alias)
        if not len(available):
            raise ValueError('Training patient has no patch')
        indices.extend(rng.choice(available,64,replace=len(available)<64))
    return np.asarray(indices,dtype=np.int64)


def prepare():
    verify_lock(); load_reference()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Preserve registered patch MIL experiment')
    proof=json.loads((CACHE/'encoder_provenance.json').read_text(encoding='utf-8'))
    for encoder in ENCODERS:
        assert proof[encoder]['frozen']
        assert digest(CACHE/f'{encoder}_roi_patches.npy')==proof[encoder]['patch_cache_sha256']
    paths=[Path(__file__),SOURCE,BASE/'patient_assignments.csv',BASE/'development_metrics.csv',BASE/'development_predictions.csv',
        AUDIT/'patient_label_linkage_private.csv',AUDIT/'kidney_label_linkage_private.csv',AUDIT/'image_labels_dmsa_private.csv',
        CACHE/'encoder_provenance.json',CACHE/'patch_offsets.npy',
        ROOT/'02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH/image_manifest.csv',
        Path(__file__).with_name('nckuh_dmsa_cluster_ci.py')]
    paths += [CACHE/f'{name}_roi_patches.npy' for name in ENCODERS]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Whole-kidney token means can dilute local abnormalities. Compare learned patch attention with an approximately capacity-matched uniform-pooling network on the same frozen ROI tokens',
        references=['https://proceedings.mlr.press/v80/ilse18a.html','https://github.com/AMLab-Amsterdam/AttentionDeepMIL',
                    'https://arxiv.org/abs/2508.10104','https://arxiv.org/abs/2510.22990'],
        method='Gated tanh/sigmoid attention from Ilse et al.2018, adapted to pretrained renal ROI patch vectors with masked padding. Not an exact MNIST/CNN reproduction or newly named architecture',
        cohort='210 source-code-available patients,414 imaged kidneys,820 images;54 unavailable outcomes excluded. One malformed date/exact DMSA timing still pending',
        encoders=ENCODERS,configurations=[(e,m) for e in ENCODERS for m in MODES],pilot_seeds=PILOT_SEEDS,
        inputs='Existing strictly validated frozen DINOv3/USF-MAE patch caches. Only patches with>=80% existing kidney-mask coverage, all selected patches from a kidney form one bag. No positive patch labels, new masks or caliper guesses',
        preprocessing='L2 normalize patch vectors. Sample64 patches/training patient, replacement only if fewer than64. Fit randomizedPCA32 with whitening on those training-patient-balanced patches only. Inner selection uses a separate100-patient fitted PCA; full126 refit gets a freshly fitted PCA',
        mean_control='32-to32 ReLU/dropout then32-to32 ReLU; uniform masked mean; linear classifier,2145 parameters',
        gated='32-to32 ReLU/dropout, gated16 attention, weighted sum and linear classifier,2161 parameters. Difference16 parameters (<1%) is disclosed, not exact matched functional capacity',
        training='Full-batch class-balanced kidney BCE; AdamW lr.001 weight_decay.001, dropout.25, gradient norm5. No negative-patch auxiliary loss or pair ranking in this experiment',
        inner='One100/26 patient split inside each126 training role, stratified by R/L/B/N,seed+4000; maximize inner mean(AUROC,AP), ties earliest, max80epochs/minimum stopping10/patience15. Reset model and refit all126 for selected epochs',
        evaluation='Three existing random patient draws, outer42development only; both sides/all views stay together, no k-fold or test scoring',
        expansion_gate='Mean AUROC>=.70 and AP>=.78, changes>=.03AUROC and>=.02AP vs corrected summary baseline, both improve>=2/3 draws. Either valid pooling method can qualify; attention is only credited against its matched mean control',
        uncertainty='10000 stratified patient-cluster fixed-prediction resamples; conditional within-cohort, not selection-adjusted',
        interpretation='Attention is feature contribution, not a lesion/scar/compartment diagnosis. Historical exploration prevents independent-cohort claims. Goal remains kidney AUROC and AP both>.8')
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'run_lock.json',dict(protocol_sha256=digest(OUT/'protocol.json'),inputs={str(p):digest(p) for p in paths}))
    print('Registered four local patch pooling candidates before evaluation',flush=True)


def require_lock():
    lock=json.loads((OUT/'run_lock.json').read_text(encoding='utf-8'))
    assert digest(OUT/'protocol.json')==lock['protocol_sha256']
    for path,sha in lock['inputs'].items():
        assert digest(Path(path))==sha,path


def local_patch_data(encoder,images,kidneys):
    raw=np.load(CACHE/f'{encoder}_roi_patches.npy',mmap_mode='r')
    offsets=np.load(CACHE/'patch_offsets.npy')
    manifest=pd.read_csv(ROOT/'02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH/image_manifest.csv')
    assert images.image_id.tolist()==manifest.image_id.tolist() and images.filename.tolist()==manifest.filename.tolist()
    assert len(offsets)==1028 and offsets[0]==0 and offsets[-1]==len(raw)==99865
    kept_images=np.flatnonzero(images.kidney_APN.notna())
    rows=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in kept_images])
    values=np.asarray(raw[rows])
    norm=np.linalg.norm(values,axis=1,keepdims=True)
    assert np.isfinite(values).all() and (norm>1e-8).all()
    values=values/norm
    image_index=np.repeat(kept_images,np.diff(offsets)[kept_images])
    patch_aliases=images.patient_alias.to_numpy()[image_index]
    patch_kidneys=images.kidney_alias.to_numpy()[image_index]
    groups=[np.flatnonzero(patch_kidneys==alias) for alias in kidneys.kidney_alias]
    assert sum(map(len,groups))==len(values) and all(len(g)>0 for g in groups)
    return values,patch_aliases,groups,rows


def prepare_bags(values,patch_aliases,groups,train_aliases,seed):
    from sklearn.decomposition import PCA
    sample=balanced_patch_sample(patch_aliases,train_aliases,seed)
    assert set(patch_aliases[sample])==set(train_aliases)
    pca=PCA(32,whiten=True,random_state=seed,svd_solver='randomized').fit(values[sample])
    transformed=pca.transform(values).astype(np.float32)
    bags=np.zeros((len(groups),max(map(len,groups)),32),dtype=np.float32)
    mask=np.zeros(bags.shape[:2],dtype=bool)
    for i,indices in enumerate(groups):
        bags[i,:len(indices)]=transformed[indices]
        mask[i,:len(indices)]=True
    assert np.isfinite(bags).all()
    proof=dict(train_aliases=sorted(train_aliases),selected_local_patch_indices=sample,
               patch_count=len(sample),pca=pca)
    return torch.from_numpy(bags).cuda(),torch.from_numpy(mask).cuda(),proof


def fit_model(bags,mask,y,fit,inner,mode,seed,epochs=None):
    torch.manual_seed(seed+37)
    model=PatchBagModel(mode).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    target=y[fit]
    class_weights=torch.where(target>0,len(target)/(2*target.sum()),len(target)/(2*(len(target)-target.sum())))
    best,chosen,bad=-float('inf'),1,0
    history=[]
    for epoch in range(1,(epochs if epochs is not None else 80)+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        logits,_=model(bags[fit],mask[fit])
        loss=(nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*class_weights).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite patch MIL objective')
        loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
        row=dict(phase='refit' if epochs is not None else 'inner',epoch=epoch,train_bce=float(loss.detach()))
        if inner is not None:
            model.eval()
            with torch.inference_mode():
                logits,_=model(bags[inner],mask[inner])
                score=torch.sigmoid(logits).cpu().numpy()
            m=rank_metrics(y[inner].cpu().numpy().astype(int),score)
            row.update(inner_auroc=m['auroc'],inner_ap=m['ap'])
            criterion=(m['auroc']+m['ap'])/2
            if criterion>best+1e-12:
                best,chosen,bad=criterion,epoch,0
            else:
                bad+=1
        history.append(row)
        if inner is not None and epoch>=10 and bad>=15:
            break
    return model.eval(),chosen if inner is not None else epochs,history


def pilot():
    from sklearn.model_selection import train_test_split
    require_lock()
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    patients,kidneys,images=load_reference()
    assignments=pd.read_csv(BASE/'patient_assignments.csv')
    y=torch.tensor(kidneys.kidney_APN.to_numpy(int),dtype=torch.float32,device='cuda')
    records,predictions,audits=[],[],[]
    (OUT/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for encoder in ENCODERS:
            values,patch_aliases,groups,global_rows=local_patch_data(encoder,images,kidneys)
            for seed in PILOT_SEEDS:
                result_path=OUT/f'result_{encoder}_{seed}.json'
                if result_path.exists():
                    saved=json.loads(result_path.read_text(encoding='utf-8'))
                    records.extend(saved['metrics']); predictions.extend(saved['predictions']); audits.append(saved['audit'])
                    continue
                mapping=assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
                outer=patients[patients.patient_alias.map(mapping).eq('train')]
                fit_alias,inner_alias=train_test_split(outer.patient_alias.to_numpy(),train_size=100,
                    random_state=seed+4000,stratify=outer.DMSA_APN)
                fit=np.flatnonzero(kidneys.patient_alias.isin(fit_alias)); inner=np.flatnonzero(kidneys.patient_alias.isin(inner_alias))
                train=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('train'))
                dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
                assert not set(kidneys.patient_alias.iloc[train])&set(kidneys.patient_alias.iloc[dev])
                inner_bags,inner_mask,inner_proof=prepare_bags(values,patch_aliases,groups,fit_alias,seed)
                selected,history_by_mode={},{}
                for mode in MODES:
                    model,epoch,history=fit_model(inner_bags,inner_mask,y,fit,inner,mode,seed)
                    selected[mode]=epoch; history_by_mode[mode]=history
                    del model
                del inner_bags,inner_mask; torch.cuda.empty_cache()
                bags,mask,full_proof=prepare_bags(values,patch_aliases,groups,outer.patient_alias.to_numpy(),seed)
                joblib.dump(dict(inner=inner_proof,full=full_proof,source_patch_rows=global_rows),
                    OUT/'models'/f'{encoder}_{seed}_preprocess.joblib',compress=3)
                current_metrics,current_predictions=[],[]
                for mode in MODES:
                    model,epoch,refit=fit_model(bags,mask,y,train,None,mode,seed,selected[mode])
                    with torch.inference_mode():
                        logits,attention=model(bags[dev],mask[dev])
                        score=torch.sigmoid(logits).cpu().numpy()
                    group=encoder+'_'+mode
                    parameters=sum(p.numel() for p in model.parameters())
                    current_metrics.append(dict(group=group,seed=seed,role='development',patients=42,kidneys=len(dev),
                        selected_epoch=epoch,parameters=parameters,**rank_metrics(y[dev].cpu().numpy().astype(int),score)))
                    for i,s in zip(dev,score):
                        k=kidneys.iloc[i]
                        current_predictions.append(dict(group=group,seed=seed,patient_alias=k.patient_alias,kidney_alias=k.kidney_alias,
                            DMSA_APN=k.DMSA_APN,side=k.side,label=int(k.kidney_APN),score=float(s)))
                    torch.save(dict(model=model.state_dict(),mode=mode,epoch=epoch),OUT/'models'/f'{group}_{seed}.pth')
                    np.savez_compressed(OUT/f'{group}_{seed}_development_attention.npz',kidney_indices=dev,
                        attention=attention.cpu().numpy(),mask=mask[dev].cpu().numpy())
                    pd.DataFrame(history_by_mode[mode]+refit).to_csv(OUT/f'history_{group}_{seed}.csv',index=False)
                    del model
                audit=dict(encoder=encoder,seed=seed,inner_fit_patients=len(fit_alias),inner_validation_patients=len(inner_alias),
                    full_refit_patients=len(outer),inner_pca_patches=inner_proof['patch_count'],full_pca_patches=full_proof['patch_count'],
                    selected_epoch=selected,fit_aliases=fit_alias.tolist(),inner_aliases=inner_alias.tolist(),
                    total_labeled_patches=len(values),kidney_bags=len(groups),max_bag_patches=max(map(len,groups)))
                write_json(result_path,dict(metrics=current_metrics,predictions=current_predictions,audit=audit))
                records.extend(current_metrics); predictions.extend(current_predictions); audits.append(audit)
                print(pd.DataFrame(current_metrics)[['group','seed','auroc','ap','selected_epoch']].to_string(index=False),flush=True)
                del bags,mask; torch.cuda.empty_cache()
            del values
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'pilot_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'pilot_predictions.csv',index=False)
    write_json(OUT/'training_audits.json',audits)
    summary=frame.groupby('group')[['auroc','ap','auprc']].mean()
    summary['mean_score']=summary[['auroc','ap']].mean(axis=1)
    summary=summary.sort_values(['mean_score','group'],ascending=[False,True])
    reference=pd.read_csv(BASE/'development_metrics.csv')
    reference=reference[reference.group.eq(REFERENCE)&reference.seed.isin(PILOT_SEEDS)].set_index('seed')[['auroc','ap']]
    eligible=[]
    for group in summary.index:
        delta=frame[frame.group.eq(group)].set_index('seed')[['auroc','ap']]-reference
        wins=int(delta.gt(0).all(axis=1).sum())
        summary.loc[group,'delta_auroc']=delta.auroc.mean(); summary.loc[group,'delta_ap']=delta.ap.mean(); summary.loc[group,'joint_wins']=wins
        if summary.loc[group,'auroc']>=.70 and summary.loc[group,'ap']>=.78 and delta.auroc.mean()>=.03 and delta.ap.mean()>=.02 and wins>=2:
            eligible.append(group)
    summary.to_csv(OUT/'pilot_summary.csv')
    write_json(OUT/'pilot_decision.json',dict(selected_for_expansion=eligible[0] if eligible else None,eligible=eligible,
        exploratory_test_scored=False,goal_achieved=False))
    print(summary.to_string(),flush=True)


def bootstrap():
    require_lock()
    frame=pd.read_csv(OUT/'pilot_predictions.csv')
    baseline=pd.read_csv(BASE/'development_predictions.csv')
    baseline=baseline[baseline.group.eq(REFERENCE)&baseline.seed.isin(PILOT_SEEDS)]
    groups=[e+'_'+m for e in ENCODERS for m in MODES]+[REFERENCE]
    ci,paired,samples=cluster_intervals(pd.concat([frame,baseline],ignore_index=True),groups,PILOT_SEEDS,REFERENCE)
    ci.to_csv(OUT/'pilot_confidence_intervals.csv',index=False); paired.to_csv(OUT/'pilot_paired_intervals.csv',index=False)
    np.save(OUT/'pilot_cluster_bootstrap.npy',samples)
    write_json(OUT/'completion_audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        clinical_source_unchanged=True,encoders_frozen=True,unknown_outcome_patients_excluded=54,
        pca_fit_inner_only_then_full_training_refit=True,test_scored=False,goal_achieved=False,
        interpretation='Conditional development evidence; patch attention is not lesion truth'))
    print('Patch-pooling pilot and patient-cluster intervals completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','pilot','bootstrap'])
    globals()[parser.parse_args().stage]()
