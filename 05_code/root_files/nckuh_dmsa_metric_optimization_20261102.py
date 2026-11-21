"""Matched warm-start linear objectives using actual LibAUC loss/optimizer APIs."""
import argparse
from datetime import datetime,timezone
import inspect
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from libauc.losses import AUCMLoss,AveragePrecisionLoss
from libauc.optimizers import PESG,SOAP

import nckuh_dmsa_label_powerset_20261102 as P
from nckuh_dmsa_cluster_ci import cluster_intervals

OUT=P.ROOT/'02_experiments/NCKUH_DMSA_metric_optimization_20261102'
MODES=['frozen_LR','BCE_SGD','BCE_Adam','AUCM_PESG','AP_SOAP']
REFERENCE='handcrafted_binary'


def prepare():
    P.require_lock()
    env=json.loads((OUT/'environment_validation.json').read_text(encoding='utf-8'))
    assert env['version']=='2.0.1' and all(c['finite_forward_backward_and_optimizer_steps'] for c in env['checks'])
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Preserve registered objective comparison')
    paths=[Path(__file__),OUT/'environment_validation.json',P.SOURCE,P.BASE/'patient_assignments.csv',
        P.OUT/'handcrafted_pairs.npy',P.OUT/'pilot_predictions.csv',P.OUT/'pilot_metrics.csv',
        P.AUDIT/'patient_label_linkage_private.csv',P.AUDIT/'kidney_label_linkage_private.csv',
        Path(__file__).with_name('nckuh_dmsa_label_powerset_20261102.py'),Path(__file__).with_name('nckuh_dmsa_cluster_ci.py')]
    paths += [Path(inspect.getsourcefile(o)) for o in [AUCMLoss,AveragePrecisionLoss,PESG,SOAP]]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Paired handcrafted input gives promising AP but weaker AUROC. Test metric-aligned optimization with identical representation and train-only LR warm start, not another encoder search',
        references=['https://arxiv.org/abs/2012.03173','https://arxiv.org/abs/2104.08736','https://docs.libauc.org/api/libauc.losses.html','https://docs.libauc.org/api/libauc.optimizers.html'],
        methods=MODES,library='Verified pinned LibAUC2.0.1 core losses/optimizers in .venvs/libauc; not a new named model',
        cohort='All210 code-available patients414 imaged kidneys,54 unknown outcomes excluded. Same three126/42/42 patient-random draws, no k-fold or test scoring',
        input='Unchanged1930 paired handcrafted features (own/opposite summaries and availability), no reference or demographic predictors',
        warm_start='Every phase fits imputer/scaler/variance filter and balanced liblinear LR C=.01 on that phase training kidneys only. Copy its coefficients/intercept into float32 torch Linear; full126 LR probabilities must reproduce the prior paired reference',
        controls='Frozen original LR; balanced BCE with SGD lr.01 momentum.9 or Adam lr.001. Matched optimizer weight_decay1e-5; coordinate gradient clipping1',
        AUCM='Actual libauc AUCMLoss versionv2 margin1 on sigmoid scores; PESG SGD lr.01 momentum.9,weight_decay1e-5,epoch_decay0,clip_value1. Native auxiliary a/b/alpha reset per phase',
        AP='Actual AveragePrecisionLoss gamma.9 margin1 squared_hinge on sigmoid scores; SOAP Adam lr.001,weight_decay1e-5,epoch_decay0,clip_value1. Full training batch contains both classes; AP memory length is phase training kidneys only, using local train indices',
        selection='One100/26 patient inner split inside126 training, R/L/B/N stratified seed+4000. Maximize inner mean(AUROC,AP), ties earliest, include epoch0 LR, max80epochs/minimum stopping10/patience15. No development-selected epoch',
        refit='Fit a fresh warm-start LR/preprocessor on all126 training patients, reset objective/optimizer state and run selected epoch count, then score only42 development patients',
        gate='Expand if gain>=.03AUROC and>=.02AP against the stronger paired LR with>=2/3 joint wins; or both pilot means>.8 with both metrics>.8 in>=2/3 draws. Pilot passing is not goal completion; require unchanged ten-draw and separately locked exploratory-test assessment',
        interpretation='Full method comparison includes native objective/optimizer state; BCE SGD/Adam controls distinguish generic gradient adaptation. Conditional patient-cluster CIs are not selection-adjusted or independent validation. Exact DMSA timing caveat remains')
    P.write_json(OUT/'protocol.json',protocol)
    P.write_json(OUT/'run_lock.json',dict(protocol_sha256=P.digest(OUT/'protocol.json'),inputs={str(path):P.digest(path) for path in paths}))
    print('Registered five matched objective/optimizer methods against the paired-input LR reference',flush=True)


def require_lock():
    lock=json.loads((OUT/'run_lock.json').read_text(encoding='utf-8'))
    assert P.digest(OUT/'protocol.json')==lock['protocol_sha256']
    for path,sha in lock['inputs'].items():
        assert P.digest(Path(path))==sha,path


def warm_start(x,y,indices,seed):
    transform=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),VarianceThreshold())
    ztrain=transform.fit_transform(x[indices])
    lr=LogisticRegression(C=.01,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=seed).fit(ztrain,y[indices])
    z=torch.from_numpy(transform.transform(x).astype(np.float32))
    return z,transform,lr


def new_network(lr):
    model=torch.nn.Linear(lr.coef_.shape[1],1)
    with torch.no_grad():
        model.weight.copy_(torch.from_numpy(lr.coef_.astype(np.float32)))
        model.bias.copy_(torch.from_numpy(lr.intercept_.astype(np.float32)))
    return model


def optimize(z,y,train,inner,lr,mode,epochs=None):
    model=new_network(lr)
    target=y[train].reshape(-1,1)
    if mode=='AUCM_PESG':
        objective=AUCMLoss(version='v2',margin=1.,device='cpu')
        optimizer=PESG(model.parameters(),loss_fn=objective,lr=.01,mode='sgd',momentum=.9,weight_decay=1e-5,epoch_decay=0,clip_value=1.,device='cpu',verbose=False)
    elif mode=='AP_SOAP':
        objective=AveragePrecisionLoss(data_len=len(train),gamma=.9,margin=1.,surr_loss='squared_hinge',device='cpu')
        optimizer=SOAP(model.parameters(),lr=.001,mode='adam',weight_decay=1e-5,epoch_decay=0,clip_value=1.,device='cpu',verbose=False)
    else:
        objective=None
        optimizer=(torch.optim.SGD(model.parameters(),lr=.01,momentum=.9,weight_decay=1e-5) if mode=='BCE_SGD'
                   else torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5))
    before=torch.cat([p.detach().flatten().clone() for p in model.parameters()])
    selected,bad,best=0,0,-float('inf')
    if inner is not None:
        with torch.inference_mode():
            initial=torch.sigmoid(model(z[inner])).numpy().ravel()
        initial_metric=P.rank_metrics(y[inner].numpy().astype(int),initial)
        best=(initial_metric['auroc']+initial_metric['ap'])/2
    history=[]
    local_index=torch.arange(len(train))
    for epoch in range(1,(epochs if epochs is not None else 80)+1):
        optimizer.zero_grad()
        logits=model(z[train])
        if mode in ['BCE_SGD','BCE_Adam']:
            weight=torch.where(target>0,len(target)/(2*target.sum()),len(target)/(2*(len(target)-target.sum())))
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')*weight).mean()
        elif mode=='AUCM_PESG':
            loss=objective(torch.sigmoid(logits),target)
        else:
            loss=objective(torch.sigmoid(logits),target,local_index)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite LibAUC objective')
        loss.backward(); torch.nn.utils.clip_grad_value_(model.parameters(),1.); optimizer.step()
        row=dict(phase='inner' if inner is not None else 'refit',epoch=epoch,train_objective=float(loss.detach()))
        if inner is not None:
            with torch.inference_mode():
                score=torch.sigmoid(model(z[inner])).numpy().ravel()
            m=P.rank_metrics(y[inner].numpy().astype(int),score)
            criterion=(m['auroc']+m['ap'])/2
            row.update(inner_auroc=m['auroc'],inner_ap=m['ap'])
            if criterion>best+1e-12:
                best,selected,bad=criterion,epoch,0
            else:
                bad+=1
        history.append(row)
        if inner is not None and epoch>=10 and bad>=15:
            break
    displacement=float((torch.cat([p.detach().flatten() for p in model.parameters()])-before).square().sum())
    audit=dict(training_kidneys=len(train),iterations=len(history),parameter_squared_displacement=displacement,
        AP_memory_length=len(train) if mode=='AP_SOAP' else None,AP_heldout_indices_used=False)
    return model,selected if inner is not None else epochs,history,audit


def pilot():
    require_lock(); torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    patients,kidneys,_=P.load_reference()
    values=np.load(P.OUT/'handcrafted_pairs.npy')
    patient_index={a:i for i,a in enumerate(patients.patient_alias)}
    x=np.stack([values[patient_index[k.patient_alias]] if k.side=='left' else P.swap_pair(values[patient_index[k.patient_alias]:patient_index[k.patient_alias]+1])[0]
                for k in kidneys.itertuples(index=False)])
    target=kidneys.kidney_APN.to_numpy(int); y=torch.from_numpy(target.astype(np.float32))
    assignments=pd.read_csv(P.BASE/'patient_assignments.csv')
    reference=pd.read_csv(P.OUT/'pilot_predictions.csv')
    reference=reference[reference.group.eq(REFERENCE)]
    outputs,metrics,audits=[],[],[]; (OUT/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in P.PILOT_SEEDS:
            result_path=OUT/f'result_{seed}.json'
            if result_path.exists():
                saved=json.loads(result_path.read_text(encoding='utf-8'))
                metrics.extend(saved['metrics']); outputs.extend(saved['predictions']); audits.extend(saved['audits'])
                continue
            mapping=assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
            outer=patients[patients.patient_alias.map(mapping).eq('train')]
            fit_alias,inner_alias=train_test_split(outer.patient_alias.to_numpy(),train_size=100,random_state=seed+4000,stratify=outer.DMSA_APN)
            fit=np.flatnonzero(kidneys.patient_alias.isin(fit_alias)); inner=np.flatnonzero(kidneys.patient_alias.isin(inner_alias))
            train=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('train')); dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
            zi,ti,li=warm_start(x,target,fit,seed)
            z,transform,lr=warm_start(x,target,train,seed)
            baseline=lr.predict_proba(transform.transform(x[dev]))[:,1]
            expected=reference[reference.seed.eq(seed)].set_index('kidney_alias').score.reindex(kidneys.kidney_alias.iloc[dev]).to_numpy()
            assert np.max(np.abs(baseline-expected))<=1e-10
            joblib.dump(dict(transform=transform,lr=lr,train_aliases=sorted(outer.patient_alias),inner_transform=ti,
                            inner_lr=li,inner_fit_aliases=fit_alias.tolist(),inner_validation_aliases=inner_alias.tolist()),OUT/'models'/f'preprocess_{seed}.joblib',compress=3)
            current_metrics,current_outputs,current_audits=[],[],[]
            for mode in MODES:
                if mode=='frozen_LR':
                    model=new_network(lr); selected=0; history=[]; refit=[]
                    audit=dict(training_kidneys=len(train),iterations=0,parameter_squared_displacement=0.,AP_memory_length=None,AP_heldout_indices_used=False)
                    score=baseline
                else:
                    _,selected,history,inner_audit=optimize(zi,y,fit,inner,li,mode)
                    model,_,refit,audit=optimize(z,y,train,None,lr,mode,selected)
                    audit['inner']=inner_audit
                    with torch.inference_mode():
                        score=torch.sigmoid(model(z[dev])).numpy().ravel()
                    if selected==0:
                        score=baseline
                current_metrics.append(dict(group=mode,seed=seed,role='development',patients=42,kidneys=len(dev),selected_epoch=selected,**P.rank_metrics(target[dev],score)))
                for i,s in zip(dev,score):
                    k=kidneys.iloc[i]
                    current_outputs.append(dict(group=mode,seed=seed,patient_alias=k.patient_alias,kidney_alias=k.kidney_alias,DMSA_APN=k.DMSA_APN,
                        side=k.side,label=int(k.kidney_APN),score=float(s)))
                audit.update(group=mode,seed=seed,selected_epoch=selected,inner_fit_patients=100,inner_validation_patients=26,full_training_patients=126)
                current_audits.append(audit)
                torch.save(dict(model=model.state_dict(),selected_epoch=selected,mode=mode),OUT/'models'/f'{mode}_{seed}.pth')
                pd.DataFrame(history+refit).to_csv(OUT/f'history_{mode}_{seed}.csv',index=False)
            P.write_json(result_path,dict(metrics=current_metrics,predictions=current_outputs,audits=current_audits))
            metrics.extend(current_metrics); outputs.extend(current_outputs); audits.extend(current_audits)
            print(pd.DataFrame(current_metrics)[['group','seed','auroc','ap','selected_epoch']].to_string(index=False),flush=True)
    frame=pd.DataFrame(metrics); frame.to_csv(OUT/'pilot_metrics.csv',index=False)
    pd.DataFrame(outputs).to_csv(OUT/'pilot_predictions.csv',index=False); P.write_json(OUT/'training_audits.json',audits)
    summary=frame.groupby('group')[['auroc','ap','auprc']].mean(); summary['mean_score']=summary[['auroc','ap']].mean(axis=1)
    summary=summary.sort_values(['mean_score','group'],ascending=[False,True]); eligible=[]
    base=frame[frame.group.eq('frozen_LR')].set_index('seed')[['auroc','ap']]
    for group in summary.index:
        delta=frame[frame.group.eq(group)].set_index('seed')[['auroc','ap']]-base
        joint=int(delta.gt(0).all(axis=1).sum())
        summary.loc[group,'delta_auroc']=delta.auroc.mean(); summary.loc[group,'delta_ap']=delta.ap.mean(); summary.loc[group,'joint_wins']=joint
        absolute=(summary.loc[group,['auroc','ap']]>.8).all() and (frame[frame.group.eq(group)][['auroc','ap']]>.8).all(axis=1).sum()>=2
        relative=delta.auroc.mean()>=.03 and delta.ap.mean()>=.02 and joint>=2
        if group!='frozen_LR' and (absolute or relative):
            eligible.append(group)
    summary.to_csv(OUT/'pilot_summary.csv')
    P.write_json(OUT/'pilot_decision.json',dict(selected_for_expansion=eligible[0] if eligible else None,eligible=eligible,test_scored=False,goal_achieved=False))
    print(summary.to_string(),flush=True)


def bootstrap():
    require_lock()
    frame=pd.read_csv(OUT/'pilot_predictions.csv')
    ci,paired,samples=cluster_intervals(frame,MODES,P.PILOT_SEEDS,'frozen_LR')
    ci.to_csv(OUT/'pilot_confidence_intervals.csv',index=False); paired.to_csv(OUT/'pilot_paired_intervals.csv',index=False)
    np.save(OUT/'pilot_cluster_bootstrap.npy',samples)
    P.write_json(OUT/'completion_audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),clinical_source_unchanged=True,
        libauc_core_used=True,supervised_unknown_outcomes=0,fit_and_validation_patient_grouped=True,
        source_LR_reproduced=True,AP_state_contains_training_kidneys_only=True,test_scored=False,goal_achieved=False))
    print('Matched LibAUC pilot and patient-cluster intervals completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','pilot','bootstrap'])
    globals()[parser.parse_args().stage]()
