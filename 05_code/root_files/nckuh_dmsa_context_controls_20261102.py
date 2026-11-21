"""Own/opposite-only matched controls of selected paired handcrafted features."""
from datetime import datetime,timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

import nckuh_dmsa_label_powerset_20261102 as P
from nckuh_dmsa_cluster_ci import cluster_intervals


def pair_order(predictions):
    output=[]
    for (group,seed),draw in predictions.groupby(['group','seed']):
        correct=[]
        for _,patient in draw.groupby('patient_alias'):
            if len(patient)==2 and patient.label.nunique()==2:
                positive=patient.loc[patient.label.eq(1),'score'].item()
                negative=patient.loc[patient.label.eq(0),'score'].item()
                correct.append(float(positive>negative)+.5*float(positive==negative))
        output.append(dict(group=group,seed=int(seed),unilateral_pairs=len(correct),pair_order_accuracy=float(np.mean(correct))))
    return pd.DataFrame(output)


def main():
    P.require_lock()
    if (P.OUT/'context_control_protocol.json').exists():
        raise FileExistsError('Preserve completed matched context controls')
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='Exploratory follow-up mechanism controls after the paired-input pilot, not independent confirmation',
        question='Does paired handcrafted prediction improve over an otherwise identical own-only head, and does it distinguish the clinically affected side within unilateral patients?',
        heads='Own-only and opposite-only use same train-only preprocessing and balanced liblinear LR C=.01 as the paired binary head',
        cohort='All210 patients414 kidneys retained; same three126/42/42 draws. Only development scoring; unilateral order is auxiliary, not a replacement endpoint',
        selection='No new hyperparameter or seed selection, no promotion decision from these controls, no test scoring')
    P.write_json(P.OUT/'context_control_protocol.json',protocol)
    paths=[Path(__file__),P.OUT/'handcrafted_pairs.npy',P.OUT/'pilot_predictions.csv',P.BASE/'patient_assignments.csv',
        P.AUDIT/'kidney_label_linkage_private.csv',P.SOURCE]
    lock=dict(inputs={str(p):P.digest(p) for p in paths},protocol_sha256=P.digest(P.OUT/'context_control_protocol.json'))
    P.write_json(P.OUT/'context_control_lock.json',lock)
    patients,kidneys,_=P.load_reference()
    values=np.load(P.OUT/'handcrafted_pairs.npy')
    half=values.shape[1]//2
    patient_index={alias:i for i,alias in enumerate(patients.patient_alias)}
    pidx=np.array([patient_index[a] for a in kidneys.patient_alias])
    side=kidneys.side.map({'left':0,'right':1}).to_numpy(int)
    ordered=np.stack([values[i] if s==0 else P.swap_pair(values[i:i+1])[0] for i,s in zip(pidx,side)])
    assignments=pd.read_csv(P.BASE/'patient_assignments.csv')
    target=kidneys.kidney_APN.to_numpy(int)
    metrics,outputs=[],[]
    with threadpool_limits(limits=4):
        for mode,x in [('own_only',ordered[:,:half]),('opposite_only',ordered[:,half:])]:
            for seed in P.PILOT_SEEDS:
                mapping=assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
                train=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('train'))
                dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
                model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),VarianceThreshold(),
                    LogisticRegression(C=.01,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=seed))
                model.fit(x[train],target[train])
                score=model.predict_proba(x[dev])[:,1]
                group='handcrafted_'+mode
                metrics.append(dict(group=group,seed=seed,**P.rank_metrics(target[dev],score)))
                for i,s in zip(dev,score):
                    k=kidneys.iloc[i]
                    outputs.append(dict(group=group,seed=seed,patient_alias=k.patient_alias,kidney_alias=k.kidney_alias,
                        DMSA_APN=k.DMSA_APN,label=int(k.kidney_APN),score=float(s)))
                joblib.dump(model,P.OUT/'models'/f'{group}_{seed}.joblib',compress=3)
    controls=pd.DataFrame(outputs)
    controls.to_csv(P.OUT/'context_control_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(P.OUT/'context_control_metrics.csv',index=False)
    original=pd.read_csv(P.OUT/'pilot_predictions.csv')
    original=original[original.group.eq('handcrafted_binary')]
    predictions=pd.concat([controls,original],ignore_index=True)
    order=pair_order(predictions)
    order.to_csv(P.OUT/'context_unilateral_order.csv',index=False)
    groups=['handcrafted_own_only','handcrafted_opposite_only','handcrafted_binary']
    ci,paired,samples=cluster_intervals(predictions,groups,P.PILOT_SEEDS,'handcrafted_binary')
    ci.to_csv(P.OUT/'context_control_confidence_intervals.csv',index=False)
    paired.to_csv(P.OUT/'context_control_paired_intervals.csv',index=False)
    np.save(P.OUT/'context_control_cluster_bootstrap.npy',samples)
    summary=pd.DataFrame(metrics).groupby('group')[['auroc','ap']].mean()
    original_metrics=pd.read_csv(P.OUT/'pilot_metrics.csv')
    summary=pd.concat([summary,original_metrics[original_metrics.group.eq('handcrafted_binary')].groupby('group')[['auroc','ap']].mean()])
    summary=summary.join(order.groupby('group').pair_order_accuracy.mean())
    summary.to_csv(P.OUT/'context_control_summary.csv')
    for path,sha in lock['inputs'].items():
        assert P.digest(Path(path))==sha
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    main()
