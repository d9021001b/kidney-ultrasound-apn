"""Two-kidney label-powerset control with kidney-level marginal scoring."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nckuh_dmsa_kidney_baseline_20261102 import ROOT,AUDIT,FEATURES,OUT as BASE,SOURCE,SEEDS,load_reference,verify_lock
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_dmsa_contralateral_probe_20261102 import rank_metrics
from nckuh_only_study_20261102 import digest,write_json

OUT=ROOT/'02_experiments/NCKUH_DMSA_label_powerset_20261102'
USF=ROOT/'02_experiments/NCKUH_ultrasound_probe_20261102'
PILOT_SEEDS=SEEDS[:3]
CODE={'N':0,'L':1,'R':2,'B':3}
SWAP=np.array([0,2,1,3])
REFERENCE='Summary three source fusion'
FAMILIES=['handcrafted','usf_class']


def swap_pair(values):
    half=values.shape[1]//2
    assert values.shape[1]%2==0
    return np.c_[values[:,half:],values[:,:half]]


def joint_marginals(model,values):
    if set(model.classes_)!={0,1,2,3}:
        raise ValueError('Four DMSA states required')
    order=[list(model.classes_).index(c) for c in range(4)]
    direct=model.predict_proba(values)[:,order]
    swapped=model.predict_proba(swap_pair(values))[:,order][:,SWAP]
    probability=(direct+swapped)/2
    return np.c_[probability[:,1]+probability[:,3],probability[:,2]+probability[:,3]]


def binary_marginals(model,values):
    positive=list(model.classes_).index(1)
    return np.c_[model.predict_proba(values)[:,positive],model.predict_proba(swap_pair(values))[:,positive]]


def patient_pairs(values,images,patients):
    width=values.shape[1]
    pairs=[]
    for alias in patients.patient_alias:
        parts=[]
        for side in ['left','right']:
            index=np.flatnonzero(images.patient_alias.eq(alias)&images.side.eq(side))
            parts.append(np.r_[values[index].mean(0),values[index].std(0),1.] if len(index) else np.r_[np.full(width*2,np.nan),0.])
        pairs.append(np.r_[parts[0],parts[1]])
    return np.asarray(pairs,dtype=np.float32)


def prepare():
    verify_lock(); load_reference()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Preserve registered joint-label control')
    paths=[Path(__file__),SOURCE,BASE/'patient_assignments.csv',BASE/'development_metrics.csv',BASE/'development_predictions.csv',
        AUDIT/'patient_label_linkage_private.csv',AUDIT/'kidney_label_linkage_private.csv',AUDIT/'image_labels_dmsa_private.csv',
        FEATURES/'radiomics_percentile_image.csv',FEATURES/'i3_image.csv',USF/'usf_class_image.npy',USF/'encoder_provenance.json',
        FEATURES/'image_manifest.csv',Path(__file__).with_name('nckuh_dmsa_cluster_ci.py')]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        motivation='Both unilateral and bilateral APN exist. Independent binary heads do not explicitly model the joint two-kidney outcome distribution. Test a standard label-powerset transformation with actual R/L/B/N source codes',
        references=['https://doi.org/10.1109/TKDE.2010.164','https://github.com/scikit-multilearn/scikit-multilearn/blob/master/skmultilearn/problem_transform/lp.py'],
        boundary='Label powerset only, not the RAkEL ensemble. Use standard sklearn multinomial LR with explicit four-state mapping and marginal probability sums. No new model name or oracle opposite-label input',
        cohort='Same210 patients414 imaged kidneys820 images;54 unknown outcomes excluded; all414 kidneys remain the scored endpoint, not a selected B/N or unilateral subset',
        representations=FAMILIES,features='Per-kidney image mean/populationSD plus availability flag; pair concatenates left and right summaries. Missing imaged side remains missing, never a synthetic image',
        fixed_heads='Train-only median imputation/standardization/variance removal, balanced LR C=.01. Binary independent control uses liblinear, joint4-state uses lbfgs multinomial; no parameter search',
        augmentation='Joint training includes original and exchanged left/right feature order. Target L/R codes exchange, N/B unchanged. These are two orders of one patient, not additional patients. Inference averages original and remapped exchanged-order joint probabilities',
        endpoint='Left marginal p(L)+p(B), right marginal p(R)+p(B). Evaluate only the414 actually imaged kidneys. Joint states can include an unimaged opposite side from the source, but no unimaged kidney enters performance',
        sampling='Three fixed126/42/42 patient draws, source-state stratification, no k-fold. All preprocessing fit on training rows only; development42 only, no test scoring',
        gate='Mean AUROC>=.70,AP>=.78, gains>=.03AUROC and>=.02AP vs corrected summary baseline with>=2/3 joint wins. Goal remains both>.8',
        exclusions='No diagnosis/reference fields, age/sex, MRN or source-row fields as image predictors; no new encoders, cross-hospital target weights or post-test choice',
        caveat='Joint output structure is an experimental adaptation, not proof of visible lesions. Conditional patient-cluster CIs do not create external validation; temporal reference caveats remain')
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'run_lock.json',dict(protocol_sha256=digest(OUT/'protocol.json'),inputs={str(p):digest(p) for p in paths}))
    print('Registered paired-input independent versus four-state LR on two existing representations',flush=True)


def require_lock():
    lock=json.loads((OUT/'run_lock.json').read_text(encoding='utf-8'))
    assert digest(OUT/'protocol.json')==lock['protocol_sha256']
    for path,sha in lock['inputs'].items():
        assert digest(Path(path))==sha,path


def pilot():
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import LogisticRegression
    require_lock()
    if (OUT/'pilot_predictions.csv').exists():
        raise FileExistsError('Preserve completed joint-label experiment')
    patients,kidneys,images=load_reference()
    manifest=pd.read_csv(FEATURES/'image_manifest.csv')
    assert images.image_id.tolist()==manifest.image_id.tolist()
    rad=pd.read_csv(FEATURES/'radiomics_percentile_image.csv').set_index('image_id').reindex(images.image_id)
    rad=rad[[c for c in rad if not c.endswith(('_Energy','_TotalEnergy'))]]
    direction=pd.read_csv(FEATURES/'i3_image.csv').set_index('image_id').reindex(images.image_id)
    sources=dict(handcrafted=np.c_[rad.to_numpy(),direction.to_numpy()],usf_class=np.load(USF/'usf_class_image.npy'))
    assignments=pd.read_csv(BASE/'patient_assignments.csv')
    patient_index={alias:i for i,alias in enumerate(patients.patient_alias)}
    kidney_patient=np.array([patient_index[a] for a in kidneys.patient_alias])
    kidney_side=kidneys.side.map({'left':0,'right':1}).to_numpy(int)
    state=patients.DMSA_APN.map(CODE).to_numpy(int)
    labels=kidneys.kidney_APN.to_numpy(int)
    outputs,metrics=[],[]
    (OUT/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for family in FAMILIES:
            values=patient_pairs(sources[family],images,patients)
            np.save(OUT/f'{family}_pairs.npy',values)
            for seed in PILOT_SEEDS:
                mapping=assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
                train=np.flatnonzero(patients.patient_alias.map(mapping).eq('train'))
                train_kidneys=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('train'))
                dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
                assert len(train)==126 and len(set(kidneys.patient_alias.iloc[dev]))==42
                for mode in ['binary','joint']:
                    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),VarianceThreshold(),
                        LogisticRegression(C=.01,solver='liblinear' if mode=='binary' else 'lbfgs',class_weight='balanced',max_iter=3000,random_state=seed))
                    if mode=='joint':
                        x=np.r_[values[train],swap_pair(values[train])]
                        target=np.r_[state[train],SWAP[state[train]]]
                        model.fit(x,target)
                        probabilities=joint_marginals(model,values)
                    else:
                        x=np.stack([values[p] if side==0 else swap_pair(values[p:p+1])[0]
                                    for p,side in zip(kidney_patient[train_kidneys],kidney_side[train_kidneys])])
                        model.fit(x,labels[train_kidneys])
                        probabilities=binary_marginals(model,values)
                    score=probabilities[kidney_patient[dev],kidney_side[dev]]
                    group=family+'_'+mode
                    metrics.append(dict(group=group,seed=seed,role='development',patients=42,kidneys=len(dev),**rank_metrics(labels[dev],score)))
                    for i,s in zip(dev,score):
                        k=kidneys.iloc[i]
                        outputs.append(dict(group=group,seed=seed,patient_alias=k.patient_alias,kidney_alias=k.kidney_alias,
                            DMSA_APN=k.DMSA_APN,side=k.side,label=int(k.kidney_APN),score=float(s)))
                    joblib.dump(model,OUT/'models'/f'{group}_{seed}.joblib',compress=3)
            print(f'Completed paired binary/joint development comparison: {family}',flush=True)
    pd.DataFrame(outputs).to_csv(OUT/'pilot_predictions.csv',index=False)
    frame=pd.DataFrame(metrics); frame.to_csv(OUT/'pilot_metrics.csv',index=False)
    summary=frame.groupby('group')[['auroc','ap','auprc']].mean(); summary['mean_score']=summary[['auroc','ap']].mean(axis=1)
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
        test_scored=False,goal_achieved=False,all_imaged_kidneys_retained=True))
    print(summary.to_string(),flush=True)


def bootstrap():
    require_lock()
    frame=pd.read_csv(OUT/'pilot_predictions.csv')
    baseline=pd.read_csv(BASE/'development_predictions.csv')
    baseline=baseline[baseline.group.eq(REFERENCE)&baseline.seed.isin(PILOT_SEEDS)]
    groups=[f+'_'+m for f in FAMILIES for m in ['binary','joint']]+[REFERENCE]
    ci,paired,samples=cluster_intervals(pd.concat([frame,baseline],ignore_index=True),groups,PILOT_SEEDS,REFERENCE)
    ci.to_csv(OUT/'pilot_confidence_intervals.csv',index=False); paired.to_csv(OUT/'pilot_paired_intervals.csv',index=False)
    np.save(OUT/'pilot_cluster_bootstrap.npy',samples)
    write_json(OUT/'completion_audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        clinical_source_unchanged=True,full_cohort_patients=210,imaged_kidneys=414,unknown_reference_patients_excluded=54,
        test_scored=False,goal_achieved=False,scored_unit='kidney',bootstrap_cluster='patient'))
    print('Joint-label pilot and conditional patient-cluster intervals completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','pilot','bootstrap'])
    globals()[parser.parse_args().stage]()
