"""Reconstruct paired-input predictions and record matched mechanism intervals."""
import json

import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_label_powerset_20261102 as P


def paired_mechanism(folder,groups,treatments):
    samples=np.load(folder/'pilot_cluster_bootstrap.npy')
    points=pd.read_csv(folder/'pilot_confidence_intervals.csv')
    rows=[]
    for treatment,control in treatments:
        a,b=groups.index(treatment),groups.index(control)
        for i,metric in enumerate(['auroc','ap','auprc']):
            low,high=np.quantile(samples[:,a,i]-samples[:,b,i],[.025,.975])
            t=points[points.group.eq(treatment)&points.metric.eq(metric)]['mean'].item()
            c=points[points.group.eq(control)&points.metric.eq(metric)]['mean'].item()
            rows.append(dict(treatment=treatment,control=control,metric=metric,difference=t-c,low=float(low),high=float(high)))
    pd.DataFrame(rows).to_csv(folder/'paired_mechanism_intervals.csv',index=False)
    return rows


def main():
    P.require_lock()
    patients,kidneys,images=P.load_reference()
    assignment=pd.read_csv(P.BASE/'patient_assignments.csv')
    seed=P.PILOT_SEEDS[0]
    mapping=assignment[assignment.seed.eq(seed)].set_index('patient_alias').role
    dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
    patient_index={a:i for i,a in enumerate(patients.patient_alias)}
    pidx=np.array([patient_index[a] for a in kidneys.patient_alias.iloc[dev]])
    side=kidneys.side.iloc[dev].map({'left':0,'right':1}).to_numpy(int)
    predictions=pd.read_csv(P.OUT/'pilot_predictions.csv')
    checks=[]
    for family in P.FAMILIES:
        values=np.load(P.OUT/f'{family}_pairs.npy')
        for mode in ['binary','joint']:
            group=family+'_'+mode
            model=joblib.load(P.OUT/'models'/f'{group}_{seed}.joblib')
            marginal=P.binary_marginals(model,values) if mode=='binary' else P.joint_marginals(model,values)
            flipped=P.binary_marginals(model,P.swap_pair(values)) if mode=='binary' else P.joint_marginals(model,P.swap_pair(values))
            equivariance=float(np.max(np.abs(marginal-flipped[:,::-1])))
            assert equivariance<=1e-12
            expected=predictions[predictions.group.eq(group)&predictions.seed.eq(seed)].set_index('kidney_alias').score.reindex(kidneys.kidney_alias.iloc[dev]).to_numpy()
            error=float(np.max(np.abs(marginal[pidx,side]-expected)))
            assert error<=1e-10
            checks.append(dict(group=group,seed=seed,reconstruction_maxabs=error,side_exchange_maxabs=equivariance))
    paired_mechanism(P.OUT,[f+'_'+m for f in P.FAMILIES for m in ['binary','joint']]+[P.REFERENCE],
        [(f+'_joint',f+'_binary') for f in P.FAMILIES])
    mil=P.ROOT/'02_experiments/NCKUH_DMSA_patch_mil_20261102'
    paired_mechanism(mil,['dinov3_mean','dinov3_gated','usf_mae_mean','usf_mae_gated',P.REFERENCE],
        [('dinov3_gated','dinov3_mean'),('usf_mae_gated','usf_mae_mean')])
    P.require_lock()
    P.write_json(P.OUT/'verification.json',dict(reconstruction=checks,clinical_source_unchanged=True,
        no_unimaged_kidney_scored=True,unknown_reference_patients_excluded=54,test_scored=False,
        notes='Joint labels are targets only; output marginals tested under left/right exchange'))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':
    main()
