"""Fixed-prediction kidney metrics with patient-cluster random-draw uncertainty."""
import numpy as np
import pandas as pd

from nckuh_only_study_20261102 import weighted_metrics_batch


def cluster_intervals(predictions, groups, seeds, reference, replicates=10000):
    required = {'patient_alias','kidney_alias','DMSA_APN','label','score','seed','group'}
    if not required.issubset(predictions.columns):
        raise ValueError('Missing prediction keys')
    selected = predictions[predictions.group.isin(groups)&predictions.seed.isin(seeds)].copy()
    if selected.duplicated(['seed','group','kidney_alias']).any():
        raise ValueError('Duplicate kidney prediction')
    if not selected.DMSA_APN.isin(['R','L','B','N']).all():
        raise ValueError('Unavailable clinical reference')
    if selected.groupby('patient_alias').DMSA_APN.nunique().max()!=1 or selected.groupby('kidney_alias').label.nunique().max()!=1:
        raise ValueError('Inconsistent reference across predictions')
    aliases = sorted(selected.patient_alias.unique())
    codes = selected.groupby('patient_alias').DMSA_APN.first().reindex(aliases).to_numpy()
    strata = [np.flatnonzero(codes==c) for c in ['R','L','B','N']]
    if not all(len(s) for s in strata):
        raise ValueError('Missing patient-level stratum')
    matrices, point = [], np.zeros((len(groups),3))
    for seed in seeds:
        draw = selected[selected.seed.eq(seed)]
        pivot = draw.pivot(index='kidney_alias',columns='group',values='score').reindex(columns=groups)
        if pivot.isna().any().any():
            raise ValueError('Configurations are not evaluated on matching kidneys')
        key = draw.groupby('kidney_alias').first().reindex(pivot.index)
        y = key.label.to_numpy(int)
        clusters = np.array([aliases.index(alias) for alias in key.patient_alias])
        scores = pivot.to_numpy(float)
        if not np.isfinite(scores).all() or set(y)!={0,1}:
            raise ValueError('Invalid scores/outcome')
        matrices.append((clusters,y,scores))
        for j in range(len(groups)):
            point[j] += weighted_metrics_batch(y,scores[:,j],np.ones((1,len(y))))[0,:3]
    point /= len(seeds)
    rng = np.random.default_rng(20261102)
    batches, total = [], 0
    while total<replicates:
        size = min(1000,replicates-total)
        sample = np.concatenate([rng.choice(s,(size,len(s)),replace=True) for s in strata],axis=1)
        counts = np.zeros((size,len(aliases)))
        np.add.at(counts,(np.arange(size)[:,None],sample),1)
        values = np.zeros((size,len(groups),3))
        keep = np.ones(size,dtype=bool)
        for clusters,y,scores in matrices:
            weights = counts[:,clusters]
            keep &= (weights[:,y==0].sum(1)>0)&(weights[:,y==1].sum(1)>0)
            for j in range(len(groups)):
                values[:,j] += weighted_metrics_batch(y,scores[:,j],weights)[:,:3]
        batches.append(values[keep]/len(seeds))
        total += int(keep.sum())
    samples = np.concatenate(batches)[:replicates]
    rows, differences = [], []
    reference_index = groups.index(reference)
    for j,group in enumerate(groups):
        for k,name in enumerate(['auroc','ap','auprc']):
            low,high = np.quantile(samples[:,j,k],[.025,.975])
            rows.append(dict(group=group,metric=name,mean=float(point[j,k]),low=float(low),high=float(high),
                distinct_patients=len(aliases),draws=len(seeds),replicates=replicates))
            if group!=reference:
                low,high = np.quantile(samples[:,j,k]-samples[:,reference_index,k],[.025,.975])
                differences.append(dict(group=group,reference=reference,metric=name,
                    difference=float(point[j,k]-point[reference_index,k]),low=float(low),high=float(high)))
    return pd.DataFrame(rows),pd.DataFrame(differences),samples
