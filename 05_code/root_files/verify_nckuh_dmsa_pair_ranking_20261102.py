"""Patient-cluster uncertainty and checkpoint reconstruction for ranking pilot."""
import copy
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_pair_ranking_20261102 as P
from nckuh_only_study_20261102 import digest, weighted_metrics_batch, write_json


def main():
    P.configure(); P.require_lock()
    patients, kidneys, images = P.load_reference()
    predictions = pd.read_csv(P.OUT/'pilot_predictions.csv')
    metrics = pd.read_csv(P.OUT/'pilot_metrics.csv')
    baseline = pd.read_csv(P.BASE/'development_predictions.csv')
    baseline = baseline[baseline.group.eq(P.REFERENCE)&baseline.seed.isin(P.PILOT_SEEDS)]
    frame = pd.concat([predictions, baseline], ignore_index=True)
    groups = [c[0] for c in P.CONFIGS]+[P.REFERENCE]
    aliases = sorted(predictions.patient_alias.unique())
    codes = predictions.groupby('patient_alias').DMSA_APN.first().reindex(aliases).to_numpy()
    strata = [np.flatnonzero(codes==code) for code in ['R','L','B','N']]
    matrices, points = [], np.zeros((len(groups),5))
    for seed in P.PILOT_SEEDS:
        draw = frame[frame.seed.eq(seed)]
        pivot = draw.pivot(index='kidney_alias', columns='group', values='score').reindex(columns=groups)
        key = draw.groupby('kidney_alias').first().reindex(pivot.index)
        assert pivot.notna().all().all() and (key.groupby('patient_alias').DMSA_APN.nunique()==1).all()
        y = key.label.to_numpy(int)
        clusters = np.array([aliases.index(alias) for alias in key.patient_alias])
        pos, neg = P.pairs_within_patient(key.patient_alias.to_numpy(),y)
        scores = pivot.to_numpy()
        pair_order = (scores[pos]>scores[neg]).astype(float)+.5*(scores[pos]==scores[neg])
        for j in range(len(groups)):
            points[j,:4] += weighted_metrics_batch(y,scores[:,j],np.ones((1,len(y))))[0]
        points[:,4] += pair_order.mean(0)
        matrices.append((clusters,y,scores,clusters[pos],pair_order))
    points /= len(P.PILOT_SEEDS)
    rng = np.random.default_rng(20261102)
    batches, total = [], 0
    while total<10000:
        size = min(1000,10000-total)
        sample = np.concatenate([rng.choice(s,(size,len(s)),replace=True) for s in strata],axis=1)
        counts = np.zeros((size,len(aliases)))
        np.add.at(counts,(np.arange(size)[:,None],sample),1)
        values = np.zeros((size,len(groups),5))
        keep = np.ones(size,dtype=bool)
        for clusters,y,scores,pair_clusters,pair_order in matrices:
            weights = counts[:,clusters]
            pair_weights = counts[:,pair_clusters]
            keep &= (weights[:,y==0].sum(1)>0)&(weights[:,y==1].sum(1)>0)&(pair_weights.sum(1)>0)
            for j in range(len(groups)):
                values[:,j,:4] += weighted_metrics_batch(y,scores[:,j],weights)
            values[:,:,4] += (pair_weights@pair_order)/pair_weights.sum(1,keepdims=True)
        batches.append(values[keep]/len(P.PILOT_SEEDS))
        total += int(keep.sum())
    samples = np.concatenate(batches)[:10000]
    np.save(P.OUT/'pilot_patient_cluster_bootstrap.npy',samples)
    names = ['auroc','ap','auprc','brier','pair_order_accuracy']
    rows, contrasts = [], []
    for j,group in enumerate(groups):
        for k,metric in enumerate(names):
            low,high = np.quantile(samples[:,j,k],[.025,.975])
            rows.append(dict(group=group,metric=metric,mean=float(points[j,k]),low=float(low),high=float(high),
                             distinct_patients=len(aliases),draws=3,replicates=10000))
    for treatment,control in [('frozen_BCE_rank','frozen_BCE'),
                               ('adapted_L2SP_BCE_rank','adapted_L2SP_BCE'),
                               ('adapted_L2SP_BCE_rank','frozen_BCE_rank'),
                               ('frozen_BCE_rank',P.REFERENCE)]:
        a,b = groups.index(treatment),groups.index(control)
        for k,metric in enumerate(names):
            low,high = np.quantile(samples[:,a,k]-samples[:,b,k],[.025,.975])
            contrasts.append(dict(treatment=treatment,control=control,metric=metric,
                difference=float(points[a,k]-points[b,k]),low=float(low),high=float(high)))
    pd.DataFrame(rows).to_csv(P.OUT/'pilot_confidence_intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(P.OUT/'pilot_paired_intervals.csv',index=False)
    for j,group in enumerate(groups[:-1]):
        observed = metrics[metrics.group.eq(group)]
        np.testing.assert_allclose(points[j,:4],observed[names[:4]].mean().to_numpy(),atol=1e-12)
        np.testing.assert_allclose(points[j,4],observed.pair_order_accuracy.mean(),atol=1e-12)

    model = P.radimagenet_densenet121().cuda().eval().requires_grad_(False)
    _,original_tail = P.T.split_encoder(model)
    maps = torch.from_numpy(np.load(P.CACHE/'prefix_maps.npy')).cuda()
    dense = torch.from_numpy(np.load(P.FEATURES/'dense_tight_image.npy')).cuda()
    assignment = pd.read_csv(P.BASE/'patient_assignments.csv')
    checks = []
    for group in ['frozen_BCE_rank','adapted_L2SP_BCE_rank']:
        seed = P.PILOT_SEEDS[0]
        checkpoint = torch.load(P.OUT/'models'/f'{group}_{seed}.pth',map_location='cuda',weights_only=True)
        head = P.T.Projection(1024).cuda().eval()
        head.load_state_dict(checkpoint['head'],strict=True)
        tail = copy.deepcopy(original_tail).cuda().eval()
        tail.load_state_dict(checkpoint['tail'],strict=True)
        mapping = assignment[assignment.seed.eq(seed)].set_index('patient_alias').role
        aliases_dev = patients.loc[patients.patient_alias.map(mapping).eq('development'),'patient_alias'].to_numpy()
        data = P.phase_data(images,kidneys,aliases_dev,maps.device)
        with torch.inference_mode():
            score = torch.sigmoid(P.logits_for(head,tail,maps,data,None if checkpoint['adapted'] else dense)).cpu().numpy()
        expected = predictions[predictions.group.eq(group)&predictions.seed.eq(seed)].set_index('kidney_alias').score.reindex(data['kidneys'].kidney_alias).to_numpy()
        error = float(np.max(np.abs(score-expected)))
        assert error<=1e-6
        checks.append(dict(group=group,seed=seed,probability_maxabs=error,weights_only=True))
    P.require_lock()
    proof = dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        previous_goal_turn_classification='progress: corrected labels/cohort/unit and baseline completed',
        current_turn_classification='progress: sixteen view/contralateral probes and four paired-ranking pilot arms completed',
        checkpoint_reconstruction=checks, clinical_source_unchanged=True,
        unique_pilot_development_patients=len(aliases),source_outcome_missing_patients_excluded=54,
        test_scored=False, new_diagnoses_invented=False,bootstrap_cluster='patient stratified R/L/B/N',
        uncertainty='Conditional fixed-prediction within-cohort intervals, no refitting or selection adjustment',
        outputs={name:digest(P.OUT/name) for name in ['pilot_metrics.csv','pilot_predictions.csv','pilot_confidence_intervals.csv','pilot_paired_intervals.csv']})
    write_json(P.OUT/'verification.json',proof)
    print(pd.DataFrame(rows)[lambda f:f.group.eq('frozen_BCE_rank')&f.metric.isin(['auroc','ap','pair_order_accuracy'])].to_string(index=False),flush=True)
    print(json.dumps(dict(reconstruction=checks,test_scored=False)),flush=True)


if __name__=='__main__':
    main()
