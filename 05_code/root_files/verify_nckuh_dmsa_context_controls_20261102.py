"""Patient-paired laterality uncertainty and training-only coefficient diagnostics."""
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

import nckuh_dmsa_label_powerset_20261102 as P


def main():
    P.require_lock()
    lock=json.loads((P.OUT/'context_control_lock.json').read_text(encoding='utf-8'))
    for path,sha in lock['inputs'].items():
        assert P.digest(Path(path))==sha
    controls=pd.read_csv(P.OUT/'context_control_predictions.csv')
    original=pd.read_csv(P.OUT/'pilot_predictions.csv')
    original=original[original.group.eq('handcrafted_binary')]
    predictions=pd.concat([controls,original],ignore_index=True)
    names=['handcrafted_own_only','handcrafted_opposite_only','handcrafted_binary']
    aliases=sorted(predictions.patient_alias.unique())
    codes=predictions.groupby('patient_alias').DMSA_APN.first().reindex(aliases).to_numpy()
    strata=[np.flatnonzero(codes==c) for c in ['R','L','B','N']]
    matrices=[]; paired_patients=set()
    for seed in P.PILOT_SEEDS:
        draw=predictions[predictions.seed.eq(seed)]
        values=[]; clusters=[]
        for alias,patient in draw.groupby('patient_alias'):
            reference=patient[patient.group.eq(names[0])]
            if len(reference)!=2 or reference.label.nunique()!=2:
                continue
            scores=patient.pivot(index='label',columns='group',values='score').reindex(columns=names)
            values.append((scores.loc[1].to_numpy()>scores.loc[0].to_numpy()).astype(float)+.5*(scores.loc[1].to_numpy()==scores.loc[0].to_numpy()))
            clusters.append(aliases.index(alias)); paired_patients.add(alias)
        matrices.append((np.array(clusters),np.stack(values)))
    point=np.mean([value.mean(0) for _,value in matrices],axis=0)
    rng=np.random.default_rng(20261102)
    batches=[]; total=0
    while total<10000:
        size=min(1000,10000-total)
        sample=np.concatenate([rng.choice(s,(size,len(s)),replace=True) for s in strata],axis=1)
        count=np.zeros((size,len(aliases)))
        np.add.at(count,(np.arange(size)[:,None],sample),1)
        values=np.zeros((size,len(names))); keep=np.ones(size,dtype=bool)
        for clusters,correct in matrices:
            weights=count[:,clusters]; keep &= weights.sum(1)>0
            values += weights@correct/weights.sum(1,keepdims=True)
        batches.append(values[keep]/len(matrices)); total+=int(keep.sum())
    samples=np.concatenate(batches)[:10000]
    rows=[]
    for i,name in enumerate(names):
        low,high=np.quantile(samples[:,i],[.025,.975])
        rows.append(dict(group=name,mean=float(point[i]),low=float(low),high=float(high),distinct_paired_patients=len(paired_patients),replicates=10000))
    pd.DataFrame(rows).to_csv(P.OUT/'context_unilateral_order_confidence_intervals.csv',index=False)
    contrasts=[]
    for control in names[:2]:
        i=names.index(control)
        low,high=np.quantile(samples[:,2]-samples[:,i],[.025,.975])
        contrasts.append(dict(treatment=names[2],control=control,difference=float(point[2]-point[i]),low=float(low),high=float(high)))
    pd.DataFrame(contrasts).to_csv(P.OUT/'context_unilateral_order_paired_intervals.csv',index=False)

    # Coefficient summaries are explanations only, never a cross-draw feature selector.
    patients,kidneys,images=P.load_reference()
    rad=pd.read_csv(P.FEATURES/'radiomics_percentile_image.csv',nrows=0).columns.tolist()[1:]
    rad=[name for name in rad if not name.endswith(('_Energy','_TotalEnergy'))]
    direction=pd.read_csv(P.FEATURES/'i3_image.csv',nrows=0).columns.tolist()[1:]
    base=['radiomics__'+name for name in rad]+['directional__'+name for name in direction]
    feature_names=['mean_'+name for name in base]+['sd_'+name for name in base]+['image_available']
    pair_values=np.load(P.OUT/'handcrafted_pairs.npy')
    assignments=pd.read_csv(P.BASE/'patient_assignments.csv')
    coefficients=[]
    for seed in P.PILOT_SEEDS:
        mapping=assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
        train=patients.patient_alias.map(mapping).eq('train').to_numpy()
        complete=train & (pair_values[:,len(feature_names)-1]==1) & (pair_values[:,-1]==1)
        left,right=pair_values[complete,:len(feature_names)],pair_values[complete,len(feature_names):]
        model=joblib.load(P.OUT/'models'/f'handcrafted_binary_{seed}.joblib')
        selected=model.named_steps['variancethreshold'].get_support()
        coefficient=np.zeros(pair_values.shape[1])
        coefficient[selected]=model.named_steps['logisticregression'].coef_[0]
        raw=coefficient/model.named_steps['standardscaler'].scale_
        own,other=raw[:len(feature_names)],raw[len(feature_names):]
        differential=(own-other)*np.nanstd(left-right,axis=0)
        shared=(own+other)*np.nanstd((left+right)/2,axis=0)
        for name,d,s in zip(feature_names,differential,shared):
            coefficients.append(dict(seed=seed,feature=name,training_standardized_side_contrast=float(d),training_standardized_shared_context=float(s)))
    coefficient_frame=pd.DataFrame(coefficients)
    coefficient_frame.to_csv(P.OUT/'training_coefficient_diagnostics.csv',index=False)
    summary=coefficient_frame.groupby('feature').agg(
        mean_side_contrast=('training_standardized_side_contrast','mean'),
        mean_abs_side_contrast=('training_standardized_side_contrast',lambda x:float(np.abs(x).mean())),
        mean_shared_context=('training_standardized_shared_context','mean'))
    summary.sort_values('mean_abs_side_contrast',ascending=False).to_csv(P.OUT/'training_coefficient_summary.csv')
    report=dict(clinical_source_unchanged=True,test_scored=False,
        pair_order_ci=rows,coefficient_features=len(feature_names),
        coefficient_boundary='Correlated training-only model coefficients, not causal biomarkers or held-out importance. Never pool this summary to select features for a split whose held-out patients appear in another draw training role.')
    P.write_json(P.OUT/'context_control_verification.json',report)
    print(json.dumps(dict(pair_order_ci=rows,top_training_side_contrast_features=summary.sort_values('mean_abs_side_contrast',ascending=False).head(3).index.tolist())),flush=True)


if __name__=='__main__':
    main()
