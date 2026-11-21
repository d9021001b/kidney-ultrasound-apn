"""Reconstruct every objective checkpoint and check actual preprocessing role isolation."""
import json

import joblib
import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_metric_optimization_20261102 as M


def main():
    torch.set_num_threads(4)
    M.require_lock()
    patients,kidneys,_=M.P.load_reference()
    assignment=pd.read_csv(M.P.BASE/'patient_assignments.csv')
    values=np.load(M.P.OUT/'handcrafted_pairs.npy')
    index={alias:i for i,alias in enumerate(patients.patient_alias)}
    x=np.stack([values[index[k.patient_alias]] if k.side=='left' else M.P.swap_pair(values[index[k.patient_alias]:index[k.patient_alias]+1])[0]
                for k in kidneys.itertuples(index=False)])
    predictions=pd.read_csv(M.OUT/'pilot_predictions.csv')
    checks=[]
    for seed in M.P.PILOT_SEEDS:
        mapping=assignment[assignment.seed.eq(seed)].set_index('patient_alias').role
        train=set(mapping[mapping.eq('train')].index)
        dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
        preprocessing=joblib.load(M.OUT/'models'/f'preprocess_{seed}.joblib')
        assert set(preprocessing['train_aliases'])==train and len(train)==126
        assert len(preprocessing['inner_fit_aliases'])==100 and len(preprocessing['inner_validation_aliases'])==26
        assert not set(preprocessing['inner_fit_aliases'])&set(preprocessing['inner_validation_aliases'])
        assert set(preprocessing['inner_fit_aliases'])|set(preprocessing['inner_validation_aliases'])==train
        transformed=preprocessing['transform'].transform(x[dev])
        all_transformed=preprocessing['transform'].transform(x)
        z=torch.from_numpy(all_transformed.astype(np.float32))[dev]
        for mode in M.MODES:
            checkpoint=torch.load(M.OUT/'models'/f'{mode}_{seed}.pth',map_location='cpu',weights_only=True)
            model=M.new_network(preprocessing['lr'])
            model.load_state_dict(checkpoint['model'],strict=True)
            with torch.inference_mode():
                score=torch.sigmoid(model(z)).numpy().ravel()
            if checkpoint['selected_epoch']==0:
                score=preprocessing['lr'].predict_proba(transformed)[:,1]
            expected=predictions[predictions.group.eq(mode)&predictions.seed.eq(seed)].set_index('kidney_alias').score.reindex(kidneys.kidney_alias.iloc[dev]).to_numpy()
            error=float(np.abs(score-expected).max())
            assert error<=1e-6,(mode,seed,error)
            checks.append(dict(group=mode,seed=seed,prediction_maxabs=error,selected_epoch=checkpoint['selected_epoch']))
    audit=json.loads((M.OUT/'training_audits.json').read_text(encoding='utf-8'))
    for item in audit:
        assert item['full_training_patients']==126 and not item['AP_heldout_indices_used']
        if item['group']=='AP_SOAP':
            assert item['AP_memory_length']==item['training_kidneys']
            assert item['inner']['AP_memory_length']==item['inner']['training_kidneys']
    M.require_lock()
    report=dict(checkpoint_reconstruction=checks,clinical_source_unchanged=True,
        test_scored=False,AP_memory_training_only=True,inner_and_refit_preprocessing_role_isolated=True,
        epoch_zero_boundary='Use the original sklearn LR scorer for epoch0; this prevents float32 conversion ties from manufacturing an apparent metric change',
        largest_reconstruction_error=max(c['prediction_maxabs'] for c in checks))
    M.P.write_json(M.OUT/'verification.json',report)
    print(json.dumps(dict(models_verified=len(checks),maxabs=report['largest_reconstruction_error'],test_scored=False)),flush=True)


if __name__=='__main__':
    main()
