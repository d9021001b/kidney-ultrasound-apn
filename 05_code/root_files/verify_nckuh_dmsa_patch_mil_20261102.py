"""Check PCA role isolation and independently reconstruct held-out patch MIL scores."""
import json

import joblib
import numpy as np
import pandas as pd
import torch

import nckuh_dmsa_patch_mil_20261102 as M


def main():
    M.require_lock()
    patients,kidneys,images=M.load_reference()
    assignments=pd.read_csv(M.BASE/'patient_assignments.csv')
    predictions=pd.read_csv(M.OUT/'pilot_predictions.csv')
    audits=[]
    seed=M.PILOT_SEEDS[0]
    mapping=assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
    train_aliases=set(mapping[mapping.eq('train')].index)
    dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
    for encoder,mode in [('usf_mae','gated'),('dinov3','mean')]:
        values,patch_aliases,groups,global_rows=M.local_patch_data(encoder,images,kidneys)
        proof=joblib.load(M.OUT/'models'/f'{encoder}_{seed}_preprocess.joblib')
        np.testing.assert_equal(proof['source_patch_rows'],global_rows)
        assert set(proof['full']['train_aliases'])==train_aliases
        assert len(proof['inner']['train_aliases'])==100
        assert set(proof['inner']['train_aliases']).issubset(train_aliases)
        for phase in ['inner','full']:
            sampled=proof[phase]['selected_local_patch_indices']
            assert set(patch_aliases[sampled])==set(proof[phase]['train_aliases'])
            assert len(sampled)==64*len(proof[phase]['train_aliases'])
        transformed=proof['full']['pca'].transform(values).astype(np.float32)
        width=max(map(len,groups))
        bags=np.zeros((len(dev),width,32),dtype=np.float32)
        mask=np.zeros((len(dev),width),dtype=bool)
        for j,i in enumerate(dev):
            bags[j,:len(groups[i])]=transformed[groups[i]]; mask[j,:len(groups[i])]=True
        checkpoint=torch.load(M.OUT/'models'/f'{encoder}_{mode}_{seed}.pth',map_location='cpu',weights_only=True)
        model=M.PatchBagModel(mode).eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        with torch.inference_mode():
            logits,attention=model(torch.from_numpy(bags),torch.from_numpy(mask))
            score=torch.sigmoid(logits).numpy()
        expected=predictions[predictions.group.eq(encoder+'_'+mode)&predictions.seed.eq(seed)].set_index('kidney_alias').score.reindex(kidneys.kidney_alias.iloc[dev]).to_numpy()
        error=float(np.abs(score-expected).max())
        assert error<=1e-5
        assert torch.all(attention[~torch.from_numpy(mask)]==0)
        torch.testing.assert_close(attention.sum(1),torch.ones(len(dev)))
        audits.append(dict(encoder=encoder,mode=mode,seed=seed,cpu_checkpoint_vs_gpu_saved_maxabs=error,
            inner_pca_patients=100,full_pca_patients=126,weights_only=True,padding_attention_zero=True))
    M.require_lock()
    M.write_json(M.OUT/'verification.json',dict(reconstruction=audits,
        clinical_source_unchanged=True,source_patch_cache_unchanged=True,test_scored=False,
        note='PCA isolation checked on actual selected patch indices, not only source metadata'))
    print(json.dumps(audits),flush=True)


if __name__=='__main__':
    main()
