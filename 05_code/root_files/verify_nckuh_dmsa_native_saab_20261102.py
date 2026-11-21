"""Reconstruct native descriptors/heads and inspect label-stratified training ROIs."""
import json

import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import nckuh_dmsa_native_saab_20261102 as S


def main():
    S.require_lock(); S.verify_patches()
    patients,kidneys,images=S.load_reference()
    images=images[images.kidney_APN.notna()].reset_index(drop=True)
    assignment=pd.read_csv(S.BASE/'patient_assignments.csv')
    seed=S.PILOT_SEEDS[0]
    mapping=assignment[assignment.seed.eq(seed)].set_index('patient_alias').role
    train_aliases=set(mapping[mapping.eq('train')].index)
    train_ids=set(images.loc[images.patient_alias.isin(train_aliases),'image_id'])
    basis=joblib.load(S.OUT/'models'/f'saab_basis_{seed}.joblib')
    assert set(basis['train_patient_aliases'])==train_aliases and set(basis['fit_image_ids'])==train_ids
    dev=np.flatnonzero(kidneys.patient_alias.map(mapping).eq('development'))
    index=dev[0]
    alias=kidneys.kidney_alias.iloc[index]
    subset=images[images.kidney_alias.eq(alias)]
    image_features=[]
    for row in subset.itertuples(index=False):
        gray=cv2.imread(str(S.FEATURES/'percentile'/f'{row.image_id}.png'),0)
        mask=(cv2.imread(str(S.FEATURES/'masks'/f'{row.image_id}.png'),0)>0).astype(np.uint8)
        vector=[]
        for k in [5,9]:
            patches,_,_=S.L.get_patches(gray,mask,k,row.image_id)
            vector.extend(S.L.local_statistics(patches,basis['units'][k],k))
        image_features.append(vector)
    actual=np.r_[np.mean(image_features,axis=0),np.std(image_features,axis=0)].astype(np.float32)
    saved=np.load(S.OUT/f'saab_{seed}_kidney.npy')
    feature_error=float(np.abs(actual-saved[index]).max())
    assert feature_error<=1e-6
    predictions=pd.read_csv(S.OUT/'pilot_predictions.csv')
    errors={}
    for head in S.HEADS:
        group='native_saab_'+head
        model=joblib.load(S.OUT/'models'/f'{group}_{seed}.joblib')
        actual=model.decision_function(saved[dev]) if head=='pca32_svm' else model.predict_proba(saved[dev])[:,1]
        expected=predictions[predictions.group.eq(group)&predictions.seed.eq(seed)].set_index('kidney_alias').score.reindex(kidneys.kidney_alias.iloc[dev]).to_numpy()
        errors[group]=float(np.max(np.abs(actual-expected)))
        assert errors[group]<=1e-6
    selected=[]
    for code in ['R','L','B','N']:
        for patient in patients[patients.DMSA_APN.eq(code)&patients.patient_alias.isin(train_aliases)].sort_values('patient_alias').itertuples(index=False):
            rows=images[images.patient_alias.eq(patient.patient_alias)]
            if set(rows['view'])=={'LL','LT','RL','RT'}:
                selected.append(rows.set_index('view').loc[['LL','LT','RL','RT']].reset_index())
                break
    assert len(selected)==4
    qa=S.OUT/'qa'; qa.mkdir(exist_ok=True)
    fig,axes=plt.subplots(4,4,figsize=(12,10))
    for row_index,subset in enumerate(selected):
        for column,row in enumerate(subset.itertuples(index=False)):
            gray=cv2.imread(str(S.FEATURES/'percentile'/f'{row.image_id}.png'),0)
            mask=cv2.imread(str(S.FEATURES/'masks'/f'{row.image_id}.png'),0)>0
            axis=axes[row_index,column]
            axis.imshow(gray,cmap='gray',vmin=0,vmax=255)
            axis.contour(mask,levels=[.5],colors=['#00B8D9'],linewidths=.6)
            axis.set_title(f'{row.patient_alias} {row.view} APN={int(row.kidney_APN)} / {row.DMSA_APN}',fontsize=9)
            axis.axis('off')
    fig.suptitle('Training-only clinical-code-stratified ROI review',fontsize=13)
    fig.tight_layout(rect=[0,0,1,.97]); fig.savefig(qa/'training_roi_review.png',dpi=150); plt.close(fig)
    pd.concat(selected).drop(columns=['patient_id','filename','key'],errors='ignore').to_csv(qa/'training_roi_review_selection.csv',index=False)
    S.require_lock()
    report=dict(feature_reconstruction_maxabs=feature_error,head_reconstruction_maxabs=errors,
        training_patients=len(train_aliases),basis_fit_source_images=len(train_ids),
        training_review_patients=4,training_review_images=16,selection='First complete four-view training patient in each clinical R/L/B/N category; no outcome score selection',
        image_review_boundary='Source-supported labels shown for QC, not a naked-eye APN or scar diagnosis',
        clinical_source_unchanged=True,test_scored=False)
    S.write_json(S.OUT/'verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['selection','image_review_boundary']}),flush=True)


if __name__=='__main__':
    main()
