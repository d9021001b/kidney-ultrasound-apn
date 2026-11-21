"""Train-only single-stage Saab on native intrarenal patches; corrected kidney endpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from nckuh_dmsa_kidney_baseline_20261102 import (
    ROOT, AUDIT, FEATURES, OUT as BASE, SOURCE, SEEDS, load_reference, verify_lock,
)
from nckuh_dmsa_contralateral_probe_20261102 import classifier, rank_metrics
from nckuh_dmsa_cluster_ci import cluster_intervals
from nckuh_only_study_20261102 import digest, write_json

sys.path.insert(0,str(ROOT/'02_experiments/CYCH_feature_research_results_20260913/source'))
import renal_local_features as L
import green_learning_cv as G

OUT = ROOT/'02_experiments/NCKUH_DMSA_native_saab_20261102'
PILOT_SEEDS = SEEDS[:3]
HEADS = ['top32_lr','full_lr','pca32_svm']
REFERENCE = 'Summary three source fusion'
CONTROL = 'acquisition_geometry_control'


def summarize_kidneys(values, images, kidneys):
    return np.stack([np.r_[values[images.kidney_alias.eq(alias)].mean(0),
                          values[images.kidney_alias.eq(alias)].std(0)] for alias in kidneys.kidney_alias]).astype(np.float32)


def prepare():
    verify_lock()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        raise FileExistsError('Preserve registered native Saab experiment')
    load_reference()
    assert L.KERNELS==(5,9) and L.TRAIN_PATCHES_PER_PATIENT==512 and L.MAX_PATCHES==2048
    paths = [Path(__file__),Path(G.__file__),Path(L.__file__),Path(G.S.__file__),
        Path(__file__).with_name('nckuh_dmsa_cluster_ci.py'),
        Path(__file__).with_name('nckuh_dmsa_contralateral_probe_20261102.py'),
        SOURCE,BASE/'patient_assignments.csv',BASE/'development_metrics.csv',BASE/'development_predictions.csv',
        AUDIT/'image_labels_dmsa_private.csv',AUDIT/'kidney_label_linkage_private.csv']
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        objective='Find APN image representations with kidney AUROC and AP both>.8 under the corrected source reference',
        motivation='Pooled frozen medical/general/ultrasound encoders did not improve both metrics. Fit a small native-pixel local texture basis inside actual kidney masks, without resize blur or source-trained target filters',
        reference='https://arxiv.org/abs/2002.03141',
        method_boundary='Reuse existing single-stage native5x5/9x9 Saab DC-anchored PCA and five response statistics. This is not a full multi-hop PixelHop++ reproduction or newly named architecture',
        reuse=dict(fit_saab_unit=str(G.__file__),patches_and_statistics=str(L.__file__)),
        data='210 patients414 imaged kidneys820 images;54 unknown reference patients excluded. Existing masks and per-image P1-P99 input remain unchanged; one malformed exam date/precise DMSA timing still pending',
        patches='Wholly intrarenal native5x5 and9x9 pixels only; deterministic existing max2048 windows/image/kernel. No interpolation, new mask, inpainting, cortex guess or diagnostic pixel label',
        filter_fit='512 patches per training patient per kernel, balanced across their images;126 training patients=>64512 patches/kernel. Fit all basis parameters on training roles only. Existing bound bias=k for AC, no channel pruning or rectifier',
        descriptor='All25+81 channels times mean,SD,P10,P50,P90=>530 image descriptors; aggregate image mean/populationSD within anatomical kidney=>1060 features',
        control='Native W,H,W/H,masked area fraction,mask-model-acceptance flag; mean/SD within kidney. Diagnostic acquisition control only, never a disease candidate or voting member',
        heads=HEADS,training='Existing fixed train-only median imputation/standardization/variance removal. ANOVA32 balanced LR C=.1; full balanced LR C=.01; whitenedPCA32 RBF-SVM C1 gamma1/32 probabilityFalse. No hidden calibration folds',
        evaluation='Three fixed pilot draws of126train/42development/42exploratory-test patients. Same DMSA-stratified patient roles, all sides/views together, no k-fold. Development only',
        expansion_gate='Mean pilot AUROC>=.70,AP>=.78; delta>=.03AUROC and>=.02AP vs corrected summary fusion, both improve>=2/3 draws, AUROC exceeds acquisition-only control by>=.03. Goal still both>.8; no test scoring before separate lock',
        uncertainty='10000 R/L/B/N-stratified patient-cluster fixed-prediction resamples, retaining both sides and repeated draw appearances; conditional within-cohort, not selection-adjusted',
        exclusions='No CYCH images/labels or fitted filters, no old NCKUH fitted filter bank, no k-fold CLI, no clinical covariates or reference fields as image predictors')
    write_json(OUT/'protocol.json',protocol)
    write_json(OUT/'run_lock.json',dict(protocol_sha256=digest(OUT/'protocol.json'),
        inputs={str(path):digest(path) for path in paths}))
    print('Registered native Saab three-head pilot and separate acquisition control',flush=True)


def require_lock():
    lock = json.loads((OUT/'run_lock.json').read_text(encoding='utf-8'))
    assert digest(OUT/'protocol.json')==lock['protocol_sha256']
    for path,sha in lock['inputs'].items():
        assert digest(Path(path))==sha,path


def make_patches():
    require_lock()
    if (OUT/'patch_audit.json').exists():
        verify_patches()
        print('Existing native patch cache verified; not recreated',flush=True)
        return
    _,_,images = load_reference()
    images = images[images.kidney_APN.notna()].reset_index(drop=True)
    chunks,offsets = {k:[] for k in L.KERNELS},{k:[0] for k in L.KERNELS}
    audit,geometry = [],[]
    for index,row in enumerate(images.itertuples(index=False)):
        gray_path,mask_path = FEATURES/'percentile'/f'{row.image_id}.png', FEATURES/'masks'/f'{row.image_id}.png'
        gray = cv2.imread(str(gray_path),0)
        mask = (cv2.imread(str(mask_path),0)>0).astype(np.uint8)
        assert gray is not None and gray.shape==mask.shape and not gray[mask==0].any()
        geometry.append([gray.shape[1],gray.shape[0],gray.shape[1]/gray.shape[0],float(mask.mean()),int(row.mask_source=='existing_yolo')])
        for k in L.KERNELS:
            patches,coords,total = L.get_patches(gray,mask,k,row.image_id)
            # Check every selected window independently of the extraction routine.
            windows = np.lib.stride_tricks.sliding_window_view(mask,(k,k))
            assert windows[coords[:,0],coords[:,1]].all()
            chunks[k].append(patches)
            offsets[k].append(offsets[k][-1]+len(patches))
            audit.append(dict(image_id=row.image_id,kernel=k,valid_windows=total,selected_windows=len(patches),
                coordinates_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
                image_sha256=digest(gray_path),mask_sha256=digest(mask_path)))
        if (index+1)%200==0:
            print(f'Wholly intrarenal native windows: {index+1}/820',flush=True)
    for k in L.KERNELS:
        np.save(OUT/f'patches_k{k}.npy',np.concatenate(chunks[k]))
        np.save(OUT/f'offsets_k{k}.npy',np.asarray(offsets[k],dtype=np.int64))
    np.save(OUT/'acquisition_geometry_image.npy',np.asarray(geometry,dtype=np.float32))
    pd.DataFrame(audit).to_csv(OUT/'native_patch_audit.csv',index=False)
    write_json(OUT/'patch_audit.json',dict(image_ids=images.image_id.tolist(),images=len(images),
        geometry_shape=[len(images),5],
        cache_sha256={name:digest(OUT/name) for name in ['patches_k5.npy','patches_k9.npy','offsets_k5.npy','offsets_k9.npy','acquisition_geometry_image.npy']},
        all_selected_windows_within_mask=True,learned_parameters_or_diagnosis_used=False))
    print('Native image-only patch cache saved; no target-domain filter fitted yet',flush=True)


def verify_patches():
    proof = json.loads((OUT/'patch_audit.json').read_text(encoding='utf-8'))
    for name,sha in proof['cache_sha256'].items():
        assert digest(OUT/name)==sha,name
    return proof


def pilot():
    require_lock(); patch_proof=verify_patches()
    _,kidneys,images = load_reference()
    images = images[images.kidney_APN.notna()].reset_index(drop=True)
    assert images.image_id.tolist()==patch_proof['image_ids']
    cache={}
    for k in L.KERNELS:
        patches = np.load(OUT/f'patches_k{k}.npy',mmap_mode='r')
        offsets = np.load(OUT/f'offsets_k{k}.npy')
        cache[k]={ident:patches[offsets[i]:offsets[i+1]] for i,ident in enumerate(images.image_id)}
    assignments = pd.read_csv(BASE/'patient_assignments.csv')
    # Only aliases and image keys reach the reusable patient-balanced sampler.
    sampling_manifest=images[['image_id','patient_alias']].rename(columns={'patient_alias':'patient_id'})
    y=kidneys.kidney_APN.to_numpy(int)
    geometry=summarize_kidneys(np.load(OUT/'acquisition_geometry_image.npy'),images,kidneys)
    records,predictions,fit_records=[],[],[]
    (OUT/'models').mkdir(exist_ok=True)
    with threadpool_limits(limits=4):
        for seed in PILOT_SEEDS:
            result_path=OUT/f'result_{seed}.json'
            if result_path.exists():
                result=json.loads(result_path.read_text(encoding='utf-8'))
                records.extend(result['metrics']); predictions.extend(result['predictions']); fit_records.append(result['fit_audit'])
                continue
            mapping=assignments[assignments.seed.eq(seed)].set_index('patient_alias').role
            train_aliases=sorted(mapping[mapping.eq('train')].index)
            roles=kidneys.patient_alias.map(mapping)
            train,dev=np.flatnonzero(roles.eq('train')),np.flatnonzero(roles.eq('development'))
            expected_fit_ids=sorted(images.loc[images.patient_alias.isin(train_aliases),'image_id'])
            units,sampling_audit={},[]
            for k in L.KERNELS:
                samples,items=L.training_patches(cache,sampling_manifest,train_aliases,seed,k)
                assert samples.shape==(126*512,k*k)
                assert {x['patient_id'] for x in items}==set(train_aliases)
                assert sorted({x['image_id'] for x in items})==expected_fit_ids
                units[k]=G.fit_saab_unit(samples)
                for item in items:
                    sampling_audit.append(dict(patient_alias=item['patient_id'],image_id=item['image_id'],kernel=k,
                        n_patches=item['n_patches'],indices_sha256=hashlib.sha256(np.asarray(item['sampled_patch_indices'],dtype=np.int64).tobytes()).hexdigest()))
            unit_state=joblib.hash(units)
            features=[]
            for i,ident in enumerate(images.image_id):
                features.append(np.r_[L.local_statistics(cache[5][ident],units[5],5),L.local_statistics(cache[9][ident],units[9],9)])
                if (i+1)%200==0:
                    print(f'{seed}: native Saab statistics {i+1}/820',flush=True)
            assert joblib.hash(units)==unit_state
            image_features=np.stack(features)
            values=summarize_kidneys(image_features,images,kidneys)
            assert values.shape==(414,1060) and np.isfinite(values).all()
            np.save(OUT/f'saab_{seed}_kidney.npy',values)
            joblib.dump(dict(units=units,biases={5:5,9:9},train_patient_aliases=train_aliases,fit_image_ids=expected_fit_ids),
                OUT/'models'/f'saab_basis_{seed}.joblib',compress=3)
            current_metrics,current_predictions=[],[]
            for head in HEADS+['control_lr']:
                group='native_saab_'+head if head!='control_lr' else CONTROL
                x=values if head!='control_lr' else geometry
                model=classifier(head,seed)
                model.fit(x[train],y[train])
                score=model.decision_function(x[dev]) if head=='pca32_svm' else model.predict_proba(x[dev])[:,1]
                current_metrics.append(dict(group=group,seed=seed,role='development',patients=42,kidneys=len(dev),**rank_metrics(y[dev],score)))
                for i,s in zip(dev,score):
                    k=kidneys.iloc[i]
                    current_predictions.append(dict(group=group,seed=seed,patient_alias=k.patient_alias,kidney_alias=k.kidney_alias,
                        DMSA_APN=k.DMSA_APN,side=k.side,label=int(y[i]),score=float(s)))
                joblib.dump(model,OUT/'models'/f'{group}_{seed}.joblib',compress=3)
            fit_record=dict(seed=seed,training_patients=126,fit_image_ids=expected_fit_ids,
                training_patient_aliases=train_aliases,per_scale_patches=64512,fit_sampling=sampling_audit,
                basis_unchanged_after_transform=True,kidney_feature_shape=list(values.shape))
            result=dict(metrics=current_metrics,predictions=current_predictions,fit_audit=fit_record)
            write_json(result_path,result)
            records.extend(current_metrics); predictions.extend(current_predictions); fit_records.append(fit_record)
            print(pd.DataFrame(current_metrics)[['group','auroc','ap']].to_string(index=False),flush=True)
    frame=pd.DataFrame(records)
    frame.to_csv(OUT/'pilot_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(OUT/'pilot_predictions.csv',index=False)
    write_json(OUT/'fit_audits.json',fit_records)
    summary=frame.groupby('group')[['auroc','ap','auprc']].mean()
    summary['mean_score']=summary[['auroc','ap']].mean(axis=1)
    reference=pd.read_csv(BASE/'development_metrics.csv')
    reference=reference[reference.group.eq(REFERENCE)&reference.seed.isin(PILOT_SEEDS)].set_index('seed')[['auroc','ap']]
    eligible=[]
    summary=summary.sort_values(['mean_score','group'],ascending=[False,True])
    for group in summary.index:
        delta=frame[frame.group.eq(group)].set_index('seed')[['auroc','ap']]-reference
        wins=int(delta.gt(0).all(axis=1).sum())
        summary.loc[group,'delta_auroc']=delta.auroc.mean(); summary.loc[group,'delta_ap']=delta.ap.mean()
        summary.loc[group,'joint_wins']=wins
        if group!=CONTROL and summary.loc[group,'auroc']>=.70 and summary.loc[group,'ap']>=.78 and delta.auroc.mean()>=.03 and delta.ap.mean()>=.02 and wins>=2 and summary.loc[group,'auroc']>=summary.loc[CONTROL,'auroc']+.03:
            eligible.append(group)
    summary.to_csv(OUT/'pilot_summary.csv')
    write_json(OUT/'pilot_decision.json',dict(selected_for_expansion=eligible[0] if eligible else None,
        eligible=eligible,exploratory_test_scored=False,goal_achieved=False,heads=len(HEADS),controls=1))
    print(summary.to_string(),flush=True)


def bootstrap():
    require_lock()
    frame=pd.read_csv(OUT/'pilot_predictions.csv')
    baseline=pd.read_csv(BASE/'development_predictions.csv')
    baseline=baseline[baseline.group.eq(REFERENCE)&baseline.seed.isin(PILOT_SEEDS)]
    groups=['native_saab_'+head for head in HEADS]+[CONTROL,REFERENCE]
    intervals,differences,samples=cluster_intervals(pd.concat([frame,baseline],ignore_index=True),groups,PILOT_SEEDS,REFERENCE)
    intervals.to_csv(OUT/'pilot_confidence_intervals.csv',index=False)
    differences.to_csv(OUT/'pilot_paired_intervals.csv',index=False)
    np.save(OUT/'pilot_cluster_bootstrap.npy',samples)
    write_json(OUT/'completion_audit.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),
        clinical_source_unchanged=True,unknown_outcome_patients_excluded=54,scored_role='development only',
        patient_grouped=True,no_fitted_target_bank_reused=True,feature_fit_patients_per_draw=126,
        conditional_cluster_replicates=10000,goal_achieved=False,
        note='A successful run or CI upper bound>.8 is not target completion'))
    print('Native Saab pilot and patient-cluster uncertainty completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','make_patches','pilot','bootstrap'])
    globals()[parser.parse_args().stage]()
