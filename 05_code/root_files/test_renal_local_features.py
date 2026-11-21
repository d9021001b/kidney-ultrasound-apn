"""Native ROI support, patient isolation, calibration and result reproduction."""
import json
import unittest

import cv2
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score

import renal_local_features as L


class LocalFeatureTests(unittest.TestCase):
    def test_native_windows_are_inside_roi_and_ignore_outside(self):
        rng=np.random.default_rng(29)
        image=rng.integers(0,256,(100,130),dtype=np.uint8)
        mask=np.zeros(image.shape,np.uint8)
        cv2.ellipse(mask,(64,50),(48,36),13,0,360,1,-1)
        altered=image.copy()
        altered[mask==0]=255-altered[mask==0]
        for k in L.KERNELS:
            a,coords,n=L.get_patches(image,mask,k,"synthetic")
            b,other,_=L.get_patches(altered,mask,k,"synthetic")
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(coords,other)
            self.assertEqual(len(coords),min(n,L.MAX_PATCHES))
            self.assertEqual(len(np.unique(coords,axis=0)),len(coords))
            for j,(y,x) in enumerate(coords):
                self.assertTrue(np.all(mask[y:y+k,x:x+k]))
                np.testing.assert_array_equal(a[j],image[y:y+k,x:x+k].reshape(-1).astype(np.float32)/255.)
            with self.assertRaises(ValueError):
                L.get_patches(image,np.zeros_like(mask),k,"empty")

    def test_training_patch_selection_is_balanced_and_isolated(self):
        from green_learning_cv import fit_saab_unit
        rng=np.random.default_rng(18)
        df=pd.DataFrame(dict(patient_id=["a","a","b","c"],image_id=["i1","i2","i3","i4"]))
        cache={5:{i:rng.random((700,25),dtype=np.float32) for i in df.image_id}}
        a,audit=L.training_patches(cache,df,["a","b"],9,5)
        self.assertEqual(a.shape,(1024,25))
        counts=pd.DataFrame(audit).groupby("patient_id").n_patches.sum().to_dict()
        self.assertEqual(counts,dict(a=512,b=512))
        cache[5]["i4"]*=500
        b,second=L.training_patches(cache,df,["a","b"],9,5)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(audit,second)
        self.assertEqual(joblib.hash(fit_saab_unit(a)),joblib.hash(fit_saab_unit(b)))

    def test_positive_temperature_cannot_flip_rank_or_threshold(self):
        p=np.array([.01,.09,.2,.4,.5,.6,.81,.95,.99])
        y=np.array([1,1,1,1,0,0,0,0,0])
        c=L.fit_temperature(p,y)
        self.assertGreaterEqual(c["temperature"],.5)
        self.assertLessEqual(c["temperature"],5)
        self.assertLessEqual(c["objective"],c["identity_objective"]+1e-12)
        for t in (.5,1.,5.,c["temperature"]):
            q=L.temperature(p,t)
            self.assertTrue(np.all(np.diff(q)>0))
            np.testing.assert_array_equal(p>.5,q>.5)
            self.assertEqual(roc_auc_score(y,p),roc_auc_score(y,q))
            self.assertEqual(average_precision_score(y,p),average_precision_score(y,q))
        for t in (0,-1,np.nan):
            with self.assertRaises(ValueError):
                L.temperature(p,t)

    def test_locked_inputs_and_every_patch_support(self):
        L.require_lock()
        df=L.read("image_manifest.csv")
        self.assertEqual(len(df),349)
        self.assertEqual(df.patient_id.nunique(),98)
        for name in ("image_manifest.csv","patients.csv","random_patient_assignments.csv"):
            self.assertEqual(L.R.sha256(L.OUT/name),L.R.sha256(L.R.OUT/name))
        audit=L.read("native_patch_audit.csv")
        self.assertEqual(len(audit),698)
        for row in df.itertuples():
            mask=cv2.imread(str(L.OUT/"masks"/f"{row.image_id}.png"),0)>0
            self.assertEqual(L.R.sha256(L.OUT/"masks"/f"{row.image_id}.png"),row.mask_sha256)
            for k in L.KERNELS:
                coords,n=L.patch_coordinates(mask,k,row.image_id)
                path=L.OUT/"patch_coordinates"/f"{row.image_id}_k{k}.npy"
                np.testing.assert_array_equal(coords,np.load(path))
                windows=L.sliding_window_view(mask,(k,k))
                self.assertTrue(windows[coords[:,0],coords[:,1]].all())
                stored=audit[audit.image_id.eq(row.image_id)&audit.kernel.eq(k)].iloc[0]
                self.assertEqual(stored.valid_windows,n)
                self.assertEqual(stored.coordinate_sha256,L.R.sha256(path))

    def test_feature_provenance_and_train_fit_reconstruction(self):
        if not (L.OUT/"features/native_saab_20260922_provenance.json").exists():
            self.skipTest("Feature stage not complete")
        from green_learning_cv import fit_saab_unit
        df=L.read("image_manifest.csv")
        splits=L.read("random_patient_assignments.csv")
        for seed in L.R.SEEDS:
            p=json.loads((L.OUT/"features"/f"native_saab_{seed}_provenance.json").read_text())
            train=set(splits.loc[splits.seed.eq(seed)&splits.role.eq("train"),"patient_id"])
            self.assertEqual(set(p["train_patients"]),train)
            self.assertEqual(set(p["fit_images"]),set(df.loc[df.patient_id.isin(train),"image_id"]))
            self.assertEqual(p["old_state_before"],p["old_state_after"])
            self.assertEqual(p["old_extractor_sha256"],L.R.sha256(p["old_extractor"]))
            a=pd.DataFrame(p["training_patches"])
            self.assertEqual(set(a.patient_id),train)
            self.assertTrue((a.groupby(["patient_id","kernel"]).n_patches.sum()==512).all())
            arrays=L.load_features(seed)
            self.assertEqual(arrays["green_roi_reused"].shape,(98,250))
            self.assertEqual(arrays["green_roi_fitted"].shape,(98,1060))
        for item in json.loads((L.OUT/"features/encoder_provenance.json").read_text()):
            self.assertEqual(item["state_before"],item["state_after"])
            self.assertEqual(item["trainable_parameters"],0)
            self.assertEqual(item["sha256"],L.R.sha256(item["checkpoint"]))
        cache=L.load_patch_cache()
        seed=L.R.SEEDS[0]
        fitted=joblib.load(L.OUT/"models"/f"native_saab_{seed}.joblib")
        for k in L.KERNELS:
            samples,_=L.training_patches(cache,df,fitted["train_patients"],seed,k)
            recomputed=fit_saab_unit(samples)
            np.testing.assert_allclose(recomputed.W,fitted["fitted"][k].W,atol=1e-10)
        expected=[]
        for image_id in df.image_id:
            expected.append(np.concatenate([L.local_statistics(cache[k][image_id],fitted["fitted"][k],k) for k in L.KERNELS]))
        np.testing.assert_allclose(np.stack(expected),np.load(L.OUT/"features"/f"green_roi_fitted_{seed}_image.npy"),atol=1e-7)

    def test_all_metrics_roles_and_fixed_votes(self):
        if not (L.OUT/"summary.csv").exists():
            self.skipTest("Evaluation not complete")
        m=L.read("metrics_per_draw.csv")
        p=L.read("patient_test_predictions.csv")
        member=L.read("member_predictions_by_role.csv")
        splits=L.read("random_patient_assignments.csv")
        self.assertEqual(len(m),250)
        self.assertEqual(len(p),5000)
        self.assertEqual(len(member),2800)
        self.assertEqual(len(L.read("development_diversity.csv")),210)
        self.assertFalse(p.duplicated(["seed","model","patient_id"]).any())
        for row in m.itertuples():
            pred=p[p.seed.eq(row.seed)&p.model.eq(row.model)]
            self.assertEqual(len(pred),20)
            ids=set(splits.loc[splits.seed.eq(row.seed)&splits.role.eq("test"),"patient_id"])
            self.assertEqual(set(pred.patient_id),ids)
            self.assertAlmostEqual(roc_auc_score(pred.label,pred.score),row.auroc,places=10)
            self.assertAlmostEqual(average_precision_score(pred.label,pred.score),row.average_precision,places=10)
            for key,value in L.V.score_metrics(pred.label,pred.score).items():
                self.assertAlmostEqual(value,getattr(row,key),places=10)
        for seed in L.R.SEEDS:
            per=member[member.seed.eq(seed)&member.role.eq("test")]
            matrix=per.pivot(index="patient_id",columns="group",values="probability")
            votes=L.fixed_votes({g:matrix[g].to_numpy() for g in L.GROUPS})
            for name,score in votes.items():
                stored=p[p.seed.eq(seed)&p.model.eq(name+"__native")].set_index("patient_id").loc[matrix.index]
                np.testing.assert_allclose(stored.score,score,atol=1e-12)
            for name in votes:
                a=m[m.seed.eq(seed)&m.model.eq(name+"__native")].iloc[0]
                b=m[m.seed.eq(seed)&m.model.eq(name+"__temperature20")].iloc[0]
                self.assertAlmostEqual(a.auroc,b.auroc,places=12)
                self.assertAlmostEqual(a.average_precision,b.average_precision,places=12)

    def test_all_calibrators_use_only_the_declared_patients(self):
        if not (L.OUT/"summary.csv").exists():
            self.skipTest("Evaluation not complete")
        cal=json.loads((L.OUT/"calibration_parameters.json").read_text())
        self.assertEqual(len(cal),130)
        member=L.read("member_predictions_by_role.csv")
        splits=L.read("random_patient_assignments.csv")
        test=L.read("patient_test_predictions.csv")
        old=pd.read_csv(L.V.OUT/"voting_patient_assignments.csv",dtype={"patient_id":str})
        for item in cal:
            seed=item["seed"]
            expected=set(splits.loc[splits.seed.eq(seed)&splits.role.eq("validation"),"patient_id"])
            if item["role"]=="temperature10":
                expected=set(old.loc[old.seed.eq(seed)&old.role.eq("calibration"),"patient_id"])
            self.assertEqual(set(item["patients"]),expected)
            self.assertEqual(item["n_calibration"],len(expected))
            per=member[member.seed.eq(seed)&member.role.eq("validation")]
            p=per.pivot(index="patient_id",columns="group",values="probability").loc[item["patients"]]
            y=per.groupby("patient_id").label.first().loc[item["patients"]]
            values=L.fixed_votes({g:p[g].to_numpy() for g in L.GROUPS})[item["workflow"]]
            c=L.fit_temperature(values,y.to_numpy())
            self.assertAlmostEqual(c["temperature"],item["temperature"],places=6)
            native=test[test.seed.eq(seed)&test.model.eq(item["workflow"]+"__native")].set_index("patient_id")
            calibrated=test[test.seed.eq(seed)&test.model.eq(item["workflow"]+"__"+item["role"])].set_index("patient_id").loc[native.index]
            np.testing.assert_allclose(L.temperature(native.score.to_numpy(),item["temperature"]),calibrated.score,atol=1e-12)

    def test_saved_heads_and_previous_strong_baselines(self):
        if not (L.OUT/"summary.csv").exists():
            self.skipTest("Evaluation not complete")
        records=json.loads((L.OUT/"head_provenance.json").read_text())
        self.assertEqual(len(records),70)
        patients=L.read("patients.csv")
        member=L.read("member_predictions_by_role.csv")
        splits=L.read("random_patient_assignments.csv")
        old=pd.read_csv(L.V.OUT/"member_predictions_by_role.csv",dtype={"patient_id":str})
        for seed in L.R.SEEDS:
            arrays=L.load_features(seed)
            for item in [r for r in records if r["seed"]==seed]:
                path=L.OUT/item["file"]
                self.assertEqual(L.R.sha256(path),item["sha256"])
                expected=set(splits.loc[splits.seed.eq(seed)&splits.role.eq("train"),"patient_id"])
                self.assertEqual(set(item["train_patients"]),expected)
                model=joblib.load(path)
                self.assertEqual(model[1].n_samples_seen_,58)
                per=member[member.seed.eq(seed)&member.group.eq(item["group"])].set_index("patient_id")
                ix=np.flatnonzero(patients.patient_id.isin(per.index))
                per=per.loc[patients.patient_id.iloc[ix]]
                probs=model.predict_proba(arrays[item["group"]][ix])[:,1]
                np.testing.assert_allclose(probs,per.probability,atol=1e-12)
            for group in ("radiomics_percentile","radimagenet_percentile","green_original"):
                a=member[member.seed.eq(seed)&member.group.eq(group)&member.role.eq("test")].set_index("patient_id")
                b=old[old.seed.eq(seed)&old.member.eq(group+"__lr")&old.role.eq("test")].set_index("patient_id").loc[a.index]
                np.testing.assert_allclose(a.probability,expit(b.raw_score),atol=1e-12)


if __name__=="__main__":
    unittest.main(verbosity=2)
