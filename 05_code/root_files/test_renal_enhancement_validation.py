"""ROI invariance, baseline equivalence, provenance and independent metric checks."""
import json
import unittest
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import renal_enhancement_validation as E


class EnhancementTests(unittest.TestCase):
    def test_outside_intensities_cannot_influence_any_mode(self):
        rng = np.random.default_rng(81)
        gray = rng.integers(0,256,(67,83),dtype=np.uint8)
        mask = np.zeros(gray.shape,np.uint8)
        cv2.ellipse(mask,(40,31),(30,22),17,0,360,1,-1)
        changed = gray.copy()
        changed[mask == 0] = 255-gray[mask == 0]
        for mode in E.MODES:
            a,b=E.enhance(gray,mask,mode),E.enhance(changed,mask,mode)
            np.testing.assert_array_equal(a,b,err_msg=mode)
            self.assertEqual(a.shape,gray.shape)
            self.assertEqual(a.dtype,np.uint8)
            self.assertTrue(np.all(a[mask == 0] == 0))
            np.testing.assert_array_equal(a,E.enhance(gray,mask,mode))

    def test_constant_empty_and_full_masks(self):
        gray=np.full((30,42),73,np.uint8)
        mask=np.zeros_like(gray)
        mask[4:24,7:35]=1
        for mode in E.MODES:
            np.testing.assert_array_equal(E.enhance(gray,mask,mode),gray*mask)
            np.testing.assert_array_equal(E.enhance(gray,np.ones_like(mask),mode),gray)
            with self.assertRaises(ValueError):
                E.enhance(gray,np.zeros_like(mask),mode)

    def test_actual_images_and_locked_protocol(self):
        d=E.read_manifest()
        self.assertEqual(len(d),349)
        self.assertEqual(d.patient_id.nunique(),98)
        for name in ("image_manifest.csv","patients.csv","random_patient_assignments.csv"):
            self.assertEqual(E.R.sha256(E.OUT/name),E.R.sha256(E.BASE/name))
        audit=pd.read_csv(E.OUT/"image_transform_audit.csv").set_index(["image_id","mode"])
        self.assertEqual(len(audit),349*len(E.MODES))
        for row in d.itertuples():
            mask=cv2.imread(str(E.OUT/"masks"/f"{row.image_id}.png"),0)>0
            self.assertEqual(E.R.sha256(E.OUT/"masks"/f"{row.image_id}.png"),row.mask_sha256)
            original=cv2.imread(str(E.BASE/"masked"/f"{row.image_id}.png"),0)
            for mode in E.MODES:
                path=E.OUT/"images"/mode/f"{row.image_id}.png"
                image=cv2.imread(str(path),0)
                self.assertTrue(np.all(image[~mask]==0))
                self.assertEqual(image.shape,mask.shape)
                self.assertEqual(E.R.sha256(path),audit.loc[(row.image_id,mode),"image_sha256"])
                if mode == "original":
                    np.testing.assert_array_equal(image,original)
        locked=json.loads((E.OUT/"protocol_locked.json").read_text())
        self.assertEqual(locked["sha256"],E.R.sha256(E.OUT/"protocol.json"))

    def test_radiomics_override_and_baseline(self):
        args=dict(image_id="I0001",gray_path=str(E.OUT/"images/original/I0001.png"),
                  mask_path=str(E.OUT/"masks/I0001.png"))
        computed=E.R.radiomics_one(args)
        cached=pd.read_csv(E.BASE/"features/radiomics.csv").set_index("image_id").loc["I0001"]
        self.assertEqual(len(computed)-1,467)
        for key,value in cached.items():
            np.testing.assert_allclose(computed[key],value,rtol=1e-7,atol=1e-8,err_msg=key)

    def test_fit_and_selection_do_not_use_test_data(self):
        rng=np.random.default_rng(93)
        x=rng.normal(size=(3,40,10)).astype(np.float32)
        y=np.tile([0,1],20)
        train,val,test=np.arange(20),np.arange(20,30),np.arange(30,40)
        _,a,_=E.R.fit_arm(x,y,train,val,test)
        altered=x.copy()
        altered[:,test]*=100
        alternate_labels=y.copy()
        alternate_labels[test]=1-alternate_labels[test]
        _,b,_=E.R.fit_arm(altered,alternate_labels,train,val,test)
        self.assertEqual(a[2],b[2])
        self.assertEqual(a[0],b[0])
        np.testing.assert_array_equal(a[1][1].mean_,b[1][1].mean_)
        np.testing.assert_array_equal(a[1][-1].support_vectors_,b[1][-1].support_vectors_)
        self.assertEqual(a[1][1].n_samples_seen_,len(train)*len(x))

    def test_results_and_validation_only_selection_when_complete(self):
        path=E.OUT/"summary.csv"
        if not path.exists():
            self.skipTest("Full evaluation not finished yet")
        result=pd.read_csv(E.OUT/"metrics_per_draw.csv")
        pred=pd.read_csv(E.OUT/"patient_test_predictions.csv",dtype={"patient_id":str})
        splits=pd.read_csv(E.OUT/"random_patient_assignments.csv",dtype={"patient_id":str})
        self.assertEqual(len(result),410)
        self.assertEqual(len(pred),8200)
        self.assertFalse(pred.duplicated(["seed","model","patient_id"]).any())
        for row in result.itertuples():
            scores=pred[pred.seed.eq(row.seed)&pred.model.eq(row.model)]
            ids=set(splits.loc[splits.seed.eq(row.seed)&splits.role.eq("test"),"patient_id"])
            self.assertEqual(set(scores.patient_id),ids)
            self.assertEqual(len(scores),20)
            self.assertAlmostEqual(roc_auc_score(scores.label,scores.score),row.auroc,places=10)
            self.assertAlmostEqual(average_precision_score(scores.label,scores.score),row.average_precision,places=10)
        selection=json.loads((E.OUT/"validation_selection.json").read_text())
        self.assertEqual(len(selection),60)
        for item in selection:
            self.assertEqual(item["selected_arm"],max(item["candidate_scores"],key=item["candidate_scores"].get))
            a=pred[pred.seed.eq(item["seed"])&pred.model.eq(item["selected_arm"])].set_index("patient_id")
            b=pred[pred.seed.eq(item["seed"])&pred.model.eq(item["selector"])].set_index("patient_id").loc[a.index]
            np.testing.assert_array_equal(a.score,b.score)
        tuning=json.loads((E.OUT/"validation_tuning.json").read_text())
        self.assertEqual(len(tuning),350)
        self.assertTrue(all(len(item["candidates"])==16 for item in tuning))
        for seed in E.R.SEEDS:
            for mode in E.GREEN_MODES[1:]:
                provenance=json.loads((E.OUT/"features"/f"green_{mode}_{seed}_provenance.json").read_text())
                self.assertEqual(set(provenance["actual_fit_images"]),set(provenance["expected_train_images"]))
                self.assertEqual(provenance["extractor_sha256"],E.R.sha256(provenance["extractor_path"]))
                self.assertTrue(np.isfinite(np.load(E.OUT/"features"/f"green_{mode}_{seed}.npy")).all())
        previous=pd.read_csv(E.BASE/"metrics_per_draw.csv")
        pairs={"original__radiomics":"masked_pyradiomics","original__radimagenet":"masked_radimagenet",
               "original__radimagenet_aug":"masked_radimagenet_aug","original__fusion_aug":"masked_radiomics_radimagenet_aug",
               "original__green_frozen":"masked_green","original__green_fusion_aug":"masked_green_radiomics_radimagenet_aug"}
        for new,old in pairs.items():
            a=result[result.model.eq(new)].set_index("seed").sort_index()
            b=previous[previous.model.eq(old)].set_index("seed").sort_index()
            np.testing.assert_allclose(a[["auroc","average_precision"]],b[["auroc","average_precision"]],atol=1e-10)


if __name__=="__main__":
    unittest.main(verbosity=2)
