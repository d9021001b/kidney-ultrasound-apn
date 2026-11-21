"""Voting arithmetic, role isolation, independent metric and provenance checks."""
import json
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import renal_voting_validation as V


class VotingTests(unittest.TestCase):
    def test_grids_and_family_balance(self):
        for shrink in (0.,.5):
            weights=np.array(V.weight_grid(shrink))
            self.assertEqual(weights.shape,(15,3))
            np.testing.assert_allclose(weights.sum(axis=1),1)
            self.assertTrue(np.all(weights>=shrink/3))
        p=np.zeros((4,15))
        p[:,V.family_indices()["green"]]=1
        f=V.family_probabilities(p)
        np.testing.assert_allclose(f,np.tile([0,0,1],(4,1)))
        rules=V.select_rules(p,[0,1,0,1])
        scores=V.combine_votes(p,p.astype(int),rules)
        np.testing.assert_allclose(scores["all_soft"],.2)
        np.testing.assert_allclose(scores["family_balanced_soft"],1/3)
        np.testing.assert_allclose(scores["without_green_soft"],0)

    def test_constant_probabilities_and_invalid_inputs(self):
        p=np.full((10,15),.5)
        rules=V.select_rules(p,np.tile([0,1],5))
        self.assertEqual(rules["selected_member"],V.MEMBERS[0])
        votes=V.combine_votes(p,np.zeros_like(p),rules)
        np.testing.assert_array_equal(votes["all_hard_calibrated"],0)
        m=V.score_metrics(np.tile([0,1],5),np.full(10,.5))
        self.assertEqual(m["tp"],0)
        self.assertEqual(m["tn"],5)
        self.assertEqual(m["auroc"],.5)
        for bad in (p[:,:14],np.full_like(p,np.nan),np.full_like(p,1.1)):
            with self.assertRaises(ValueError):
                V.family_probabilities(bad)
        rules[V.PRIMARY]["weights"]=[1,1,1]
        with self.assertRaises(ValueError):
            V.combine_votes(p,np.zeros_like(p),rules)

    def test_unused_patients_cannot_change_fitted_models(self):
        rng=np.random.default_rng(43)
        x=rng.normal(size=(80,12))
        y=np.tile([0,1],40)
        train,cal=np.arange(48),np.arange(48,60)
        changed=x.copy()
        changed[60:]*=1000
        labels=y.copy()
        labels[60:]=1-labels[60:]
        for algorithm in V.ALGORITHMS:
            a,b=V.fit_member(x,y,train,cal,algorithm,73)
            c,d=V.fit_member(changed,labels,train,cal,algorithm,73)
            self.assertEqual(joblib.hash(a),joblib.hash(c),algorithm)
            self.assertEqual(joblib.hash(b),joblib.hash(d),algorithm)
            changed_cal=y.copy()
            changed_cal[cal]=1-changed_cal[cal]
            e,_=V.fit_member(x,changed_cal,train,cal,algorithm,73)
            self.assertEqual(joblib.hash(a),joblib.hash(e),algorithm)
        with self.assertRaises(ValueError):
            V.fit_member(x,y,train,np.arange(40,60),"lr",73)

    def test_locked_protocol_disjoint_roles_and_features(self):
        old=V.frame("random_patient_assignments.csv")
        splits=V.frame("voting_patient_assignments.csv")
        self.assertEqual(len(splits),980)
        self.assertFalse(splits.duplicated(["seed","patient_id"]).any())
        for seed in V.R.SEEDS:
            per=splits[splits.seed.eq(seed)]
            self.assertEqual(per.role.value_counts().to_dict(),dict(train=58,test=20,calibration=10,vote_selection=10))
            for role in ("train","test"):
                self.assertEqual(set(per.loc[per.role.eq(role),"patient_id"]),set(old.loc[old.seed.eq(seed)&old.role.eq(role),"patient_id"]))
            for role in ("calibration","vote_selection"):
                self.assertEqual(per.loc[per.role.eq(role),"label"].nunique(),2)
            arrays=V.load_features(seed)
            self.assertEqual({k:v.shape[1] for k,v in arrays.items()},dict(radiomics_original=914,radiomics_percentile=914,radimagenet_original=2048,radimagenet_percentile=2048,green_original=8466))
        locked=json.loads((V.OUT/"protocol_locked.json").read_text())
        self.assertEqual(locked["sha256"],V.R.sha256(V.OUT/"protocol.json"))
        for item in json.loads((V.OUT/"features/provenance.json").read_text()):
            self.assertEqual(V.R.sha256(item["source"]),item["source_sha256"])
            self.assertEqual(V.R.sha256(V.OUT/item["output"]),item["output_sha256"])
            if "seed" in item:
                ids=set(splits.loc[splits.seed.eq(item["seed"])&splits.role.eq("train"),"patient_id"])
                self.assertEqual(ids,set(item["train_patients"]))

    def test_all_results_recomputed_from_predictions(self):
        if not (V.OUT/"summary.csv").exists():
            self.skipTest("Full evaluation not finished")
        metrics=V.frame("metrics_per_draw.csv")
        pred=V.frame("patient_test_predictions.csv")
        splits=V.frame("voting_patient_assignments.csv")
        self.assertEqual(len(metrics),280)
        self.assertEqual(len(pred),5600)
        self.assertFalse(pred.duplicated(["seed","model","patient_id"]).any())
        for row in metrics.itertuples():
            p=pred[pred.seed.eq(row.seed)&pred.model.eq(row.model)]
            self.assertEqual(len(p),20)
            self.assertEqual(set(p.patient_id),set(splits.loc[splits.seed.eq(row.seed)&splits.role.eq("test"),"patient_id"]))
            self.assertAlmostEqual(roc_auc_score(p.label,p.score),row.auroc,places=11)
            self.assertAlmostEqual(average_precision_score(p.label,p.score),row.average_precision,places=11)
            values=V.score_metrics(p.label,p.score,not row.model.startswith("all_hard"))
            for key,value in values.items():
                if value is not None:
                    self.assertAlmostEqual(value,getattr(row,key),places=10,msg=f"{row.model} {key}")
            np.testing.assert_array_equal(p.prediction,p.score>.5)

    def test_weights_and_ensembles_reproduced_from_selection_only(self):
        if not (V.OUT/"summary.csv").exists():
            self.skipTest("Full evaluation not finished")
        member=V.frame("member_predictions_by_role.csv")
        pred=V.frame("patient_test_predictions.csv")
        splits=V.frame("voting_patient_assignments.csv")
        self.assertEqual(len(member),6000)
        self.assertEqual(len(V.frame("vote_selection_diversity.csv")),1050)
        self.assertEqual(len(V.frame("raw_member_metrics.csv")),150)
        selections=json.loads((V.OUT/"ensemble_selection.json").read_text())
        self.assertEqual(len(selections),10)
        for item in selections:
            seed=item["seed"]
            per=member[member.seed.eq(seed)]
            for role in ("calibration","vote_selection","test"):
                self.assertEqual(set(per.loc[per.role.eq(role),"patient_id"]),set(splits.loc[splits.seed.eq(seed)&splits.role.eq(role),"patient_id"]))
            vote=per[per.role.eq("vote_selection")]
            matrix=vote.pivot(index="patient_id",columns="member",values="probability").loc[:,list(V.MEMBERS)]
            y=vote.groupby("patient_id").label.first().loc[matrix.index]
            regenerated=V.select_rules(matrix.to_numpy(),y.to_numpy())
            self.assertEqual(regenerated["selected_member"],item["selected_member"])
            self.assertEqual(set(item["vote_selection_patients"]),set(matrix.index))
            for rule in ("validation_weighted_soft",V.PRIMARY):
                np.testing.assert_allclose(regenerated[rule]["weights"],item[rule]["weights"])
                self.assertAlmostEqual(regenerated[rule]["validation_score"],item[rule]["validation_score"],places=10)
            test=per[per.role.eq("test")]
            p=test.pivot(index="patient_id",columns="member",values="probability").loc[:,list(V.MEMBERS)]
            raw=test.pivot(index="patient_id",columns="member",values="raw_label").loc[p.index,list(V.MEMBERS)]
            combined=V.combine_votes(p.to_numpy(),raw.to_numpy(),regenerated)
            self.assertEqual(set(combined),set(V.ENSEMBLES))
            for name,scores in combined.items():
                stored=pred[pred.seed.eq(seed)&pred.model.eq(name)].set_index("patient_id").loc[p.index]
                np.testing.assert_allclose(stored.score,scores,atol=1e-12)

    def test_saved_models_refit_counts_and_exact_predictions(self):
        if not (V.OUT/"summary.csv").exists():
            self.skipTest("Full evaluation not finished")
        provenance=json.loads((V.OUT/"model_provenance.json").read_text())
        self.assertEqual(len(provenance),150)
        member=V.frame("member_predictions_by_role.csv")
        splits=V.frame("voting_patient_assignments.csv")
        patients=V.frame("patients.csv")
        for seed in V.R.SEEDS:
            arrays=V.load_features(seed)
            roles=splits[splits.seed.eq(seed)]
            for item in [v for v in provenance if v["seed"]==seed]:
                path=V.OUT/item["model_file"]
                self.assertEqual(V.R.sha256(path),item["sha256"])
                for role,key in (("train","train_patients"),("calibration","calibration_patients")):
                    self.assertEqual(set(item[key]),set(roles.loc[roles.role.eq(role),"patient_id"]))
                fitted=joblib.load(path)
                head,cal=fitted["head"],fitted["calibrator"]
                self.assertEqual(head[1].n_samples_seen_,58)
                self.assertEqual(cal[0].n_samples_seen_,10)
                self.assertEqual(len(item["selected_indices"]),32)
                p=member[member.seed.eq(seed)&member.member.eq(item["member"])].set_index("patient_id")
                ix=patients.index[patients.patient_id.isin(p.index)]
                p=p.loc[patients.patient_id.iloc[ix]]
                raw=V.raw_score(head,arrays[item["feature_group"]][ix])
                np.testing.assert_allclose(raw,p.raw_score,atol=1e-12)
                np.testing.assert_allclose(cal.predict_proba(raw[:,None])[:,1],p.probability,atol=1e-12)

    def test_followup_probabilities_and_paired_metrics(self):
        import renal_voting_probability_control as C
        if not (C.OUT/"summary.csv").exists():
            self.skipTest("Follow-up not finished")
        locked=json.loads((C.OUT/"protocol_locked.json").read_text())
        self.assertEqual(locked["sha256"],V.R.sha256(C.OUT/"protocol.json"))
        parent=V.frame("member_predictions_by_role.csv")
        member=pd.read_csv(C.OUT/"member_test_probabilities.csv",dtype={"patient_id":str})
        preds=pd.read_csv(C.OUT/"patient_test_predictions.csv",dtype={"patient_id":str})
        metrics=pd.read_csv(C.OUT/"metrics_per_draw.csv")
        self.assertEqual(len(member),2000)
        self.assertEqual(len(preds),1200)
        self.assertEqual(len(metrics),60)
        self.assertEqual(len(C.MEMBERS),10)
        self.assertTrue(all(not name.endswith("__svm") for name in C.MEMBERS))
        joined=member.merge(parent[parent.role.eq("test")],on=["seed","patient_id","member","label"],validate="one_to_one")
        self.assertEqual(len(joined),2000)
        np.testing.assert_allclose(joined.calibrated_probability,joined.probability,atol=1e-12)
        from scipy.special import expit
        expected=np.where(joined.member.str.endswith("__lr"),expit(joined.raw_score),joined.raw_score)
        np.testing.assert_allclose(joined.native_probability,expected,atol=1e-12)
        for seed in V.R.SEEDS:
            per=member[member.seed.eq(seed)]
            for arm in ("native","calibrated"):
                p=per.pivot(index="patient_id",columns="member",values=arm+"_probability").loc[:,list(C.MEMBERS)]
                scores=C.combine(p.to_numpy())
                for name,values in scores.items():
                    stored=preds[preds.seed.eq(seed)&preds.model.eq(arm+"__"+name)].set_index("patient_id").loc[p.index]
                    np.testing.assert_allclose(values,stored.score,atol=1e-12)
        for row in metrics.itertuples():
            p=preds[preds.seed.eq(row.seed)&preds.model.eq(row.model)]
            self.assertEqual(len(p),20)
            self.assertAlmostEqual(roc_auc_score(p.label,p.score),row.auroc,places=11)
            self.assertAlmostEqual(average_precision_score(p.label,p.score),row.average_precision,places=11)


if __name__=="__main__":
    unittest.main(verbosity=2)
