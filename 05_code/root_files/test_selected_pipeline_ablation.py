from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "CYCH_selected_pipeline_ablation_20260914"
LOCKED = ROOT / "CYCH_reliable_negative_graph_mil_development_20260914" / "development_predictions.csv"


class SelectedPipelineAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.predictions = pd.read_csv(OUT / "development_predictions_deidentified.csv")
        cls.metrics = pd.read_csv(OUT / "development_metrics_per_draw.csv")
        cls.summary = pd.read_csv(OUT / "development_summary.csv").set_index("group")

    def test_only_fixed_development_rows_are_present(self) -> None:
        self.assertEqual(set(self.predictions.role), {"development_validation"})
        self.assertEqual(sorted(self.predictions.seed.unique()), list(range(20260913, 20260923)))
        counts = self.predictions.groupby(["group", "seed"]).patient_alias.nunique()
        self.assertTrue((counts == 20).all())
        self.assertNotIn("patient_id", self.predictions.columns)
        self.assertTrue(self.predictions.patient_alias.str.fullmatch(r"P\d{3}").all())

    def test_final_scores_reproduce_locked_candidate(self) -> None:
        final = self.predictions.loc[
            self.predictions.group.eq("FINAL-EQUAL-VOTE"),
            ["seed", "patient_alias", "label", "score"],
        ]
        locked = pd.read_csv(LOCKED)
        locked = locked.loc[
            locked.group.eq("RELIABLE-NEG-GRAPH+I3"),
            ["seed", "patient_alias", "label", "score"],
        ]
        joined = final.merge(
            locked,
            on=["seed", "patient_alias", "label"],
            suffixes=("_ablation", "_locked"),
            validate="one_to_one",
        )
        self.assertEqual(len(joined), 200)
        self.assertTrue(np.allclose(joined.score_ablation, joined.score_locked, atol=1e-12))

    def test_locked_summary_metrics_are_reproduced(self) -> None:
        final = self.summary.loc["FINAL-EQUAL-VOTE"]
        self.assertAlmostEqual(final.mean_auroc, 0.805050505050505, places=12)
        self.assertAlmostEqual(final.mean_average_precision, 0.8435240330776369, places=9)
        self.assertAlmostEqual(final.mean_auprc_trapezoid, 0.8320882400818371, places=9)
        self.assertTrue(self.predictions.score.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
