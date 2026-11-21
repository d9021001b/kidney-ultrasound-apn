"""Focused tests for blinded semantic-region packet helpers and structural QC."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import build_internal_region_annotation_packet as B
import validate_internal_region_annotations as V


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    index = 0
    for view in B.VIEWS:
        for source in ("existing_yolo", "manual_polygon_fallback"):
            for _ in range(12):
                index += 1
                rows.append(
                    {
                        "image_id": f"I{index:04d}",
                        "patient_id": f"patient_{index:04d}",
                        "patient_alias": f"P{index:03d}",
                        "view": view,
                        "mask_source": source,
                        "mask_sha256": f"hash_{index}",
                        "label": index % 2,
                        "filename": f"source_{index}.png",
                    }
                )
    return pd.DataFrame(rows)


class AnnotationPacketTests(unittest.TestCase):
    def test_core_selection_is_unique_patient_balanced_and_label_blind(self):
        manifest = synthetic_manifest()
        selected = B.select_core_tasks(manifest)
        changed = manifest.copy()
        changed["label"] = 1 - changed["label"]
        changed["filename"] = "different_source_filename.png"
        selected_changed = B.select_core_tasks(changed)
        self.assertEqual(len(selected), 32)
        self.assertEqual(selected.patient_id.nunique(), 32)
        self.assertListEqual(selected.image_id.tolist(), selected_changed.image_id.tolist())
        counts = selected.groupby(["view", "mask_source"]).size().to_dict()
        for view in B.VIEWS:
            self.assertEqual(counts[(view, "existing_yolo")], 5)
            self.assertEqual(counts[(view, "manual_polygon_fallback")], 3)

    def test_hidden_repeats_and_display_order_have_expected_size(self):
        core = B.select_core_tasks(synthetic_manifest())
        repeats = B.select_hidden_repeats(core)
        ordered = B.order_task_items(core, repeats)
        self.assertEqual(len(repeats), 8)
        self.assertEqual(len(ordered), 40)
        self.assertEqual(sum(item["kind"] == "hidden_repeat" for item in ordered), 8)
        for index, item in enumerate(ordered):
            recent = ordered[max(0, index - B.MIN_REPEAT_SEPARATION) : index]
            self.assertNotIn(item["image_id"], [other["image_id"] for other in recent])

    def test_validator_reports_outer_roi_leakage_overlap_and_status_conflict(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "packet"
            annotations = Path(temp) / "returned"
            output = Path(temp) / "qc"
            for name in ("images", "outer_roi_masks"):
                (root / name).mkdir(parents=True, exist_ok=True)
            annotations.mkdir()
            pd.DataFrame([{"task_id": "A001", "display_order": 1, "view": "LL"}]).to_csv(
                root / "annotation_tasks_blinded.csv", index=False
            )
            image = np.zeros((24, 24), np.uint8)
            outer = np.zeros_like(image)
            cv2.rectangle(outer, (4, 4), (19, 19), 255, -1)
            cv2.imwrite(str(root / "images" / "A001.png"), image)
            cv2.imwrite(str(root / "outer_roi_masks" / "A001.png"), outer)
            cec = np.zeros_like(image)
            cec[6:12, 6:12] = 255
            medulla = np.zeros_like(image)
            medulla[9:15, 9:15] = 255
            cortex = np.zeros_like(image)
            uncertain = np.zeros_like(image)
            uncertain[0:3, 0:3] = 255
            for name, mask in (("cec", cec), ("medulla", medulla), ("cortex", cortex), ("uncertain", uncertain)):
                cv2.imwrite(str(annotations / f"A001_{name}.png"), mask)
            pd.DataFrame(
                [
                    {
                        "task_id": "A001",
                        "cec_status": "present",
                        "medulla_status": "absent",
                        "cortex_status": "not_identifiable",
                    }
                ]
            ).to_csv(annotations / "annotation_status.csv", index=False)
            table = V.validate_submission(root, annotations, output)
            errors = ";".join(table.errors.fillna(""))
            self.assertIn("class_overlap", errors)
            self.assertIn("outside_outer_roi", errors)
            self.assertIn("nonempty_mask_conflicts_with_status", errors)
            self.assertTrue((output / "annotation_qc.csv").exists())
            self.assertTrue((output / "qc_overlays" / "A001.png").exists())

    def test_dice_does_not_treat_two_empty_masks_as_anatomical_agreement(self):
        empty = np.zeros((3, 3), np.uint8)
        full = np.ones((3, 3), np.uint8)
        self.assertIsNone(V.dice(empty, empty))
        self.assertEqual(V.dice(full, full), 1.0)
        self.assertEqual(V.dice(empty, full), 0.0)


if __name__ == "__main__":
    unittest.main()
