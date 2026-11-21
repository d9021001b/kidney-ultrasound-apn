"""Check the actual released ensemble when single-checkpoint CEC approximation failed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import nckuh_internal_regions_20261102 as engine
from nckuh_cec_reference_20261102 import largest_component

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_cec_ensemble_confirmation_20261102"
VARIANTS = ["annotator_1", "annotator_2"]


def configure():
    engine.OUT = OUT
    engine.MODES = ["full_frame"]


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(ROOT / "02_experiments/NCKUH_cec_reference_20261102/selected_images_private.csv", dtype=str)
    selected = previous[previous.technical_role.eq("new_confirmation")].copy()
    assert len(selected) == selected.patient_alias.nunique() == 32 and "label" not in selected.columns
    selected.to_csv(OUT / "selected_images_private.csv", index=False)
    records = {variant: engine.K.validate_model_family(variant) for variant in VARIANTS}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), diagnostic_labels_read=False,
                    reason="The single-checkpoint approximation failed its own gate and did not reproduce the five-checkpoint CEC masks sufficiently. Evaluate the original five-checkpoint source ensemble on the same new32 technical confirmation patients before abandoning that distinct inference method",
                    source_method="https://github.com/rsingla92/kidneyUS", class_of_interest=1,
                    clinical_boundary="Model-predicted central echo region only; no cortex or medulla claim, no APN diagnosis or localization labels derived",
                    inputs=["full_frame"], image_ids=selected.image_id.tolist(), checkpoints=records,
                    selection_sha256=engine.K.sha256(OUT / "selected_images_private.csv"),
                    necessary_CEC_gates=dict(each_model_nonempty_rate_at_least=.85, each_model_area_fraction_02_to_50_rate_at_least=.8,
                                             both_models_nonempty_rate_at_least=.8, median_intermodel_CEC_dice_at_least=.7),
                    interpretation="Neither prior failed gate is reclassified as passed. This separate ensemble test contains no outcome-based selection or APN classification",
                    wrapper_source_sha256=engine.K.sha256(Path(__file__)))
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    engine.write_json(OUT / "protocol.json", protocol)
    engine.write_json(OUT / "run_lock.json", dict(source_sha256=engine.K.sha256(Path(engine.__file__)),
                                                  protocol_sha256=engine.K.sha256(OUT / "protocol.json")))
    print("Original five-checkpoint CEC ensemble confirmation fixed; no APN columns", flush=True)


def summarize():
    engine.require_lock()
    selection = pd.read_csv(OUT / "selected_images_private.csv", dtype=str)
    summaries = []
    images = {}
    for variant in VARIANTS:
        fractions, present = [], []
        images[variant] = {}
        for ident in selection.image_id:
            labels = cv2.imread(str(OUT / variant / "region_masks/full_frame" / f"{ident}.png"), 0)
            outer = cv2.imread(str(engine.FEATURES / "masks" / f"{ident}.png"), 0) > 0
            cec = largest_component(labels == 1)
            images[variant][ident] = cec
            fractions.append(cec.sum()/outer.sum()); present.append(cec.sum() >= 64)
        fractions = np.asarray(fractions)
        summaries.append(dict(variant=variant, nonempty_rate=float(np.mean(present)),
                              area_fraction_in_range_rate=float(np.mean((fractions >= .02)&(fractions <= .5))),
                              median_area_fraction=float(np.median(fractions))))
    dice, both = [], []
    for ident in selection.image_id:
        a, b = images[VARIANTS[0]][ident], images[VARIANTS[1]][ident]
        both.append(a.sum() >= 64 and b.sum() >= 64)
        if a.any() or b.any():
            dice.append(engine.K.dice(a, b))
    median_dice = float(np.median(dice)) if dice else 0.
    both_rate = float(np.mean(both))
    passed = bool(all(m["nonempty_rate"] >= .85 and m["area_fraction_in_range_rate"] >= .8 for m in summaries) and both_rate >= .8 and median_dice >= .7)
    result = dict(diagnostic_labels_read=False, models=summaries, median_intermodel_CEC_dice=median_dice,
                  both_nonempty_rate=both_rate, technical_CEC_gates_passed=passed,
                  independent_anatomical_truth_available=False, APN_classification_performed=False,
                  action="Review CEC consensus and predefine full-cohort features" if passed else "Exclude the released compartment model from APN features without reliable target anatomy")
    engine.write_json(OUT / "decision.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "summarize"])
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    configure()
    if args.stage == "run":
        if args.variant is None:
            parser.error("run requires --variant")
        protocol = engine.require_lock()
        assert engine.K.sha256(Path(__file__)) == protocol["wrapper_source_sha256"]
        engine.run(args.variant)
    else:
        globals()[args.stage]()
