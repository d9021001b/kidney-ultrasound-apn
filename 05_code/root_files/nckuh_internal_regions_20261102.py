"""Outcome-blind NCKUH compatibility test for released kidneyUS compartments."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch

import kidneyus_internal_region_probe as K
from kidneyus_context_region_probe import context_zoom_canvas, invert_context_labels

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "02_experiments/NCKUH_internal_regions_20261102"
BASE = ROOT / "02_experiments/NCKUH_only_study_20261102"
FEATURES = ROOT / "02_experiments/CYCH_honest_reevaluation_20261003/feat_NCKUH"
K.WEIGHT_ROOT = ROOT / "03_data_models/external_models/kidneyUS_official_20260914/weights"
MODES = ["full_frame", "context2"]
VARIANTS = ["annotator_1", "annotator_2"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def select_images():
    frame = pd.read_csv(BASE / "image_manifest_private.csv", usecols=["image_id", "patient_alias", "filename", "view"], dtype=str)
    assert "label" not in frame.columns
    rng = np.random.default_rng(20261102)
    selected, used = [], set()
    for view in ["LL", "LT", "RL", "RT"]:
        pool = frame[frame["view"].eq(view)].iloc[rng.permutation(int(frame["view"].eq(view).sum()))]
        count = 0
        for row in pool.itertuples(index=False):
            if row.patient_alias in used:
                continue
            selected.append(dict(zip(frame.columns, row)))
            used.add(row.patient_alias)
            count += 1
            if count == 8:
                break
        assert count == 8
    return pd.DataFrame(selected)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    selected = select_images()
    selected.to_csv(OUT / "selected_images_private.csv", index=False)
    annotation_script = ROOT / "03_data_models/third_party/kidneyUS/src/annotation_analysis/create_masks.py"
    source = annotation_script.read_text(encoding="utf-8")
    assert '"Central Echo Complex": 2' in source and '"Medulla": 3' in source and '"Cortex": 4' in source
    assert "color=region-1" in source
    checkpoints = {variant: K.validate_model_family(variant) for variant in VARIANTS}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), diagnostic_labels_read=False,
                    source="https://github.com/rsingla92/kidneyUS", task="Task002_KidneyRegions",
                    class_map={1: "central_echo_complex", 2: "medulla", 3: "cortex"},
                    label_map_evidence=dict(source_sha256=K.sha256(annotation_script), capsule_excluded=True, region_labels_shifted_minus_one=True),
                    sample="32 NCKUH images from 32 distinct patient aliases, eight per LL/LT/RL/RT, selected with seed20261102 without diagnosis columns",
                    image_ids=selected.image_id.tolist(), variants=VARIANTS, inputs=MODES,
                    geometry="full_frame: aspect preserving resize/letterbox to 768x1024; context2: fixed two-times kidney-centred full-frame context; neither uses percentile remapping",
                    outer_mask="Exact localization of saved native grayscale crop in its canonical clean frame, with pixel equality asserted; paste unchanged accepted mask at that location",
                    inference="Mean of five released source checkpoints per annotator; no TTA, target adaptation or APN classification. These source checkpoint indices are not validation folds on NCKUH",
                    necessary_gates=dict(each_model_nonempty_rate_at_least=.8, each_model_median_outer_coverage_at_least=.5,
                                         each_model_two_region_rate_at_least=.8, median_union_dice_when_either_nonempty_at_least=.6,
                                         median_class_dice_when_either_nonempty_at_least=.6),
                    interpretation="Model agreement and technical coverage are necessary checks, not anatomical ground truth. Do not fabricate cortex/medulla from brightness if transfer fails",
                    selection_sha256=K.sha256(OUT / "selected_images_private.csv"), checkpoints=checkpoints)
    if (OUT / "protocol.json").exists():
        raise FileExistsError("Preserve existing protocol")
    write_json(OUT / "protocol.json", protocol)
    write_json(OUT / "run_lock.json", dict(source_sha256=K.sha256(Path(__file__)), protocol_sha256=K.sha256(OUT / "protocol.json")))
    print("NCKUH outcome-blind 32-patient region probe fixed", flush=True)


def require_lock():
    lock = json.loads((OUT / "run_lock.json").read_text(encoding="utf-8"))
    assert lock["source_sha256"] == K.sha256(Path(__file__))
    assert lock["protocol_sha256"] == K.sha256(OUT / "protocol.json")
    return json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))


def locate_outer(row):
    image_path = ROOT / "03_data_models/original_png_nckuh" / row.filename.replace("_red_", "_", 1)
    color = cv2.imread(str(image_path))
    assert color is not None
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    native = cv2.imread(str(FEATURES / "native" / f"{row.image_id}.png"), 0)
    local_mask = cv2.imread(str(FEATURES / "masks" / f"{row.image_id}.png"), 0)
    assert native is not None and native.shape == local_mask.shape
    _, _, position, _ = cv2.minMaxLoc(cv2.matchTemplate(gray, native, cv2.TM_SQDIFF))
    x, y = position
    h, w = native.shape
    assert np.array_equal(gray[y:y+h, x:x+w], native), row.image_id
    outer = np.zeros_like(gray)
    outer[y:y+h, x:x+w] = (local_mask > 0).astype(np.uint8)
    return gray, outer, dict(x=x, y=y, width=w, height=h, source_sha256=K.sha256(image_path))


def full_frame_canvas(gray, outer):
    h, w = gray.shape
    scale = min(768/h, 1024/w)
    nh, nw = round(h*scale), round(w*scale)
    y, x = (768-nh)//2, (1024-nw)//2
    canvas = np.zeros((768, 1024), dtype=np.uint8)
    canvas[y:y+nh, x:x+nw] = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, dict(original_height=h, original_width=w, height=nh, width=nw, x=x, y=y)


def invert_full_frame(labels, transform):
    t = transform
    crop = labels[t["y"]:t["y"]+t["height"], t["x"]:t["x"]+t["width"]]
    return cv2.resize(crop, (t["original_width"], t["original_height"]), interpolation=cv2.INTER_NEAREST)


def run(variant):
    protocol = require_lock()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    selected = pd.read_csv(OUT / "selected_images_private.csv", dtype=str)
    assert K.sha256(OUT / "selected_images_private.csv") == protocol["selection_sha256"]
    from nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
    run_out = OUT / variant
    trainer = K.build_trainer(variant, run_out)
    checkpoints = K.validate_model_family(variant)
    assert [r["sha256"] for r in checkpoints] == [r["sha256"] for r in protocol["checkpoints"][variant]]
    items = []
    for row in selected.itertuples(index=False):
        gray, outer, location = locate_outer(row)
        for mode in MODES:
            if mode == "full_frame":
                canvas, transform = full_frame_canvas(gray, outer)
            else:
                canvas, _, transform = context_zoom_canvas(gray, outer, 2.)
            nifti = run_out / "inputs_nifti" / f"{row.image_id}_{mode}_0000.nii.gz"
            K.array_to_nifti(canvas, nifti)
            data, _, properties = trainer.preprocess_patient([str(nifti)])
            items.append(dict(image_id=row.image_id, view=row.view, mode=mode, gray=gray, outer=outer,
                              location=location, transform=transform, data=data, properties=properties, softmax=None))
    for record in checkpoints:
        checkpoint = K.load_checkpoint(Path(record["path"]))
        trainer.load_checkpoint_ram(checkpoint, train=False)
        trainer.network.eval()
        with torch.inference_mode():
            for item in items:
                softmax = trainer.predict_preprocessed_data_return_seg_and_softmax(item["data"], do_mirroring=False, mirror_axes=(),
                    use_sliding_window=True, step_size=.5, use_gaussian=True, all_in_gpu=False, verbose=False, mixed_precision=True)[1].astype(np.float32)
                item["softmax"] = softmax if item["softmax"] is None else item["softmax"]+softmax
        print(f"{variant} source checkpoint {record['fold']} completed ({len(items)} fixed inputs)", flush=True)
        del checkpoint
        torch.cuda.empty_cache()
    metrics = []
    for item in items:
        softmax = (item["softmax"]/len(checkpoints)).transpose([0]+[axis+1 for axis in trainer.transpose_backward])
        exported = run_out / "output_nifti" / f"{item['image_id']}_{item['mode']}.nii.gz"
        exported.parent.mkdir(parents=True, exist_ok=True)
        export = trainer.plans.get("segmentation_export_params", {})
        save_segmentation_nifti_from_softmax(softmax, str(exported), item["properties"], export.get("interpolation_order", 1),
            trainer.regions_class_order, None, None, None, None, force_separate_z=export.get("force_separate_z"), interpolation_order_z=export.get("interpolation_order_z", 0))
        canvas_labels = sitk.GetArrayFromImage(sitk.ReadImage(str(exported)))
        assert canvas_labels.shape == (1, 768, 1024)
        inverse = invert_full_frame if item["mode"] == "full_frame" else invert_context_labels
        raw = inverse(canvas_labels[0], item["transform"])
        outer = item["outer"] > 0
        predicted = raw.copy(); predicted[~outer] = 0
        location = item["location"]
        x, y, w, h = [location[k] for k in ["x", "y", "width", "height"]]
        native = predicted[y:y+h, x:x+w]
        folder = run_out / "region_masks" / item["mode"]
        folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(folder / f"{item['image_id']}.png"), native)
        K.write_overlay(item["gray"][y:y+h, x:x+w], outer[y:y+h, x:x+w], native,
                        run_out / "overlays" / f"{item['image_id']}_{item['mode']}.png", f"{item['image_id']} {variant} {item['mode']}")
        record = dict(image_id=item["image_id"], view=item["view"], variant=variant, mode=item["mode"],
                      outer_pixels=int(outer.sum()), inside_region_pixels=int((native > 0).sum()),
                      outer_coverage=float((predicted > 0).sum()/outer.sum()),
                      raw_outside_fraction=float(((raw > 0)&~outer).sum()/max(1, (raw > 0).sum())),
                      classes_present=int(sum(np.count_nonzero(native == c) >= 16 for c in [1, 2, 3])), **location)
        metrics.append(record)
    pd.DataFrame(metrics).to_csv(run_out / "technical_metrics.csv", index=False)
    print(f"{variant} technical outputs saved; no APN diagnosis loaded", flush=True)


def summarize():
    require_lock()
    summaries, agreements = [], []
    passed = []
    for mode in MODES:
        model_summary = []
        frames = {}
        for variant in VARIANTS:
            frame = pd.read_csv(OUT / variant / "technical_metrics.csv")
            frame = frame[frame["mode"].eq(mode)]
            assert len(frame) == 32
            summary = dict(mode=mode, variant=variant, images=len(frame), nonempty_rate=float(frame.inside_region_pixels.ge(64).mean()),
                           median_outer_coverage=float(frame.outer_coverage.median()), two_region_rate=float(frame.classes_present.ge(2).mean()))
            model_summary.append(summary); summaries.append(summary)
            frames[variant] = frame
        dice_union, dice_class = [], []
        for image_id in frames[VARIANTS[0]].image_id:
            masks = [cv2.imread(str(OUT / variant / "region_masks" / mode / f"{image_id}.png"), 0) for variant in VARIANTS]
            assert masks[0].shape == masks[1].shape
            if np.any(masks[0]) or np.any(masks[1]):
                dice_union.append(K.dice(masks[0] > 0, masks[1] > 0))
                per_class = [K.dice(masks[0] == c, masks[1] == c) for c in [1, 2, 3] if (masks[0] == c).any() or (masks[1] == c).any()]
                dice_class.append(float(np.mean(per_class)))
        union = float(np.median(dice_union)) if dice_union else 0.
        classes = float(np.median(dice_class)) if dice_class else 0.
        decision = bool(all(s["nonempty_rate"] >= .8 and s["median_outer_coverage"] >= .5 and s["two_region_rate"] >= .8 for s in model_summary) and union >= .6 and classes >= .6)
        agreements.append(dict(mode=mode, median_union_dice=union, median_present_class_dice=classes,
                               either_nonempty_images=len(dice_union), necessary_technical_gates_passed=decision))
        if decision:
            passed.append(mode)
    result = dict(diagnostic_labels_read=False, source_models=summaries, intermodel_agreement=agreements,
                  eligible_input_modes=passed, classification_performed=False,
                  next_action="Review predicted anatomy before any APN classification" if passed else "Exclude unverified model compartments; do not manufacture anatomy using brightness")
    write_json(OUT / "decision.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "summarize"])
    parser.add_argument("--variant", choices=VARIANTS)
    args = parser.parse_args()
    if args.stage == "run":
        if args.variant is None:
            parser.error("run requires --variant")
        run(args.variant)
    else:
        globals()[args.stage]()
