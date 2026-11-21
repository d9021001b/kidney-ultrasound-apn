"""Paper 2, step 13: results under paper 1's validation design (mean over three fixed development sets).

Reads the three draws in `draws_paper1_split/` (seeds 20261102-04; each draw: 60% train, 20% development
scored, 20% held out and never scored; four-fold patient-grouped selection inside the training part only)
and writes the mean and SD of AUROC and AP over the three development sets for every model.
"""

import json

import numpy as np
import pandas as pd

from p2_common import DRAW_SEEDS, P2

BASE = P2 / "draws_paper1_split"
rows = []
for d, seed in enumerate(DRAW_SEEDS):
    folder = BASE / f"draw{d}_seed{seed}"
    table = pd.read_csv(folder / "summary_table.csv")
    for r in table.itertuples():
        rows.append((r.probe, r.model.strip(), d, r.auroc, r.ap))
    for name, sub in (("green_saab", "results.json"), ("nested_saab", "results.json")):
        path = folder / name / sub
        if not path.exists():
            continue
        groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
        for key, v in groups.items():
            if name == "green_saab":
                probe, model = key.split("/")
                label = {"saab_own": "Saab (Green learning), own view", "saab_paired": "Saab (Green learning), paired view",
                         "fusion_saab_frozen": "Fixed rank mean: Saab + nested frozen"}.get(model, model)
            else:
                probe, label = key, "Nested selection with Saab candidate added"
            rows.append((probe, label, d, v["auroc"], v["ap"]))
frame = pd.DataFrame(rows, columns=["probe", "model", "draw", "auroc", "ap"])
out = (frame.groupby(["probe", "model"], sort=False)
       .agg(n_draws=("draw", "count"), auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
            ap_mean=("ap", "mean"), ap_sd=("ap", "std")).reset_index())
out.to_csv(BASE / "paper1_design_summary.csv", index=False)
frame.to_csv(BASE / "paper1_design_per_draw.csv", index=False)
for probe, sub in out.groupby("probe", sort=False):
    print(f"\n{probe}")
    for r in sub.itertuples():
        print(f"| {r.model} | {r.auroc_mean:.3f} ({r.auroc_sd:.3f}) | {r.ap_mean:.3f} ({r.ap_sd:.3f}) | n={r.n_draws} |")
