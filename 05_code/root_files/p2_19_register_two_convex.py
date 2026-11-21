"""Route the 5 added images with the probe detector (five fold models, as in p2_02) and append the two patients to the paper-2 cohort.
Patients: P384 (workbook id redacted, DMSA L, age 16.09), P385 (workbook id redacted, DMSA N, age 17.38); role 'added_388' (no earlier role)."""
import json
import cv2, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import p2_02_probe_detector as d
from nckuh_probe_blind_sector import image_frame
from p2_common import SEED, P2, STATE_SIDES, load_cohort
from p2_18_add_two_convex_patients import EXT, PATIENTS, new_images

patients, kidneys, images = load_cohort()
assert "P384" not in set(patients.patient_alias)
images = images.merge(patients[["patient_alias", "role", "workbook_probe"]], on="patient_alias", validate="many_to_one")
frames = d.load_frames(images)
single = images.workbook_probe.isin(("convex", "linear")).to_numpy()
y = images.workbook_probe.eq("convex").to_numpy().astype(np.float32)
held = images.role.eq("exploratory_test").to_numpy()
pool = np.flatnonzero(single & ~held)
new = new_images()
nf = np.zeros((len(new), d.SIZE, d.SIZE), np.uint8)
for i, p in enumerate(new.path):
    nf[i] = cv2.resize(cv2.cvtColor(image_frame(p, panel=True), cv2.COLOR_BGR2GRAY), (d.SIZE, d.SIZE), interpolation=cv2.INTER_AREA)
allf = np.concatenate([frames, nf]); idx = np.arange(len(frames), len(allf))
prob = np.zeros(len(new))
for k, (fit_idx, _) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(pool, y[pool], images.patient_alias.iloc[pool])):
    prob += d.predict(d.fit(frames, y, pool[fit_idx], SEED + k), allf, idx) / 5
new["p_convex"] = prob.round(4)
new["route"] = np.where(prob >= d.UPPER, "convex", np.where(prob <= d.LOWER, "linear", "unknown"))
print(new.drop(columns="path").to_string())
assert (new.route == "convex").all(), "an added image was not routed to convex"

qc = pd.read_csv(EXT / "extraction_qc_added.csv")[["image_id", "height", "width", "roi_height", "roi_width", "mask_pixels"]]
wb = {"P384": ("L", 16.09), "P385": ("N", 17.38)}
add_p = pd.DataFrame([{"patient_alias": a, "DMSA_APN_v1": np.nan, "role": "added_388", "age": wb[a][1], "DMSA_APN": wb[a][0], "workbook_probe": "convex"} for a in PATIENTS])
add_i = new.drop(columns=["path", "p_convex", "route"]).merge(qc, on="image_id")
add_i["cache_row"] = add_i.image_id.str[1:].astype(int) - 1
side = {"left": 0, "right": 1}
add_i["kidney_APN"] = [STATE_SIDES[wb[p][0]][side[s]] for p, s in zip(add_i.patient_alias, add_i.side)]
add_k = add_i[["kidney_alias", "patient_alias", "side", "kidney_APN"]].drop_duplicates()
cohort = P2 / "cohort"
for name, extra in (("patients.csv", add_p), ("kidneys.csv", add_k), ("images.csv", add_i)):
    old = pd.read_csv(cohort / name)
    pd.concat([old, extra[old.columns]]).to_csv(cohort / name, index=False)
routes = pd.read_csv(P2 / "router/image_routes.csv")
pd.concat([routes, pd.DataFrame({"image_id": new.image_id, "p_convex": new.p_convex, "route": new.route}).reindex(columns=routes.columns)]).to_csv(P2 / "router/image_routes.csv", index=False)
pd.DataFrame([{"patient_alias": a, "patient_id": pid} for a, pid in PATIENTS.items()]).to_csv(EXT / "added_patient_linkage_private.csv", index=False)
s = json.loads((cohort / "cohort_summary.json").read_text())
s.update(patients=len(pd.read_csv(cohort / "patients.csv")), kidneys=len(pd.read_csv(cohort / "kidneys.csv")), images=len(pd.read_csv(cohort / "images.csv")),
         added_convex_patients=2, note_added="Two convex-only patients of the (2) workbook (not in the 383 lock) were added on 2026-11-17 with role added_388.")
(cohort / "cohort_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
print(s["patients"], s["kidneys"], s["images"])
