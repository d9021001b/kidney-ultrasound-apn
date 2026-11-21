"""Route the added linear-patient images with the probe detector (as p2_19) and append the 41 patients to the paper-2 cohort."""
import json
import cv2, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import p2_02_probe_detector as d
from nckuh_probe_blind_sector import image_frame
from p2_common import SEED, P2, STATE_SIDES, load_cohort
from p2_18_add_two_convex_patients import EXT
from p2_20_add_linear_patients import assemble

imgs, wb, _ = assemble()
patients, kidneys, images = load_cohort()
assert not set(imgs.patient_alias) & set(patients.patient_alias)
images = images.merge(patients[["patient_alias", "role", "workbook_probe"]], on="patient_alias", validate="many_to_one")
from p2_18_add_two_convex_patients import new_images
_old = images.image_id.str[1:].astype(int) <= 1362
_conv = new_images().set_index("image_id").path
frames = np.zeros((len(images), d.SIZE, d.SIZE), np.uint8)
frames[_old.to_numpy()] = d.load_frames(images[_old])
for i in np.flatnonzero(~_old.to_numpy()):
    frames[i] = cv2.resize(cv2.cvtColor(image_frame(_conv[images.image_id.iloc[i]], panel=True), cv2.COLOR_BGR2GRAY), (d.SIZE, d.SIZE), interpolation=cv2.INTER_AREA)
single = images.workbook_probe.isin(("convex", "linear")).to_numpy()
y = images.workbook_probe.eq("convex").to_numpy().astype(np.float32)
held = images.role.eq("exploratory_test").to_numpy()
pool = np.flatnonzero(single & ~held & ~images.role.eq("added_388").to_numpy())
nf = np.zeros((len(imgs), d.SIZE, d.SIZE), np.uint8)
for i, r in enumerate(imgs.itertuples()):
    nf[i] = cv2.resize(cv2.cvtColor(image_frame(r.path, panel=r.source_type == "panel_with_mask"), cv2.COLOR_BGR2GRAY), (d.SIZE, d.SIZE), interpolation=cv2.INTER_AREA)
allf = np.concatenate([frames, nf]); idx = np.arange(len(frames), len(allf))
prob = np.zeros(len(imgs))
for k, (fit_idx, _) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(pool, y[pool], images.patient_alias.iloc[pool])):
    prob += d.predict(d.fit(frames, y, pool[fit_idx], SEED + k), allf, idx) / 5
imgs["p_convex"] = prob.round(4)
imgs["route"] = np.where(prob >= d.UPPER, "convex", np.where(prob <= d.LOWER, "linear", "unknown"))
print(imgs.groupby("source_type").route.value_counts())

qc = pd.read_csv(EXT / "extraction_qc_added_add41lin.csv")[["image_id", "height", "width", "roi_height", "roi_width", "mask_pixels"]]
wbp = wb.loc[imgs.workbook_key.unique()]
pat = pd.DataFrame({"patient_alias": imgs.drop_duplicates("patient_alias").patient_alias.to_numpy()})
key = imgs.drop_duplicates("patient_alias").set_index("patient_alias").workbook_key
pat["DMSA_APN_v1"] = np.nan; pat["role"] = "added_388"
pat["age"] = pat.patient_alias.map(key).map(wb.age).astype(float)
pat["DMSA_APN"] = pat.patient_alias.map(key).map(wb.DMSA_APN).str.upper()
pat["workbook_probe"] = pat.patient_alias.map(key).map(wb.Probe).str.lower()
add_i = imgs[["image_id", "patient_alias", "kidney_alias", "side", "view", "source_type"]].merge(qc, on="image_id")
add_i["cache_row"] = add_i.image_id.str[1:].astype(int) - 1
state = pat.set_index("patient_alias").DMSA_APN
side = {"left": 0, "right": 1}
add_i["kidney_APN"] = [STATE_SIDES[state[p]][side[s]] for p, s in zip(add_i.patient_alias, add_i.side)]
add_k = add_i[["kidney_alias", "patient_alias", "side", "kidney_APN"]].drop_duplicates()
cohort = P2 / "cohort"
for name, extra in (("patients.csv", pat), ("kidneys.csv", add_k), ("images.csv", add_i)):
    old = pd.read_csv(cohort / name)
    pd.concat([old, extra[old.columns]]).to_csv(cohort / name, index=False)
rt = pd.read_csv(P2 / "router/image_routes.csv")
r = imgs[["image_id", "patient_alias", "kidney_alias"]].copy()
r["role"] = "added_388"; r["workbook_probe"] = "linear"; r["detector_convex_probability"] = imgs.p_convex; r["geometry_convex_probability"] = np.nan; r["route"] = imgs.route
pd.concat([rt, r[rt.columns]]).to_csv(P2 / "router/image_routes.csv", index=False)
pd.DataFrame({"patient_alias": pat.patient_alias, "workbook_id": pat.patient_alias.map(key)}).to_csv(EXT / "added_linear_patient_linkage_private.csv", index=False)
s = json.loads((cohort / "cohort_summary.json").read_text())
s.update(patients=len(pd.read_csv(cohort / "patients.csv")), kidneys=len(pd.read_csv(cohort / "kidneys.csv")), images=len(pd.read_csv(cohort / "images.csv")), added_linear_patients=len(pat),
         note_added_linear="41 linear-only patients added on 2026-11-17: 36 with predicted component masks, 5 with panel masks; role added_388.")
(cohort / "cohort_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
print(s["patients"], s["kidneys"], s["images"])
