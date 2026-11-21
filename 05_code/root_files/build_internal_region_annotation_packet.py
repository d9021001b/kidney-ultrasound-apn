"""Build a label-blinded kidney internal-region annotation packet.

The packet is a local validation instrument, not a disease-classification
experiment and not a prefilled semantic segmentation result.  It selects only
from the frozen whole-kidney ROI assets in ``renal_random_validation`` and
never reads or exports diagnosis labels while selecting tasks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import renal_random_validation as R


ROOT = R.ROOT
BASE = R.OUT
OUT = ROOT / "CYCH_internal_region_annotation_packet_20260914"
ARCHIVE = ROOT / f"{OUT.name}_blinded.zip"
VALIDATOR_SOURCE = ROOT / "validate_internal_region_annotations.py"
SELECTION_SEED = 20260914
VIEWS = ("LL", "LT", "RL", "RT")
CORE_TARGETS = (("existing_yolo", 5), ("manual_polygon_fallback", 3))
REPEATS_PER_VIEW = 2
MIN_REPEAT_SEPARATION = 5
PUBLIC_FILES = (
    "README.md",
    "PROVENANCE.md",
    "MODEL_CANDIDATE_AUDIT.md",
    "annotation_tasks_blinded.csv",
    "ANNOTATION_STATUS_TEMPLATE.csv",
    "annotation_schema.json",
    "selection_summary_blinded.json",
    "figure_resolution_audit.json",
    "validate_internal_region_annotations.py",
)
PUBLIC_DIRECTORIES = ("images", "outer_roi_masks", "outline_previews", "annotation_templates", "figures_600dpi")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _rank(*parts: object) -> int:
    message = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest(), "big")


def load_frozen_manifest() -> pd.DataFrame:
    manifest_path = BASE / "image_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"patient_id": str})
    required = {"image_id", "patient_id", "patient_alias", "view", "mask_source", "mask_sha256"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Frozen image manifest lacks columns: {sorted(missing)}")
    if manifest.image_id.duplicated().any():
        raise ValueError("Frozen image manifest contains duplicate image ids")
    if not set(manifest.view).issuperset(VIEWS):
        raise ValueError("Frozen image manifest does not contain all required views")
    return manifest.copy()


def select_core_tasks(manifest: pd.DataFrame) -> pd.DataFrame:
    """Deterministically select 32 unique-patient tasks without using labels.

    Selection is balanced by view and outer-ROI source solely so the annotation
    validation includes both stored whole-kidney contour sources.  No outcome,
    diagnosis, date, filename, or patient characteristic is used in ranking.
    """
    columns = ["image_id", "patient_id", "patient_alias", "view", "mask_source", "mask_sha256"]
    working = manifest.loc[:, columns].copy()
    selected: list[pd.Series] = []
    used_patients: set[str] = set()
    for view in VIEWS:
        for mask_source, target in CORE_TARGETS:
            candidates = working.loc[(working.view == view) & (working.mask_source == mask_source)].copy()
            candidates["_rank"] = candidates.image_id.map(
                lambda image_id: _rank(SELECTION_SEED, "core", view, mask_source, image_id)
            )
            candidates = candidates.sort_values(["_rank", "image_id"])
            picked = 0
            for _, candidate in candidates.iterrows():
                patient_id = str(candidate.patient_id)
                if patient_id in used_patients:
                    continue
                selected.append(candidate.drop(labels="_rank"))
                used_patients.add(patient_id)
                picked += 1
                if picked == target:
                    break
            if picked != target:
                raise RuntimeError(
                    f"Could not select {target} unique-patient images for view={view}, mask_source={mask_source}; got {picked}"
                )
    core = pd.DataFrame(selected).reset_index(drop=True)
    expected = len(VIEWS) * sum(target for _, target in CORE_TARGETS)
    if len(core) != expected or core.patient_id.duplicated().any():
        raise AssertionError("Core selection invariant failed")
    return core


def select_hidden_repeats(core: pd.DataFrame) -> pd.DataFrame:
    repeats: list[pd.Series] = []
    for view in VIEWS:
        candidates = core.loc[core.view.eq(view)].copy()
        candidates["_rank"] = candidates.image_id.map(lambda image_id: _rank(SELECTION_SEED, "repeat", view, image_id))
        candidates = candidates.sort_values(["_rank", "image_id"])
        if len(candidates) < REPEATS_PER_VIEW:
            raise RuntimeError(f"Insufficient core tasks to create repeats for {view}")
        for _, candidate in candidates.iloc[:REPEATS_PER_VIEW].iterrows():
            repeats.append(candidate.drop(labels="_rank"))
    result = pd.DataFrame(repeats).reset_index(drop=True)
    if len(result) != len(VIEWS) * REPEATS_PER_VIEW:
        raise AssertionError("Repeat selection invariant failed")
    return result


def order_task_items(core: pd.DataFrame, repeats: pd.DataFrame) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for _, row in core.iterrows():
        items.append({"kind": "core", **row.to_dict()})
    for _, row in repeats.iterrows():
        items.append({"kind": "hidden_repeat", **row.to_dict()})
    pending = sorted(
        items,
        key=lambda item: (_rank(SELECTION_SEED, "display", item["kind"], item["image_id"]), str(item["image_id"])),
    )
    ordered: list[dict[str, object]] = []
    while pending:
        recent_image_ids = {str(item["image_id"]) for item in ordered[-MIN_REPEAT_SEPARATION:]}
        eligible = [item for item in pending if str(item["image_id"]) not in recent_image_ids]
        choice = eligible[0] if eligible else pending[0]
        pending.remove(choice)
        ordered.append(choice)
    if len(ordered) != len(core) + len(repeats):
        raise AssertionError("Display ordering invariant failed")
    return ordered


def build_task_registry(manifest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    core = select_core_tasks(manifest)
    repeats = select_hidden_repeats(core)
    ordered = order_task_items(core, repeats)
    for index, item in enumerate(ordered, 1):
        item["task_id"] = f"A{index:03d}"
        item["display_order"] = index
    core_task_id = {str(item["image_id"]): str(item["task_id"]) for item in ordered if item["kind"] == "core"}
    public_rows: list[dict[str, object]] = []
    private_rows: list[dict[str, object]] = []
    for item in ordered:
        duplicate = core_task_id[str(item["image_id"])] if item["kind"] == "hidden_repeat" else ""
        public_rows.append(
            {
                "task_id": item["task_id"],
                "display_order": item["display_order"],
                "view": item["view"],
            }
        )
        private_rows.append(
            {
                "task_id": item["task_id"],
                "display_order": item["display_order"],
                "image_id": item["image_id"],
                "patient_id": item["patient_id"],
                "patient_alias": item["patient_alias"],
                "view": item["view"],
                "mask_source": item["mask_source"],
                "task_type": item["kind"],
                "duplicate_of_task_id": duplicate,
                "mask_sha256": item["mask_sha256"],
            }
        )
    public = pd.DataFrame(public_rows)
    private = pd.DataFrame(private_rows)
    summary = {
        "selection_seed": SELECTION_SEED,
        "core_unique_images": int(len(core)),
        "core_unique_patients": int(core.patient_id.nunique()),
        "hidden_repeat_tasks": int(len(repeats)),
        "total_blinded_tasks": int(len(public)),
        "selection_rule": "Deterministic ranking by image_id, balanced by view and stored outer-ROI source; diagnosis labels, dates, filenames, and patient characteristics were not used.",
        "core_by_view": core.view.value_counts().sort_index().to_dict(),
        "core_by_outer_roi_source": core.mask_source.value_counts().sort_index().to_dict(),
    }
    return public, private, summary


def outline_preview(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def _fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Resize both small and large source images to a consistently reviewable panel."""
    scale = min(width / image.width, height / image.height)
    dimensions = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(dimensions, Image.Resampling.LANCZOS)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def build_selection_figure(output: Path, task_registry: pd.DataFrame) -> dict[str, object]:
    """Create an internal 600-dpi QA figure without diagnosis labels or identities."""
    sample_tasks = task_registry.sort_values("display_order").head(8)
    tile_width, tile_height = 2400, 900
    canvas = Image.new("RGB", (tile_width * 2, tile_height * 4), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = _font(48)
    label_font = _font(36)
    for index, row in enumerate(sample_tasks.itertuples(index=False)):
        column, line = index % 2, index // 2
        task_id = str(row.task_id)
        raw = Image.open(output / "images" / f"{task_id}.png").convert("RGB")
        preview = Image.open(output / "outline_previews" / f"{task_id}.png").convert("RGB")
        raw = _fit_image(raw, 1120, 700)
        preview = _fit_image(preview, 1120, 700)
        x0, y0 = column * tile_width, line * tile_height
        draw.text((x0 + 30, y0 + 22), f"{task_id} ({row.view})", fill="black", font=title_font)
        canvas.paste(raw, (x0 + (1120 - raw.width) // 2 + 30, y0 + 110 + (700 - raw.height) // 2))
        canvas.paste(preview, (x0 + 1240 + (1120 - preview.width) // 2, y0 + 110 + (700 - preview.height) // 2))
        draw.text((x0 + 30, y0 + 830), "source image", fill="black", font=label_font)
        draw.text((x0 + 1240, y0 + 830), "provided whole-kidney outline", fill="black", font=label_font)
    figure_dir = output / "figures_600dpi"
    png_path = figure_dir / "Figure_AR1_blinded_annotation_selection.png"
    tiff_path = figure_dir / "Figure_AR1_blinded_annotation_selection.tif"
    canvas.save(png_path, dpi=(600, 600))
    canvas.save(tiff_path, dpi=(600, 600), compression="tiff_lzw")
    with Image.open(tiff_path) as image:
        tiff_dpi = tuple(float(value) for value in image.info.get("dpi", (0, 0)))
        dimensions = list(image.size)
    return {
        "figure": "Figure_AR1_blinded_annotation_selection",
        "purpose": "Internal label-blinded annotation packet QA only; not a manuscript result.",
        "png": {"path": str(png_path.name), "pixels": dimensions, "requested_dpi": [600, 600]},
        "tiff": {"path": str(tiff_path.name), "pixels": dimensions, "embedded_dpi": tiff_dpi},
    }


def annotation_readme() -> str:
    return """# 腎臟超音波內部區域盲化標註包

這份資料包是用於建立本地、可驗證的腎臟內部區域標註基準。它不是疾病分類實驗，未包含診斷標籤、病人代號、日期、原始檔名或任何已填入的內部區域分割結果。

## 標註範圍

每個工作項目有三個檔案：

- `images/Axxx.png`：灰階來源影像。
- `outer_roi_masks/Axxx.png`：既有的全腎臟外輪廓，白色為可標註範圍。
- `outline_previews/Axxx.png`：同一影像加上外輪廓，僅供定位。

請只在白色全腎臟外輪廓內標註。請將結果另存為與 `annotation_templates/` 中相同大小、單通道、二值 PNG：背景為 0，標註為 255。檔名必須為 `Axxx_cec.png`、`Axxx_medulla.png`、`Axxx_cortex.png` 及 `Axxx_uncertain.png`。四個遮罩不可重疊。

## 解剖術語與保守原則

本包採用 Open Kidney Ultrasound Data Set 的三個內部區域術語：`Central Echo Complex (CEC)`、`Medulla`、`Cortex`。CEC 為超音波中的廣義中央回音複合區，不能改稱腎竇、腎盂或病灶。不要依亮暗、量測十字、虛線、標尺或模型顯著區自行推論發炎、疤痕或其他病理。

- `CEC`：僅標註在影像中可辨識的中央回音複合區。
- `Medulla`：僅標註可辨識的髓質／腎錐體區域；可有多個互不相連區域。
- `Cortex`：僅標註可辨識的皮質實質區域；不可用全腎邊緣侵蝕或幾何環帶替代。
- `Uncertain`：可標示在無法可靠歸類、但需要排除於其他三類之外的像素；可留白。

若某類在該影像中不可辨識，保留該類遮罩為空白，並於 `annotation_status.csv` 填寫 `not_identifiable`。請勿為了完整性而強制描繪。外輪廓若不適合標註，請在同一表格的 `outer_roi_status` 填寫 `requires_correction` 或 `unusable`，並在註解說明原因。

## 建議交付結構

請為每位標註者建立獨立資料夾，例如 `returned/rater_1/`，內含 40 組遮罩與一份由 `ANNOTATION_STATUS_TEMPLATE.csv` 改名而成的 `annotation_status.csv`。同一位標註者應依顯示順序逐一完成所有工作項目，將每張圖視為獨立案例。

完成後可執行：

```powershell
python validate_internal_region_annotations.py validate --annotations returned/rater_1 --output returned/rater_1_qc
```

驗證器會檢查尺寸、全腎外輪廓外漏、類別重疊、遮罩狀態矛盾，並輸出覆蓋圖。結構檢查通過不代表解剖學真值成立；至少需要兩名獨立標註者，再比較 inter-rater Dice。一小部分工作項目經過隱藏重複，僅供內部評估同一標註者一致性。

## 使用界線

此包只適用於本專案的本地標註與方法學驗證。它不能作為外部驗證、臨床診斷或疾病病灶標記的證據。未來若有安全且可追溯的公開內部區域模型，只能在這份固定、盲化的本地基準上評估，且不得以疾病結果挑選分割版本。
"""


def provenance_text() -> str:
    return """# Provenance and Scope

The anatomical vocabulary in this packet follows the Open Kidney Ultrasound Data Set: capsule, Central Echo Complex, medulla, and cortex. This project copied no Open Kidney ultrasound image, annotation mask, or model weight into the packet. The packet contains only re-encoded CYCH cropped images and the pre-existing CYCH whole-kidney ROI.

Reference: Singla R, et al. The Open Kidney Ultrasound Data Set. arXiv:2206.06657. https://arxiv.org/abs/2206.06657

Official repository: https://github.com/rsingla92/kidneyUS

The cloned reference repository is pinned locally at commit `73d671ee6143b72e0606884eb9ab3353f0224be0` and declares CC BY-NC-SA 4.0. Any future use of its data, annotations, or weights must first be reviewed for attribution, non-commercial, and ShareAlike obligations.
"""


def candidate_audit_text() -> str:
    return """# Semantic Segmentation Candidate Audit

## Eligible research candidate

`Open Kidney Ultrasound Data Set / kidneyUS` is the only candidate located during the current audit that has a documented B-mode ultrasound taxonomy matching the required internal regions: Central Echo Complex, medulla, and cortex, alongside a kidney capsule. The repository reports expert polygon annotations and points to pre-trained segmentation weights.

Status: **not imported, not executed, and not used to generate any packet mask.** The publicly linked archive is large and was not available through a reproducible verified download path in this environment. A browser safety-screen indication was also present, which is not a malware determination but requires manual verification and a safety scan before any download/import. The process must not bypass that protection.

## Preconditions before model evaluation

1. Obtain a direct, authorized copy with a recorded source URL, archive hash, model-file inventory, license review, and local malware scan.
2. Verify the model architecture and label-index mapping against the source implementation.
3. Evaluate only against the fixed local clinician annotations from this packet, without diagnosis labels or outcome-based model selection.
4. Report per-class Dice, absence/not-identifiable handling, and failure cases separately by view and outer-ROI source.
5. Do not call CEC renal sinus/pelvis, do not infer pathology from segmentation, and do not use the result as an external clinical validation.

Other reviewed directions were not activated: broad promptable medical SAM variants do not provide validated renal CEC/medulla/cortex semantics without task-specific supervision; CT renal-structure models do not transfer as a validated B-mode ultrasound compartment model; and published internal-region models without released, verifiable weights are research references rather than executable candidates.
"""


def write_packet_assets(output: Path, public: pd.DataFrame, private: pd.DataFrame) -> None:
    for name in ("images", "outer_roi_masks", "outline_previews", "annotation_templates", "figures_600dpi"):
        (output / name).mkdir(parents=True, exist_ok=True)
    private_by_task = private.set_index("task_id")
    for task in public.itertuples(index=False):
        task_id = str(task.task_id)
        source = private_by_task.loc[task_id]
        image_id = str(source.image_id)
        gray = cv2.imread(str(BASE / "native" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(BASE / "masks" / f"{image_id}.png"), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape != mask.shape:
            raise RuntimeError(f"Frozen image/mask asset unavailable or incompatible: {image_id}")
        if sha256(BASE / "masks" / f"{image_id}.png") != str(source.mask_sha256):
            raise RuntimeError(f"Frozen outer-ROI hash mismatch: {image_id}")
        binary_mask = (mask > 0).astype(np.uint8)
        if int(binary_mask.sum()) < 64:
            raise RuntimeError(f"Frozen outer ROI is unusably small: {image_id}")
        # Re-encoding strips source file metadata and uses only blinded task names.
        cv2.imwrite(str(output / "images" / f"{task_id}.png"), gray)
        cv2.imwrite(str(output / "outer_roi_masks" / f"{task_id}.png"), binary_mask * 255)
        cv2.imwrite(str(output / "outline_previews" / f"{task_id}.png"), outline_preview(gray, binary_mask))
        for class_name in ("cec", "medulla", "cortex", "uncertain"):
            cv2.imwrite(str(output / "annotation_templates" / f"{task_id}_{class_name}.png"), np.zeros_like(gray))


def write_public_files(output: Path, public: pd.DataFrame, private: pd.DataFrame, summary: dict[str, object]) -> None:
    public.to_csv(output / "annotation_tasks_blinded.csv", index=False)
    private.to_csv(output / "private_task_key.csv", index=False)
    (output / "README.md").write_text(annotation_readme(), encoding="utf-8")
    (output / "PROVENANCE.md").write_text(provenance_text(), encoding="utf-8")
    (output / "MODEL_CANDIDATE_AUDIT.md").write_text(candidate_audit_text(), encoding="utf-8")
    status = pd.DataFrame(
        {
            "task_id": public.task_id,
            "image_quality": "",
            "outer_roi_status": "",
            "cec_status": "not_annotated",
            "medulla_status": "not_annotated",
            "cortex_status": "not_annotated",
            "comments": "",
        }
    )
    status.to_csv(output / "ANNOTATION_STATUS_TEMPLATE.csv", index=False)
    write_json(
        output / "annotation_schema.json",
        {
            "image": "images/Axxx.png",
            "outer_roi": "outer_roi_masks/Axxx.png",
            "required_annotation_masks": [
                "Axxx_cec.png",
                "Axxx_medulla.png",
                "Axxx_cortex.png",
                "Axxx_uncertain.png",
            ],
            "encoding": {"background": 0, "foreground": 255, "channels": 1, "format": "PNG"},
            "inside_outer_roi_required": True,
            "class_overlap_allowed": False,
            "semantic_classes": {
                "cec": "Central Echo Complex. Broad ultrasound central echogenic region; not renal sinus, renal pelvis, or pathology.",
                "medulla": "Only visually identifiable medullary/pyramidal tissue.",
                "cortex": "Only visually identifiable cortical parenchyma; not a geometric outer ring.",
                "uncertain": "Optional pixels that should not be assigned to a semantic class.",
            },
            "scope": "Local annotation validation only; no disease labels were used for selection.",
        },
    )
    write_json(output / "selection_summary_blinded.json", summary)
    private_note = """# Confidential linkage key

`private_task_key.csv` maps blinded task IDs to the internal CYCH image and patient records. It is intentionally excluded from the blinded ZIP. It contains no diagnosis label, but it must remain inside the controlled project workspace and must not be sent to annotators or external collaborators.
"""
    (output / "PRIVATE_LINKAGE_KEY_DO_NOT_SHARE.md").write_text(private_note, encoding="utf-8")
    shutil.copy2(VALIDATOR_SOURCE, output / "validate_internal_region_annotations.py")


def build_archive(output: Path, archive: Path) -> None:
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for name in PUBLIC_FILES:
            path = output / name
            package.write(path, arcname=path.name)
        for directory in PUBLIC_DIRECTORIES:
            for path in sorted((output / directory).rglob("*")):
                if path.is_file():
                    package.write(path, arcname=str(path.relative_to(output)))


def require_clean_target(output: Path, archive: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if archive.exists():
        raise FileExistsError(f"Archive already exists: {archive}")
    if not VALIDATOR_SOURCE.exists():
        raise FileNotFoundError(f"Validator source is missing: {VALIDATOR_SOURCE}")


def build(output: Path = OUT, archive: Path = ARCHIVE) -> tuple[Path, Path]:
    output, archive = Path(output), Path(archive)
    require_clean_target(output, archive)
    manifest = load_frozen_manifest()
    public, private, summary = build_task_registry(manifest)
    output.mkdir(parents=True)
    try:
        write_public_files(output, public, private, summary)
        write_packet_assets(output, public, private)
        figure_audit = build_selection_figure(output, public)
        write_json(output / "figure_resolution_audit.json", figure_audit)
        write_json(
            output / "source_lock.json",
            {
                "frozen_image_manifest": str(BASE / "image_manifest.csv"),
                "frozen_image_manifest_sha256": sha256(BASE / "image_manifest.csv"),
                "builder_sha256": sha256(Path(__file__)),
                "validator_sha256": sha256(VALIDATOR_SOURCE),
                "source_scope": "Only frozen native image and whole-kidney ROI assets were used. Diagnosis labels were not read or exported.",
            },
        )
        build_archive(output, archive)
    except Exception:
        # Retain partial outputs for inspection rather than silently deleting a clinical-data artifact.
        raise
    return output, archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, archive = build(args.output, args.archive)
    print(f"Built blinded annotation packet: {output}")
    print(f"Built blinded archive: {archive}")


if __name__ == "__main__":
    main()
