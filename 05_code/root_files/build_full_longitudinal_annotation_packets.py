"""Build label-blinded full-cohort longitudinal anatomy annotation packets.

This builder reads only patient linkage, view, image alias, and frozen mask
provenance columns. It explicitly does not read diagnosis labels, dates, or
source filenames. The public packets contain re-encoded Bxxx images only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "CYCH_random_validation_20260913"
SOURCE_MANIFEST = BASE / "image_manifest.csv"
VALIDATOR_SOURCE = ROOT / "CYCH_anatomy_artifact_annotation_packet_v2_20260914" / "validate_internal_region_annotations.py"
OUT = ROOT / "CYCH_full_longitudinal_annotation_packets_v1_20260914"
PRIMARY_ARCHIVE = ROOT / "CYCH_full_longitudinal_primary_v1_20260914_public.zip"
QC_ARCHIVE = ROOT / "CYCH_full_longitudinal_second_rater_qc_v1_20260914_public.zip"

SELECTION_SEED = 20260914
LONGITUDINAL_VIEWS = ("LL", "RL")
EXPECTED_IMAGES = 190
EXPECTED_PATIENTS = 98
QC_IMAGES = 38
MIN_PATIENT_SEPARATION = 5
CLASSES = ("cec", "medulla", "cortex", "artifact", "uncertain")
PUBLIC_ROOT_FILES = (
    "README.md",
    "ANNOTATION_STATUS_TEMPLATE.csv",
    "annotation_schema.json",
    "annotation_tasks_blinded.csv",
    "selection_summary_blinded.json",
    "figure_resolution_audit.json",
    "validate_internal_region_annotations.py",
)
PUBLIC_DIRECTORIES = (
    "images",
    "outer_roi_masks",
    "outline_previews",
    "annotation_templates",
    "figures_600dpi",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rank(*parts: object) -> int:
    message = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(message).digest(), "big")


def load_label_free_manifest() -> pd.DataFrame:
    columns = ["patient_id", "view", "image_id", "mask_source", "mask_sha256"]
    manifest = pd.read_csv(SOURCE_MANIFEST, usecols=columns, dtype={"patient_id": str})
    if manifest.image_id.duplicated().any():
        raise ValueError("Frozen image manifest contains duplicate image ids")
    longitudinal = manifest.loc[manifest.view.isin(LONGITUDINAL_VIEWS)].copy()
    if len(longitudinal) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES} longitudinal images, found {len(longitudinal)}")
    if longitudinal.patient_id.nunique() != EXPECTED_PATIENTS:
        raise ValueError(
            f"Expected longitudinal coverage of {EXPECTED_PATIENTS} patients, "
            f"found {longitudinal.patient_id.nunique()}"
        )
    return longitudinal.reset_index(drop=True)


def order_images(manifest: pd.DataFrame) -> pd.DataFrame:
    pending = manifest.copy()
    pending["_rank"] = pending.image_id.map(lambda value: rank(SELECTION_SEED, "primary", value))
    pending = pending.sort_values(["_rank", "image_id"]).to_dict("records")
    ordered: list[dict[str, object]] = []
    while pending:
        recent_patients = {str(item["patient_id"]) for item in ordered[-MIN_PATIENT_SEPARATION:]}
        eligible_index = next(
            (index for index, item in enumerate(pending) if str(item["patient_id"]) not in recent_patients),
            0,
        )
        ordered.append(pending.pop(eligible_index))
    table = pd.DataFrame(ordered).drop(columns="_rank")
    table.insert(0, "display_order", np.arange(1, len(table) + 1))
    table.insert(0, "task_id", [f"B{index:03d}" for index in range(1, len(table) + 1)])
    return table


def select_qc_subset(private: pd.DataFrame) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    used_patients: set[str] = set()
    target_per_view = QC_IMAGES // len(LONGITUDINAL_VIEWS)
    for view in LONGITUDINAL_VIEWS:
        candidates = private.loc[private.view.eq(view)].copy()
        candidates["_qc_rank"] = candidates.image_id.map(
            lambda value: rank(SELECTION_SEED, "qc-image", view, value)
        )
        candidates = candidates.sort_values(["_qc_rank", "image_id"])
        picked = 0
        for _, candidate in candidates.iterrows():
            patient_id = str(candidate.patient_id)
            if patient_id in used_patients:
                continue
            selected_rows.append(candidate)
            used_patients.add(patient_id)
            picked += 1
            if picked == target_per_view:
                break
        if picked != target_per_view:
            raise RuntimeError(f"Could not select {target_per_view} distinct-patient QC images for {view}")
    selected = pd.DataFrame(selected_rows)
    if len(selected) != QC_IMAGES or selected.patient_id.nunique() != QC_IMAGES:
        raise AssertionError("Second-rater QC subset must contain 38 distinct patients")
    selected["_qc_order"] = selected.image_id.map(lambda value: rank(SELECTION_SEED, "qc-order", value))
    selected = selected.sort_values(["_qc_order", "task_id"]).reset_index(drop=True)
    selected["display_order"] = np.arange(1, len(selected) + 1)
    return selected.drop(columns=["_qc_rank", "_qc_order"])


def public_registry(private: pd.DataFrame) -> pd.DataFrame:
    return private.loc[:, ["task_id", "display_order", "view"]].copy()


def outline_preview(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    preview = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(preview, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
    return preview


def create_asset_directories(packet: Path) -> None:
    for directory in PUBLIC_DIRECTORIES:
        (packet / directory).mkdir(parents=True, exist_ok=True)


def write_assets(packet: Path, private: pd.DataFrame) -> None:
    create_asset_directories(packet)
    for position, row in enumerate(private.itertuples(index=False), 1):
        task_id = str(row.task_id)
        image_id = str(row.image_id)
        image_path = BASE / "native" / f"{image_id}.png"
        mask_path = BASE / "masks" / f"{image_id}.png"
        if sha256(mask_path) != str(row.mask_sha256):
            raise RuntimeError(f"Frozen outer-ROI hash mismatch: {image_id}")
        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if gray is None or mask is None or gray.shape != mask.shape:
            raise RuntimeError(f"Frozen image/mask unavailable or incompatible: {image_id}")
        binary = (mask > 0).astype(np.uint8)
        if int(binary.sum()) < 64:
            raise RuntimeError(f"Whole-kidney ROI is too small: {image_id}")
        cv2.imwrite(str(packet / "images" / f"{task_id}.png"), gray)
        cv2.imwrite(str(packet / "outer_roi_masks" / f"{task_id}.png"), binary * 255)
        cv2.imwrite(str(packet / "outline_previews" / f"{task_id}.png"), outline_preview(gray, binary))
        for class_name in CLASSES:
            cv2.imwrite(
                str(packet / "annotation_templates" / f"{task_id}_{class_name}.png"),
                np.zeros_like(gray),
            )
        if position % 25 == 0 or position == len(private):
            print(f"{packet.name}: wrote {position}/{len(private)} tasks", flush=True)


def annotation_schema() -> dict[str, object]:
    return {
        "image": "images/Bxxx.png",
        "outer_roi": "outer_roi_masks/Bxxx.png",
        "required_annotation_masks": [f"Bxxx_{name}.png" for name in CLASSES],
        "encoding": {"background": 0, "foreground": 255, "channels": 1, "format": "PNG"},
        "inside_outer_roi_required": True,
        "class_overlap_rule": (
            "cec, medulla, cortex, and uncertain are mutually exclusive; artifact may overlap anatomy "
            "because display overlays are removed before feature extraction"
        ),
        "semantic_classes": {
            "cec": "Central Echo Complex; not renal sinus, renal pelvis, or pathology.",
            "medulla": "Only visually identifiable medullary or pyramidal tissue.",
            "cortex": "Only visually identifiable cortical parenchyma; not a geometric outer ring.",
            "artifact": "Only burned-in calipers, measurement lines, crosses, letters, or display overlays.",
            "uncertain": "Optional pixels that cannot be assigned confidently to an anatomy class.",
        },
        "scope": "Full-cohort longitudinal anatomy annotation; no diagnosis labels were read or exported.",
    }


def readme_text(packet_role: str, task_count: int) -> str:
    second_rater_note = (
        "本包是固定的第二評者品質控制子集，共 38 張影像且來自 38 位不同病人。"
        if packet_role == "second_rater_qc"
        else "本包是主評者完整縱切影像包，共 190 張影像，涵蓋嘉基論文確認的 98 位病人。"
    )
    return f"""# 嘉基全隊列縱切腎臟解剖與 artifact 盲標註包

{second_rater_note}

本包未包含疾病 label、病人代號、日期或原始檔名。每張影像均以 `Bxxx`
代號重新編碼。這是方法學標註資料，不是外部驗證資料，也不能單獨證明診斷效能。

## 需要繪製的遮罩

- `CEC`：僅標註可辨識的 Central Echo Complex，不得改稱腎竇、腎盂或病灶。
- `Medulla`：僅標註可辨識的髓質或腎錐體。
- `Cortex`：僅標註可辨識的皮質實質，不可用幾何外環替代。
- `Artifact`：只標註 caliper、量測十字、虛線、字母或儀器顯示疊印。
- `Uncertain`：無法可靠歸類且需要排除的像素，可留白。

所有遮罩必須是與來源影像相同尺寸的單通道二值 PNG，背景 0、前景 255，
並限制在 `outer_roi_masks/` 的白色全腎輪廓內。CEC、Medulla、Cortex 與
Uncertain 彼此不可重疊；Artifact 可與解剖區重疊，因為分析時會從解剖區
排除 artifact 與固定兩像素邊界。

## 回傳方式

將 `ANNOTATION_STATUS_TEMPLATE.csv` 政名為 `annotation_status.csv`，完成全部
{task_count} 個 task。每個類別狀態只能填 `present`、`absent` 或
`not_identifiable`，不得保留 `not_annotated`。若全腎外輪廓不適合使用，請在
`outer_roi_status` 填 `requires_correction` 或 `unusable` 並註明原因。

請把遮罩與 `annotation_status.csv` 放在同一個獨立評者資料夾。不要查看疾病
結果、模型分數或另一位評者的遮罩。可執行下列結構檢查：

```powershell
python validate_internal_region_annotations.py validate --annotations <returned-folder> --output <qc-folder>
```

結構檢查通過不等於解剖學有效性；正式特徵分析仍受 v2 protocol 的雙評者
可靠度與開發組門檻約束。
"""


def write_status_template(packet: Path, public: pd.DataFrame) -> None:
    table = pd.DataFrame(
        {
            "task_id": public.task_id,
            "image_quality": "",
            "outer_roi_status": "",
            "cec_status": "not_annotated",
            "medulla_status": "not_annotated",
            "cortex_status": "not_annotated",
            "artifact_status": "not_annotated",
            "comments": "",
        }
    )
    table.to_csv(packet / "ANNOTATION_STATUS_TEMPLATE.csv", index=False)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def build_montage(packet: Path, public: pd.DataFrame) -> dict[str, object]:
    samples = public.sort_values("display_order").head(8)
    tile_width, tile_height = 2400, 900
    canvas = Image.new("RGB", (tile_width * 2, tile_height * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(samples.itertuples(index=False)):
        column, line = index % 2, index // 2
        x0, y0 = column * tile_width, line * tile_height
        task_id = str(row.task_id)
        source = fit_image(Image.open(packet / "images" / f"{task_id}.png").convert("RGB"), 1120, 700)
        preview = fit_image(
            Image.open(packet / "outline_previews" / f"{task_id}.png").convert("RGB"),
            1120,
            700,
        )
        draw.text((x0 + 30, y0 + 22), f"{task_id} ({row.view})", fill="black", font=font(48))
        canvas.paste(source, (x0 + 30 + (1120 - source.width) // 2, y0 + 110 + (700 - source.height) // 2))
        canvas.paste(preview, (x0 + 1240 + (1120 - preview.width) // 2, y0 + 110 + (700 - preview.height) // 2))
        draw.text((x0 + 30, y0 + 830), "source image", fill="black", font=font(36))
        draw.text((x0 + 1240, y0 + 830), "whole-kidney outline", fill="black", font=font(36))
    png = packet / "figures_600dpi" / "Figure_FL1_blinded_longitudinal_packet.png"
    tif = packet / "figures_600dpi" / "Figure_FL1_blinded_longitudinal_packet.tif"
    canvas.save(png, dpi=(600, 600))
    canvas.save(tif, dpi=(600, 600), compression="tiff_lzw")
    with Image.open(tif) as image:
        embedded_dpi = [float(value) for value in image.info.get("dpi", (0, 0))]
        pixels = list(image.size)
    return {
        "purpose": "Blinded packet QA only; not a diagnostic-result figure.",
        "pixels": pixels,
        "png_requested_dpi": [600, 600],
        "tiff_embedded_dpi": embedded_dpi,
    }


def write_public_packet(packet: Path, private: pd.DataFrame, role: str) -> None:
    public = public_registry(private)
    packet.mkdir(parents=True)
    write_assets(packet, private)
    public.to_csv(packet / "annotation_tasks_blinded.csv", index=False)
    write_status_template(packet, public)
    write_json(packet / "annotation_schema.json", annotation_schema())
    summary = {
        "selection_seed": SELECTION_SEED,
        "packet_role": role,
        "total_blinded_tasks": int(len(public)),
        "views": public.view.value_counts().sort_index().to_dict(),
        "cohort_scope": (
            "All longitudinal images covering all 98 paper patients"
            if role == "primary"
            else "Fixed 20% image-level QC subset with one image from each of 38 distinct patients"
        ),
        "selection_uses_diagnosis_labels": False,
    }
    write_json(packet / "selection_summary_blinded.json", summary)
    (packet / "README.md").write_text(readme_text(role, len(public)), encoding="utf-8")
    shutil.copy2(VALIDATOR_SOURCE, packet / "validate_internal_region_annotations.py")
    figure = build_montage(packet, public)
    write_json(packet / "figure_resolution_audit.json", figure)


def build_archive(packet: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as package:
        for filename in PUBLIC_ROOT_FILES:
            package.write(packet / filename, arcname=filename)
        for directory in PUBLIC_DIRECTORIES:
            for path in sorted((packet / directory).rglob("*")):
                if path.is_file():
                    package.write(path, arcname=path.relative_to(packet).as_posix())


def build(output: Path, primary_archive: Path, qc_archive: Path) -> None:
    for path in (output, primary_archive, qc_archive):
        if path.exists():
            raise FileExistsError(f"Target already exists: {path}")
    if not VALIDATOR_SOURCE.exists():
        raise FileNotFoundError(f"Validator unavailable: {VALIDATOR_SOURCE}")
    manifest = load_label_free_manifest()
    private = order_images(manifest)
    qc_private = select_qc_subset(private)
    output.mkdir(parents=True)
    private.to_csv(output / "private_task_key.csv", index=False)
    qc_private.loc[:, ["task_id", "patient_id", "image_id", "view"]].to_csv(
        output / "private_second_rater_qc_key.csv",
        index=False,
    )
    write_json(
        output / "source_lock.json",
        {
            "source_manifest_sha256": sha256(SOURCE_MANIFEST),
            "builder_sha256": sha256(Path(__file__)),
            "validator_sha256": sha256(VALIDATOR_SOURCE),
            "manifest_columns_read": ["patient_id", "view", "image_id", "mask_source", "mask_sha256"],
            "diagnosis_label_read": False,
            "date_read": False,
            "source_filename_read": False,
            "primary_images": EXPECTED_IMAGES,
            "covered_patients": EXPECTED_PATIENTS,
            "second_rater_qc_images_and_distinct_patients": QC_IMAGES,
        },
    )
    primary_packet = output / "primary_public"
    qc_packet = output / "second_rater_qc_public"
    write_public_packet(primary_packet, private, "primary")
    write_public_packet(qc_packet, qc_private, "second_rater_qc")
    build_archive(primary_packet, primary_archive)
    build_archive(qc_packet, qc_archive)
    print(f"Primary packet: {primary_packet}")
    print(f"Primary archive: {primary_archive}")
    print(f"Second-rater QC packet: {qc_packet}")
    print(f"Second-rater QC archive: {qc_archive}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--primary-archive", type=Path, default=PRIMARY_ARCHIVE)
    parser.add_argument("--qc-archive", type=Path, default=QC_ARCHIVE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.output, args.primary_archive, args.qc_archive)


if __name__ == "__main__":
    main()
