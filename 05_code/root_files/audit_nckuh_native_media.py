"""Inventory native-media candidates without exporting filenames or metadata values."""
import io
import json
import subprocess
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "03_data_models/ki-data"
OUT = ROOT / "02_experiments/NCKUH_native_media_audit_20261103"
DIRECT = {".dcm", ".dicom", ".ima", ".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}


def zip_inventory(path):
    extensions, frames = Counter(), Counter()
    preamble_hits = 0
    with ZipFile(path) as archive:
        for entry in archive.infolist():
            if entry.is_dir():
                continue
            suffix = Path(entry.filename).suffix.lower()
            extensions[suffix or "(none)"] += 1
            with archive.open(entry) as stream:
                head = stream.read(132)
                preamble_hits += int(head[128:132] == b"DICM")
                if suffix in {".tif", ".tiff"}:
                    with Image.open(io.BytesIO(head + stream.read())) as raster:
                        frames[raster.n_frames] += 1
    return dict(extensions=dict(extensions), dicom_preamble_hits=preamble_hits,
                tiff_frame_count=dict(frames))


def run():
    if OUT.exists():
        raise FileExistsError(OUT)
    names = ["1.zip", "2.zip", "3.zip", "before-labeled.zip", "NCKUH3.zip",
             "test_predict-e.zip", "ezdicom-20260910T035043Z-1-001.zip"]
    zips = {name: zip_inventory(DATA / name) for name in names}
    rar = ROOT / "04_quality_archive/tmp/NCKUH.rar"
    contents = subprocess.run(["tar", "-tf", str(rar)], check=True,
                              capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    rar_suffixes = Counter(Path(line).suffix.lower() or "(none)" for line in contents.splitlines()
                           if line and not line.endswith("/"))
    unpacked = Counter(p.suffix.lower() for top in ("03_data_models", "04_quality_archive")
                       for p in (ROOT / top).rglob("*") if p.is_file() and p.suffix.lower() in DIRECT)
    data = dict(source_archives=zips, legacy_nckuh_rar_extension_counts=dict(rar_suffixes),
                unpacked_dicom_or_video_extension_counts=dict(unpacked))
    OUT.mkdir()
    (OUT / "media_inventory.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("DICOM preambles in ZIP entries:", sum(z["dicom_preamble_hits"] for z in zips.values()))
    print("TIFF frame counts:", sum((Counter(z["tiff_frame_count"]) for z in zips.values()), Counter()))
    print("Legacy NCKUH RAR extensions:", dict(rar_suffixes))
    print("Unpacked DICOM/video extension counts:", dict(unpacked))


if __name__ == "__main__":
    run()
