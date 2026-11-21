"""Ultrasound field shape from pixels alone; no kidney mask or APN label."""

from pathlib import Path

import cv2
import numpy as np


def image_frame(path: Path, panel: bool) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable ultrasound image: {path}")
    if panel:
        if image.shape[1] % 2:
            raise ValueError(f"Odd-width image-mask panel: {path}")
        image = image[:, :image.shape[1] // 2]
    return image


def blind_sector(frame: np.ndarray) -> tuple[np.ndarray, bool]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    height, width = gray.shape
    active = (gray > 8).astype(np.uint8)
    active[:max(2, round(height * .06))] = 0
    active = cv2.morphologyEx(active, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    if count <= 1:
        return np.ones((height, width), dtype=bool), True
    candidates = [(stats[j, cv2.CC_STAT_AREA], j) for j in range(1, count)
                  if stats[j, cv2.CC_STAT_WIDTH] >= .12 * width
                  and stats[j, cv2.CC_STAT_HEIGHT] >= .15 * height]
    if not candidates:
        return np.ones((height, width), dtype=bool), True
    _, selected = max(candidates)
    return labels == selected, False


def sector_features(frame: np.ndarray) -> np.ndarray:
    sector, fallback = blind_sector(frame)
    height, width = sector.shape
    ys, xs = np.nonzero(sector)
    y0, y1 = ys.min(), ys.max() + 1
    rows = np.linspace(y0, y1, 17, dtype=int)
    widths, centers = [], []
    for a, b in zip(rows[:-1], rows[1:]):
        selected = sector[a:max(a + 1, b)]
        sy, sx = np.nonzero(selected)
        if len(sx):
            widths.append((sx.max() - sx.min() + 1) / width)
            centers.append(sx.mean() / width - .5)
        else:
            widths.append(0.0)
            centers.append(0.0)
    positive = [value for value in widths if value > 0]
    values = np.asarray([
        height / width, sector.mean(), (xs.max() - xs.min() + 1) / width,
        (y1 - y0) / height, np.mean(widths[:4]), np.mean(widths[6:10]),
        np.mean(widths[-4:]), np.std(widths),
        np.max(widths) - min(positive or [0]), np.std(centers), float(fallback),
        *widths,
    ], dtype=np.float32)
    assert np.isfinite(values).all()
    return values
