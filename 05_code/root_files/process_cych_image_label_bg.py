import os
import cv2
import numpy as np

src_dir = "cych_image_label"
dst_white = "cych_bg_white"
dst_black = "cych_bg_black"
os.makedirs(dst_white, exist_ok=True)
os.makedirs(dst_black, exist_ok=True)

def read_yolo_seg_label(txt_path, img_shape):
    h, w = img_shape[:2]
    polygons = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue  # skip degenerate
            coords = list(map(float, parts[1:]))
            pts = []
            for i in range(0, len(coords), 2):
                x = int(round(coords[i] * w))
                y = int(round(coords[i+1] * h))
                pts.append([x, y])
            polygons.append(np.array(pts, dtype=np.int32))
    return polygons

for fname in os.listdir(src_dir):
    if fname.endswith(".png"):
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(src_dir, base + ".txt")
        img_path = os.path.join(src_dir, fname)
        if not os.path.exists(txt_path):
            print(f"Label not found for {fname}")
            continue
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        polygons = read_yolo_seg_label(txt_path, img.shape)
        if not polygons:
            print(f"No valid polygons for {fname}")
            continue
        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, polygons, 255)
        # White background
        white_bg = np.ones_like(img, dtype=np.uint8) * 255
        img_white = np.where(mask[..., None] == 255, img, white_bg)
        cv2.imwrite(os.path.join(dst_white, fname), img_white)
        # Black background
        black_bg = np.zeros_like(img, dtype=np.uint8)
        img_black = np.where(mask[..., None] == 255, img, black_bg)
        cv2.imwrite(os.path.join(dst_black, fname), img_black)

print("All images processed. Results saved to cych_bg_white and cych_bg_black.")
