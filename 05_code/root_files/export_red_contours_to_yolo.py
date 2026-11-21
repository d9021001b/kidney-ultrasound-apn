import os
import cv2
import numpy as np
import shutil

# Source and destination directories
src_dir = os.path.join("CYCH", "Data")
dst_dir = "cych_image_label"
os.makedirs(dst_dir, exist_ok=True)

# YOLOv11-Seg expects: class_id x1 y1 x2 y2 ... xn yn (normalized, polygon)
# For this example, we use class_id 0 for the red contour

def find_red_contours(image):
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red can wrap around the hue, so use two ranges
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def save_yolo_seg_label(txt_path, contours, img_shape):
    h, w = img_shape[:2]
    with open(txt_path, "w") as f:
        for contour in contours:
            if len(contour) < 6:
                continue  # skip too small polygons
            # Flatten and normalize
            points = contour.squeeze().astype(float)
            if points.ndim == 1:
                continue  # skip degenerate
            norm_points = []
            for x, y in points:
                norm_points.append(f"{x/w:.6f}")
                norm_points.append(f"{y/h:.6f}")
            # YOLOv11-Seg: class_id x1 y1 x2 y2 ... xn yn
            line = "0 " + " ".join(norm_points)
            f.write(line + "\n")

for fname in os.listdir(src_dir):
    if fname.endswith(".png") and "_red_" in fname:
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        contours = find_red_contours(img)
        # Save label
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(dst_dir, base + ".txt")
        save_yolo_seg_label(label_path, contours, img.shape)
        # Copy image
        shutil.copy(img_path, os.path.join(dst_dir, fname))

print("Export complete. All _red_ images and labels are in cych_image_label/")
