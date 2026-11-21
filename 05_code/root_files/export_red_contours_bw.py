import os
import cv2
import numpy as np

src_dir = os.path.join("CYCH", "Data")
dst_dir = "cych_bw"
os.makedirs(dst_dir, exist_ok=True)

def find_red_contour(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Only keep the largest contour (should be only one)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    return largest

def save_yolo_seg_label(txt_path, contour, img_shape):
    h, w = img_shape[:2]
    if contour is None or len(contour) < 6:
        return
    points = contour.squeeze().astype(float)
    if points.ndim == 1:
        return
    norm_points = []
    for x, y in points:
        norm_points.append(f"{x/w:.6f}")
        norm_points.append(f"{y/h:.6f}")
    line = "0 " + " ".join(norm_points)
    with open(txt_path, "w") as f:
        f.write(line + "\n")

for fname in os.listdir(src_dir):
    if fname.endswith(".png") and "_red_" in fname:
        img_path = os.path.join(src_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {img_path}")
            continue
        contour = find_red_contour(img)
        # Create black image and fill contour with white
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        if contour is not None:
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        # Save binary image
        bw_path = os.path.join(dst_dir, fname)
        cv2.imwrite(bw_path, mask)
        # Save label
        base = os.path.splitext(fname)[0]
        label_path = os.path.join(dst_dir, base + ".txt")
        save_yolo_seg_label(label_path, contour, img.shape)

print("Export complete. All binary images and labels are in cych_bw/")
