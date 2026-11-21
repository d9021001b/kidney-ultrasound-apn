import os
import cv2
import numpy as np

SRC_ROOTS = ['nckuh_0_1', 'cych_0_1']
DST_ROOTS = ['nckuh_0_1_inner', 'cych_0_1_inner']

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_image(src_path, dst_path):
    img = cv2.imread(src_path)
    if img is None:
        print(f"Failed to read {src_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found in {src_path}")
        return

    # Find largest contour
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Compute rotation matrix to make the long axis horizontal
    width = rect[1][0]
    height = rect[1][1]
    angle = rect[2]
    # OpenCV minAreaRect: angle is [-90, 0), width is along the angle
    # If width < height, swap width/height and add 90 to angle
    if width < height:
        width, height = height, width
        angle = angle + 90
    # Now, width is always the long axis, and angle is the rotation needed to make it horizontal
    center = tuple(map(int, rect[0]))
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)

    # Get the upright rectangle region (no black border, fully inside object)
    # 1. Create mask of the rotated main object
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mask_rotated = cv2.warpAffine(mask, rot_mat, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST)

    # 2. Find the largest rectangle fully inside the white region of the mask
    # Dynamic programming to find largest rectangle of 1s in mask
    def max_rectangle_area(binary_mask):
        h, w = binary_mask.shape
        max_area = 0
        max_rect = (0, 0, 0, 0)  # x, y, width, height
        height = [0] * w
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j]:
                    height[j] += 1
                else:
                    height[j] = 0
            stack = []
            j = 0
            while j <= w:
                cur_height = height[j] if j < w else 0
                if not stack or cur_height >= height[stack[-1]]:
                    stack.append(j)
                    j += 1
                else:
                    top = stack.pop()
                    width_rect = j if not stack else j - stack[-1] - 1
                    area = height[top] * width_rect
                    if area > max_area:
                        max_area = area
                        x1 = (stack[-1] + 1) if stack else 0
                        y1 = i - height[top] + 1
                        max_rect = (x1, y1, width_rect, height[top])
        return max_rect  # x, y, w, h

    rect_x, rect_y, rect_w, rect_h = max_rectangle_area(mask_rotated)
    cropped = rotated[rect_y:rect_y+rect_h, rect_x:rect_x+rect_w]

    ensure_dir(os.path.dirname(dst_path))
    cv2.imwrite(dst_path, cropped)

def process_all():
    for src_root, dst_root in zip(SRC_ROOTS, DST_ROOTS):
        for dirpath, dirnames, filenames in os.walk(src_root):
            rel_dir = os.path.relpath(dirpath, src_root)
            dst_dir = os.path.join(dst_root, rel_dir)
            for fname in filenames:
                if fname.lower().endswith('.png'):
                    src_path = os.path.join(dirpath, fname)
                    dst_path = os.path.join(dst_dir, fname)
                    process_image(src_path, dst_path)
                    print(f"Processed: {src_path} -> {dst_path}")

if __name__ == '__main__':
    process_all()
