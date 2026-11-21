import os
import hashlib
import random
from collections import Counter, defaultdict
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIRS = ['nckuh_0_1', 'cych_0_1']
LABELS = ['0', '1']
SHOW_IMAGES = True  # Set to False if running on a headless server

def get_image_hash(image_path):
    """Compute SHA256 hash of image file."""
    with open(image_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def scan_dataset(base_dirs):
    stats = defaultdict(lambda: defaultdict(list))  # {base_dir: {label: [img_paths]}}
    all_images = []
    for base_dir in base_dirs:
        for label in LABELS:
            label_dir = os.path.join(base_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for img_file in os.listdir(label_dir):
                if img_file.lower().endswith('.png'):
                    img_path = os.path.join(label_dir, img_file)
                    stats[base_dir][label].append(img_path)
                    all_images.append((img_path, base_dir, label))
    return stats, all_images

def print_distribution(stats):
    print("=== 圖片數量分布 ===")
    total = 0
    for base_dir in stats:
        print(f"\n資料夾: {base_dir}")
        for label in LABELS:
            n = len(stats[base_dir][label])
            print(f"  標籤 {label}: {n} 張")
            total += n
    print(f"\n總圖片數: {total}")

def show_random_images(stats, n_per_class=3):
    print("\n=== 隨機抽樣檢查每類圖片 ===")
    for base_dir in stats:
        for label in LABELS:
            imgs = stats[base_dir][label]
            if not imgs:
                continue
            sample_imgs = random.sample(imgs, min(n_per_class, len(imgs)))
            for img_path in sample_imgs:
                img = Image.open(img_path)
                plt.imshow(img)
                plt.title(f"{base_dir} | Label: {label}\n{os.path.basename(img_path)}")
                plt.axis('off')
                plt.show()

def check_duplicates(all_images):
    print("\n=== 檢查重複圖片（檔案內容完全相同） ===")
    hash_map = defaultdict(list)
    for img_path, base_dir, label in all_images:
        h = get_image_hash(img_path)
        hash_map[h].append((img_path, base_dir, label))
    duplicates = [v for v in hash_map.values() if len(v) > 1]
    if not duplicates:
        print("未發現重複圖片。")
    else:
        print(f"發現 {len(duplicates)} 組重複圖片：")
        for group in duplicates:
            print("  - " + " | ".join([f"{os.path.basename(p)} ({b}/{l})" for p, b, l in group]))

def main():
    stats, all_images = scan_dataset(BASE_DIRS)
    print_distribution(stats)
    if SHOW_IMAGES:
        show_random_images(stats, n_per_class=3)
    check_duplicates(all_images)

if __name__ == '__main__':
    main()
