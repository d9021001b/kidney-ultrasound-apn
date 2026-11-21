import os
import shutil

src_dir = os.path.join("CYCH", "Data")
dst_dir = "original_png"
os.makedirs(dst_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    if fname.endswith(".png") and "_red_" not in fname:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)
        shutil.move(src_path, dst_path)

print("All non-_red_ .png files have been moved to original_png/")
