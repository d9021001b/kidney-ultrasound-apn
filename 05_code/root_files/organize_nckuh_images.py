import os
import shutil
import json

def organize_images_by_label():
    """
    Organizes images from the NCKUH dataset into subfolders based on their labels.
    """
    # --- Configuration ---
    metadata_path = 'nckuh_metadata.json'
    source_dir = 'nckuh_bg_black'
    base_dest_dir = 'nckuh_0_1'
    
    # --- Setup ---
    # Create destination directories
    dest_dir_0 = os.path.join(base_dest_dir, '0')
    dest_dir_1 = os.path.join(base_dest_dir, '1')
    os.makedirs(dest_dir_0, exist_ok=True)
    os.makedirs(dest_dir_1, exist_ok=True)
    
    print(f"Destination folders '{dest_dir_0}' and '{dest_dir_1}' are ready.")

    # --- Load Metadata ---
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Successfully loaded metadata from '{metadata_path}'.")
    except FileNotFoundError:
        print(f"Error: Metadata file not found at '{metadata_path}'. Aborting.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{metadata_path}'. Aborting.")
        return

    # --- Process and Copy Files ---
    copied_count = 0
    skipped_count = 0
    total_items = len(metadata)
    for i, item in enumerate(metadata):
        filename = item.get('filename')
        label = item.get('label')

        if filename is None or label is None:
            print(f"Skipping invalid metadata entry: {item}")
            continue

        source_path = os.path.join(source_dir, filename)
        
        if not os.path.exists(source_path):
            skipped_count += 1
            continue
            
        # Determine destination directory based on label
        if str(label) == '0':
            dest_path = os.path.join(dest_dir_0, filename)
        elif str(label) == '1':
            dest_path = os.path.join(dest_dir_1, filename)
        else:
            print(f"Skipping file '{filename}' with unknown label: {label}")
            continue
            
        # Copy the file
        try:
            shutil.copy2(source_path, dest_path)
            copied_count += 1
        except Exception as e:
            print(f"Error copying '{source_path}' to '{dest_path}': {e}")
        
        # Print progress
        print(f"Processing... {i+1}/{total_items}", end='\r')

    print("\n--- Summary ---")
    print(f"Total files copied: {copied_count}")
    print(f"Total files skipped (source not found): {skipped_count}")
    print("Image organization complete.")

if __name__ == '__main__':
    organize_images_by_label()
