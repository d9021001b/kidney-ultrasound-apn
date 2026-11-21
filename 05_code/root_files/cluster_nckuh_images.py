import os
import shutil
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

def read_images_with_vgg16(folder_path, device):
    """Reads images from a folder and extracts features using a pre-trained VGG16 model."""
    model = models.vgg16(pretrained=True).features.to(device).eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    file_paths = []
    features = []
    print(f"Reading images from '{folder_path}'...")
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        return [], None
        
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(folder_path, file_name)
            try:
                image = Image.open(file_path).convert('RGB')
                input_tensor = preprocess(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(input_tensor)
                    feature = output.cpu().numpy().flatten()
                features.append(feature)
                file_paths.append(file_path)
            except Exception as e:
                print(f"Could not process file {file_path}: {e}")
    
    if not features:
        return [], None
        
    features_tensor = torch.tensor(features, dtype=torch.float32)
    return file_paths, features_tensor

def compute_distance_matrix(features):
    """Computes the cosine distance matrix from a set of features."""
    features_np = features.cpu().numpy()
    distance_matrix = cosine_distances(features_np)
    distance_matrix = np.maximum(distance_matrix, 0.0)
    if np.isnan(distance_matrix).any() or np.isinf(distance_matrix).any():
        print("Warning: NaN or Inf values found in distance matrix. Replacing with 0.")
        distance_matrix = np.nan_to_num(distance_matrix, copy=True, nan=0.0, posinf=1.0, neginf=0.0)
    return distance_matrix

def cluster_and_save_fixed_k(file_paths, distance_matrix, output_folder_base, n_clusters=10):
    """Performs clustering with a fixed k and saves images to numbered subfolders."""
    if not file_paths:
        print("No files to cluster.")
        return

    print(f"Clustering {len(file_paths)} images into {n_clusters} groups...")
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)

    os.makedirs(output_folder_base, exist_ok=True)

    for i, file_path in enumerate(file_paths):
        cluster_id = cluster_labels[i]
        cluster_folder = os.path.join(output_folder_base, str(cluster_id))
        os.makedirs(cluster_folder, exist_ok=True)
        shutil.copy(file_path, cluster_folder)

    print(f"Images have been clustered and saved into subfolders inside '{output_folder_base}'.")

def main():
    """Main function to run the clustering tasks."""
    tasks = [
        {'input_folder': 'nckuh_0_1/0', 'output_folder_base': 'nckuh_0', 'n_clusters': 10},
        {'input_folder': 'nckuh_0_1/1', 'output_folder_base': 'nckuh_1', 'n_clusters': 10}
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for task in tasks:
        input_folder = task['input_folder']
        output_folder_base = task['output_folder_base']
        n_clusters = task['n_clusters']

        print(f"\n--- Processing Task: {input_folder} ---")

        if not os.path.isdir(input_folder):
            print(f"Input folder '{input_folder}' not found. Skipping task.")
            continue

        if os.path.exists(output_folder_base):
            print(f"Removing existing output folder: '{output_folder_base}'")
            shutil.rmtree(output_folder_base)

        file_paths, features = read_images_with_vgg16(input_folder, device)
        
        if not file_paths:
            print(f"No images found or processed in '{input_folder}'. Skipping clustering.")
            continue
            
        print(f"Extracted features for {len(file_paths)} images.")

        print("Computing distance matrix...")
        distance_matrix = compute_distance_matrix(features)

        cluster_and_save_fixed_k(file_paths, distance_matrix, output_folder_base, n_clusters=n_clusters)

    print("\nAll clustering tasks are complete.")

if __name__ == "__main__":
    main()
