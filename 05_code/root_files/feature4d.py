import os  
import shutil  
import torch  
import numpy as np  
from torchvision import models, transforms  
from PIL import Image  
from sklearn.cluster import AgglomerativeClustering  
from sklearn.metrics.pairwise import cosine_distances  
  
def read_images_with_vgg16(folder_path, device):  
    # Load the pre-trained VGG16 model and set it to evaluation mode  
    model = models.vgg16(pretrained=True).features.to(device).eval()  
  
    # Define the preprocessing transformations  
    preprocess = transforms.Compose([  
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        # Normalize using ImageNet's mean and standard deviation  
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  
                             std= [0.229, 0.224, 0.225])  
    ])  
  
    file_paths = []  
    features = []  
  
    for file_name in os.listdir(folder_path):  
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):  
            file_path = os.path.join(folder_path, file_name)  
            # Open image and convert to RGB  
            image = Image.open(file_path).convert('RGB')  
            # Apply preprocessing  
            input_tensor = preprocess(image).unsqueeze(0).to(device)  
            with torch.no_grad():  
                # Extract features using VGG16  
                output = model(input_tensor)  
                # Flatten the output tensor  
                feature = output.cpu().numpy().flatten()  
            features.append(feature)  
            file_paths.append(file_path)  
  
    # Convert features to a tensor  
    features_tensor = torch.tensor(features, dtype=torch.float32)  
    return file_paths, features_tensor  
  
def compute_distance_matrix(features):  
    features_np = features.cpu().numpy()  
    # Compute cosine distance matrix  
    distance_matrix = cosine_distances(features_np)  
    # Ensure no negative distances  
    distance_matrix = np.maximum(distance_matrix, 0.0)  
    # Check for NaNs or Infs  
    if np.isnan(distance_matrix).any():  
        print("NaN values found in distance matrix!")  
    if np.isinf(distance_matrix).any():  
        print("Inf values found in distance matrix!")  
    return distance_matrix  
  
def cluster_and_save(file_paths, distance_matrix, output_folder, distance_threshold=0.5):  
    # Perform Agglomerative Clustering with the specified distance threshold  
    clustering = AgglomerativeClustering(  
        n_clusters=None,  # Let the algorithm determine the number of clusters  
        metric='precomputed',  # Use precomputed distance matrix  
        linkage='average',  # Use average linkage  
        distance_threshold=distance_threshold  # Set distance threshold for clustering  
    )  
  
    cluster_labels = clustering.fit_predict(distance_matrix)  
  
    # # Count the number of clusters  
    n_clusters = len(set(cluster_labels))  
    print(f"Number of clusters: {n_clusters}")  
  
    # Create a dictionary to hold clusters  
    clusters = {}  
    for file_path, label in zip(file_paths, cluster_labels):  
        clusters.setdefault(label, []).append(file_path)  
  
    # Sort clusters by size in descending order and keep the largest 5 clusters  
    sorted_clusters = sorted(clusters.items(), key=lambda item: len(item[1]), reverse=True)  
    largest_clusters = sorted_clusters[:5]  
  
    # Create the output folder if it doesn't exist  
    os.makedirs(output_folder, exist_ok=True)  
  
    # Save images into the output folder for the largest clusters without creating subfolders  
    for label, files in largest_clusters:  
        for file_path in files:  
            # Copy each image to the output folder  
            shutil.copy(file_path, output_folder)  
  
    print(f"Images have been clustered and saved to '{output_folder}'.")  
    print(f"Only the largest {len(largest_clusters)} clusters have been saved.")   
  
def main():  
    
  
    # 指定資料夾路徑  
    directory_path = 'b1'  
      
    # 獲取子資料夾列表  
    subdirectories = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]  
      
    # 逐一列印子資料夾名稱  
    for subdir in subdirectories:  
        #print(subdir)  
    
        input_folder = './'+directory_path+'/'+f'{subdir}'  # Replace with your input folder path  
        output_folder = f'./a1/{subdir}'+'_ecg1'  # Output folder for clusters  
        distance_threshold = 0.05  # Adjust the distance threshold as needed  
      
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print(f"Using device: {device}")  
      
        # Delete output_folder if it exists to start fresh  
        if os.path.exists(output_folder):  
            shutil.rmtree(output_folder)  
      
        # Read images and extract features using VGG16  
        file_paths, features = read_images_with_vgg16(input_folder, device)  
        print(f"Extracted features for {len(file_paths)} images.")  
      
        # Compute distance matrix using cosine distances  
        distance_matrix = compute_distance_matrix(features)  
      
        # Perform clustering and save the results  
        cluster_and_save(file_paths, distance_matrix, output_folder, distance_threshold=distance_threshold)  
  
if __name__ == "__main__":  
    main()  