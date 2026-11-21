import os
import cv2
import numpy as np
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd

# --- 1. Configuration ---
BASE_DIRS = ['nckuh_0_1', 'cych_0_1']
VOCAB_SIZE = 128  # Using a fixed vocab size for fair comparison
FEATURE_TYPES = ['SIFT', 'ORB', 'SURF'] # Add SURF to the list

# --- 2. Data Loading ---
def load_image_paths_and_labels(base_dirs):
    image_paths, labels = [], []
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir): continue
        for label_str in ['0', '1']:
            label_dir = os.path.join(base_dir, label_str)
            if not os.path.isdir(label_dir): continue
            for img_file in os.listdir(label_dir):
                if img_file.lower().endswith('.png'):
                    image_paths.append(os.path.join(label_dir, img_file))
                    labels.append(int(label_str))
    return image_paths, labels

# --- 3. Feature Extraction Factory ---
def get_feature_extractor(feature_type):
    """Factory function to get a feature extractor object."""
    if feature_type == 'SIFT':
        return cv2.SIFT_create()
    if feature_type == 'ORB':
        return cv2.ORB_create(nfeatures=200) # ORB usually needs more features
    if feature_type == 'SURF':
        try:
            # This call will fail if SURF is not enabled in the build
            return cv2.xfeatures2d.SURF_create(400)
        except cv2.error as e:
            # Check if the error message contains the specific patent issue text
            if 'not implemented' in str(e) or 'nonfree' in str(e):
                print("SURF not available due to patent restrictions in this OpenCV build. Skipping.")
                return None
            else:
                # Re-raise other unexpected errors
                raise e
    raise ValueError(f"Unknown feature type: {feature_type}")

def extract_descriptors(image_paths, labels, feature_type):
    """Extracts descriptors and their corresponding labels."""
    extractor = get_feature_extractor(feature_type)
    if extractor is None:
        return None, None
        
    successful_descriptors = []
    corresponding_labels = []
    print(f"Extracting {feature_type} descriptors...")
    for i, img_path in enumerate(tqdm(image_paths)):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        _, des = extractor.detectAndCompute(img, None)
        if des is not None:
            successful_descriptors.append(des)
            corresponding_labels.append(labels[i])
    return successful_descriptors, corresponding_labels

# --- 4. BoVW and Evaluation ---
def create_bovw_and_evaluate(descriptors, filtered_labels, feature_type):
    """Creates BoVW features and evaluates them using XGBoost."""
    if descriptors is None or len(descriptors) < 2:
        print(f"Not enough descriptors for {feature_type}. Skipping.")
        return None

    # BoVW Creation
    flat_descriptors = np.vstack([d for d in descriptors if d is not None])
    # ORB descriptors are binary, need to convert to float for KMeans
    if flat_descriptors.dtype == np.uint8:
        flat_descriptors = flat_descriptors.astype(float)

    print(f"\nClustering {len(flat_descriptors)} {feature_type} descriptors...")
    kmeans = MiniBatchKMeans(n_clusters=VOCAB_SIZE, batch_size=2048, random_state=42).fit(flat_descriptors)
    
    print("Building histograms...")
    histograms = []
    for desc_list in tqdm(descriptors):
        if desc_list is None:
            histograms.append(np.zeros(VOCAB_SIZE))
        else:
            if desc_list.dtype == np.uint8:
                desc_list = desc_list.astype(float)
            labels_pred = kmeans.predict(desc_list)
            hist, _ = np.histogram(labels_pred, bins=np.arange(VOCAB_SIZE + 1))
            histograms.append(hist)
    
    X = np.array(histograms)
    y = np.array(filtered_labels)

    # Train and Evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    y_pred = xgb_model.predict(X_test_scaled)
    
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        return min(tpr, tnr)
    except ValueError:
        return 0

# --- 5. Main Execution ---
def main():
    image_paths, labels = load_image_paths_and_labels(BASE_DIRS)
    if not image_paths:
        print("No images found.")
        return
    print(f"Found {len(image_paths)} total images.")

    results = {}
    for feature_type in FEATURE_TYPES:
        print(f"\n{'='*20} Testing Feature Type: {feature_type} {'='*20}")
        descriptors, filtered_labels = extract_descriptors(image_paths, labels, feature_type)
        if descriptors:
            score = create_bovw_and_evaluate(descriptors, filtered_labels, feature_type)
            if score is not None:
                results[feature_type] = score

    print(f"\n{'='*20} Comparison Complete {'='*20}")
    if not results:
        print("No feature descriptors could be tested.")
        return
        
    # Print sorted results
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    print("Performance based on min(TPR, TNR):")
    for feature, score in sorted_results:
        print(f"- {feature}: {score:.4f}")

if __name__ == '__main__':
    main()
