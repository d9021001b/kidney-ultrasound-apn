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
VOCAB_SIZE = 128
FEATURE_TYPES = ['SIFT', 'ORB', 'SURF']

# --- 2. Data Loading ---
def load_image_paths_and_labels(base_dirs):
    # (Same as before)
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

# --- 3. Feature Extraction & BoVW ---
def get_feature_extractor(feature_type):
    # (Same robust version as before)
    if feature_type == 'SIFT':
        return cv2.SIFT_create()
    if feature_type == 'ORB':
        return cv2.ORB_create(nfeatures=200)
    if feature_type == 'SURF':
        try:
            return cv2.xfeatures2d.SURF_create(400)
        except cv2.error:
            print(f"SURF not available in this OpenCV build. Skipping {feature_type}.")
            return None
    raise ValueError(f"Unknown feature type: {feature_type}")

def get_bovw_features_for_type(image_paths, feature_type, vocab_size):
    """Extracts features and returns a BoVW histogram matrix for a single feature type."""
    print(f"\nProcessing feature type: {feature_type}")
    extractor = get_feature_extractor(feature_type)
    if extractor is None:
        return None

    # Step 1: Extract descriptors for all images
    all_descriptors = []
    # Keep track of which images yield descriptors
    descriptor_map = {} 
    print(f"Extracting {feature_type} descriptors...")
    for i, img_path in enumerate(tqdm(image_paths)):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        _, des = extractor.detectAndCompute(img, None)
        if des is not None:
            all_descriptors.append(des)
            descriptor_map[i] = des

    if not all_descriptors:
        print(f"No {feature_type} descriptors found. Skipping.")
        return None

    # Step 2: Build vocabulary from all found descriptors
    flat_descriptors = np.vstack(all_descriptors)
    if flat_descriptors.dtype == np.uint8: # For ORB
        flat_descriptors = flat_descriptors.astype(float)
    
    print(f"Clustering {len(flat_descriptors)} descriptors into {vocab_size} words...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=4096, random_state=42).fit(flat_descriptors)

    # Step 3: Build histogram for every image
    histograms = np.zeros((len(image_paths), vocab_size))
    print("Building histograms for each image...")
    for i in tqdm(range(len(image_paths))):
        if i in descriptor_map:
            des = descriptor_map[i]
            if des.dtype == np.uint8:
                des = des.astype(float)
            labels = kmeans.predict(des)
            hist, _ = np.histogram(labels, bins=np.arange(vocab_size + 1))
            histograms[i] = hist
    
    return histograms

# --- 4. Main Execution ---
def main():
    image_paths, labels = load_image_paths_and_labels(BASE_DIRS)
    if not image_paths:
        print("No images found.")
        return
    print(f"Found {len(image_paths)} total images.")

    # Collect feature matrices from all available types
    all_feature_matrices = []
    feature_names = []
    for f_type in FEATURE_TYPES:
        hist_matrix = get_bovw_features_for_type(image_paths, f_type, VOCAB_SIZE)
        if hist_matrix is not None:
            all_feature_matrices.append(hist_matrix)
            feature_names.extend([f"{f_type}_word_{i}" for i in range(VOCAB_SIZE)])

    if not all_feature_matrices:
        print("Could not generate features from any descriptor type.")
        return

    # Fuse features
    X_fused = np.hstack(all_feature_matrices)
    y = np.array(labels)
    print(f"\nFeature fusion complete. Final feature vector dimension: {X_fused.shape[1]}")

    # Train and Evaluate with fused features
    X_train, X_test, y_train, y_test = train_test_split(X_fused, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining XGBoost on FUSED features...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = xgb_model.predict(X_test_scaled)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        min_tpr_tnr = min(tpr, tnr)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    except ValueError:
        tpr, tnr, min_tpr_tnr, accuracy = 0, 0, 0, 0

    print(f"\n--- Performance with Fused Features ---")
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"min(TPR, TNR): {min_tpr_tnr:.4f}")

    # Feature Importance Analysis
    feature_importances = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature_Name': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\n--- Top 20 Most Important Fused Features ---")
    print(importance_df.head(20))
    
    output_path = "fused_feature_importance.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\nFull fused feature importance list saved to '{output_path}'")

if __name__ == '__main__':
    main()
