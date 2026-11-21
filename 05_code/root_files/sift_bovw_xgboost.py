import os
import cv2
import numpy as np
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import confusion_matrix

# --- 1. Configuration ---
BASE_DIRS = ['nckuh_0_1', 'cych_0_1']
SIFT_FEATURES = 100  # SIFT features per image
VOCAB_SIZE = 128     # Number of "visual words"

# --- 2. Data Loading ---
def load_image_paths_and_labels(base_dirs):
    """Scans base directories to get image paths and their corresponding labels."""
    image_paths = []
    labels = []
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            continue
        for label_str in ['0', '1']:
            label_dir = os.path.join(base_dir, label_str)
            if not os.path.isdir(label_dir):
                continue
            for img_file in os.listdir(label_dir):
                if img_file.lower().endswith('.png'):
                    image_paths.append(os.path.join(label_dir, img_file))
                    labels.append(int(label_str))
    return image_paths, labels

# --- 3. SIFT Feature Extraction ---
def extract_sift_descriptors(image_paths):
    """Extracts SIFT descriptors from a list of image paths."""
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    all_descriptors = []
    print("Extracting SIFT descriptors...")
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            all_descriptors.append(des)
    return all_descriptors

# --- 4. Bag of Visual Words (BoVW) Creation ---
def create_bovw_features(descriptors, vocab_size):
    """Creates a visual vocabulary and transforms descriptors into histograms."""
    # Flatten all descriptors to build the vocabulary
    flat_descriptors = np.vstack([d for d in descriptors if d is not None])
    
    print(f"\nClustering {len(flat_descriptors)} descriptors into {vocab_size} visual words...")
    kmeans = MiniBatchKMeans(n_clusters=vocab_size, batch_size=2048, random_state=42).fit(flat_descriptors)
    
    print("Building histograms for each image...")
    histograms = []
    for desc_list in tqdm(descriptors):
        if desc_list is None:
            histograms.append(np.zeros(vocab_size))
        else:
            labels = kmeans.predict(desc_list)
            hist, _ = np.histogram(labels, bins=np.arange(vocab_size + 1))
            histograms.append(hist)
            
    return np.array(histograms), kmeans

# --- 5. Main Execution ---
def main():
    # Load data
    image_paths, labels = load_image_paths_and_labels(BASE_DIRS)
    if not image_paths:
        print("No images found. Please check BASE_DIRS configuration.")
        return
    print(f"Found {len(image_paths)} images across {len(set(labels))} classes.")

    # Extract SIFT descriptors
    all_descriptors = extract_sift_descriptors(image_paths)

    # Create BoVW features (histograms)
    X, kmeans_model = create_bovw_features(all_descriptors, VOCAB_SIZE)
    y = np.array(labels)

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # Scale the features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining XGBoost classifier...")
    xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_classifier.fit(X_train_scaled, y_train)

    # --- 6. Evaluation ---
    print("\nEvaluating model performance...")
    y_pred = xgb_classifier.predict(X_test_scaled)
    accuracy = np.mean(y_pred == y_test)
    
    # Calculate TPR, TNR, and min(TPR, TNR)
    try:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        min_tpr_tnr = min(tpr, tnr)
    except ValueError: # Happens if only one class is predicted
        tpr, tnr, min_tpr_tnr = 0, 0, 0

    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"TPR (Sensitivity): {tpr:.4f}")
    print(f"TNR (Specificity): {tnr:.4f}")
    print(f"min(TPR, TNR): {min_tpr_tnr:.4f}")

    # --- 7. Feature Importance Analysis ---
    print("\nAnalyzing feature importances from the trained model...")
    feature_importances = xgb_classifier.feature_importances_
    
    importance_df = pd.DataFrame({
        'Visual_Word_ID': range(VOCAB_SIZE),
        'Importance': feature_importances
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\n--- Top 20 Most Important Visual Words (Features) ---")
    print(importance_df.head(20))
    
    # Save to CSV
    output_path = "sift_bovw_feature_importance.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\nFull feature importance list saved to '{output_path}'")

if __name__ == '__main__':
    main()
