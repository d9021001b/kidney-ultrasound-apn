import os
import cv2
import numpy as np
import xgboost as xgb
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, make_scorer
from tqdm import tqdm
import pandas as pd

# --- 1. Configuration ---
BASE_DIRS = ['nckuh_0_1', 'cych_0_1']
SIFT_FEATURES = 100
# We will iterate over these vocabulary sizes
VOCAB_SIZE_OPTIONS = [64, 128, 256] 

# --- 2. Data Loading & SIFT (Same as before) ---
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

def extract_sift_descriptors(image_paths):
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    all_descriptors = []
    print("Extracting SIFT descriptors...")
    for img_path in tqdm(image_paths):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        _, des = sift.detectAndCompute(img, None)
        if des is not None: all_descriptors.append(des)
    return all_descriptors

# --- 3. BoVW Creation (Same as before) ---
def create_bovw_features(descriptors, vocab_size):
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
    return np.array(histograms)

# --- 4. Custom Scorer for Hyperparameter Tuning ---
def min_tpr_tnr_score(y_true, y_pred):
    """Calculates min(TPR, TNR) for use as a scorer."""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        return min(tpr, tnr)
    except ValueError:
        return 0

min_tpr_tnr_scorer = make_scorer(min_tpr_tnr_score, greater_is_better=True)

# --- 5. Main Execution ---
def main():
    # Load data paths
    image_paths, labels = load_image_paths_and_labels(BASE_DIRS)
    if not image_paths:
        print("No images found.")
        return
    print(f"Found {len(image_paths)} total images.")

    # Extract SIFT descriptors once
    all_descriptors = extract_sift_descriptors(image_paths)
    y = np.array(labels)

    best_overall_score = -1
    best_overall_params = {}

    # Outer loop for VOCAB_SIZE
    for vocab_size in VOCAB_SIZE_OPTIONS:
        print(f"\n{'='*20} Testing VOCAB_SIZE = {vocab_size} {'='*20}")
        
        # Create BoVW features for the current vocab size
        X = create_bovw_features(all_descriptors, vocab_size)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- RandomizedSearchCV for XGBoost ---
        param_dist = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }

        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

        print("\nRunning RandomizedSearchCV for XGBoost...")
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_dist,
            n_iter=50,  # Number of parameter settings that are sampled
            cv=3,       # 3-fold cross-validation
            scoring=min_tpr_tnr_scorer,
            verbose=1,
            n_jobs=-1,  # Use all available cores
            random_state=42
        )
        random_search.fit(X_train_scaled, y_train)

        print(f"\nBest parameters for VOCAB_SIZE={vocab_size}:")
        print(random_search.best_params_)
        
        # Evaluate the best estimator on the test set
        best_estimator = random_search.best_estimator_
        y_pred = best_estimator.predict(X_test_scaled)
        
        test_score = min_tpr_tnr_score(y_test, y_pred)
        print(f"Test Set min(TPR, TNR) for best estimator: {test_score:.4f}")

        if test_score > best_overall_score:
            best_overall_score = test_score
            best_overall_params = {
                'VOCAB_SIZE': vocab_size,
                **random_search.best_params_
            }

    print(f"\n{'='*20} Tuning Complete {'='*20}")
    print(f"Best overall min(TPR, TNR) score: {best_overall_score:.4f}")
    print("Best overall parameters found:")
    print(best_overall_params)

if __name__ == '__main__':
    main()
