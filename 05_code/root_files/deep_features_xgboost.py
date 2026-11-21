import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd

# --- 1. Configuration ---
BASE_DIRS = ['nckuh_0_1', 'cych_0_1']
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_LAYER = 'avgpool'  # For ResNet50, use 'avgpool' before fc

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

# --- 3. Image Preprocessing ---
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- 4. Deep Feature Extraction ---
def extract_deep_features(image_paths, batch_size=32):
    # Load pre-trained ResNet50 and remove final fc layer
    model = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove last fc, keep avgpool
    model.eval()
    model.to(DEVICE)

    transform = get_transform()
    features = []
    valid_labels = []
    batch_imgs = []
    batch_indices = []

    print("Extracting deep features with ResNet50...")
    for idx, img_path in enumerate(tqdm(image_paths)):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            batch_imgs.append(img_tensor)
            batch_indices.append(idx)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")

        if len(batch_imgs) == batch_size or (idx == len(image_paths) - 1 and batch_imgs):
            batch_tensor = torch.stack(batch_imgs).to(DEVICE)
            with torch.no_grad():
                out = model(batch_tensor).squeeze(-1).squeeze(-1)  # Shape: (B, 2048)
                features.append(out.cpu().numpy())
            batch_imgs = []
            batch_indices = []

    features = np.concatenate(features, axis=0)
    return features

# --- 5. Main Execution ---
def main():
    image_paths, labels = load_image_paths_and_labels(BASE_DIRS)
    if not image_paths:
        print("No images found.")
        return
    print(f"Found {len(image_paths)} total images.")

    # Extract deep features
    X = extract_deep_features(image_paths, batch_size=BATCH_SIZE)
    y = np.array(labels)
    print(f"Deep feature extraction complete. Feature dimension: {X.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining XGBoost on deep features...")
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

    print(f"\n--- Performance with Deep Features ---")
    print(f"Classification Accuracy: {accuracy * 100:.2f}%")
    print(f"min(TPR, TNR): {min_tpr_tnr:.4f}")

    # Feature Importance Analysis
    feature_importances = xgb_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature_Index': np.arange(len(feature_importances)),
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    print("\n--- Top 20 Most Important Deep Features ---")
    print(importance_df.head(20))
    
    output_path = "deep_feature_importance.csv"
    importance_df.to_csv(output_path, index=False)
    print(f"\nFull deep feature importance list saved to '{output_path}'")

if __name__ == '__main__':
    main()
