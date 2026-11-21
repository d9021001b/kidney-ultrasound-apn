import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Config ---
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "A_nckuh_model.pth"
DATA_DIR = "nckuh_0_1_inner"
METADATA_PATH = "nckuh_metadata.json"

# --- Dataset ---
class FeatureExtractionDataset(Dataset):
    def __init__(self, file_label_items, transform=None):
        self.file_label_items = file_label_items
        self.transform = transform
    def __len__(self):
        return len(self.file_label_items)
    def __getitem__(self, idx):
        path, label = self.file_label_items[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(path)
        return img, label, filename

# --- Transform ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_feature_extractor(pretrained_path):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    if os.path.exists(pretrained_path):
        print(f"Loading model: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    else:
        raise FileNotFoundError(f"Model not found: {pretrained_path}")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.add_module("flatten", nn.Flatten())
    return feature_extractor.to(DEVICE).eval()

def extract_image_features(model, data_dir):
    file_label_items = []
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        file_label_items.extend([(f, int(label)) for f in files])
    dataset = FeatureExtractionDataset(file_label_items, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    features_dict = {}
    labels_dict = {}
    with torch.no_grad():
        for imgs, labels, filenames in tqdm(loader, desc="Extracting image features"):
            imgs = imgs.to(DEVICE)
            features = model(imgs).cpu().numpy()
            for i, filename in enumerate(filenames):
                features_dict[filename] = features[i]
                labels_dict[filename] = labels[i].item()
    return features_dict, labels_dict

def load_metadata(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    metadata_dict = {}
    for item in metadata_list:
        filename = item.get("filename")
        if filename:
            metadata_dict[filename] = item
    return metadata_dict

def combine_features(image_features, labels_dict, metadata):
    combined_X = []
    combined_y = []
    sorted_filenames = sorted(image_features.keys())
    for filename in sorted_filenames:
        if filename in metadata:
            img_feat = image_features[filename]
            combined_X.append(img_feat)
            combined_y.append(labels_dict[filename])
    return np.array(combined_X), np.array(combined_y)

if __name__ == "__main__":
    # 1. Extract image features
    feature_extractor = get_feature_extractor(MODEL_PATH)
    image_features, labels_dict = extract_image_features(feature_extractor, DATA_DIR)

    # 2. Load Metadata (for filename matching only)
    metadata = load_metadata(METADATA_PATH)

    # 3. Combine features (image only)
    X, y = combine_features(image_features, labels_dict, metadata)

    if len(X) == 0:
        print("Error: No features combined. Please check filenames and metadata.")
    else:
        print(f"Combined features for {len(X)} samples.")
        print(f"Original image feature dimension: {X.shape[1]}")

        # --- PCA to 2D ---
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(X)
        print("Image features reduced to 2D by PCA.")

        # --- Logistic Regression ---
        print("\nTraining logistic regression model (cross-validation)...")
        pipeline = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=10,
                cv=5,
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            )
        )
        pipeline.fit(X_2d, y)

        model = pipeline.named_steps['logisticregressioncv']
        print("\n--- Logistic Regression Model Analysis (Image 2D Only) ---")
        print(f"Best C after CV: {model.C_[0]}")
        print(f"Mean CV accuracy: {np.mean(model.scores_[1]):.4f}")

        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        print(f"\nIntercept: {intercept:.4f}")
        print("\nCoefficients for PCA1 and PCA2:")
        print(f"  - PCA1: {coefficients[0]:.4f}")
        print(f"  - PCA2: {coefficients[1]:.4f}")
