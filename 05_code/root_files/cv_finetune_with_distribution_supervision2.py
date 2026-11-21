import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.linalg import norm

# --- Config ---
import itertools

DATA_DIRS = ['nckuh_0_1_inner', 'cych_0_1_inner']
IMG_SIZE = 224
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'densenet121'
FEATURES_NPZ = 'densenet121_all_features.npz'
N_SPLITS = 5
PATIENCE = 7
DIST_THRESHOLD = 0.5  # 可依實際分布距離調整

# --- Grid Search 超參數組合 ---
param_grid = {
    "LR": [1e-3, 1e-4],
    "BATCH_SIZE": [8, 16],
    "WEIGHT_DECAY": [0, 1e-4]
}
param_list = list(itertools.product(
    param_grid["LR"],
    param_grid["BATCH_SIZE"],
    param_grid["WEIGHT_DECAY"]
))

# --- Dataset ---
class KidneyDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.samples = []
        self.labels = []
        for root_dir in root_dirs:
            for label in ['0', '1']:
                label_dir = os.path.join(root_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith('.png'):
                        self.samples.append(os.path.join(label_dir, fname))
                        self.labels.append(int(label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Model ---
def get_model():
    # Use weights="IMAGENET1K_V1" to avoid deprecation warning
    model = models.densenet121(weights="IMAGENET1K_V1")
    # Replace classifier head for binary classification
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    feature_layer = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    return model, feature_layer

def extract_features(model, feature_layer, loader):
    features = []
    labels = []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(DEVICE)
            feats = feature_layer(imgs)
            feats = feats.view(feats.size(0), -1)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels

def frechet_distance(mu1, cov1, mu2, cov2):
    # Fréchet distance between two Gaussians
    return norm(mu1 - mu2) + norm(cov1 - cov2)

def main():
    # 1. Load reference features
    ref = np.load(FEATURES_NPZ)
    ref_features = ref['features']
    ref_labels = ref['labels']
    ref_mu = ref_features.mean(axis=0)
    ref_cov = np.cov(ref_features, rowvar=False)

    # 2. Prepare dataset and CV
    dataset = KidneyDataset(DATA_DIRS, transform)
    X = np.arange(len(dataset))
    y = np.array(dataset.labels)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    results = []
    for lr, batch_size, weight_decay in param_list:
        print(f"\n=== Grid Search: LR={lr}, BATCH_SIZE={batch_size}, WEIGHT_DECAY={weight_decay} ===")
        fold_metrics = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold+1}/{N_SPLITS} ---")
            train_set = Subset(dataset, train_idx)
            val_set = Subset(dataset, val_idx)
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

            model, feature_layer = get_model()
            model = model.to(DEVICE)
            feature_layer = feature_layer.to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            best_min_tpr_tnr = 0
            best_model_state = None
            best_epoch = 0
            best_dist = float('inf')
            epochs_no_improve = 0

            for epoch in range(EPOCHS):
                model.train()
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Evaluate on val set
                model.eval()
                all_labels = []
                all_preds = []
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs = imgs.to(DEVICE)
                        outputs = model(imgs)
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(labels.numpy())
                y_true = np.array(all_labels)
                y_pred = np.array(all_preds)
                tp = np.sum((y_true == 1) & (y_pred == 1))
                tn = np.sum((y_true == 0) & (y_pred == 0))
                fp = np.sum((y_true == 0) & (y_pred == 1))
                fn = np.sum((y_true == 1) & (y_pred == 0))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
                min_tpr_tnr = min(tpr, tnr)

                # Extract features on val set
                feats, _ = extract_features(model, feature_layer, val_loader)
                mu = feats.mean(axis=0)
                cov = np.cov(feats, rowvar=False)
                dist = frechet_distance(mu, cov, ref_mu, ref_cov)
                print(f"Epoch {epoch+1}: min(TPR,TNR)={min_tpr_tnr:.4f}, FréchetDist={dist:.4f}")

                # Early stop by distribution distance
                improved = False
                if min_tpr_tnr > best_min_tpr_tnr or dist < best_dist:
                    best_min_tpr_tnr = min_tpr_tnr
                    best_dist = dist
                    best_model_state = model.state_dict()
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    improved = True
                else:
                    epochs_no_improve += 1

                if dist > DIST_THRESHOLD:
                    print(f"Warning: Feature distribution deviates from reference (FréchetDist={dist:.4f})")
                if epochs_no_improve >= PATIENCE:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in {PATIENCE} epochs).")
                    break

            if best_model_state is not None:
                model_path = f"cv_fold{fold+1}_best_densenet121_lr{lr}_bs{batch_size}_wd{weight_decay}.pth"
                torch.save(best_model_state, model_path)
                print(f"Fold {fold+1}: Saved best model at epoch {best_epoch}, min(TPR,TNR)={best_min_tpr_tnr:.4f}, FréchetDist={best_dist:.4f}")
                fold_metrics.append((best_min_tpr_tnr, best_dist))
            else:
                fold_metrics.append((0, float('inf')))

        # 平均分數
        avg_tpr_tnr = np.mean([m[0] for m in fold_metrics])
        avg_dist = np.mean([m[1] for m in fold_metrics])
        results.append({
            "lr": lr,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "avg_min_tpr_tnr": avg_tpr_tnr,
            "avg_dist": avg_dist
        })
        print(f"Grid Search Result: LR={lr}, BATCH_SIZE={batch_size}, WEIGHT_DECAY={weight_decay}, avg_min(TPR,TNR)={avg_tpr_tnr:.4f}, avg_FrechetDist={avg_dist:.4f}")

    # 輸出最佳組合
    results = sorted(results, key=lambda x: (-x["avg_min_tpr_tnr"], x["avg_dist"]))
    print("\n=== Grid Search Summary ===")
    for r in results:
        print(r)
    print(f"\nBest Hyperparameters: {results[0]}")

if __name__ == '__main__':
    main()
