import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Config ---
DATA_DIRS = ['nckuh_0_1_inner', 'cych_0_1_inner']
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'densenet121'
FEATURES_NPZ = 'densenet121_all_features.npz'
FRECHET_LOSS_WEIGHT = 0.1  # 可調整

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
    model = models.densenet121(weights="IMAGENET1K_V1")
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    feature_layer = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    return model, feature_layer

def gaussian_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)  # (N,1,D)
    y = y.unsqueeze(0)  # (1,M,D)
    dist = ((x - y) ** 2).sum(2)
    return torch.exp(-dist / (2 * sigma ** 2))

def mmd_loss(x, y, sigma=1.0):
    # x: (N, D), y: (M, D)
    Kxx = gaussian_kernel(x, x, sigma)
    Kyy = gaussian_kernel(y, y, sigma)
    Kxy = gaussian_kernel(x, y, sigma)
    mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return mmd

def extract_features_tensor(model, feature_layer, imgs):
    feats = feature_layer(imgs)
    feats = feats.view(feats.size(0), -1)
    return feats

def main():
    # 若有舊模型則直接載入並結束
    model_path = "finetuned_with_frechet_loss.pth"
    model, feature_layer = get_model()
    model = model.to(DEVICE)
    feature_layer = feature_layer.to(DEVICE)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Found existing model: {model_path} 已載入，未重新訓練。")
        return

    # 1. Load reference features
    ref = np.load(FEATURES_NPZ)
    ref_features = torch.tensor(ref['features'], dtype=torch.float32, device=DEVICE)
    ref_mu = ref_features.mean(dim=0)
    ref_cov = torch.cov(ref_features.T)

    # 2. Prepare dataset and split train/val
    dataset = KidneyDataset(DATA_DIRS, transform)
    indices = np.arange(len(dataset))
    labels = np.array(dataset.labels)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, stratify=labels, random_state=42)
    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience = 7
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_cls = 0
        total_frechet = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            cls_loss = criterion(outputs, labels)

            # 特徵分布
            feats = extract_features_tensor(model, feature_layer, imgs)
            # MMD loss: batch feats vs. reference feats
            mmd = mmd_loss(feats, ref_features, sigma=1.0)

            loss = cls_loss + FRECHET_LOSS_WEIGHT * mmd
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_cls += cls_loss.item()
            total_frechet += mmd.item()

        # 驗證集監控
        model.eval()
        val_loss = 0
        val_cls = 0
        val_frechet = 0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                cls_loss = criterion(outputs, labels)
                feats = extract_features_tensor(model, feature_layer, imgs)
                mmd = mmd_loss(feats, ref_features, sigma=1.0)
                loss = cls_loss + FRECHET_LOSS_WEIGHT * mmd
                val_loss += loss.item()
                val_cls += cls_loss.item()
                val_frechet += mmd.item()
                # collect for min(TPR,TNR)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        # min(TPR,TNR)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        min_tpr_tnr = min(tpr, tnr)
        print(f"Epoch {epoch+1}: train_loss={total_loss:.4f}, val_loss={val_loss:.4f}, val_cls={val_cls:.4f}, val_mmd={val_frechet:.4f}, min(TPR,TNR)={min_tpr_tnr:.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs).")
                break

    if best_model_state is not None:
        torch.save(best_model_state, model_path)
        print(f"Saved best model at epoch {best_epoch} to {model_path}")
    else:
        print("No improvement, model not saved.")

if __name__ == '__main__':
    main()
