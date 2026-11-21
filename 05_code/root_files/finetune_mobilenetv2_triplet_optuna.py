import os
import json
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import optuna

# Paths
IMG_DIR = "nckuh_bg_black"
META_PATH = "nckuh_metadata.json"
MODEL_SAVE_PATH = "best_mobilenetv2_triplet.pth"

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Load metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Encode gender
gender_map = {"male": 0, "female": 1}
for m in meta:
    m["gender_enc"] = gender_map[m["gender"]]

# Filter meta to only include entries with existing images
meta = [m for m in meta if os.path.exists(os.path.join(IMG_DIR, m["filename"]))]
print(f"Filtered metadata: {len(meta)} samples with existing images.")

# Dataset
class KidneyDataset(Dataset):
    def __init__(self, meta, img_dir, transform=None):
        self.meta = meta
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        entry = self.meta[idx]
        img_path = os.path.join(self.img_dir, entry["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # Metadata: gender (1), long_axis (1), short_axis (1)
        meta_feat = np.array([
            entry["gender_enc"],
            entry["long_axis"],
            entry["short_axis"]
        ], dtype=np.float32)
        label = entry["label"]
        return image, meta_feat, label, entry["filename"]

# Model
class MobileNetV2WithMeta(nn.Module):
    def __init__(self, meta_dim=3, emb_dim=128):
        super().__init__()
        # Use new torchvision API for weights to avoid deprecation warning
        try:
            from torchvision.models import MobileNet_V2_Weights
            self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        except ImportError:
            self.backbone = models.mobilenet_v2(pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        # Dynamically determine backbone output features
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            backbone_feat = self.backbone(dummy)
            backbone_out_dim = backbone_feat.shape[1]
        self.final_fc = nn.Linear(backbone_out_dim + 8, emb_dim)

    def forward(self, x_img, x_meta):
        img_feat = self.backbone(x_img)  # (B, backbone_out_dim)
        meta_feat = self.meta_fc(x_meta) # (B, 8)
        feat = torch.cat([img_feat, meta_feat], dim=1)
        emb = self.final_fc(feat)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

# Triplet Loss with margin
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        return self.loss_fn(anchor, positive, negative)

# Triplet Sampler
def get_triplets(meta, labels):
    label_to_indices = {}
    for idx, label in enumerate(labels):
        label_to_indices.setdefault(label, []).append(idx)
    triplets = []
    for idx, label in enumerate(labels):
        pos_idx = idx
        while pos_idx == idx:
            pos_idx = random.choice(label_to_indices[label])
        neg_label = 1 - label
        neg_idx = random.choice(label_to_indices[neg_label])
        triplets.append((idx, pos_idx, neg_idx))
    return triplets

# Training and validation
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, metas, labels, _ in loader:
        imgs, metas = imgs.to(device), metas.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Triplet mining
        triplets = get_triplets(metas, labels.cpu().numpy())
        if not triplets:
            continue
        anchor_idx, pos_idx, neg_idx = zip(*triplets)
        anchor_imgs = imgs[list(anchor_idx)]
        anchor_metas = metas[list(anchor_idx)]
        pos_imgs = imgs[list(pos_idx)]
        pos_metas = metas[list(pos_idx)]
        neg_imgs = imgs[list(neg_idx)]
        neg_metas = metas[list(neg_idx)]
        anchor_emb = model(anchor_imgs, anchor_metas)
        pos_emb = model(pos_imgs, pos_metas)
        neg_emb = model(neg_imgs, neg_metas)
        loss = criterion(anchor_emb, pos_emb, neg_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def eval_epoch(model, loader, device):
    model.eval()
    all_labels = []
    all_embs = []
    with torch.no_grad():
        for imgs, metas, labels, _ in loader:
            imgs, metas = imgs.to(device), metas.to(device)
            emb = model(imgs, metas)
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_embs = np.concatenate(all_embs, axis=0)
    all_labels = np.array(all_labels)
    # Compute mean intra-class and inter-class distances
    dists = []
    for i in range(len(all_embs)):
        for j in range(i+1, len(all_embs)):
            dist = np.linalg.norm(all_embs[i] - all_embs[j])
            same = all_labels[i] == all_labels[j]
            dists.append((dist, same))
    intra = [d for d, s in dists if s]
    inter = [d for d, s in dists if not s]
    # Wasserstein distance as a metric
    try:
        from scipy.stats import wasserstein_distance
        wd = wasserstein_distance(intra, inter)
    except ImportError:
        wd = np.mean(inter) - np.mean(intra)
    return wd

# Optuna objective
def objective(trial):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    margin = trial.suggest_float("margin", 0.5, 2.0)
    emb_dim = trial.suggest_categorical("emb_dim", [64, 128, 256])
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # Prepare data
    labels = [m["label"] for m in meta]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_scores = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(meta, labels)):
        train_meta = [meta[i] for i in train_idx]
        val_meta = [meta[i] for i in val_idx]
        train_ds = KidneyDataset(train_meta, IMG_DIR, transform)
        val_ds = KidneyDataset(val_meta, IMG_DIR, transform)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        model = MobileNetV2WithMeta(meta_dim=3, emb_dim=emb_dim).to(device)
        # Load previous best model if exists and emb_dim matches
        meta_path = MODEL_SAVE_PATH.replace(".pth", "_meta.json")
        if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta_info = json.load(f)
                if meta_info.get("emb_dim") == emb_dim:
                    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
                    print(f"Loaded previous best model from {MODEL_SAVE_PATH} for continued fine-tuning (emb_dim={emb_dim}).")
                else:
                    print(f"Previous best model emb_dim ({meta_info.get('emb_dim')}) does not match current emb_dim ({emb_dim}), skipping load.")
            except Exception as e:
                print(f"Failed to load previous best model: {e}")
        criterion = TripletLoss(margin=margin)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_score = -np.inf
        for epoch in range(10):  # 10 epochs per fold
            train_epoch(model, train_loader, optimizer, criterion, device)
            score = eval_epoch(model, val_loader, device)
            if score > best_score:
                best_score = score
                if fold == 0:  # Save best model of first fold
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                    with open(meta_path, "w") as f:
                        json.dump({"emb_dim": emb_dim}, f)
        val_scores.append(best_score)
    return np.mean(val_scores)

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:")
    print(study.best_trial)
    print("Best model saved to", MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()
