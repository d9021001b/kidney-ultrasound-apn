import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

IMG_DIR = "nckuh_bg_black"
META_PATH = "nckuh_metadata.json"
MODEL_SAVE_PATH = "best_mobilenetv2_triplet.pth"
MODEL_META_PATH = "best_mobilenetv2_triplet_meta.json"

# Load emb_dim from meta file
with open(MODEL_META_PATH, "r") as f:
    emb_dim = int(json.load(f)["emb_dim"])

# Load metadata
with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

gender_map = {"male": 0, "female": 1}
for m in meta:
    m["gender_enc"] = gender_map[m["gender"]]

# Filter meta to only include entries with existing images
meta = [m for m in meta if os.path.exists(os.path.join(IMG_DIR, m["filename"]))]
print(f"Filtered metadata: {len(meta)} samples with existing images.")

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
        meta_feat = np.array([
            entry["gender_enc"],
            entry["long_axis"],
            entry["short_axis"]
        ], dtype=np.float32)
        label = entry["label"]
        return image, meta_feat, label, entry["filename"]

class MobileNetV2WithMeta(nn.Module):
    def __init__(self, meta_dim=3, emb_dim=128):
        super().__init__()
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
        img_feat = self.backbone(x_img)
        meta_feat = self.meta_fc(x_meta)
        feat = torch.cat([img_feat, meta_feat], dim=1)
        emb = self.final_fc(feat)
        emb = nn.functional.normalize(emb, p=2, dim=1)
        return emb

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    tpr = recall_score(y_true, y_pred)  # Sensitivity, Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    f1 = f1_score(y_true, y_pred)
    return {
        "TPR": tpr,
        "TNR": tnr,
        "ACC": acc,
        "Precision": prec,
        "F1": f1
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    labels = [m["label"] for m in meta]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics_list = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(meta, labels)):
        print(f"\nFold {fold+1}")
        train_meta = [meta[i] for i in train_idx]
        val_meta = [meta[i] for i in val_idx]
        train_ds = KidneyDataset(train_meta, IMG_DIR, transform)
        val_ds = KidneyDataset(val_meta, IMG_DIR, transform)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        # Load model
        model = MobileNetV2WithMeta(meta_dim=3, emb_dim=emb_dim).to(device)
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        model.eval()
        # Extract embeddings
        def get_embs_and_labels(loader):
            embs = []
            labs = []
            with torch.no_grad():
                for imgs, metas, labels, _ in loader:
                    imgs, metas = imgs.to(device), metas.to(device)
                    emb = model(imgs, metas)
                    embs.append(emb.cpu().numpy())
                    labs.extend(labels.numpy())
            return np.concatenate(embs, axis=0), np.array(labs)
        X_train, y_train = get_embs_and_labels(train_loader)
        X_val, y_val = get_embs_and_labels(val_loader)
        # Train classifier on embeddings
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        print({k: f"{v:.4f}" for k, v in metrics.items()})
        metrics_list.append(metrics)
    # Print mean metrics
    print("\nMean 5-fold metrics:")
    mean_metrics = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
    print({k: f"{v:.4f}" for k, v in mean_metrics.items()})

if __name__ == "__main__":
    main()
