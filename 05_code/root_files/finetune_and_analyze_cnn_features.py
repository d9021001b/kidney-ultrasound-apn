"""
Recommended CNNs for Ultrasound Feature Extraction (PyTorch, torchvision.models):
1. resnet50 (ResNet-50) - High accuracy, widely used, robust.
2. efficientnet_b0 - Modern, efficient, good performance/size tradeoff.
3. densenet121 - Dense connections, good for medical images.
4. mobilenet_v3_large - Lightweight, fast, suitable for limited resources.
5. swin_v2_s (Swin Transformer V2 Small) - Transformer-based, strong on vision tasks.

All support feature extraction and fine-tuning via torchvision.models.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance

# 1. Dataset
class USImageDataset(Dataset):
    def __init__(self, img_dir, metadata_json, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(metadata_json, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.meta_map = {os.path.splitext(m["filename"])[0]: m for m in self.metadata}
        # Preprocess gender to int, axes to float
        for m in self.metadata:
            m["gender_num"] = 1 if m.get("gender", "").lower() in ["m", "male", "男"] else 0
            m["long_axis"] = float(m.get("long_axis", 0))
            m["short_axis"] = float(m.get("short_axis", 0))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        base = os.path.splitext(img_name)[0]
        meta = self.meta_map.get(base, {})
        gender = meta.get("gender_num", 0)
        long_axis = meta.get("long_axis", 0.0)
        short_axis = meta.get("short_axis", 0.0)
        if self.transform:
            img = self.transform(img)
        meta_vec = torch.tensor([gender, long_axis, short_axis], dtype=torch.float32)
        return img, meta_vec, img_name

# 2. Model selection
def get_model(model_name):
    if model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        feat_dim = model.classifier.in_features
        model.classifier = nn.Identity()
    elif model_name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        feat_dim = model.classifier[3].in_features
        model.classifier = nn.Identity()
    elif model_name == "swin_v2_s":
        model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.IMAGENET1K_V1)
        feat_dim = model.head.in_features
        model.head = nn.Identity()
    else:
        raise ValueError("Unknown model")
    return model, feat_dim

# 3. Feature extraction
def extract_features(model, dataloader, device):
    model.eval()
    feats = []
    metas = []
    names = []
    with torch.no_grad():
        for imgs, meta_vecs, img_names in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            out = out.cpu().numpy()
            feats.append(out)
            metas.append(meta_vecs.numpy())
            names.extend(img_names)
    feats = np.concatenate(feats, axis=0)
    metas = np.concatenate(metas, axis=0)
    return feats, metas, names

# 4. Integration: concatenate CNN features with normalized meta
def integrate_features(cnn_feats, meta_feats):
    # Normalize meta features (gender: already 0/1, axes: z-score)
    meta_norm = meta_feats.copy()
    if meta_norm.shape[1] > 1:
        meta_norm[:,1:] = (meta_norm[:,1:] - meta_norm[:,1:].mean(axis=0)) / (meta_norm[:,1:].std(axis=0) + 1e-8)
    return np.concatenate([cnn_feats, meta_norm], axis=1)

# 5. Visualization and distance
def plot_tsne(feats, meta, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    emb = tsne.fit_transform(feats)
    plt.figure(figsize=(8,6))
    plt.scatter(emb[:,0], emb[:,1], c=meta[:,0], cmap="coolwarm", alpha=0.7, label="gender")
    plt.colorbar(label="gender (0=female, 1=male)")
    plt.title(title)
    plt.savefig(save_path)
    plt.close()

def compute_wasserstein(feats1, feats2):
    # Use mean of each dimension, then average 1D Wasserstein distances
    dists = []
    for i in range(min(feats1.shape[1], feats2.shape[1])):
        d = wasserstein_distance(feats1[:,i], feats2[:,i])
        dists.append(d)
    return float(np.mean(dists))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_names = ["resnet50", "efficientnet_b0", "densenet121", "mobilenet_v3_large", "swin_v2_s"]
    cych_img_dir = "cych_bg_black"
    cych_meta_json = "cych_metadata.json"
    nckuh_img_dir = "nckuh_bg_black"
    nckuh_meta_json = "nckuh_metadata.json"
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    cych_ds = USImageDataset(cych_img_dir, cych_meta_json, transform)
    nckuh_ds = USImageDataset(nckuh_img_dir, nckuh_meta_json, transform)
    cych_loader = DataLoader(cych_ds, batch_size=32, shuffle=False)
    nckuh_loader = DataLoader(nckuh_ds, batch_size=32, shuffle=False)

    for model_name in model_names:
        print(f"Processing {model_name} ...")
        model, feat_dim = get_model(model_name)
        model = model.to(device)
        # Optionally: fine-tune here (not implemented for brevity)
        # Extract features
        cych_feats, cych_metas, cych_names = extract_features(model, cych_loader, device)
        nckuh_feats, nckuh_metas, nckuh_names = extract_features(model, nckuh_loader, device)
        # Integrate
        cych_all = integrate_features(cych_feats, cych_metas)
        nckuh_all = integrate_features(nckuh_feats, nckuh_metas)
        # Save features
        np.save(f"cych_{model_name}_features.npy", cych_all)
        np.save(f"nckuh_{model_name}_features.npy", nckuh_all)
        # Plot
        plot_tsne(cych_all, cych_metas, f"CYCH {model_name} t-SNE", f"cych_{model_name}_tsne.png")
        plot_tsne(nckuh_all, nckuh_metas, f"NCKUH {model_name} t-SNE", f"nckuh_{model_name}_tsne.png")
        # Quantify distance
        dist = compute_wasserstein(cych_all, nckuh_all)
        print(f"Wasserstein distance between CYCH and NCKUH ({model_name}): {dist:.4f}")

"""
Integration suggestion:
- Concatenate CNN feature vector with normalized gender (0/1), long_axis, short_axis.
- For downstream analysis, use the combined vector for clustering, classification, or visualization.
- t-SNE/UMAP can visualize the distribution, and Wasserstein distance quantifies the difference.
"""
