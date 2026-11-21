import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from sklearn.manifold import TSNE
from scipy.stats import wasserstein_distance

# Dataset class (same as before)
class USImageDataset(Dataset):
    def __init__(self, img_dir, metadata_json, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        with open(metadata_json, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]
        self.meta_map = {os.path.splitext(m["filename"])[0]: m for m in self.metadata}
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

# Helper to get all classification models from torchvision
def get_classification_models():
    import torchvision.models
    all_models = []
    for name in dir(torchvision.models):
        if name.startswith("_"):
            continue
        obj = getattr(torchvision.models, name)
        if callable(obj) and hasattr(obj, "__module__") and "models" in obj.__module__:
            # Try to filter out detection/segmentation/quantized/experimental
            if "quantized" in name or "segmentation" in name or "detection" in name or "video" in name or "get_" in name:
                continue
            all_models.append(name)
    return sorted(set(all_models))

# Helper to adapt model for feature extraction
def adapt_model(model, name):
    # Try common classifier attributes
    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            model.classifier = nn.Identity()
        else:
            try:
                model.classifier[-1] = nn.Identity()
            except Exception:
                model.classifier = nn.Identity()
    elif hasattr(model, "head"):
        model.head = nn.Identity()
    return model

def extract_features(model, dataloader, device):
    model.eval()
    feats = []
    metas = []
    with torch.no_grad():
        for imgs, meta_vecs, _ in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)
            out = out.cpu().numpy()
            feats.append(out)
            metas.append(meta_vecs.numpy())
    feats = np.concatenate(feats, axis=0)
    metas = np.concatenate(metas, axis=0)
    return feats, metas

def integrate_features(cnn_feats, meta_feats):
    meta_norm = meta_feats.copy()
    if meta_norm.shape[1] > 1:
        meta_norm[:,1:] = (meta_norm[:,1:] - meta_norm[:,1:].mean(axis=0)) / (meta_norm[:,1:].std(axis=0) + 1e-8)
    return np.concatenate([cnn_feats, meta_norm], axis=1)

def compute_wasserstein(feats1, feats2):
    dists = []
    for i in range(min(feats1.shape[1], feats2.shape[1])):
        d = wasserstein_distance(feats1[:,i], feats2[:,i])
        dists.append(d)
    return float(np.mean(dists))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    model_names = get_classification_models()
    results = []
    for model_name in model_names:
        try:
            # Try to get weights enum if available
            model_func = getattr(models, model_name)
            weights = None
            # Try to get default weights
            if hasattr(model_func, "default"):
                weights = model_func.default
            elif hasattr(model_func, "Weights"):
                weights = list(model_func.Weights)[0]
            elif hasattr(model_func, "__annotations__") and "weights" in model_func.__annotations__:
                weights = "IMAGENET1K_V1"
            # Try to load model with weights
            try:
                if weights is not None:
                    model = model_func(weights=weights)
                else:
                    model = model_func(pretrained=True)
            except Exception:
                model = model_func()
            model = adapt_model(model, model_name)
            model = model.to(device)
            # Extract features
            cych_feats, cych_metas = extract_features(model, cych_loader, device)
            nckuh_feats, nckuh_metas = extract_features(model, nckuh_loader, device)
            cych_all = integrate_features(cych_feats, cych_metas)
            nckuh_all = integrate_features(nckuh_feats, nckuh_metas)
            dist = compute_wasserstein(cych_all, nckuh_all)
            print(f"Wasserstein distance between CYCH and NCKUH ({model_name}): {dist:.4f}")
            results.append((model_name, dist))
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")

    if results:
        results.sort(key=lambda x: x[1])
        best_model, best_dist = results[0]
        print(f"\nRecommended model: {best_model} (Wasserstein distance: {best_dist:.4f})")
    else:
        print("No models successfully evaluated.")
