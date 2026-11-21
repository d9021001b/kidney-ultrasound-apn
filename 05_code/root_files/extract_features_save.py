import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Config ---
DATA_DIRS = ['nckuh_0_1_inner', 'cych_0_1_inner']
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'densenet121'
WEIGHTS_PATH = 'best_densenet121_min_tpr_tnr_0.963.pth'
FEATURES_OUT = 'densenet121_all_features.npz'

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
def get_feature_extractor(model_name, weights_path):
    model_func = getattr(models, model_name)
    # Use weights=None to avoid deprecation warning
    model = model_func(weights=None)
    # For densenet121: classifier is nn.Linear, features is nn.Sequential
    if model_name == "densenet121":
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
        feature_layer = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    else:
        raise ValueError(f"Unknown model head for {model_name}")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
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

def main():
    dataset = KidneyDataset(DATA_DIRS, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model, feature_layer = get_feature_extractor(MODEL_NAME, WEIGHTS_PATH)
    features, labels = extract_features(model, feature_layer, loader)
    np.savez(FEATURES_OUT, features=features, labels=labels)
    print(f"Saved features to {FEATURES_OUT}")

if __name__ == '__main__':
    main()
