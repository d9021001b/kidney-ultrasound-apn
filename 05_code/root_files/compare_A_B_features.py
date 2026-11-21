import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

# --- Config ---
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_TO_COMPARE = [
    {
        "name": "A_nckuh",
        "path": "A_nckuh_model.pth",
        "data_dir": "nckuh_0_1_inner"
    },
    {
        "name": "B_cych",
        "path": "B_cych_model.pth",
        "data_dir": "cych_0_1_inner"
    }
]

# --- Dataset ---
class CustomFileDataset(Dataset):
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
        return img, label

# --- Transform ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(pretrained_path):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    if os.path.exists(pretrained_path):
        print(f"載入模型: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    else:
        print(f"警告: 找不到模型 {pretrained_path}，將使用隨機初始化權重。")
        return None
    
    # 我們只需要特徵提取部分
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.add_module("flatten", nn.Flatten())
    
    return feature_extractor.to(DEVICE).eval()

def extract_features(model, data_dir):
    file_label_items = []
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        file_label_items.extend([(f, int(label)) for f in files])
        
    dataset = CustomFileDataset(file_label_items, transform=val_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"提取特徵 from {data_dir}"):
            imgs = imgs.to(DEVICE)
            features = model(imgs)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.concatenate(all_features, axis=0), np.array(all_labels)

def plot_tsne(features, labels, title):
    print(f"開始 t-SNE 降維 for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e'] # 藍色 for class 0, 橘色 for class 1
    
    for label_val in np.unique(labels):
        idx = labels == label_val
        plt.scatter(
            features_2d[idx, 0], 
            features_2d[idx, 1], 
            c=colors[label_val], 
            label=f'Class {label_val}',
            alpha=0.7,
            s=20
        )
        
    plt.title(f't-SNE of Features - {title}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)
    
    save_path = f"{title}_feature_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"特徵分佈圖已儲存至: {save_path}")
    plt.close()

if __name__ == "__main__":
    for model_info in MODELS_TO_COMPARE:
        print(f"\n--- 處理模型: {model_info['name']} ---")
        
        model = get_model(model_info['path'])
        if model is None:
            continue
            
        features, labels = extract_features(model, model_info['data_dir'])
        
        plot_tsne(features, labels, model_info['name'])
