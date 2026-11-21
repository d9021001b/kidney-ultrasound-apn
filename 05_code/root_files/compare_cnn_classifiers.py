import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from PIL import Image
import json
import os
import numpy as np
import pandas as pd

# --- 設定參數 ---
DATASETS = {
    "nckuh": {
        "img_dir": "nckuh_bg_black",
        "metadata_path": "nckuh_metadata.json"
    },
    "cych": {
        "img_dir": "cych_bg_black",
        "metadata_path": "cych_metadata.json"
    }
}
MODELS_TO_TEST = ['resnet50', 'densenet121', 'efficientnet_b0', 'mobilenet_v3_large']
NUM_FOLDS = 5
NUM_EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. 資料集定義 ---
class KidneyDataset(Dataset):
    def __init__(self, metadata, img_dirs, transform=None):
        self.metadata = metadata
        self.img_dirs = img_dirs
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        filename = item['filename']
        label = int(item['label'])
        
        # 找到對應的圖片路徑
        img_path = None
        for dir_name, dir_path in self.img_dirs.items():
            potential_path = os.path.join(dir_path, filename)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            # 如果找不到檔案，回傳 None，由 collate_fn 處理
            return None

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 處理 metadata
        gender = 1 if item['gender'] == 'male' else 0
        meta_features = torch.tensor([gender, item['long_axis'], item['short_axis']], dtype=torch.float32)

        return image, meta_features, torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    """過濾掉資料集中找不到檔案的 None 項目"""
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # 如果整個批次都被過濾掉了，返回空的 Tensors
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    return torch.utils.data.dataloader.default_collate(batch)

# --- 2. 模型定義 ---
class CombinedModel(nn.Module):
    def __init__(self, backbone_name, num_meta_features=3, pretrained=True):
        super(CombinedModel, self).__init__()
        # 影像特徵提取
        if backbone_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone_name == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_ftrs = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone_name == 'mobilenet_v3_large':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            num_ftrs = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Metadata 處理
        self.meta_fc = nn.Sequential(
            nn.Linear(num_meta_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )

        # 分類器
        self.classifier = nn.Linear(num_ftrs + 8, 2)

    def forward(self, image, meta):
        img_features = self.backbone(image)
        meta_features = self.meta_fc(meta)
        combined_features = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined_features)
        return output

# --- 3. 訓練與評估函數 ---
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for images, metas, labels in loader:
        # 如果批次是空的，則跳過
        if images.nelement() == 0:
            continue
            
        images, metas, labels = images.to(device), metas.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images, metas)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    if total_samples == 0:
        return 0.0
    return running_loss / total_samples

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, metas, labels in loader:
            # 如果批次是空的，則跳過
            if images.nelement() == 0:
                continue
            images, metas, labels = images.to(device), metas.to(device), labels.to(device)
            outputs = model(images, metas)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if not all_labels: # 如果沒有任何有效的樣本被評估
        return {
            "accuracy": 0, "tpr": 0, "tnr": 0, "precision": 0, "f1": 0
        }
            
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall / Sensitivity
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "tnr": tnr,
        "precision": precision,
        "f1": f1
    }

# --- 4. 主執行流程 ---
def main():
    # 載入並合併 metadata
    all_metadata = []
    img_dirs = {}
    for name, paths in DATASETS.items():
        img_dirs[name] = paths['img_dir']
        with open(paths['metadata_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 加上來源標記，方便追蹤
            for item in data:
                item['source'] = name
            all_metadata.extend(data)

    # 影像轉換
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 建立完整資料集
    full_dataset = KidneyDataset(all_metadata, img_dirs, transform)
    
    # 準備紀錄結果
    results = []

    for model_name in MODELS_TO_TEST:
        print(f"--- Testing Model: {model_name} ---")
        
        fold_metrics = []
        
        # 5-Fold CV
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
        labels = [item['label'] for item in all_metadata]

        for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
            print(f"  Fold {fold+1}/{NUM_FOLDS}")
            
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

            model = CombinedModel(model_name).to(DEVICE)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

            for epoch in range(NUM_EPOCHS):
                train_loss = train_model(model, train_loader, criterion, optimizer, DEVICE)
                print(f"    Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}")

            metrics = evaluate_model(model, val_loader, DEVICE)
            fold_metrics.append(metrics)
            print(f"    Fold {fold+1} Metrics: {metrics}")

        # 計算平均指標
        avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
        print(f"\n  Average Metrics for {model_name}: {avg_metrics}\n")
        
        results.append({
            "model": model_name,
            "avg_metrics": avg_metrics,
            "fold_metrics": fold_metrics
        })

    # 儲存結果
    df = pd.DataFrame([{
        "Model": r["model"],
        "Accuracy": r["avg_metrics"]["accuracy"],
        "TPR": r["avg_metrics"]["tpr"],
        "TNR": r["avg_metrics"]["tnr"],
        "Precision": r["avg_metrics"]["precision"],
        "F1-Score": r["avg_metrics"]["f1"]
    } for r in results])
    
    df.to_csv("cnn_comparison_results.csv", index=False)
    print("Comparison finished. Results saved to cnn_comparison_results.csv")
    print(df)

if __name__ == '__main__':
    main()
