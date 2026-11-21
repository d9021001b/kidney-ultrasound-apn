import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-5  # 使用較小的學習率進行微調
WEIGHT_DECAY = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAINED_MODEL_PATH = "A_nckuh_model.pth"
DATA_DIR = "nckuh_0_1_inner"
K_FOLDS = 5
PATIENCE = 50 # Early stopping 的耐心值

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
        return img, torch.tensor(int(label), dtype=torch.long)

# --- Transform ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model(pretrained_path):
    model = models.densenet121(weights=None) # 不使用預設的 ImageNet 權重
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    if os.path.exists(pretrained_path):
        print(f"載入預訓練模型: {pretrained_path}")
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE))
    else:
        print(f"警告: 找不到預訓練模型 {pretrained_path}，將使用隨機初始化權重。")
    return model.to(DEVICE)

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for imgs, labels in loader:
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
    return min_tpr_tnr, tpr, tnr

if __name__ == "__main__":
    # 1. 準備資料
    file_label_items = []
    for label in ['0', '1']:
        label_dir = os.path.join(DATA_DIR, label)
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        file_label_items.extend([(f, int(label)) for f in files])
    
    file_label_items = np.array(file_label_items)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(file_label_items)):
        print(f"\n--- Fold {fold+1}/{K_FOLDS} ---")
        
        # 2. 準備資料集和 DataLoader
        train_items = file_label_items[train_idx]
        val_items = file_label_items[val_idx]
        
        train_dataset = CustomFileDataset(train_items, transform=train_transform)
        val_dataset = CustomFileDataset(val_items, transform=val_transform)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 3. 載入模型和設定
        model = get_model(PRETRAINED_MODEL_PATH)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        best_min_tpr_tnr = 0
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            model.train()
            for imgs, labels in tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{EPOCHS}"):
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # 4. 驗證
            min_val, tpr_val, tnr_val = evaluate(model, val_loader)
            print(f"Epoch {epoch+1}: Val min(TPR,TNR)={min_val:.4f} (TPR={tpr_val:.4f}, TNR={tnr_val:.4f})")
            
            if min_val > best_min_tpr_tnr:
                best_min_tpr_tnr = min_val
                epochs_no_improve = 0
                best_model_state = model.state_dict()
                print(f"New best model found at epoch {epoch+1} with min(TPR,TNR)={best_min_tpr_tnr:.4f}")
            else:
                epochs_no_improve += 1
            
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} after {PATIENCE} epochs with no improvement.")
                break
        
        print(f"Fold {fold+1} Best min(TPR,TNR): {best_min_tpr_tnr:.4f}")
        fold_results.append(best_min_tpr_tnr)

    print("\n--- K-Fold Cross-Validation Summary ---")
    for i, result in enumerate(fold_results):
        print(f"Fold {i+1}: {result:.4f}")
    print(f"Average min(TPR,TNR): {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
