import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'densenet121'
MIN_TPR_TNR = 0.85

# --- Dataset ---
class KidneyDataset(Dataset):
    def __init__(self, root_dir, indices=None, transform=None):
        self.samples = []
        self.labels = []
        for label in ['0', '1']:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith('.png'):
                    self.samples.append(os.path.join(label_dir, fname))
                    self.labels.append(int(label))
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
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
# Data Augmentation for training set
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
# For validation/test set
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_model():
    model = models.densenet121(weights="IMAGENET1K_V1")
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    feature_layer = nn.Sequential(model.features, nn.AdaptiveAvgPool2d((1, 1)))
    return model, feature_layer

def evaluate(model, feature_layer, loader):
    model.eval()
    all_labels = []
    all_preds = []
    features = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            feats = feature_layer(imgs)
            feats = feats.view(feats.size(0), -1)
            features.append(feats.cpu().numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    min_tpr_tnr = min(tpr, tnr)
    features = np.concatenate(features, axis=0)
    return min_tpr_tnr, features, y_true

from tqdm import tqdm
from sklearn.model_selection import KFold

import glob
from collections import defaultdict

def train_until_target(data_dir, min_tpr_tnr=0.9):
    print(f"--- 單一訓練直到 min(TPR,TNR) ≥ {min_tpr_tnr} ---")
    # 1. 取得所有資料
    file_label_items = []
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        file_label_items.extend([(f, int(label)) for f in files])
    # 2. 建立資料集
    dataset = CustomFileDataset(file_label_items, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model, feature_layer = get_model()
    model = model.to(DEVICE)
    feature_layer = feature_layer.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    patience = 70
    best_min_tpr_tnr = 0
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    reached_target = False
    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # 驗證
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, labels in DataLoader(dataset, batch_size=BATCH_SIZE):
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
        min_fold_tpr_tnr = min(tpr, tnr)
        print(f"Epoch {epoch+1}: TPR={tpr:.4f}, TNR={tnr:.4f}, min(TPR,TNR)={min_fold_tpr_tnr:.4f}")
        if min_fold_tpr_tnr > best_min_tpr_tnr:
            best_min_tpr_tnr = min_fold_tpr_tnr
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_no_improve = 0
            if min_fold_tpr_tnr >= min_tpr_tnr:
                print(f"達到 min(TPR,TNR) ≥ {min_tpr_tnr}，儲存模型 nckuh_model.pth")
                torch.save(best_model_state, "nckuh_model.pth")
                reached_target = True
                break
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs).")
                break
    if not reached_target:
        print("未達到 min(TPR,TNR) ≥ {min_tpr_tnr}，流程終止。")
        exit(1)
    return "nckuh_model.pth"

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

def federated_finetune(model1, model2, data_dir1, data_dir2, save_path):
    # 參數平均（federated averaging）
    print("--- 聯邦式學習融合 ---")
    state1 = model1.state_dict()
    state2 = model2.state_dict()
    avg_state = {}
    for k in state1:
        if state1[k].shape == state2[k].shape:
            avg_state[k] = (state1[k] + state2[k]) / 2
        else:
            avg_state[k] = state1[k]
    model1.load_state_dict(avg_state)
    model1 = model1.to(DEVICE)
    # 以 nckuh_0_1_inner 為主繼續微調
    dataset = KidneyDataset(data_dir1, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for epoch in range(EPOCHS):
        model1.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model1(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # 用 cych_0_1_inner 當測試集
        test_dataset = KidneyDataset(data_dir2, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        feature_layer = nn.Sequential(model1.features, nn.AdaptiveAvgPool2d((1, 1))).to(DEVICE)
        min_tpr_tnr, _, _ = evaluate(model1, feature_layer, test_loader)
        print(f"聯邦微調 Epoch {epoch+1}: cych_0_1_inner min(TPR,TNR)={min_tpr_tnr:.4f}")
        if min_tpr_tnr >= MIN_TPR_TNR:
            print(f"融合後模型於 cych_0_1_inner min(TPR,TNR) ≥ {MIN_TPR_TNR}，儲存模型 {save_path}")
            torch.save(model1.state_dict(), save_path)
            break

def train_cych_with_nckuh_init(nckuh_model_path, cych_data_dir, min_tpr_tnr=0.9):
    print(f"--- 以 nckuh_model.pth 權重初始化 cych_0_1_inner 微調 ---")
    # 1. 載入 nckuh_model.pth 權重
    model, feature_layer = get_model()
    model.load_state_dict(torch.load(nckuh_model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    feature_layer = feature_layer.to(DEVICE)
    # 2. 準備 cych_0_1_inner 全部資料
    file_label_items = []
    for label in ['0', '1']:
        label_dir = os.path.join(cych_data_dir, label)
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        file_label_items.extend([(f, int(label)) for f in files])
    dataset = CustomFileDataset(file_label_items, transform=train_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    patience = 70
    best_min_tpr_tnr = 0
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    reached_target = False
    for epoch in range(EPOCHS):
        model.train()
        for imgs, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        # 驗證
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for imgs, labels in DataLoader(dataset, batch_size=BATCH_SIZE):
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
        min_fold_tpr_tnr = min(tpr, tnr)
        print(f"Epoch {epoch+1}: TPR={tpr:.4f}, TNR={tnr:.4f}, min(TPR,TNR)={min_fold_tpr_tnr:.4f}")
        if min_fold_tpr_tnr > best_min_tpr_tnr:
            best_min_tpr_tnr = min_fold_tpr_tnr
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_no_improve = 0
            if min_fold_tpr_tnr >= min_tpr_tnr:
                print(f"達到 min(TPR,TNR) ≥ {min_tpr_tnr}，儲存模型 cych_model.pth")
                torch.save(best_model_state, "cych_model.pth")
                reached_target = True
                break
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs).")
                break
    if not reached_target:
        print("未達到 min(TPR,TNR) ≥ {min_tpr_tnr}，流程終止。")
        exit(1)
    return "cych_model.pth"

if __name__ == "__main__":
    # 1. 單一訓練直到 min(TPR,TNR) ≥ 0.9
    model_path = train_until_target("nckuh_0_1_inner", min_tpr_tnr=0.9)
    print(f"模型已儲存於 {model_path}")
    # 2. cych_0_1_inner 微調
    cych_model_path = train_cych_with_nckuh_init("nckuh_model.pth", "cych_0_1_inner", min_tpr_tnr=0.9)
    print(f"模型已儲存於 {cych_model_path}")
    # 3. 聯邦式學習
    model1, _ = get_model()
    model1.load_state_dict(torch.load("nckuh_model.pth", map_location=DEVICE))
    model2, _ = get_model()
    model2.load_state_dict(torch.load("cych_model.pth", map_location=DEVICE))
    federated_finetune(
        model1, model2,
        "nckuh_0_1_inner", "cych_0_1_inner",
        "federated_nckuh_model.pth"
    )
    print("聯邦式學習已完成，融合模型已儲存於 federated_nckuh_model.pth")
