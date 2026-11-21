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
MIN_TPR_TNR = 0.95

# --- Dataset ---
class KidneyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
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

def finetune_on_dataset(data_dir, model_save_path, features_save_path):
    print(f"--- 微調 {data_dir} ---")
    dataset = KidneyDataset(data_dir, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    patience = 7
    # 每次都重新初始化模型
    model, feature_layer = get_model()
    model = model.to(DEVICE)
    feature_layer = feature_layer.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_min_tpr_tnr = 0
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    reached_target = False

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0
        for imgs, labels in tqdm(loader, desc=f"{data_dir} Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(dataset)

        # Evaluate on all data (no validation split, like finetune_cnn_on_inner.py)
        model.eval()
        all_labels = []
        all_preds = []
        features = []
        with torch.no_grad():
            for imgs, labels in DataLoader(dataset, batch_size=BATCH_SIZE):
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
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, TPR={tpr:.4f}, TNR={tnr:.4f}, min(TPR,TNR)={min_tpr_tnr:.4f}")

        if min_tpr_tnr > best_min_tpr_tnr:
            best_min_tpr_tnr = min_tpr_tnr
            best_model_state = model.state_dict()
            best_epoch = epoch + 1
            epochs_no_improve = 0
            if min_tpr_tnr >= MIN_TPR_TNR:
                print(f"達到 min(TPR,TNR) ≥ {MIN_TPR_TNR}，儲存模型 {model_save_path}")
                torch.save(best_model_state, model_save_path)
                reached_target = True
                break
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {patience} epochs).")
                break

    if not reached_target:
        print(f"未達到 min(TPR,TNR) ≥ {MIN_TPR_TNR}，不儲存模型與特徵。")
        return None, None

    # 用全資料集微調並儲存特徵
    model.load_state_dict(torch.load(model_save_path, map_location=DEVICE))
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    min_tpr_tnr_full, features, y_true = evaluate(model, feature_layer, full_loader)
    if min_tpr_tnr_full >= MIN_TPR_TNR:
        print(f"全資料集 min(TPR,TNR)={min_tpr_tnr_full:.4f}，儲存特徵 {features_save_path}")
        np.savez(features_save_path, features=features, labels=y_true)
    else:
        print(f"全資料集 min(TPR,TNR)={min_tpr_tnr_full:.4f}，未儲存特徵")
    return model, feature_layer

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
    # 以 nckuh_0_1_inner 為主繼續微調
    dataset = KidneyDataset(data_dir1, transform)
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
        test_dataset = KidneyDataset(data_dir2, transform)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        min_tpr_tnr, _, _ = evaluate(model1, nn.Sequential(model1.features, nn.AdaptiveAvgPool2d((1, 1))).to(DEVICE), test_loader)
        print(f"聯邦微調 Epoch {epoch+1}: cych_0_1_inner min(TPR,TNR)={min_tpr_tnr:.4f}")
        if min_tpr_tnr >= MIN_TPR_TNR:
            print(f"融合後模型於 cych_0_1_inner min(TPR,TNR) ≥ {MIN_TPR_TNR}，儲存模型 {save_path}")
            torch.save(model1.state_dict(), save_path)
            break

if __name__ == "__main__":
    # 第一階段：nckuh_0_1_inner
    model1, feature_layer1 = finetune_on_dataset(
        "nckuh_0_1_inner",
        "nckuh_model.pth",
        "nckuh_features.npz"
    )
    # 第二階段：cych_0_1_inner
    model2, feature_layer2 = finetune_on_dataset(
        "cych_0_1_inner",
        "cych_model.pth",
        "cych_features.npz"
    )
    # 聯邦式學習融合
    federated_finetune(
        model1, model2,
        "nckuh_0_1_inner", "cych_0_1_inner",
        "federated_nckuh_model.pth"
    )
