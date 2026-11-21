import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 1  # 每次只訓練一個 epoch
MAX_ROUNDS = 1000
LR = 1e-4
WEIGHT_DECAY = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MIN_TPR_TNR = 0.82

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

def get_file_label_items(data_dir):
    items = []
    for label in ['0', '1']:
        label_dir = os.path.join(data_dir, label)
        files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.lower().endswith('.png')]
        items.extend([(f, int(label)) for f in files])
    return items

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

def get_model():
    model = models.densenet121(weights="IMAGENET1K_V1")
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)
    return model

def evaluate(model, dataset_dir, transform):
    model.eval()
    file_label_items = get_file_label_items(dataset_dir)
    loader = DataLoader(CustomFileDataset(file_label_items, transform=transform), batch_size=BATCH_SIZE, shuffle=False)
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

def train_one_epoch(model, dataset_dir, transform):
    model.train()
    file_label_items = get_file_label_items(dataset_dir)
    loader = DataLoader(CustomFileDataset(file_label_items, transform=transform), batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model = model.to(DEVICE)
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

def federated_average(modelA, modelB):
    stateA = modelA.state_dict()
    stateB = modelB.state_dict()
    avg_state = {}
    for k in stateA:
        if stateA[k].shape == stateB[k].shape:
            avg_state[k] = (stateA[k] + stateB[k]) / 2
        else:
            avg_state[k] = stateA[k]
    new_model = get_model().to(DEVICE)
    new_model.load_state_dict(avg_state)
    return new_model

if __name__ == "__main__":
    nckuh_dir = "nckuh_0_1_inner"
    cych_dir = "cych_0_1_inner"
    # 初始化模型
    modelA = get_model().to(DEVICE)
    modelB = get_model().to(DEVICE)
    fed_model = get_model().to(DEVICE)
    round_num = 0
    while round_num < MAX_ROUNDS:
        print(f"\n=== Round {round_num+1} ===")
        # 1. nckuh_0_1_inner 微調A一代
        modelA = train_one_epoch(modelA, nckuh_dir, train_transform)
        # 2. cych_0_1_inner 微調A一代得B
        modelB.load_state_dict(modelA.state_dict())
        modelB = train_one_epoch(modelB, cych_dir, train_transform)
        # 3. 聯邦式學習融合A,B得新A
        fed_model = federated_average(modelA, modelB)
        modelA = fed_model  # 讓A進入下一輪
        # 4. 驗證三模型
        minA, tprA, tnrA = evaluate(modelA, nckuh_dir, val_transform)
        minB, tprB, tnrB = evaluate(modelB, cych_dir, val_transform)
        minFed, tprFed, tnrFed = evaluate(fed_model, cych_dir, val_transform)
        # print(f"A(nckuh) min(TPR,TNR)={minA:.4f} (TPR={tprA:.4f}, TNR={tnrA:.4f})")
        # print(f"B(cych) min(TPR,TNR)={minB:.4f} (TPR={tprB:.4f}, TNR={tnrB:.4f})")
        # print(f"Fed(cych) min(TPR,TNR)={minFed:.4f} (TPR={tprFed:.4f}, TNR={tnrFed:.4f})")
        # 5. 判斷是否全部達標
        if minA >= MIN_TPR_TNR and minB >= MIN_TPR_TNR: #and minFed >= MIN_TPR_TNR:
            MIN_TPR_TNR = min(minA, minB)
            #print("三模型 min(TPR,TNR) 均達標，儲存模型並結束。")
            print(f"A(nckuh) min(TPR,TNR)={minA:.4f} (TPR={tprA:.4f}, TNR={tnrA:.4f})")
            print(f"B(cych) min(TPR,TNR)={minB:.4f} (TPR={tprB:.4f}, TNR={tnrB:.4f})")
            print(f"Fed(cych) min(TPR,TNR)={minFed:.4f} (TPR={tprFed:.4f}, TNR={tnrFed:.4f})")
            torch.save(modelA.state_dict(), "A_nckuh_model.pth")
            torch.save(modelB.state_dict(), "B_cych_model.pth")
            torch.save(fed_model.state_dict(), "federated_model.pth")
            #break
        round_num += 1
    else:
        print("未在最大輪數內全部達標。")
