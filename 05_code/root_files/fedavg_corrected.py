"""
Corrected Federated Averaging (FedAvg)
=======================================
修正原版兩個問題：
  1. modelA = fed_model 是 reference copy → A 和 Fed 存出相同權重
  2. B 從 A 的權重開始 → 非標準 FedAvg

標準 FedAvg 流程：
  每輪：global_model → 複製給 A 和 B → A 在 NCKUH 訓練、B 在 CYCH 訓練
       → federated_average(A, B) → 新的 global_model

三個模型在存檔時保證不同：
  - A_nckuh_model.pth:  NCKUH site-specific model（global + 1 epoch NCKUH）
  - B_cych_model.pth:   CYCH site-specific model（global + 1 epoch CYCH）
  - federated_model.pth: global averaged model（A 和 B 的平均）
"""

import os
import json
import copy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# --- Config ---
BATCH_SIZE = 16
MAX_ROUNDS = 1000
LR = 1e-4
WEIGHT_DECAY = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INIT_MIN_TPR_TNR = 0.82   # 初始門檻
PATIENCE = 100             # global model 在兩院區都無改善的輪數上限

NCKUH_DIR = "nckuh_0_1_inner"
CYCH_DIR = "cych_0_1_inner"

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
        if not os.path.isdir(label_dir):
            continue
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
    loader = DataLoader(CustomFileDataset(file_label_items, transform=transform),
                        batch_size=BATCH_SIZE, shuffle=False)
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        "min_tpr_tnr": round(min(tpr, tnr), 4),
        "tpr": round(tpr, 4),
        "tnr": round(tnr, 4),
        "accuracy": round((tp + tn) / len(y_true), 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }

def train_one_epoch(model, dataset_dir, transform):
    model.train()
    file_label_items = get_file_label_items(dataset_dir)
    loader = DataLoader(CustomFileDataset(file_label_items, transform=transform),
                        batch_size=BATCH_SIZE, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    return model

def federated_average(modelA, modelB):
    """回傳全新的模型，權重為 A 和 B 的逐層平均"""
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
    print(f"Device: {DEVICE}")
    print(f"NCKUH: {NCKUH_DIR}, CYCH: {CYCH_DIR}")
    print(f"Max rounds: {MAX_ROUNDS}, Init threshold: {INIT_MIN_TPR_TNR}")

    # ====== 初始化 global model ======
    global_model = get_model().to(DEVICE)

    best_global_score = 0      # min of (global on NCKUH, global on CYCH)
    best_round = 0
    best_states = {}           # 存最佳輪次的三個模型 state_dict
    no_improve_count = 0
    history = []

    for round_num in range(MAX_ROUNDS):
        # ====== 1. 兩個 client 各自從 global model 出發 ======
        modelA = get_model().to(DEVICE)
        modelA.load_state_dict(copy.deepcopy(global_model.state_dict()))

        modelB = get_model().to(DEVICE)
        modelB.load_state_dict(copy.deepcopy(global_model.state_dict()))

        # ====== 2. Site A: NCKUH 訓練 1 epoch ======
        modelA = train_one_epoch(modelA, NCKUH_DIR, train_transform)

        # ====== 3. Site B: CYCH 訓練 1 epoch ======
        modelB = train_one_epoch(modelB, CYCH_DIR, train_transform)

        # ====== 4. Federated Averaging → 新 global model ======
        global_model = federated_average(modelA, modelB)

        # ====== 5. 評估三個模型 ======
        # A 在自己院區
        evalA_nckuh = evaluate(modelA, NCKUH_DIR, val_transform)
        # B 在自己院區
        evalB_cych = evaluate(modelB, CYCH_DIR, val_transform)
        # Global 在兩個院區
        evalG_nckuh = evaluate(global_model, NCKUH_DIR, val_transform)
        evalG_cych = evaluate(global_model, CYCH_DIR, val_transform)

        # Global 的跨院區綜合分數
        global_cross_score = min(evalG_nckuh["min_tpr_tnr"], evalG_cych["min_tpr_tnr"])

        record = {
            "round": round_num + 1,
            "A_on_NCKUH": evalA_nckuh,
            "B_on_CYCH": evalB_cych,
            "Global_on_NCKUH": evalG_nckuh,
            "Global_on_CYCH": evalG_cych,
            "global_cross_score": round(global_cross_score, 4),
        }
        history.append(record)

        # 印出進度（每 10 輪或有改善時印）
        improved = global_cross_score > best_global_score
        if (round_num + 1) % 10 == 0 or improved:
            tag = " ★ NEW BEST" if improved else ""
            print(f"Round {round_num+1:4d} | "
                  f"A(NCKUH) min={evalA_nckuh['min_tpr_tnr']:.4f} | "
                  f"B(CYCH) min={evalB_cych['min_tpr_tnr']:.4f} | "
                  f"Global(NCKUH) min={evalG_nckuh['min_tpr_tnr']:.4f} "
                  f"Global(CYCH) min={evalG_cych['min_tpr_tnr']:.4f} | "
                  f"cross={global_cross_score:.4f}{tag}")

        # ====== 6. 追蹤最佳模型 ======
        if global_cross_score > best_global_score:
            best_global_score = global_cross_score
            best_round = round_num + 1
            # deep copy 三個模型的權重
            best_states = {
                "A_nckuh": copy.deepcopy(modelA.state_dict()),
                "B_cych": copy.deepcopy(modelB.state_dict()),
                "federated": copy.deepcopy(global_model.state_dict()),
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        # ====== 7. Early stopping ======
        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping at round {round_num+1} "
                  f"(no improvement in {PATIENCE} rounds)")
            break

    # ====== 儲存最佳模型 ======
    print(f"\n{'='*60}")
    print(f"Best round: {best_round} (global cross-site score: {best_global_score:.4f})")
    print(f"{'='*60}")

    torch.save(best_states["A_nckuh"], "A_nckuh_model.pth")
    torch.save(best_states["B_cych"], "B_cych_model.pth")
    torch.save(best_states["federated"], "federated_model.pth")
    print("Saved: A_nckuh_model.pth, B_cych_model.pth, federated_model.pth")

    # ====== 載入最佳模型做最終報告 ======
    final_report = {}
    for name, state in best_states.items():
        model = get_model().to(DEVICE)
        model.load_state_dict(state)
        final_report[name] = {
            "on_NCKUH": evaluate(model, NCKUH_DIR, val_transform),
            "on_CYCH": evaluate(model, CYCH_DIR, val_transform),
        }

    print(f"\n{'Model':<15} {'Test Site':<8} {'Acc':>6} {'TPR':>6} {'TNR':>6} {'min':>6}")
    print("-" * 50)
    for name in ["A_nckuh", "B_cych", "federated"]:
        for site in ["on_NCKUH", "on_CYCH"]:
            m = final_report[name][site]
            print(f"{name:<15} {site:<8} "
                  f"{m['accuracy']:>6.4f} {m['tpr']:>6.4f} "
                  f"{m['tnr']:>6.4f} {m['min_tpr_tnr']:>6.4f}")

    # ====== 儲存完整結果 ======
    os.makedirs("results", exist_ok=True)
    output = {
        "best_round": best_round,
        "best_global_cross_score": best_global_score,
        "final_report": final_report,
    }
    with open("results/fedavg_corrected_result.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    with open("results/fedavg_corrected_history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print("\nResults saved to results/fedavg_corrected_*.json")
