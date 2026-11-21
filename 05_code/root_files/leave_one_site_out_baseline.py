"""
Leave-One-Site-Out Baseline
============================
實驗 A: Train on NCKUH only → Test on CYCH (CYCH 完全未參與訓練)
實驗 B: Train on CYCH only → Test on NCKUH (NCKUH 完全未參與訓練)

用途：作為 federated learning 的 baseline，量化單院區模型的跨院區 domain shift。
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- Config ---
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-4
WEIGHT_DECAY = 0
PATIENCE = 70
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NCKUH_DIR = "nckuh_0_1_inner"
CYCH_DIR = "cych_0_1_inner"

# --- Dataset ---
class KidneyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        for label in ['0', '1']:
            label_dir = os.path.join(root_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in sorted(os.listdir(label_dir)):
                if fname.lower().endswith('.png'):
                    self.samples.append(os.path.join(label_dir, fname))
                    self.labels.append(int(label))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# --- Transforms ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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


def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    acc = (tp + tn) / len(y_true)
    return {
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "tpr": round(tpr, 4), "tnr": round(tnr, 4),
        "min_tpr_tnr": round(min(tpr, tnr), 4),
        "accuracy": round(acc, 4),
        "n_samples": len(y_true),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }


def run_experiment(train_dir, test_dir, train_name, test_name, save_prefix):
    print(f"\n{'='*60}")
    print(f"Train: {train_name} ({train_dir})")
    print(f"Test:  {test_name} ({test_dir})  ← 完全未參與訓練")
    print(f"{'='*60}")

    train_dataset = KidneyDataset(train_dir, transform=train_transform)
    test_dataset = KidneyDataset(test_dir, transform=val_transform)
    # 訓練集也做一份 val_transform 版本，用來評估 train performance
    train_eval_dataset = KidneyDataset(train_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    model = get_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_train_min = 0
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0
    history = []

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        running_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # --- Evaluate on train set (用 val_transform，無 augmentation) ---
        train_metrics = evaluate(model, train_eval_loader)
        # --- Evaluate on test set ---
        test_metrics = evaluate(model, test_loader)

        history.append({
            "epoch": epoch + 1,
            "train": train_metrics,
            "test": test_metrics,
        })

        print(f"Epoch {epoch+1:3d} | "
              f"Train min(TPR,TNR)={train_metrics['min_tpr_tnr']:.4f} "
              f"(TPR={train_metrics['tpr']:.4f}, TNR={train_metrics['tnr']:.4f}) | "
              f"Test min(TPR,TNR)={test_metrics['min_tpr_tnr']:.4f} "
              f"(TPR={test_metrics['tpr']:.4f}, TNR={test_metrics['tnr']:.4f})")

        # Early stopping 依據 train set 的 min(TPR,TNR)，與你原本的邏輯一致
        if train_metrics['min_tpr_tnr'] > best_train_min:
            best_train_min = train_metrics['min_tpr_tnr']
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1} (no improvement in {PATIENCE} epochs)")
                break

    # --- 載入最佳模型，做最終評估 ---
    if best_model_state:
        model.load_state_dict(best_model_state)
    final_train = evaluate(model, train_eval_loader)
    final_test = evaluate(model, test_loader)

    print(f"\n--- Best model (epoch {best_epoch}) ---")
    print(f"Train ({train_name}): min(TPR,TNR)={final_train['min_tpr_tnr']:.4f}, "
          f"TPR={final_train['tpr']:.4f}, TNR={final_train['tnr']:.4f}, Acc={final_train['accuracy']:.4f}")
    print(f"Test  ({test_name}): min(TPR,TNR)={final_test['min_tpr_tnr']:.4f}, "
          f"TPR={final_test['tpr']:.4f}, TNR={final_test['tnr']:.4f}, Acc={final_test['accuracy']:.4f}")

    # --- 儲存 ---
    torch.save(best_model_state, f"{save_prefix}_model.pth")

    result = {
        "experiment": f"Train={train_name}, Test={test_name}",
        "best_epoch": best_epoch,
        "train_final": final_train,
        "test_final": final_test,
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{save_prefix}_result.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # 也存完整 history 供畫圖用
    with open(f"results/{save_prefix}_history.json", "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return result


if __name__ == "__main__":
    results = {}

    # 實驗 A: NCKUH → CYCH
    results["A"] = run_experiment(
        train_dir=NCKUH_DIR, test_dir=CYCH_DIR,
        train_name="NCKUH", test_name="CYCH",
        save_prefix="loso_trainNCKUH_testCYCH"
    )

    # 實驗 B: CYCH → NCKUH
    results["B"] = run_experiment(
        train_dir=CYCH_DIR, test_dir=NCKUH_DIR,
        train_name="CYCH", test_name="NCKUH",
        save_prefix="loso_trainCYCH_testNCKUH"
    )

    # --- 總結 ---
    print(f"\n{'='*60}")
    print("Leave-One-Site-Out Baseline Summary")
    print(f"{'='*60}")
    for key, r in results.items():
        t = r["test_final"]
        print(f"Exp {key} [{r['experiment']}]")
        print(f"  Cross-site test: min(TPR,TNR)={t['min_tpr_tnr']:.4f}, "
              f"TPR={t['tpr']:.4f}, TNR={t['tnr']:.4f}, Acc={t['accuracy']:.4f}")

    with open("results/loso_baseline_summary.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nAll results saved to results/loso_*.json")
