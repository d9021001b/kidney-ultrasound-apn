"""
Evaluate Federated Learning Models
====================================
對三個模型 (A_nckuh, B_cych, federated) 在兩個院區分別評估：
- Confusion matrix, Accuracy, TPR, TNR, Precision, Recall, F1
- ROC curve + AUC
- PR curve + AP (Average Precision)

同時加入 LOSO baseline 模型做對比（如果存在）。
輸出：
  results/federated_evaluation.json  — 所有數字
  results/federated_roc_curves.png   — ROC 圖
  results/federated_pr_curves.png    — PR 圖
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# --- Config ---
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NCKUH_DIR = "nckuh_0_1_inner"
CYCH_DIR = "cych_0_1_inner"

# 要評估的模型
MODELS = {
    "A_nckuh":    "A_nckuh_model.pth",
    "B_cych":     "B_cych_model.pth",
    "Federated":  "federated_model.pth",
}

# LOSO baseline 模型（如果存在就加入對比）
LOSO_MODELS = {
    "LOSO_trainNCKUH": "loso_trainNCKUH_testCYCH_model.pth",
    "LOSO_trainCYCH":  "loso_trainCYCH_testNCKUH_model.pth",
}

DATASETS = {
    "NCKUH": NCKUH_DIR,
    "CYCH":  CYCH_DIR,
}

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


def predict(model, loader):
    """回傳 y_true, y_pred, y_prob (positive class probability)"""
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
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_true, y_pred, y_prob):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # sensitivity / recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # specificity
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    acc = (tp + tn) / len(y_true)

    # ROC
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr_arr, tpr_arr)

    # PR
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": round(acc, 4),
        "tpr_sensitivity_recall": round(tpr, 4),
        "tnr_specificity": round(tnr, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "min_tpr_tnr": round(min(tpr, tnr), 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(ap, 4),
        "n_samples": len(y_true),
        "n_pos": int(np.sum(y_true == 1)),
        "n_neg": int(np.sum(y_true == 0)),
    }, fpr_arr, tpr_arr, prec_arr, rec_arr


def main():
    os.makedirs("results", exist_ok=True)

    # 準備 dataloaders
    loaders = {}
    for ds_name, ds_dir in DATASETS.items():
        dataset = KidneyDataset(ds_dir, transform=val_transform)
        loaders[ds_name] = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
        print(f"Dataset {ds_name}: {len(dataset)} samples")

    # 收集所有模型（聯邦 + LOSO baseline）
    all_models = dict(MODELS)
    for name, path in LOSO_MODELS.items():
        if os.path.exists(path):
            all_models[name] = path
            print(f"Found LOSO baseline: {path}")

    # 評估每個模型在每個資料集上的結果
    all_results = {}
    curve_data = {}  # 存 ROC/PR curve 資料

    for model_name, model_path in all_models.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name} ({model_path})")
        print(f"{'='*50}")

        model = get_model()
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model = model.to(DEVICE)

        all_results[model_name] = {}
        curve_data[model_name] = {}

        for ds_name, loader in loaders.items():
            y_true, y_pred, y_prob = predict(model, loader)
            metrics, fpr_arr, tpr_arr, prec_arr, rec_arr = compute_metrics(y_true, y_pred, y_prob)

            all_results[model_name][ds_name] = metrics
            curve_data[model_name][ds_name] = {
                "fpr": fpr_arr, "tpr": tpr_arr,
                "precision": prec_arr, "recall": rec_arr,
            }

            print(f"\n  → Tested on {ds_name}:")
            print(f"    Acc={metrics['accuracy']:.4f}  "
                  f"TPR={metrics['tpr_sensitivity_recall']:.4f}  "
                  f"TNR={metrics['tnr_specificity']:.4f}  "
                  f"Prec={metrics['precision']:.4f}  "
                  f"F1={metrics['f1']:.4f}")
            print(f"    min(TPR,TNR)={metrics['min_tpr_tnr']:.4f}  "
                  f"AUC={metrics['roc_auc']:.4f}  "
                  f"AP={metrics['average_precision']:.4f}")
            print(f"    CM: TP={metrics['tp']} TN={metrics['tn']} FP={metrics['fp']} FN={metrics['fn']}")

    # --- 儲存數字結果 ---
    with open("results/federated_evaluation.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nMetrics saved to results/federated_evaluation.json")

    # ========== 畫圖 ==========

    # 定義顏色和線型
    style_map = {
        "A_nckuh":          {"color": "#2196F3", "ls": "-",  "label": "Client A (NCKUH)"},
        "B_cych":           {"color": "#FF9800", "ls": "-",  "label": "Client B (CYCH)"},
        "Federated":        {"color": "#E91E63", "ls": "-",  "lw": 2.5, "label": "Federated"},
        "LOSO_trainNCKUH":  {"color": "#9E9E9E", "ls": "--", "label": "LOSO (Train NCKUH)"},
        "LOSO_trainCYCH":   {"color": "#BDBDBD", "ls": "--", "label": "LOSO (Train CYCH)"},
    }

    for ds_name in DATASETS:
        # --- ROC ---
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        for model_name in curve_data:
            if ds_name not in curve_data[model_name]:
                continue
            cd = curve_data[model_name][ds_name]
            st = style_map.get(model_name, {"color": "gray", "ls": "-", "label": model_name})
            auc_val = all_results[model_name][ds_name]["roc_auc"]
            lw = st.get("lw", 1.8)
            ax.plot(cd["fpr"], cd["tpr"], color=st["color"], ls=st["ls"], lw=lw,
                    label=f'{st["label"]} (AUC={auc_val:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curve — Tested on {ds_name}", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = f"results/federated_roc_{ds_name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"ROC saved: {path}")

        # --- PR ---
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        for model_name in curve_data:
            if ds_name not in curve_data[model_name]:
                continue
            cd = curve_data[model_name][ds_name]
            st = style_map.get(model_name, {"color": "gray", "ls": "-", "label": model_name})
            ap_val = all_results[model_name][ds_name]["average_precision"]
            lw = st.get("lw", 1.8)
            ax.plot(cd["recall"], cd["precision"], color=st["color"], ls=st["ls"], lw=lw,
                    label=f'{st["label"]} (AP={ap_val:.3f})')
        # baseline: prevalence
        n_pos = all_results[list(all_results.keys())[0]][ds_name]["n_pos"]
        n_total = all_results[list(all_results.keys())[0]][ds_name]["n_samples"]
        prevalence = n_pos / n_total
        ax.axhline(y=prevalence, color='k', ls='--', lw=0.8, alpha=0.5,
                    label=f'Prevalence ({prevalence:.3f})')
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(f"Precision-Recall Curve — Tested on {ds_name}", fontsize=13)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = f"results/federated_pr_{ds_name}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print(f"PR saved: {path}")

    # ========== 合併對比表（方便論文用）==========
    print(f"\n{'='*80}")
    print("Summary Table (for paper)")
    print(f"{'='*80}")
    header = f"{'Model':<20} {'Test Site':<8} {'Acc':>6} {'TPR':>6} {'TNR':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'AUC':>6} {'AP':>6}"
    print(header)
    print("-" * len(header))
    for model_name in all_results:
        for ds_name in all_results[model_name]:
            m = all_results[model_name][ds_name]
            print(f"{model_name:<20} {ds_name:<8} "
                  f"{m['accuracy']:>6.4f} {m['tpr_sensitivity_recall']:>6.4f} "
                  f"{m['tnr_specificity']:>6.4f} {m['precision']:>6.4f} "
                  f"{m['tpr_sensitivity_recall']:>6.4f} {m['f1']:>6.4f} "
                  f"{m['roc_auc']:>6.4f} {m['average_precision']:>6.4f}")

    print(f"\nDone! All results in results/federated_*.json and results/federated_*.png")


if __name__ == "__main__":
    main()
