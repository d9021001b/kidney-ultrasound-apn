import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import collections

# --- 1. Configuration ---
BASE_DIRS = ['nckuh_0', 'nckuh_1', 'cych_0', 'cych_1']
NUM_CLUSTERS = 10
MIN_SAMPLES_FOR_VAL = 10
NUM_FOLDS = 5
NUM_EPOCHS = 50 # Increased epochs, will be managed by Early Stopping
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EARLY_STOPPING_PATIENCE = 5 # Stop if validation F1-score doesn't improve for 5 epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_DIR = "best_models"
RESULTS_CSV_PATH = "advanced_cnn_comparison_results_v3.csv"

# --- 2. Custom Dataset ---
class ClusteredImageDataset(Dataset):
    def __init__(self, base_dirs, transform=None, is_train=False):
        self.transform = transform
        self.is_train = is_train
        self.samples = []
        # Data Augmentation for training set
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Standard transform for validation set
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for base_dir in base_dirs:
            if not os.path.isdir(base_dir):
                continue
            # Master label is 0 if dir name contains '_0', else 1
            master_label = 0 if '_0' in base_dir else 1
            for cluster_id in range(NUM_CLUSTERS):
                cluster_dir = os.path.join(base_dir, str(cluster_id))
                if not os.path.isdir(cluster_dir):
                    continue
                
                image_files = [f for f in os.listdir(cluster_dir) if f.lower().endswith('.png')]
                for img_file in image_files:
                    self.samples.append({
                        'path': os.path.join(cluster_dir, img_file),
                        'label': master_label,
                        'group': f"{base_dir}/{cluster_id}"
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample['path']
        label = sample['label']
        
        image = Image.open(path).convert('RGB')
        
        # Apply different transforms for train and val
        if self.is_train:
            image = self.train_transform(image)
        else:
            image = self.val_transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# --- 3. Custom Cross-Validation Splitter ---
def get_custom_kfold_splits(samples, n_splits=5):
    group_counts = collections.Counter(s['group'] for s in samples)
    
    # Separate indices by group size
    small_group_indices = [i for i, s in enumerate(samples) if group_counts[s['group']] < MIN_SAMPLES_FOR_VAL]
    large_group_indices = [i for i, s in enumerate(samples) if group_counts[s['group']] >= MIN_SAMPLES_FOR_VAL]
    
    print(f"Found {len(small_group_indices)} samples in small groups (always in train set).")
    print(f"Found {len(large_group_indices)} samples in large groups for k-fold splitting.")

    if not large_group_indices:
        raise ValueError("No groups have enough samples for validation. Cannot perform k-fold splitting.")

    large_group_labels = [samples[i]['label'] for i in large_group_indices]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for train_split_lg, val_split_lg in skf.split(large_group_indices, large_group_labels):
        # Convert local indices from skf back to original sample indices
        val_indices = [large_group_indices[i] for i in val_split_lg]
        train_indices_lg = [large_group_indices[i] for i in train_split_lg]
        
        # Final train set is the combination of small groups and the large group's training split
        final_train_indices = sorted(small_group_indices + train_indices_lg)
        
        yield final_train_indices, val_indices

# --- 4. Model Definition and Helpers ---
def get_classification_models():
    # Helper to get all classification models from torchvision
    all_models = []
    for name in dir(models):
        if name.startswith("_") or not name[0].islower(): continue
        obj = getattr(models, name)
        if callable(obj) and "models" in obj.__module__:
            if any(k in name for k in ["quantized", "segmentation", "detection", "video", "get_", "list"]):
                continue
            all_models.append(name)
    return sorted(set(all_models))

def get_model(model_name, pretrained=True):
    model_func = getattr(models, model_name)
    model = model_func(pretrained=pretrained)
    
    # Replace classifier
    if hasattr(model, 'fc'):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            if hasattr(model.classifier[-1], 'in_features'):
                 num_ftrs = model.classifier[-1].in_features
                 model.classifier[-1] = nn.Linear(num_ftrs, 2)
            else: # Fallback for models like MobileNetV2
                 model.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=True), nn.Linear(model.last_channel, 2))
        else:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, 2)
    elif hasattr(model, 'head'):
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError(f"Cannot adapt model: {model_name}")
        
    return model

# --- 5. Training and Evaluation ---
def train_model_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_samples = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
    return running_loss / total_samples if total_samples > 0 else 0.0

def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if not all_labels:
        return {"accuracy": 0, "tpr": 0, "tnr": 0, "precision": 0, "f1": 0}

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    min_tpr_tnr = min(tpr, tnr)
    
    return {"accuracy": accuracy, "tpr": tpr, "tnr": tnr, "precision": precision, "f1": f1, "min_tpr_tnr": min_tpr_tnr}

# --- 6. Main Execution ---
def main():
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    # Initialize results file with header if it doesn't exist
    if not os.path.exists(RESULTS_CSV_PATH):
        pd.DataFrame(columns=["Model", "Accuracy", "TPR", "TNR", "Precision", "F1-Score", "Min_TPR_TNR"]).to_csv(RESULTS_CSV_PATH, index=False)

    print("Loading dataset metadata...")
    # Pass transform=None initially, it will be handled by __getitem__
    full_dataset = ClusteredImageDataset(BASE_DIRS, transform=None)
    print(f"Found {len(full_dataset)} total samples.")

    model_names = get_classification_models()
    print(f"Found {len(model_names)} models to test.")
    
    all_results = []

    # Read already completed models to skip them
    completed_models = set()
    if os.path.exists(RESULTS_CSV_PATH):
        df_results = pd.read_csv(RESULTS_CSV_PATH)
        completed_models = set(df_results['Model'])
    
    for model_name in model_names:
        if model_name in completed_models:
            print(f"\n--- Skipping already tested Model: {model_name} ---")
            continue

        print(f"\n--- Testing Model: {model_name} ---")
        try:
            fold_metrics = []
            # We need to re-assign the dataset to the subsets to set the is_train flag correctly
            full_dataset.is_train = True
            train_dataset = Subset(full_dataset, range(len(full_dataset)))
            full_dataset.is_train = False
            val_dataset = Subset(full_dataset, range(len(full_dataset)))

            splitter = get_custom_kfold_splits(full_dataset.samples, n_splits=NUM_FOLDS)

            for fold, (train_idx, val_idx) in enumerate(splitter):
                print(f"  Fold {fold+1}/{NUM_FOLDS} | Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
                
                train_subset = Subset(train_dataset, train_idx)
                val_subset = Subset(val_dataset, val_idx)
                
                train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

                model = get_model(model_name).to(DEVICE)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

                best_metric_val = -1
                epochs_no_improve = 0
                best_model_path = os.path.join(BEST_MODEL_DIR, f"{model_name}_fold{fold+1}.pth")

                for epoch in range(NUM_EPOCHS):
                    train_loss = train_model_epoch(model, train_loader, criterion, optimizer, DEVICE)
                    val_metrics = evaluate_model(model, val_loader, DEVICE)
                    current_metric_val = val_metrics['min_tpr_tnr']
                    
                    print(f"    Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val min(TPR,TNR): {current_metric_val:.4f}")

                    if current_metric_val > best_metric_val:
                        best_metric_val = current_metric_val
                        torch.save(model.state_dict(), best_model_path)
                        epochs_no_improve = 0
                        print(f"      New best min(TPR,TNR) score. Model saved to {best_model_path}")
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"      Early stopping triggered after {epoch+1} epochs.")
                        break
                
                # Load best model for final evaluation on this fold
                model.load_state_dict(torch.load(best_model_path))
                final_metrics = evaluate_model(model, val_loader, DEVICE)
                fold_metrics.append(final_metrics)
                print(f"    Fold {fold+1} Best Metrics: {final_metrics}")

            avg_metrics = {key: np.mean([m[key] for m in fold_metrics]) for key in fold_metrics[0]}
            print(f"\n  Average Metrics for {model_name}: {avg_metrics}\n")
            
            # Append result to CSV immediately
            result_entry = {"Model": model_name}
            result_entry.update(avg_metrics)
            df_entry = pd.DataFrame([result_entry])
            df_entry.rename(columns={
                'accuracy': 'Accuracy', 'tpr': 'TPR', 'tnr': 'TNR', 
                'precision': 'Precision', 'f1': 'F1-Score', 'min_tpr_tnr': 'Min_TPR_TNR'
            }, inplace=True)
            df_entry.to_csv(RESULTS_CSV_PATH, mode='a', header=False, index=False)
            print(f"Result for {model_name} appended to {RESULTS_CSV_PATH}")

        except Exception as e:
            print(f"Could not test model {model_name}. Error: {e}")

    # Final summary
    if os.path.exists(RESULTS_CSV_PATH):
        df_final = pd.read_csv(RESULTS_CSV_PATH).sort_values(by="Min_TPR_TNR", ascending=False)
        print("\n--- Final Results Summary ---")
        print(df_final)
    else:
        print("No models were successfully evaluated.")

if __name__ == '__main__':
    main()
