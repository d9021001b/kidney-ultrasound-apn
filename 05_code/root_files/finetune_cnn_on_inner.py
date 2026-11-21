import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np

# --- Config ---
DATA_DIRS = ['nckuh_0_1_inner', 'cych_0_1_inner']
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 500
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# --- Get all classification model names from torchvision ---
from torchvision import models

def get_classification_models():
    all_models = []
    for name in dir(models):
        if name.startswith("_"):
            continue
        obj = getattr(models, name)
        if callable(obj) and "classification" in (getattr(obj, "__module__", "") or ""):
            all_models.append(name)
        # Heuristic: include common CNNs, exclude segmentation/detection
        elif name.lower().startswith(("resnet", "densenet", "mobilenet", "efficientnet", "vgg", "squeezenet", "shufflenet", "mnasnet", "convnext", "regnet", "vit")):
            all_models.append(name)
    # Remove duplicates and sort
    return sorted(set(all_models))

# --- Dataset ---
class KidneyDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        self.samples = []
        for root_dir in root_dirs:
            for label in ['0', '1']:
                label_dir = os.path.join(root_dir, label)
                if not os.path.isdir(label_dir):
                    continue
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith('.png'):
                        self.samples.append((os.path.join(label_dir, fname), int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Model ---
def get_model(model_name, num_classes=2):
    # Dynamically load model and replace final layer for binary classification
    model_func = getattr(models, model_name)
    model = model_func(pretrained=True)
    # Replace classifier head depending on model type
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Sequential):
            # e.g. mobilenet_v2, vgg, densenet
            last_idx = -1
            for i in reversed(range(len(model.classifier))):
                if isinstance(model.classifier[i], nn.Linear):
                    last_idx = i
                    break
            if last_idx >= 0:
                in_features = model.classifier[last_idx].in_features
                model.classifier[last_idx] = nn.Linear(in_features, num_classes)
        elif isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown model head for {model_name}")
    return model

# --- Metrics ---
def compute_tpr_tnr(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return tpr, tnr, min(tpr, tnr)

# --- Training ---
def train_all_models():
    dataset = KidneyDataset(DATA_DIRS, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    model_names = get_classification_models()
    print(f"Found {len(model_names)} models to test: {model_names}")

    patience = 7  # Early stopping patience
    for model_name in model_names:
        print(f"\n=== Training model: {model_name} ===")
        try:
            model = get_model(model_name).to(DEVICE)
        except Exception as e:
            print(f"Skipping {model_name} due to error: {e}")
            continue
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        best_min_tpr_tnr = 0
        best_model_state = None
        best_epoch = 0
        epochs_no_improve = 0

        for epoch in range(EPOCHS):
            model.train()
            running_loss = 0
            for imgs, labels in tqdm(loader, desc=f"{model_name} Epoch {epoch+1}/{EPOCHS}"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
            avg_loss = running_loss / len(dataset)

            # Evaluate on all data (since no validation set)
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
            tpr, tnr, min_tpr_tnr = compute_tpr_tnr(all_labels, all_preds)
            print(f"{model_name} Epoch {epoch+1}: Loss={avg_loss:.4f}, TPR={tpr:.4f}, TNR={tnr:.4f}, min(TPR,TNR)={min_tpr_tnr:.4f}")

            if min_tpr_tnr > best_min_tpr_tnr:
                best_min_tpr_tnr = min_tpr_tnr
                best_model_state = model.state_dict()
                best_epoch = epoch + 1
                epochs_no_improve = 0
                if min_tpr_tnr >= 0.95:
                    print(f"{model_name}: Achieved min(TPR,TNR) >= 0.95, saving model and stopping training.")
                    torch.save(best_model_state, f"best_{model_name}_min_tpr_tnr_{min_tpr_tnr:.3f}.pth")
                    break
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"{model_name}: Early stopping at epoch {epoch+1} (no improvement in {patience} epochs).")
                    break

        if best_model_state is not None and best_min_tpr_tnr < 0.95:
            print(f"{model_name}: Best min(TPR,TNR) achieved: {best_min_tpr_tnr:.4f} at epoch {best_epoch}, saving model.")
            torch.save(best_model_state, f"best_{model_name}_min_tpr_tnr_{best_min_tpr_tnr:.3f}.pth")
        elif best_model_state is None:
            print(f"{model_name}: No model achieved min(TPR,TNR) >= 0.95.")

if __name__ == '__main__':
    train_all_models()
