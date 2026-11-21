"""
Generate Grad-CAM Heatmaps for Federated Model
================================================
Produces heatmap visualizations showing model attention on kidney ultrasound images.
Selects representative samples from each site and class (TP, TN, FP, FN).

Output:
  results/gradcam_panel_nckuh.png  — 2x3 panel (3 positive, 3 negative) from NCKUH
  results/gradcam_panel_cych.png   — 2x3 panel from CYCH
  results/gradcam_panel_combined.png — 2x4 panel (TP/TN/FP/FN from each site)

Uses the federated_model.pth (global model).
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "federated_model.pth"
NCKUH_DIR = "nckuh_0_1_inner"
CYCH_DIR = "cych_0_1_inner"

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


def generate_gradcam(model, img_tensor, target_class=None):
    """Generate Grad-CAM heatmap for a single image.
    Uses tensor-level hook to avoid DenseNet in-place ReLU conflict.
    """
    model.eval()

    # Hook to capture features and gradients from the last conv layer
    features = []
    gradients = []

    def forward_hook(module, input, output):
        # Store output and register tensor-level gradient hook
        features.append(output)
        output.register_hook(lambda g: gradients.append(g))

    # Register forward hook only (no backward hook on module)
    target_layer = model.features.norm5
    fh = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    img_tensor = img_tensor.unsqueeze(0).to(DEVICE)
    img_tensor.requires_grad_(True)
    output = model(img_tensor)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0, pred_class].item()

    # Backward pass
    model.zero_grad()
    output[0, target_class].backward()

    # Compute Grad-CAM
    feat = features[0].detach()
    grad = gradients[0].detach()

    weights = grad.mean(dim=[2, 3], keepdim=True)  # Global average pooling of gradients
    cam = (weights * feat).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = cam.squeeze().cpu().numpy()

    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()

    # Remove hook
    fh.remove()

    return cam, pred_class, confidence


def overlay_heatmap(img_pil, cam, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image."""
    img_np = np.array(img_pil.resize((224, 224))) / 255.0

    # Resize CAM to image size
    from PIL import Image as PILImage
    cam_resized = np.array(PILImage.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))) / 255.0

    # Apply JET colormap
    heatmap = cm.jet(cam_resized)[:, :, :3]

    # Overlay
    overlay = (1 - alpha) * img_np + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return overlay, cam_resized


def collect_samples(model, data_dir, site_name, max_per_category=3):
    """Collect TP, TN, FP, FN samples with predictions."""
    categories = {'TP': [], 'TN': [], 'FP': [], 'FN': []}

    for label_str in ['0', '1']:
        label = int(label_str)
        label_dir = os.path.join(data_dir, label_str)
        if not os.path.isdir(label_dir):
            continue
        fnames = sorted([f for f in os.listdir(label_dir) if f.lower().endswith('.png')])

        for fname in fnames:
            fpath = os.path.join(label_dir, fname)
            img_pil = Image.open(fpath).convert('RGB')
            img_tensor = val_transform(img_pil)

            cam, pred, conf = generate_gradcam(model, img_tensor)

            if label == 1 and pred == 1:
                cat = 'TP'
            elif label == 0 and pred == 0:
                cat = 'TN'
            elif label == 0 and pred == 1:
                cat = 'FP'
            else:
                cat = 'FN'

            if len(categories[cat]) < max_per_category:
                overlay, cam_resized = overlay_heatmap(img_pil, cam)
                categories[cat].append({
                    'fname': fname,
                    'label': label,
                    'pred': pred,
                    'conf': conf,
                    'overlay': overlay,
                    'original': np.array(img_pil.resize((224, 224))) / 255.0,
                    'cam': cam_resized,
                })

            # Check if we have enough
            if all(len(v) >= max_per_category for v in categories.values()):
                break

    return categories


def make_site_panel(categories, site_name, save_path):
    """Create a 2x3 panel: top row = positive (TP), bottom row = negative (TN)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for row, (cat_key, row_label) in enumerate([('TP', 'Positive (TP)'), ('TN', 'Negative (TN)')]):
        samples = categories[cat_key]
        for col in range(3):
            ax = axes[row, col]
            if col < len(samples):
                s = samples[col]
                ax.imshow(s['overlay'])
                label_text = "Inflamed" if s['label'] == 1 else "Normal"
                pred_text = "Inflamed" if s['pred'] == 1 else "Normal"
                ax.set_title(f"True: {label_text}\nPred: {pred_text} ({s['conf']:.2f})",
                            fontsize=9, pad=4)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')

        # Row label
        axes[row, 0].set_ylabel(row_label, fontsize=12, fontweight='bold', labelpad=10)
        axes[row, 0].yaxis.set_label_position("left")

    fig.suptitle(f'Grad-CAM Heatmaps — {site_name}\n(Federated Global Model)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")


def make_combined_panel(nckuh_cats, cych_cats, save_path):
    """Create a 2x4 panel showing TP, TN, FP, FN from both sites."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    cat_keys = ['TP', 'TN', 'FP', 'FN']
    cat_labels = ['True Positive', 'True Negative', 'False Positive', 'False Negative']

    for row, (cats, site_name) in enumerate([(nckuh_cats, 'NCKUH'), (cych_cats, 'CYCH')]):
        for col, (cat_key, cat_label) in enumerate(zip(cat_keys, cat_labels)):
            ax = axes[row, col]
            samples = cats[cat_key]
            if len(samples) > 0:
                s = samples[0]  # take first sample
                ax.imshow(s['overlay'])
                label_text = "Inflamed" if s['label'] == 1 else "Normal"
                pred_text = "Inflamed" if s['pred'] == 1 else "Normal"
                ax.set_title(f"{cat_label}\nTrue={label_text}, Pred={pred_text}\nConf={s['conf']:.3f}",
                            fontsize=8, pad=4)
            else:
                ax.text(0.5, 0.5, f'No {cat_key}\nsample', ha='center', va='center',
                        fontsize=10, transform=ax.transAxes, color='gray')
            ax.axis('off')

        # Row label
        axes[row, 0].set_ylabel(site_name, fontsize=14, fontweight='bold',
                                labelpad=15, rotation=0, ha='right')

    fig.suptitle('Grad-CAM Visualization — Federated Global Model\nRepresentative Cases from Both Sites',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")


def make_comparison_panel(nckuh_cats, cych_cats, save_path):
    """
    Create a 3x4 panel: Original | Heatmap | Overlay for both sites.
    Rows: NCKUH-TP, NCKUH-TN, CYCH-TP, CYCH-TN
    """
    rows_data = []
    for cats, site in [(nckuh_cats, 'NCKUH'), (cych_cats, 'CYCH')]:
        for cat_key, label in [('TP', 'Positive'), ('TN', 'Negative')]:
            if len(cats[cat_key]) > 0:
                s = cats[cat_key][0]
                rows_data.append((s, f'{site} — {label}'))

    n_rows = len(rows_data)
    fig, axes = plt.subplots(n_rows, 3, figsize=(10, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]

    col_titles = ['Original Image', 'Grad-CAM Heatmap', 'Overlay']

    for i, (s, row_label) in enumerate(rows_data):
        # Original
        axes[i][0].imshow(s['original'])
        axes[i][0].set_ylabel(row_label, fontsize=10, fontweight='bold', rotation=0,
                              labelpad=80, ha='right', va='center')
        axes[i][0].axis('off')

        # Heatmap only
        axes[i][1].imshow(s['cam'], cmap='jet', vmin=0, vmax=1)
        axes[i][1].axis('off')

        # Overlay
        axes[i][2].imshow(s['overlay'])
        pred_text = "Inflamed" if s['pred'] == 1 else "Normal"
        axes[i][2].set_title(f"Pred: {pred_text} ({s['conf']:.3f})", fontsize=9)
        axes[i][2].axis('off')

    # Column titles
    for j, title in enumerate(col_titles):
        axes[0][j].set_title(title, fontsize=11, fontweight='bold', pad=10)

    fig.suptitle('Grad-CAM Attention Maps — Federated Model',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    # Load federated model
    print(f"Loading model: {MODEL_PATH}")
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    # Collect samples from both sites
    print("\nCollecting NCKUH samples...")
    nckuh_cats = collect_samples(model, NCKUH_DIR, "NCKUH", max_per_category=3)
    for k, v in nckuh_cats.items():
        print(f"  {k}: {len(v)} samples")

    print("\nCollecting CYCH samples...")
    cych_cats = collect_samples(model, CYCH_DIR, "CYCH", max_per_category=3)
    for k, v in cych_cats.items():
        print(f"  {k}: {len(v)} samples")

    # Generate panels
    print("\nGenerating panels...")
    make_site_panel(nckuh_cats, "NCKUH", "results/gradcam_panel_nckuh.png")
    make_site_panel(cych_cats, "CYCH", "results/gradcam_panel_cych.png")
    make_combined_panel(nckuh_cats, cych_cats, "results/gradcam_panel_combined.png")
    make_comparison_panel(nckuh_cats, cych_cats, "results/gradcam_panel_comparison.png")

    print("\nDone! All Grad-CAM figures saved to results/")
