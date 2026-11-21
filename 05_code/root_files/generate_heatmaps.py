import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

# --- Grad-CAM Implementation ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        loss = output[0, class_idx]
        loss.backward()
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        weights = gradients.mean(dim=(1, 2))  # [C]
        cam = (weights[:, None, None] * activations).sum(0)
        cam = torch.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# --- Dataset ---
class ImageFolderWithPaths(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.labels = []
        self.paths = []
        for label in ['0', '1']:
            label_dir = os.path.join(root, label)
            if not os.path.isdir(label_dir):
                continue
            for fname in os.listdir(label_dir):
                if fname.lower().endswith('.png'):
                    self.samples.append(os.path.join(label_dir, fname))
                    self.labels.append(int(label))
                    self.paths.append((label, fname))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img_tensor = self.transform(img)
        else:
            img_tensor = transforms.ToTensor()(img)
        label, fname = self.paths[idx]
        return img_tensor, self.labels[idx], label, fname, np.array(img)

# --- Utility ---
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def overlay_heatmap_on_image(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlayed

def process_folder(data_dir, model_path, out_dir, device, model_type="densenet"):
    # Load model
    if model_type == "densenet":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, 2)
        target_layer = model.features[-1]
    else:
        raise NotImplementedError("Only densenet121 is supported in this script.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    # GradCAM
    gradcam = GradCAM(model, target_layer)

    # Data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = ImageFolderWithPaths(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for img_tensor, label, label_str, fname, orig_img in tqdm(loader, desc=f"Processing {data_dir}"):
        img_tensor = img_tensor.to(device)
        cam = gradcam(img_tensor)
        # orig_img is a batch of 1, shape (1, H, W, 3) or (H, W, 3)
        if isinstance(orig_img, list):
            orig_img = orig_img[0]
        if hasattr(orig_img, "shape") and len(orig_img.shape) == 4:
            orig_img = np.squeeze(orig_img, axis=0)
        if not isinstance(orig_img, np.ndarray):
            orig_img = np.array(orig_img)
        orig_img_uint8 = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        overlayed = overlay_heatmap_on_image(orig_img_uint8, cam)
        # label_str 可能是 batch of 1，需取出字串
        if isinstance(label_str, (list, tuple)):
            label_str = label_str[0]
        out_subdir = os.path.join(out_dir, str(label_str))
        ensure_dir(out_subdir)
        # fname 可能是 batch of 1，需取出字串
        if isinstance(fname, (list, tuple)):
            fname = fname[0]
        out_path = os.path.join(out_subdir, fname)
        cv2.imwrite(out_path, overlayed)
    gradcam.remove_hooks()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # NCKUH
    process_folder(
        data_dir="nckuh_0_1_inner",
        model_path="A_nckuh_model.pth",
        out_dir="nckuh_heatmap",
        device=device,
        model_type="densenet"
    )

    # CYCH
    process_folder(
        data_dir="cych_0_1_inner",
        model_path="B_cych_model.pth",
        out_dir="cych_heatmap",
        device=device,
        model_type="densenet"
    )
