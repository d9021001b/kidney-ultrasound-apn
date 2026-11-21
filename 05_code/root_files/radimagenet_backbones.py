"""
Verified loaders for the RadImageNet PyTorch checkpoints, plus the other pretrained
encoders that are actually present on this machine (no internet required).

Every claim in this file was measured, not assumed.  See the header of each loader for
the key-match counts obtained on torch 2.5.1+cu121 / torchvision 0.20.1+cu121.

NORMALISATION -- READ THIS FIRST
--------------------------------
All three RadImageNet checkpoints expect input scaled to [-1, 1], i.e.

    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])   # after ToTensor()

which is Keras `preprocess_input(mode="tf")`, the preprocessing RadImageNet was trained
with.  This was established empirically by comparing each BatchNorm layer's STORED
running statistics against the statistics actually observed when real kidney crops are
pushed through the net (scratch_transfer/probe_norm.py).  Using ImageNet mean/std
instead drives the DenseNet121 BN layers to a median |z| of 13.5 and a median
variance ratio of 197 -- the features are garbage.  The sibling project at
lung_cancerimage/codex_experiments/test_github_pretrained_backbones.py pairs these
weights with ImageNet normalisation; do not copy that.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tvm

RADIMAGENET_DIR = Path(
    r"C:/Users/udoo_w2/Desktop/lung_cancerimage/codex_experiments/pretrained/RadImageNet_pytorch"
)

# RadImageNet preprocessing: x/127.5 - 1.  With torchvision ToTensor() (which already
# divides by 255) this is exactly Normalize(0.5, 0.5).
RADIMAGENET_MEAN = [0.5, 0.5, 0.5]
RADIMAGENET_STD = [0.5, 0.5, 0.5]
RADIMAGENET_INPUT = 224           # all three; InceptionV3 also verified fine at 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

FEATURE_DIM = {"densenet121": 1024, "resnet50": 2048, "inception_v3": 2048}


# ----------------------------------------------------------------------------------
# RadImageNet DenseNet121
#   ckpt keys 725 / model keys 727 / MATCHED 725 / missing 2 (classifier.*) / unexpected 0
#   Key form: "backbone.0.<torchvision features.* name>"  ->  strip to "features.<...>"
# ----------------------------------------------------------------------------------
def radimagenet_densenet121(num_classes: int | None = None) -> nn.Module:
    sd = torch.load(RADIMAGENET_DIR / "DenseNet121.pt", map_location="cpu", weights_only=True)
    renamed = OrderedDict((k.replace("backbone.0.", "features.", 1), v) for k, v in sd.items())
    m = tvm.densenet121(weights=None)
    res = m.load_state_dict(renamed, strict=False)
    _check(res, expect_missing={"classifier.weight", "classifier.bias"}, tag="densenet121")
    m.classifier = nn.Identity() if num_classes is None else nn.Linear(1024, num_classes)
    return m


# ----------------------------------------------------------------------------------
# RadImageNet ResNet50
#   ckpt keys 318 / model keys 320 / MATCHED 318 / missing 2 (fc.*) / unexpected 0
#   Key form: "backbone.<i>.<...>" where <i> indexes
#   nn.Sequential(*list(resnet50.children())[:-2]) -- 0 conv1, 1 bn1, (2 relu, 3 maxpool
#   carry no parameters), 4..7 layer1..layer4.
# ----------------------------------------------------------------------------------
_R50 = {"0": "conv1", "1": "bn1", "4": "layer1", "5": "layer2", "6": "layer3", "7": "layer4"}


def radimagenet_resnet50(num_classes: int | None = None) -> nn.Module:
    sd = torch.load(RADIMAGENET_DIR / "ResNet50.pt", map_location="cpu", weights_only=True)
    renamed = OrderedDict()
    for k, v in sd.items():
        parts = k.split(".")
        renamed[".".join([_R50[parts[1]]] + parts[2:])] = v
    m = tvm.resnet50(weights=None)
    res = m.load_state_dict(renamed, strict=False)
    _check(res, expect_missing={"fc.weight", "fc.bias"}, tag="resnet50")
    m.fc = nn.Identity() if num_classes is None else nn.Linear(2048, num_classes)
    return m


# ----------------------------------------------------------------------------------
# RadImageNet InceptionV3
#   ckpt keys 564 / model keys 566 / MATCHED 564 / missing 2 (fc.*) / unexpected 0
#   Key form: "backbone.<i>.<...>" where <i> indexes the named children of
#   inception_v3(aux_logits=False) up to Mixed_7c.  AuxLogits is absent from the
#   checkpoint, which is why aux_logits=False is required for the indices to line up.
#   transform_input MUST be False: RadImageNet did not use the TF-Slim channel remap.
# ----------------------------------------------------------------------------------
_INC = ["Conv2d_1a_3x3", "Conv2d_2a_3x3", "Conv2d_2b_3x3", "maxpool1", "Conv2d_3b_1x1",
        "Conv2d_4a_3x3", "maxpool2", "Mixed_5b", "Mixed_5c", "Mixed_5d", "Mixed_6a",
        "Mixed_6b", "Mixed_6c", "Mixed_6d", "Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c"]


def radimagenet_inception_v3(num_classes: int | None = None) -> nn.Module:
    sd = torch.load(RADIMAGENET_DIR / "InceptionV3.pt", map_location="cpu", weights_only=True)
    renamed = OrderedDict()
    for k, v in sd.items():
        parts = k.split(".")
        renamed[".".join([_INC[int(parts[1])]] + parts[2:])] = v
    m = tvm.inception_v3(weights=None, aux_logits=False, init_weights=False,
                         transform_input=False)
    res = m.load_state_dict(renamed, strict=False)
    _check(res, expect_missing={"fc.weight", "fc.bias"}, tag="inception_v3")
    m.fc = nn.Identity() if num_classes is None else nn.Linear(2048, num_classes)
    return m


# ----------------------------------------------------------------------------------
# PanEcho 2D frame encoder -- ConvNeXt-Tiny trained on ~1M echocardiography frames.
#   Cardiac, not renal, but it is the only B-mode-ultrasound-pretrained encoder on disk.
#   ckpt 2D-encoder keys 180 / model keys 182 / MATCHED 180 / missing 2 (classifier.2.*)
#   Feature dim 768.  Normalisation: ImageNet mean/std (PanEcho's own preprocessing).
# ----------------------------------------------------------------------------------
PANECHO_PT = Path(r"C:/Users/udoo_w2/.cache/torch/hub/checkpoints/panecho.pt")


def panecho_convnext_tiny(num_classes: int | None = None) -> nn.Module:
    blob = torch.load(PANECHO_PT, map_location="cpu", weights_only=False)["weights"]
    prefix = "encoder.encoder.model."
    renamed = OrderedDict((k[len(prefix):], v) for k, v in blob.items() if k.startswith(prefix))
    m = tvm.convnext_tiny(weights=None)
    res = m.load_state_dict(renamed, strict=False)
    _check(res, expect_missing={"classifier.2.weight", "classifier.2.bias"}, tag="panecho")
    m.classifier[2] = nn.Identity() if num_classes is None else nn.Linear(768, num_classes)
    return m


def _check(res, expect_missing: set, tag: str) -> None:
    missing, unexpected = set(res.missing_keys), set(res.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"{tag}: {len(unexpected)} unexpected keys, e.g. {sorted(unexpected)[:3]}")
    if missing != expect_missing:
        raise RuntimeError(f"{tag}: missing keys {sorted(missing)} != expected {sorted(expect_missing)}")


# ----------------------------------------------------------------------------------
# Partial-freeze helpers for ARM 2 (mostly-frozen fine-tuning).
# ----------------------------------------------------------------------------------
def freeze_all_but(model: nn.Module, trainable_prefixes: tuple[str, ...]) -> dict:
    """Freeze every parameter whose qualified name does not start with a listed prefix."""
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(pre) for pre in trainable_prefixes)
        if p.requires_grad:
            n_train += p.numel()
        else:
            n_frozen += p.numel()
    return {"trainable": n_train, "frozen": n_frozen,
            "trainable_frac": n_train / max(1, n_train + n_frozen)}


def freeze_bn_running_stats(model: nn.Module) -> int:
    """Put every BatchNorm in eval() so its running stats stay at the RadImageNet values.

    With batches of 16 on a few hundred images the batch estimates are far noisier than
    the stored ones, and letting them drift is a large part of why full fine-tuning of
    these backbones collapses on this cohort.  Call this inside model.train() overrides,
    or after every .train() call.
    """
    n = 0
    for mod in model.modules():
        if isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            mod.eval()
            n += 1
    return n


if __name__ == "__main__":
    for fn in (radimagenet_densenet121, radimagenet_resnet50,
               radimagenet_inception_v3, panecho_convnext_tiny):
        m = fn()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            out = m(torch.zeros(2, 3, 224, 224))
        print(f"{fn.__name__:28s} OK  params {n/1e6:6.1f}M  feature dim {tuple(out.shape)}")
