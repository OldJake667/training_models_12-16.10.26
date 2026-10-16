"""Model definitions used by the notebooks. No training or download at import time."""
from pathlib import Path

import torch
from torch import nn
import timm

from lab_code.runtime import FrozenClassifier, sha256_file
from lab_code.scratch_models import ScratchCNN, VisionTransformerTiny


def build_cnn(root, config):
    return ScratchCNN(classes=2)


def build_vit_tiny(root, config):
    return VisionTransformerTiny(classes=2)


def build_standard_model(root, config):
    # All six new models start from scratch. The task has two output classes.
    kwargs = {"pretrained": False, "num_classes": 2, "in_chans": 3}
    if config["architecture"].startswith("vit_"):
        kwargs.update(img_size=224, drop_rate=0.1)
    return timm.create_model(config["architecture"], **kwargs)


def build_dinov2(root, config):
    path = Path(root) / "pretrained/dinov2_small.safetensors"
    if not path.is_file():
        raise FileNotFoundError("Copy the entire package, including pretrained/dinov2_small.safetensors")
    if sha256_file(path) != "04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081":
        raise ValueError("DINOv2 checkpoint differs from the verified pilot backbone")
    # Local pretrained file: no Hugging Face cache or internet is required.
    backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                 num_classes=0, img_size=224,
                                 pretrained_cfg_overlay={"file": str(path)})
    return FrozenClassifier(backbone, feature_dim=384, train_backbone=config["train_backbone"])


def build_vmamba(root, config):
    from lab_code import vmamba_official as official
    # The full release needs the fast selective-scan kernel. Fail early instead
    # of silently running 111,000 layouts through the Python reference loop.
    if config["require_fast_vmamba"] and not official.WITH_SELECTIVESCAN_MAMBA:
        raise RuntimeError("VMamba selective_scan_cuda is missing. Complete VMamba setup "
                           "in START_HERE.md on Linux/WSL2 before launching this notebook.")
    path = Path(root) / "pretrained/vmamba_tiny_imagenet.pth"
    if not path.is_file():
        raise FileNotFoundError("Copy the entire package, including pretrained/vmamba_tiny_imagenet.pth")
    if sha256_file(path) != "d855b57224a652bc692804932f88dc3a39b9028e64be99f745a7fd6792ca672a":
        raise ValueError("VMamba checkpoint differs from the verified pilot backbone")
    backbone = official.vmamba_tiny_s1l8(pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(state.get("model", state), strict=True)
    backbone.classifier.head = nn.Identity()
    # Use the installed CUDA scan and PyTorch cross-scan/merge. Avoid relying on
    # the older vendored Triton kernels for a Blackwell GPU.
    from functools import partial
    for module in backbone.modules():
        if isinstance(module, official.SS2D):
            module.forward_core = partial(module.forward_corev2, force_fp32=False,
                                          no_einsum=True, selective_scan_backend="mamba",
                                          scan_force_torch=True)
    return FrozenClassifier(backbone, feature_dim=768, train_backbone=config["train_backbone"])
