"""Randomly initialized DINOv2-style ViT and VMamba architectures; no weight loading."""
import torch
from torch import nn
import timm


class ScratchBackboneClassifier(nn.Module):
    """Train the randomly initialized backbone and two-class head together."""
    def __init__(self, backbone, feature_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 2)

    def forward(self, images):
        return self.head(self.backbone(images))


def build_dinov2_from_scratch(root, config):
    if not config["train_backbone"] or config.get("initialization") != "random":
        raise ValueError("The scratch ViT requires random initialization and all layers trainable")
    backbone = timm.create_model(config["architecture"], pretrained=False,
                                 num_classes=0, img_size=config["image_size"])
    return ScratchBackboneClassifier(backbone, feature_dim=backbone.num_features)


def build_vmamba_from_scratch(root, config):
    from lab_code import vmamba_official as official
    if not config["train_backbone"] or config.get("initialization") != "random":
        raise ValueError("Scratch VMamba requires random initialization and all layers trainable")
    if config["require_fast_vmamba"] and not official.WITH_SELECTIVESCAN_MAMBA:
        raise RuntimeError("VMamba selective_scan_cuda is missing; complete the kernel setup first")
    backbone = official.vmamba_tiny_s1l8(pretrained=False)
    backbone.classifier.head = nn.Identity()
    from functools import partial
    for module in backbone.modules():
        if isinstance(module, official.SS2D):
            module.forward_core = partial(module.forward_corev2, force_fp32=False,
                                          no_einsum=True, selective_scan_backend="mamba",
                                          scan_force_torch=True)
        if isinstance(module, official.VSSBlock):
            module.use_checkpoint = config.get("activation_checkpointing", True)
    return ScratchBackboneClassifier(backbone, feature_dim=backbone.num_features)
