"""Restore the exact original classifier architectures and trained weights."""
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from .scratch_models import ScratchCNN, VisionTransformerTiny

MODEL_NAMES = ('cnn', 'vit_tiny', 'dinov2', 'vmamba')


class PretrainedVMamba(nn.Module):
    def __init__(self, root, require_fast=True):
        super().__init__()
        from . import vmamba_official as official
        if require_fast and (not torch.cuda.is_available() or not official.WITH_SELECTIVESCAN_MAMBA):
            raise RuntimeError('VMamba LIME requires CUDA and selective_scan_cuda. See START_HERE.md.')
        self.backbone = official.vmamba_tiny_s1l8(pretrained=False)
        state = torch.load(Path(root)/'checkpoints/vmamba_imagenet.pth', map_location='cpu', weights_only=True)
        self.backbone.load_state_dict(state.get('model', state), strict=True)
        self.backbone.classifier.head = nn.Identity()
        self.head = nn.Linear(768, 2)
        # On CUDA use the installed scan kernel and torch cross-scan/merge;
        # avoid the older vendored Triton cross-scan kernels on RTX 5090.
        if require_fast:
            for module in self.backbone.modules():
                if isinstance(module, official.SS2D):
                    module.forward_core = partial(module.forward_corev2, force_fp32=False,
                        no_einsum=True, selective_scan_backend='mamba', scan_force_torch=True)

    def forward(self, images):
        return self.head(self.backbone(images))


def build_model(root, model_name, device, require_fast_vmamba=True):
    if model_name == 'cnn':
        model = ScratchCNN(classes=2)
    elif model_name == 'vit_tiny':
        model = VisionTransformerTiny(classes=2)
    elif model_name == 'dinov2':
        import timm
        # The full original trained checkpoint already includes the backbone.
        model = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                                  num_classes=2, img_size=224)
    elif model_name == 'vmamba':
        model = PretrainedVMamba(root, require_fast=require_fast_vmamba)
    else:
        raise ValueError(model_name)
    state = torch.load(Path(root)/f'checkpoints/{model_name}_best_accuracy.pth',
                       map_location='cpu', weights_only=True)
    (model.head if model_name == 'vmamba' else model).load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model.eval().to(device=device, dtype=torch.float32)


def make_transform(model_name):
    size = 256 if model_name == 'cnn' else 224
    steps = [transforms.Resize((size, size)), transforms.ToTensor()]
    if model_name == 'vit_tiny':
        steps.append(transforms.Normalize((0.5,), (0.5,)))
    elif model_name in ('dinov2', 'vmamba'):
        steps.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    return transforms.Compose(steps)


class Predictor:
    """LIME callback: recompute the ENTIRE model for every perturbed image."""
    def __init__(self, model, model_name, device, batch_size=8):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.model, self.model_name = model, model_name
        self.device, self.batch_size = torch.device(device), batch_size
        self.transform = make_transform(model_name)

    def tensor(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or not np.isfinite(rgb).all():
            raise ValueError('Expected finite H x W x 3 image, in the 0..255 range')
        if rgb.min() < 0 or rgb.max() > 255:
            raise ValueError('LIME images must use 0..255 pixel values')
        image = Image.fromarray(np.rint(rgb).astype(np.uint8)).convert('L')
        if self.model_name in ('dinov2', 'vmamba'):
            image = image.convert('RGB')
        return self.transform(image)

    @torch.inference_mode()
    def __call__(self, images):
        probabilities = []
        for start in range(0, len(images), self.batch_size):
            batch = torch.stack([self.tensor(x) for x in images[start:start+self.batch_size]])
            logits = self.model(batch.to(self.device))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
        return np.concatenate(probabilities, axis=0) if probabilities else np.empty((0, 2))
