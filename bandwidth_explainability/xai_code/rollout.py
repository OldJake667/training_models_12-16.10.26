"""Mean-head attention rollout with residual connections (Abnar & Zuidema)."""
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .runtime import read_rgb, write_json, save_figure, summarize


def combine_attention(attentions, grid, prefix_tokens=1):
    """CLS-to-patch flow: normalize(A + I), then multiply in layer order."""
    if not attentions:
        raise ValueError('No attention layers were captured')
    tokens = attentions[0].shape[-1]
    if tokens - prefix_tokens != grid[0]*grid[1]:
        raise ValueError('Attention token count does not match the patch grid')
    identity = torch.eye(tokens, dtype=torch.float64)
    joint = identity.clone()
    layer_means = []
    for attention in attentions:
        if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[-2:] != (tokens, tokens):
            raise ValueError('Expected one-image attention with shape [1, heads, tokens, tokens]')
        mean = attention[0].detach().to(device='cpu', dtype=torch.float64).mean(dim=0)
        if not torch.isfinite(mean).all() or (mean < 0).any():
            raise ValueError('Invalid attention probabilities')
        augmented = mean + identity
        augmented = augmented/augmented.sum(dim=-1, keepdim=True)
        joint = augmented @ joint
        layer_means.append(mean.numpy())
    raw = joint[0, prefix_tokens:].reshape(grid).numpy()
    relative = raw/raw.max() if raw.max() > 0 else np.zeros_like(raw)
    return raw.astype(np.float32), relative.astype(np.float32), np.stack(layer_means).astype(np.float32)


@torch.inference_mode()
def capture_attention(predictor, rgb):
    model = predictor.model
    name = predictor.model_name
    if name not in ('vit_tiny', 'dinov2'):
        raise ValueError('Ordinary transformer attention rollout is defined here only for ViT-Tiny/DINOv2')
    tensor = predictor.tensor(rgb).unsqueeze(0).to(predictor.device)
    base = model(tensor).softmax(dim=1).cpu().numpy()[0]
    attentions, handles, fused_settings = [], [], []
    try:
        if name == 'vit_tiny':
            def request_weights(module, args, kwargs):
                kwargs = dict(kwargs)
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                return args, kwargs
            def capture(module, args, result):
                if result[1] is None:
                    raise RuntimeError('MultiheadAttention did not return attention weights')
                attentions.append(result[1].detach().cpu())
            for block in model.blocks:
                handles.append(block.attention.register_forward_pre_hook(request_weights, with_kwargs=True))
                handles.append(block.attention.register_forward_hook(capture))
            grid, prefix = (14, 14), 1
        else:
            def capture(module, args, result):
                attentions.append(result.detach().cpu())
            for block in model.blocks:
                attention = block.attn
                fused_settings.append((attention, attention.fused_attn))
                # Hook the actual softmax output through attn_drop, preserving
                # timm's q/k normalization and token layout. Dropout is off in eval.
                attention.fused_attn = False
                handles.append(attention.attn_drop.register_forward_hook(capture))
            grid = tuple(model.patch_embed.grid_size)
            prefix = model.num_prefix_tokens
        hooked = model(tensor).softmax(dim=1).cpu().numpy()[0]
    finally:
        for handle in handles:
            handle.remove()
        for module, setting in fused_settings:
            module.fused_attn = setting
    if len(attentions) != len(model.blocks):
        raise RuntimeError('One attention matrix per transformer block was not captured')
    if int(base.argmax()) != int(hooked.argmax()) or not np.allclose(base, hooked, atol=1e-4, rtol=1e-4):
        raise RuntimeError('Attention capture changed inference materially; inspect software/precision before continuing')
    raw, relative, layers = combine_attention(attentions, grid, prefix_tokens=prefix)
    native = np.asarray(Image.fromarray(relative).resize(
        (rgb.shape[1], rgb.shape[0]), resample=Image.Resampling.BILINEAR), dtype=np.float32)
    return raw, relative, native, layers, float(np.max(np.abs(base-hooked)))


def run_rollout(root, rows, audit, predictor, output):
    for index, row in enumerate(rows, start=1):
        destination = output/Path(row['file']).stem
        if all((destination/name).is_file() for name in ['done.json', 'maps.npz', 'explanation.png']):
            print(f"[{index}/45] Already complete: {row['file']}")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        rgb = read_rgb(root, row)
        record = audit[row['file']]
        print(f"[{index}/45] Attention rollout: {row['file']}")
        raw, relative, native, layers, delta = capture_attention(predictor, rgb)
        np.savez_compressed(destination/'maps.npz', raw_patch_flow=raw,
            relative_patch_flow=relative, native_heatmap=native, layer_mean_attention=layers)
        Image.fromarray(rgb).save(destination/'original.png')
        save_figure(rgb, native, 'Attention flow from CLS token', destination/'explanation.png')
        write_json(destination/'done.json', {**record, 'method': 'attention_rollout',
            'class_specific': False, 'head_fusion': 'mean', 'residual': 'add identity then row normalize',
            'patch_rows': raw.shape[0], 'patch_columns': raw.shape[1],
            'attention_capture_probability_difference': delta})
    return summarize(root, rows, audit, output, method='attention_rollout')
