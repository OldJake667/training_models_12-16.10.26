"""Class-specific gradient-weighted rollout; no training or weight updates."""
from pathlib import Path
from types import MethodType
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .runtime import read_rgb, write_json, write_csv


def combine_gradient_attention(attentions, gradients, grid, prefix_tokens=1):
    """mean_heads(A * d(score)/dA), clip negatives, add I, normalize, roll out."""
    if not attentions or len(attentions) != len(gradients):
        raise ValueError('One gradient tensor is required per attention layer')
    tokens = attentions[0].shape[-1]
    if tokens - prefix_tokens != grid[0]*grid[1]:
        raise ValueError('Attention token count does not match the patch grid')
    identity = torch.eye(tokens, dtype=torch.float64)
    joint = identity.clone()
    weighted_layers, gradient_maxima = [], []
    for attention, gradient in zip(attentions, gradients):
        if attention.ndim != 4 or attention.shape[0] != 1 or attention.shape[-2:] != (tokens, tokens):
            raise ValueError('Expected [1, heads, tokens, tokens] attention')
        if gradient is None or gradient.shape != attention.shape:
            raise RuntimeError('Attention is disconnected from the selected class score')
        attention = attention.detach().to(device='cpu', dtype=torch.float64)
        gradient = gradient.detach().to(device='cpu', dtype=torch.float64)
        if not torch.isfinite(attention).all() or not torch.isfinite(gradient).all() or (attention < 0).any():
            raise ValueError('Nonfinite or invalid attention/gradient')
        weighted = (attention[0]*gradient[0]).mean(dim=0).clamp_min(0)
        augmented = weighted + identity
        augmented = augmented/augmented.sum(dim=-1, keepdim=True)
        joint = augmented @ joint
        weighted_layers.append(weighted.numpy())
        gradient_maxima.append(float(gradient.abs().max()))
    raw = joint[0, prefix_tokens:].reshape(grid).numpy()
    maximum = float(raw.max())
    relative = raw/maximum if maximum > 0 else np.zeros_like(raw)
    return (raw.astype(np.float32), relative.astype(np.float32),
            np.stack(weighted_layers).astype(np.float32), gradient_maxima)


def capture_gradient_attention(predictor, rgb, targets=(0, 1)):
    """Differentiate logits through the actual attention probabilities used @ V.

    Frozen model weights stay frozen; only the input needs requires_grad=True.
    Returned nn.MultiheadAttention weights may be a view outside the output's
    ancestry, so this custom ViT-Tiny uses an equivalent explicit Q/K/V forward
    temporarily, rather than differentiating disconnected returned weights.
    """
    model, name = predictor.model, predictor.model_name
    if name not in ('vit_tiny', 'dinov2') or model.training:
        raise ValueError('Requires an eval-mode ViT-Tiny or DINOv2 classifier')
    targets = tuple(int(t) for t in targets)
    if not targets or len(set(targets)) != len(targets) or any(t not in (0, 1) for t in targets):
        raise ValueError('Targets must be unique class IDs 0 and/or 1')
    if torch.is_inference_mode_enabled():
        raise RuntimeError('Gradient rollout must run outside torch.inference_mode()')
    base = predictor([rgb])[0]
    attentions, handles, restored_forwards, fused_settings = [], [], [], []
    try:
        if name == 'vit_tiny':
            def explicit_self_attention(module, query, key, value, **kwargs):
                if query is not key or query is not value or not module.batch_first:
                    raise ValueError('This adapter supports the original batch-first self-attention only')
                if (module.training or module.bias_k is not None or module.bias_v is not None
                    or module.add_zero_attn or not module._qkv_same_embed_dim
                    or kwargs.get('attn_mask') is not None or kwargs.get('key_padding_mask') is not None
                    or kwargs.get('is_causal', False)):
                    raise ValueError('Unsupported attention configuration')
                batch, tokens, dimension = query.shape
                heads, head_size = module.num_heads, dimension//module.num_heads
                qkv = F.linear(query, module.in_proj_weight, module.in_proj_bias)
                qkv = qkv.reshape(batch, tokens, 3, heads, head_size).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(dim=0)
                probabilities = ((q*(head_size**-0.5)) @ k.transpose(-2, -1)).softmax(dim=-1)
                attentions.append(probabilities)
                attended = (probabilities @ v).transpose(1, 2).reshape(batch, tokens, dimension)
                attended = F.linear(attended, module.out_proj.weight, module.out_proj.bias)
                return attended, probabilities
            for block in model.blocks:
                module = block.attention
                restored_forwards.append((module, 'forward' in module.__dict__, module.forward))
                module.forward = MethodType(explicit_self_attention, module)
            grid, prefix = (14, 14), 1
        else:
            def capture(module, args, result):
                attentions.append(result)
            for block in model.blocks:
                module = block.attn
                fused_settings.append((module, module.fused_attn))
                module.fused_attn = False
                handles.append(module.attn_drop.register_forward_hook(capture))
            grid, prefix = tuple(model.patch_embed.grid_size), model.num_prefix_tokens
        with torch.enable_grad():
            tensor = predictor.tensor(rgb).unsqueeze(0).to(predictor.device).detach().requires_grad_(True)
            logits = model(tensor)
            captured = logits.detach().softmax(dim=1).cpu().numpy()[0]
            delta = float(np.max(np.abs(base-captured)))
            if int(base.argmax()) != int(captured.argmax()) or not np.allclose(base, captured, atol=1e-4, rtol=1e-4):
                raise RuntimeError('Gradient capture changed inference materially; inspect precision/software')
            if len(attentions) != len(model.blocks) or not all(a.requires_grad for a in attentions):
                raise RuntimeError('Attention capture is incomplete or gradients are disabled')
            maps = {}
            for index, target in enumerate(targets):
                gradients = torch.autograd.grad(logits[0, target], attentions,
                    retain_graph=index < len(targets)-1, allow_unused=False)
                raw, relative, layers, maxima = combine_gradient_attention(attentions, gradients, grid, prefix)
                native = np.asarray(Image.fromarray(relative).resize(
                    (rgb.shape[1], rgb.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
                maps[target] = {'raw_patch_flow': raw, 'relative_patch_flow': relative,
                    'native_heatmap': native, 'positive_weighted_layers': layers,
                    'gradient_maxima': maxima, 'target_logit': float(logits[0, target].detach().cpu()),
                    'empty_positive_map': bool(not np.any(raw > 0))}
        return maps, delta
    finally:
        for handle in handles:
            handle.remove()
        for module, setting in fused_settings:
            module.fused_attn = setting
        for module, had_instance_forward, forward in restored_forwards:
            if had_instance_forward:
                module.forward = forward
            else:
                del module.forward


def save_gradient_figure(rgb, low, high, record, path):
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"True: {'High' if record['true_label'] else 'Low'}; predicted: {'High' if record['predicted_label'] else 'Low'}")
    for axis, heatmap, title in [(axes[1], low, 'Target: Low score'), (axes[2], high, 'Target: High score')]:
        axis.imshow(rgb)
        axis.imshow(np.ma.masked_where(heatmap <= 0, heatmap), cmap='viridis', vmin=0, vmax=1, alpha=0.55)
        axis.set_title(title)
    for axis in axes:
        axis.axis('off')
    figure.suptitle('Gradient attention rollout — each target map normalized separately')
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)


def summarize_gradient(rows, audit, output):
    from matplotlib import pyplot as plt
    completed = [row for row in rows if (output/Path(row['file']).stem/'done.json').is_file()]
    import json
    summaries = [json.loads((output/Path(row['file']).stem/'done.json').read_text()) for row in completed]
    write_csv(output/'summary.csv', summaries)
    for target, target_name in [(0, 'low'), (1, 'high')]:
        for grouping, label_key in [('predicted', 'predicted_label'), ('true', 'true_label')]:
            for group, group_name in [(0, 'low'), (1, 'high')]:
                members = [r for r in completed if audit[r['file']][label_key] == group]
                if not members:
                    continue
                maps = []
                for row in members:
                    with np.load(output/Path(row['file']).stem/'maps.npz', allow_pickle=False) as data:
                        maps.append(data[f'target_{target}_native_heatmap'])
                mean = np.mean(maps, axis=0)
                stem = f'target_{target_name}_group_{grouping}_{group_name}'
                np.save(output/f'{stem}_mean.npy', mean, allow_pickle=False)
                plt.imsave(output/f'{stem}_mean.png', mean, cmap='viridis', vmin=0, vmax=1)
                write_json(output/f'{stem}_members.json', {'target_class': target, 'grouped_by': grouping,
                    'group_class': group, 'includes_errors': True, 'image_count': len(members),
                    'files': [r['file'] for r in members],
                    'definition': 'mean separately maximum-normalized target-logit gradient rollout'})
    print(f'{len(completed)}/45 images complete, with both Low and High target maps.')
    return summaries


def run_gradient_rollout(root, rows, audit, predictor, output):
    from matplotlib import pyplot as plt
    for index, row in enumerate(rows, start=1):
        destination = output/Path(row['file']).stem
        required = ['done.json', 'maps.npz', 'explanation.png', 'target_low.png', 'target_high.png']
        if all((destination/name).is_file() for name in required):
            print(f"[{index}/45] Already complete: {row['file']}")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        rgb, record = read_rgb(root, row), audit[row['file']]
        print(f"[{index}/45] Gradient rollout for Low AND High: {row['file']}")
        maps, delta = capture_gradient_attention(predictor, rgb, targets=(0, 1))
        arrays = {f'target_{target}_{key}': values[key] for target, values in maps.items()
                  for key in ['raw_patch_flow', 'relative_patch_flow', 'native_heatmap', 'positive_weighted_layers']}
        np.savez_compressed(destination/'maps.npz', **arrays)
        Image.fromarray(rgb).save(destination/'original.png')
        for target, name in [(0, 'low'), (1, 'high')]:
            plt.imsave(destination/f'target_{name}.png', maps[target]['native_heatmap'], cmap='viridis', vmin=0, vmax=1)
        save_gradient_figure(rgb, maps[0]['native_heatmap'], maps[1]['native_heatmap'], record, destination/'explanation.png')
        details = {**record, 'method': 'gradient_attention_rollout', 'class_specific': True,
            'target_score': 'pre-softmax class logit', 'head_fusion': 'mean(A * gradient), then clamp negatives',
            'attention_capture_probability_difference': delta}
        for target, name in [(0, 'low'), (1, 'high')]:
            details[f'{name}_target_logit'] = maps[target]['target_logit']
            details[f'{name}_empty_positive_map'] = maps[target]['empty_positive_map']
            details[f'{name}_gradient_maximum'] = max(maps[target]['gradient_maxima'])
        write_json(destination/'done.json', details)
    return summarize_gradient(rows, audit, output)
