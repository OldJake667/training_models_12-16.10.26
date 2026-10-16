"""Static/input/weight checks and synthetic rollout checks; no real-image forwards."""
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import nbformat
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xai_code.models import build_model, Predictor
from xai_code.rollout import combine_attention, capture_attention
from xai_code.gradient_rollout import combine_gradient_attention, capture_gradient_attention
from xai_code.runtime import samples, read_rgb


def main():
    checks = []
    for path in sorted((ROOT/'xai_code').glob('*.py')):
        ast.parse(path.read_text(), filename=str(path))
    for path in sorted(ROOT.glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                assert cell.execution_count is None and not cell.outputs
                compile(cell.source, f'{path.name}:cell{index}', 'exec')
    assert len(list(ROOT.glob('*.ipynb'))) == 9
    checks.append('Nine valid, unexecuted notebooks; all code cells and shared modules compile.')
    rows = samples(ROOT)
    shapes = set()
    for row in rows:
        image = ROOT/row['relative_path']
        assert hashlib.sha256(image.read_bytes()).hexdigest() == row['sha256']
        with Image.open(image) as value:
            shapes.add(value.size)
    assert shapes == {(770, 760)}
    original = list(csv.DictReader((ROOT/'reference/original_176_45_split_manifest.csv').open()))
    train = {r['file'] for r in original if r['split'] == 'train'}
    validation = {r['file'] for r in original if r['split'] == 'validation'}
    assert len(train) == 176 and len(validation) == 45 and not train & validation
    assert validation == {r['file'] for r in rows}
    checks.append('All 45 native 770x760 BMP hashes and the original disjoint 176/45 split match.')
    reference_counts = {}
    for name, expected in [('cnn', 44), ('vit_tiny', 43), ('dinov2', 43), ('vmamba', 44)]:
        archived = list(csv.DictReader((ROOT/f'reference/{name}/best_accuracy_predictions.csv').open()))
        assert {r['file'] for r in archived} == validation
        correct = sum(r['true_label'] == r['predicted_label'] for r in archived)
        assert correct == expected
        reference_counts[name] = correct
        # This restores parameters only. No validation image is sent to a model.
        model = build_model(ROOT, name, 'cpu', require_fast_vmamba=False)
        assert not model.training and all(not p.requires_grad for p in model.parameters())
        predictor = Predictor(model, name, 'cpu')
        tensor = predictor.tensor(read_rgb(ROOT, rows[0]))
        channels, size = (1, 256) if name == 'cnn' else ((1, 224) if name == 'vit_tiny' else (3, 224))
        assert tensor.shape == (channels, size, size) and torch.isfinite(tensor).all()
        del model, predictor, tensor
    checks.append('All four original checkpoints load strictly; expected preprocessing shapes and archived accuracy counts verified.')
    # Noncommuting synthetic layers check multiplication direction and residuals.
    a = torch.tensor([[[[0.2, 0.8], [0.4, 0.6]]]])
    b = torch.tensor([[[[0.7, 0.3], [0.9, 0.1]]]])
    identity = np.eye(2)
    expected_joint = ((b[0, 0].numpy()+identity)/2) @ ((a[0, 0].numpy()+identity)/2)
    raw, relative, layers = combine_attention([a, b], (1, 1), prefix_tokens=1)
    np.testing.assert_allclose(raw[0, 0], expected_joint[0, 1], rtol=1e-6)
    assert relative[0, 0] == 1 and layers.shape == (2, 2, 2)
    checks.append('Synthetic attention aggregation checks residual normalization and layer multiplication order.')
    # Zero score gradients must produce no patch attribution, not a NaN map.
    raw, relative, weighted, maxima = combine_gradient_attention([a, b], [torch.zeros_like(a), torch.zeros_like(b)], (1, 1))
    assert not np.any(raw) and not np.any(relative) and np.isfinite(relative).all()
    # With gradients of one and nonnegative attention, this variant matches
    # standard mean-head rollout. Negative contributions are discarded.
    raw_grad, _, _, _ = combine_gradient_attention([a, b], [torch.ones_like(a), torch.ones_like(b)], (1, 1))
    raw_plain, _, _ = combine_attention([a, b], (1, 1))
    np.testing.assert_allclose(raw_grad, raw_plain, rtol=1e-6)
    raw_negative, _, _, _ = combine_gradient_attention([a], [-torch.ones_like(a)], (1, 1))
    assert not np.any(raw_negative)
    checks.append('Synthetic gradient rollout checks zero/negative gradients, finite empty maps and positive weighting.')
    # Only synthetic tiny images below. This exercises actual hook paths without
    # reading/forwarding real validation data or producing explanation figures.
    from xai_code.scratch_models import VisionTransformerTiny
    import timm
    torch.manual_seed(42)
    synthetic = np.random.RandomState(42).randint(0, 256, (32, 32, 3)).astype(np.uint8)
    for name, model in [
        ('vit_tiny', VisionTransformerTiny().eval().requires_grad_(False)),
        ('dinov2', timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False,
                                    num_classes=2, img_size=224).eval().requires_grad_(False))]:
        predictor = Predictor(model, name, 'cpu')
        raw, relative, native, layers, delta = capture_attention(predictor, synthetic)
        assert raw.shape == ((14, 14) if name == 'vit_tiny' else (16, 16))
        assert native.shape == (32, 32) and np.isfinite(native).all()
        assert len(layers) == 12
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        normal_probabilities = predictor([synthetic])
        maps, delta = capture_gradient_attention(predictor, synthetic, targets=(0, 1))
        assert set(maps) == {0, 1}
        for target in (0, 1):
            assert maps[target]['native_heatmap'].shape == (32, 32)
            assert np.isfinite(maps[target]['native_heatmap']).all()
            assert len(maps[target]['gradient_maxima']) == 12
            assert all(value > 0 for value in maps[target]['gradient_maxima'])
        assert not np.allclose(maps[0]['raw_patch_flow'], maps[1]['raw_patch_flow'], rtol=1e-4, atol=1e-12)
        np.testing.assert_allclose(predictor([synthetic]), normal_probabilities, rtol=1e-5, atol=1e-6)
        assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        if name == 'vit_tiny':
            assert all('forward' not in block.attention.__dict__ for block in model.blocks)
        del before
    checks.append('Both real transformer hook paths pass on random-weight models with synthetic inputs; hooks are cleaned up.')
    checks.append('Both gradient rollout adapters produce target-specific maps and nonzero gradients at every layer on synthetic inputs; weights and predictions stay unchanged and hooks/forwards are restored.')
    report = {'checks_passed': checks, 'archived_correct_out_of_45': reference_counts,
              'training_executed': False, 'real_image_explanations_executed': False,
              'real_image_model_forwards_executed': False,
              'lime_execution_tested': False, 'RTX_5090_execution_tested': False,
              'note': 'LIME/scikit-image are not installed locally; full LIME and CUDA execution must be checked on the lab computer.'}
    (ROOT/'reference/preparation_checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()
