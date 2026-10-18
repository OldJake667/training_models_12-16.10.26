"""Regenerate clean notebooks from the packaged explanation implementations."""
from pathlib import Path
import ast
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.strip()+'\n')


ROOT_CELL = '''
from pathlib import Path
import sys
import numpy as np
import torch

# Usually leave this as None. Set a path only if Jupyter starts elsewhere.
PACKAGE_ROOT = None
if PACKAGE_ROOT is None:
    candidates = [Path.cwd(), *Path.cwd().parents]
    ROOT = next((p for p in candidates if (p/'package_integrity.json').is_file()
                 and (p/'xai_code').is_dir()), None)
    if ROOT is None:
        raise FileNotFoundError('Start Jupyter inside the extracted package, or set PACKAGE_ROOT.')
else:
    ROOT = Path(PACKAGE_ROOT).expanduser().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from xai_code.models import build_model, Predictor
from xai_code.runtime import prepare, audit_predictions, verify_package, environment, samples
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Package:', ROOT)
print('Device:', DEVICE)
'''


def implementation_cells(filename):
    text = (ROOT/'xai_code'/filename).read_text().replace('from .runtime import', 'from xai_code.runtime import')
    tree = ast.parse(text)
    lines = text.splitlines()
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    first = min([functions[0].lineno] + [d.lineno for d in functions[0].decorator_list])
    result = [code('\n'.join(lines[:first-1]))]
    for node in functions:
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        result.append(code('\n'.join(lines[start-1:node.end_lineno])))
    return result


def save(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {'kernelspec': {'display_name': 'Bandwidth XAI CUDA', 'language': 'python',
                                       'name': 'bandwidth-xai'},
                         'language_info': {'name': 'python', 'version': '3.11'}}
    nbf.validate(notebook)
    nbf.write(notebook, ROOT/name)


def main():
    env = [markdown('''# Environment and packaged inputs

Run this on the computer before the model notebooks. This package contains
the original 45 validation images, original trained best-accuracy checkpoints,
and archived predictions. It does not contain new training runs or explanations.
Read [START_HERE.md](START_HERE.md) for installation, particularly VMamba.
'''), code(ROOT_CELL), code('''
import json
verify_package(ROOT)
print(json.dumps(environment(), indent=2))
assert len(samples(ROOT)) == 45
print('45 validation images; classes 0=Low bandwidth, 1=High bandwidth')
if DEVICE.type == 'cuda':
    value = torch.randn(32, 32, device=DEVICE)
    assert torch.isfinite(value @ value.T).all()
    print('Basic CUDA operation passed:', torch.cuda.get_device_name(0))
else:
    print('CUDA is unavailable. Resolve this before VMamba LIME; other models support CPU.')
'''), markdown('''## Optional VMamba scan check

After installing the CUDA extension, set `CHECK_VMAMBA_KERNEL=True` and run
this cell. It checks a small synthetic forward against the reference scan;
it does not train a model or explain a validation image.
'''), code('''
CHECK_VMAMBA_KERNEL = False
if CHECK_VMAMBA_KERNEL:
    from xai_code import vmamba_official as official
    if DEVICE.type != 'cuda' or not official.WITH_SELECTIVESCAN_MAMBA:
        raise RuntimeError('CUDA / selective_scan_cuda missing; see START_HERE.md')
    torch.manual_seed(42)
    values = [torch.randn((1, 8, 16), device=DEVICE)*0.1,
              torch.randn((1, 8, 16), device=DEVICE)*0.1,
              -torch.rand((8, 1), device=DEVICE),
              torch.randn((1, 4, 1, 16), device=DEVICE)*0.1,
              torch.randn((1, 4, 1, 16), device=DEVICE)*0.1,
              torch.randn(8, device=DEVICE)*0.1,
              torch.randn(8, device=DEVICE)*0.1]
    with torch.inference_mode():
        reference = official.selective_scan_fn(*values, delta_softplus=True, oflex=True, backend='torch')
        actual = official.selective_scan_fn(*values, delta_softplus=True, oflex=True, backend='mamba')
    torch.testing.assert_close(actual.float(), reference.float(), rtol=1e-4, atol=1e-4)
    print('VMamba CUDA scan matches the float32 reference on synthetic inputs.')
else:
    print('Optional VMamba check skipped; enable it before running VMamba LIME.')
'''), markdown('''Next, open a model notebook and use **Run All**. Start with CNN LIME.
Each model first audits all 45 predictions against the saved results. Keep the
package folder together when transferring it; absolute paths from the old
computer are not needed.''')]
    save('00_environment_and_checkpoint_check.ipynb', env)
    specifications = [('01', 'cnn', 'CNN', 'lime', 8, 44),
                      ('02', 'vit_tiny', 'ViT-Tiny', 'lime', 8, 43),
                      ('03', 'dinov2', 'DINOv2-S/14', 'lime', 8, 43),
                      ('04', 'vmamba', 'VMamba-T', 'lime', 4, 44),
                      ('05', 'vit_tiny', 'ViT-Tiny', 'attention_rollout', 1, 43),
                      ('06', 'dinov2', 'DINOv2-S/14', 'attention_rollout', 1, 43),
                      ('07', 'vit_tiny', 'ViT-Tiny', 'gradient_attention_rollout', 1, 43),
                      ('08', 'dinov2', 'DINOv2-S/14', 'gradient_attention_rollout', 1, 43)]
    for number, name, title, method, batch_size, correct in specifications:
        cells = [markdown(f'''# {title}: {method.replace('_', ' ')}

Explain all **45 original bandwidth validation images**, including mistakes.
Restore the existing best-validation-accuracy checkpoint: archived result
**{correct}/45**. No training is performed. Classes: **0 Low, 1 High bandwidth**.
Use [START_HERE.md](START_HERE.md) and notebook 00 to set up the computer.
These validation images were used in checkpoint selection; they are not an
independent test set.
'''), code(ROOT_CELL)]
        configuration = f'''
MODEL_NAME = {name!r}
METHOD = {method!r}
BATCH_SIZE = {batch_size}  # Reduce this if GPU memory is insufficient.
ALLOW_PREDICTION_DIFFERENCES = False
SETTINGS = {{
    'seed': 42,
    'batch_size': BATCH_SIZE,
    'allow_prediction_differences': ALLOW_PREDICTION_DIFFERENCES,
}}
'''
        if method == 'lime':
            configuration += '''
SETTINGS.update({
    'n_segments': 100,
    'compactness': 10,
    'num_samples': 1000,
    'hide_color': 255,
    'kernel_width': 0.25,
    'display_features': None,  # None = all positive segments; or set a top-k.
    'segmentation_resolution': 'native, common across models',
    'feature_selection': 'none',
    'distance_metric': 'cosine',
})
'''
        elif method == 'attention_rollout':
            configuration += '''
SETTINGS.update({'head_fusion': 'mean', 'discard_ratio': 0.0,
                 'residual': 'add identity then row normalize', 'class_specific': False})
'''
        else:
            configuration += '''
SETTINGS.update({'head_fusion': 'mean(A * gradient), then clamp negatives',
                 'discard_ratio': 0.0, 'target_classes': [0, 1],
                 'target_score': 'pre-softmax class logit',
                 'residual': 'add identity then row normalize', 'class_specific': True})
'''
        cells.extend([markdown('## Configuration'), code(configuration),
                      markdown('## Restore the fixed checkpoint and verify all predictions'), code('''
OUTPUT = prepare(ROOT, MODEL_NAME, METHOD, SETTINGS, DEVICE)
model = build_model(ROOT, MODEL_NAME, DEVICE)
predictor = Predictor(model, MODEL_NAME, DEVICE, batch_size=BATCH_SIZE)
ROWS, AUDIT = audit_predictions(ROOT, MODEL_NAME, predictor, OUTPUT,
    allow_prediction_differences=ALLOW_PREDICTION_DIFFERENCES)
print('Ready to explain ALL', len(ROWS), 'validation images.')
''')])
        if method == 'lime':
            cells.append(markdown('''## LIME implementation

The following cells contain the actual explainer and per-image execution code.
Reported paper settings: SLIC ≈100 segments, compactness 10, connectivity,
1,000 perturbations, white masking, positive evidence for the predicted class.
Shared native-resolution segmentation and the recorded surrogate settings
are explicit choices for comparing our four classifiers. Every perturbation
passes through the full original preprocessing, backbone and head.
'''))
            cells.extend(implementation_cells('lime_workflow.py'))
            cells.extend([markdown('## Run LIME for all 45 images; resume completed images'),
                          code('summaries = run_lime(ROOT, ROWS, AUDIT, predictor, OUTPUT, SETTINGS)')])
        elif method == 'attention_rollout':
            cells.append(markdown('''## Attention rollout implementation

This is an extension, separate from the original paper's LIME. Average heads,
add residual identity, normalize rows, multiply layers, then read CLS-to-patch
flow. Standard rollout is **not class-specific** and does not explain changes
in a particular output score. Capture probabilities are checked against the
original forward. Per-image maximum normalization makes the display relative.
'''))
            cells.extend(implementation_cells('rollout.py'))
            cells.extend([markdown('## Run rollout for all 45 images; resume completed images'),
                          code('summaries = run_rollout(ROOT, ROWS, AUDIT, predictor, OUTPUT)')])
        else:
            cells.append(markdown('''## Gradient attention rollout implementation

For each image, explain **both Low and High pre-softmax logits**, independently
of its prediction or ground truth. Weight attention by the target-logit gradient,
average heads, retain positive weights, add residual identity, normalize rows,
and multiply layers. No low-value weights are discarded. This gradient-weighted
variant is distinct from standard rollout and from full Chefer relevance propagation.

The original weights remain frozen and unchanged; gradients are enabled through
the input and attention computation. Do not put these cells inside
`torch.inference_mode()`. ViT-Tiny temporarily uses an equivalent explicit Q/K/V
forward to expose the attention tensor actually used in the output. DINOv2
temporarily disables fused attention. Both paths verify prediction agreement
and restore normal forwards/hooks afterward.

High and Low maps are separately maximum-normalized for visibility. Equally
bright regions in the two maps do not mean equal evidence, and these maps do
not directly explain the High-minus-Low logit margin. Raw maps and logits are
saved. Empty positive maps are recorded rather than rescaled into invented signal.
'''))
            cells.extend(implementation_cells('gradient_rollout.py'))
            cells.extend([markdown('## Run both target maps for all 45 images; resume completed images'),
                          code('summaries = run_gradient_rollout(ROOT, ROWS, AUDIT, predictor, OUTPUT)')])
        cells.extend([markdown('## Inspect saved examples, including errors'), code('''
from IPython.display import display, Image as DisplayImage
print('Saved results:', OUTPUT)
error_names = [row['file'] for row in ROWS if not AUDIT[row['file']]['correct']]
example_names = list(dict.fromkeys(error_names + [ROWS[0]['file']]))
for filename in example_names:
    print(filename, AUDIT[filename])
    figure = OUTPUT/Path(filename).stem/'explanation.png'
    if figure.is_file():
        display(DisplayImage(filename=str(figure)))
'''), markdown('''Outputs include per-image raw maps and metadata, a prediction audit,
summary CSV and class means grouped by predicted class, including mistakes.
Read the local fit scores for LIME. Heatmaps alone do not establish physical
causality. The earlier mean-brightness diagnostic is included in `reference`.

References: [photonic LIME paper](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adpr.202500284),
[LIME](https://github.com/marcotcr/lime),
[attention rollout](https://arxiv.org/abs/2005.00928),
[gradient-weighted rollout variant](https://github.com/jacobgil/vit-explain/blob/main/vit_grad_rollout.py).
''')])
        save(f'{number}_{name}_{method}.ipynb', cells)
    print('Generated nine clean, unexecuted notebooks.')


if __name__ == '__main__':
    main()
