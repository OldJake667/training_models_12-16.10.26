"""Portable inputs, integrity checks, prediction audit, and output bookkeeping."""
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
from PIL import Image
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_package(root):
    root = Path(root)
    expected = json.loads((root/'package_integrity.json').read_text())
    checked = 0
    for relative, digest in expected.items():
        # Notebook outputs and editable configuration cells change when used.
        # Preserve their distribution hashes, but do not treat normal saving as
        # input corruption. Their current code-cell source is recorded below.
        if relative == 'reference/preparation_checks.json':
            continue
        if relative.endswith('.ipynb'):
            if not (root/relative).is_file():
                raise ValueError(f'Missing notebook: {relative}')
            continue
        path = root/relative
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'Missing or changed packaged file: {relative}')
        checked += 1
    print(f'Integrity OK: {checked} fixed files; notebook existence checked separately')


def samples(root):
    with (Path(root)/'data/validation_manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 45 or len({r['file'] for r in rows}) != 45:
        raise ValueError('Expected the original 45 unique validation images')
    for row in rows:
        row['label'] = int(row['label'])
        if row['split'] != 'validation' or row['label'] not in (0, 1):
            raise ValueError('Invalid validation manifest')
    return rows


def read_rgb(root, row):
    with Image.open(Path(root)/row['relative_path']) as image:
        return np.asarray(image.convert('L').convert('RGB')).copy()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def environment():
    versions = {}
    for name in ['torch', 'torchvision', 'timm', 'lime', 'scikit-image', 'numpy', 'scikit-learn', 'Pillow']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'missing'
    return {'python': platform.python_version(), 'system': platform.system(),
            'versions': versions, 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def prepare(root, model_name, method, settings, device):
    root = Path(root)
    verify_package(root)
    if int(settings['seed']) < 0 or int(settings['seed']) >= 2**32:
        raise ValueError('Seed must be in [0, 2**32)')
    np.random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    # Use float32, with no AMP or TF32, to preserve the original preprocessing/inference.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    code_hashes = {p.name: sha256(p) for p in sorted((root/'xai_code').glob('*.py'))}
    notebook_sources = {}
    for path in sorted(root.glob('*.ipynb')):
        notebook = json.loads(path.read_text())
        source = [''.join(cell.get('source', [])) for cell in notebook['cells']
                  if cell['cell_type'] == 'code']
        notebook_sources[path.name] = hashlib.sha256(json.dumps(source).encode()).hexdigest()
    config = {'model': model_name, 'method': method, 'settings': settings,
              'checkpoint': f'checkpoints/{model_name}_best_accuracy.pth',
              'checkpoint_sha256': sha256(root/f'checkpoints/{model_name}_best_accuracy.pth'),
              'manifest_sha256': sha256(root/'data/validation_manifest.csv'),
              'code_sha256': code_hashes, 'notebook_source_sha256': notebook_sources,
              'environment': environment(), 'device': str(device),
              'dataset': 'Original bandwidth, 176 training / 45 validation / no test',
              'classes': {'0': 'Low bandwidth', '1': 'High bandwidth'}}
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    output = root/'outputs'/model_name/method/run_id
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'configuration.json', config)
    print('Outputs:', output)
    return output


def audit_predictions(root, model_name, predictor, output, allow_prediction_differences=False):
    with (Path(root)/f'reference/{model_name}/best_accuracy_predictions.csv').open(newline='') as stream:
        archived = {r['file']: r for r in csv.DictReader(stream)}
    rows = samples(root)
    if set(archived) != {r['file'] for r in rows}:
        raise ValueError('Archived predictions and validation manifest differ')
    report, differences = [], []
    for row in rows:
        probs = predictor([read_rgb(root, row)])[0]
        old = archived[row['file']]
        if int(old['true_label']) != row['label']:
            raise ValueError('Archived ground truth differs from manifest')
        prediction = int(np.argmax(probs))
        changed = prediction != int(old['predicted_label'])
        delta = float(np.max(np.abs(probs - np.array([float(old['probability_low']), float(old['probability_high'])]))))
        report.append({'file': row['file'], 'true_label': row['label'],
            'predicted_label': prediction, 'probability_low': float(probs[0]),
            'probability_high': float(probs[1]), 'correct': prediction == row['label'],
            'archived_prediction': int(old['predicted_label']), 'prediction_changed': changed,
            'max_probability_difference': delta})
        if changed:
            differences.append(row['file'])
    write_csv(output/'prediction_audit.csv', report)
    summary = {'correct': sum(r['correct'] for r in report), 'total': len(report),
               'archived_correct': sum(int(archived[r['file']]['predicted_label']) == r['label'] for r in rows),
               'changed_predictions': differences,
               'maximum_probability_difference': max(r['max_probability_difference'] for r in report),
               'allow_prediction_differences': allow_prediction_differences}
    write_json(output/'prediction_audit.json', summary)
    print(f"Validation audit: {summary['correct']}/45; {len(differences)} changed predictions")
    if summary['maximum_probability_difference'] > 5e-4:
        print('Probability drift exceeds 0.0005; inspect prediction_audit.csv and the software/device versions.')
    if differences and not allow_prediction_differences:
        raise RuntimeError('Predictions changed: inspect prediction_audit.csv before explaining. '
            'Do not describe this run as reproducing the archived accuracy. '
            'An intentional override is available in the notebook configuration.')
    return rows, {r['file']: r for r in report}


def image_seed(filename, base_seed):
    return (base_seed + int(hashlib.sha256(filename.encode()).hexdigest()[:8], 16)) % 2**32


def summarize(root, rows, audit, output, method):
    """Class means include ALL completed images, grouped by actual predicted class."""
    from matplotlib import pyplot as plt
    summaries = []
    for row in rows:
        path = output/Path(row['file']).stem
        if (path/'done.json').is_file():
            metadata = json.loads((path/'done.json').read_text())
            summaries.append(metadata)
    write_csv(output/'summary.csv', summaries)
    for label, name in [(0, 'Low'), (1, 'High')]:
        members = [r for r in rows if audit[r['file']]['predicted_label'] == label
                   and (output/Path(r['file']).stem/'done.json').is_file()]
        if not members:
            continue
        maps = []
        for row in members:
            with np.load(output/Path(row['file']).stem/'maps.npz', allow_pickle=False) as data:
                maps.append(data['positive_mask' if method == 'lime' else 'native_heatmap'])
        mean = np.mean(maps, axis=0)
        np.save(output/f'predicted_{name.lower()}_mean.npy', mean, allow_pickle=False)
        plt.imsave(output/f'predicted_{name.lower()}_mean.png', mean, cmap='viridis', vmin=0, vmax=1)
        write_json(output/f'predicted_{name.lower()}_members.json', {
            'grouped_by': 'predicted class, including misclassified images',
            'image_count': len(members), 'files': [r['file'] for r in members],
            'correct': sum(audit[r['file']]['correct'] for r in members),
            'map_definition': 'positive-mask frequency' if method == 'lime' else 'mean per-image maximum-normalized attention flow'})
    print(f'{len(summaries)}/45 explanations completed; summaries and class mean maps saved.')
    return summaries


def save_figure(rgb, heatmap, title, path, binary=False):
    from matplotlib import pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title('Original validation image')
    axes[1].imshow(heatmap, cmap='viridis', vmin=0, vmax=1)
    axes[1].set_title('Positive mask' if binary else 'Relative attention flow')
    axes[2].imshow(rgb)
    masked = np.ma.masked_where(heatmap <= 0, heatmap)
    axes[2].imshow(masked, cmap='viridis', vmin=0, vmax=1, alpha=0.5)
    axes[2].set_title(title)
    for axis in axes:
        axis.axis('off')
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(figure)
