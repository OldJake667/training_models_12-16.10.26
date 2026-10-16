"""Paper-style white masking / SLIC LIME, extended to the four saved models."""
from pathlib import Path
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.segmentation import slic, mark_boundaries
from .runtime import read_rgb, image_seed, write_json, save_figure, summarize


def explain_one(rgb, predictor, predicted_class, seed, settings):
    # Segment the original native-resolution image ONCE. All models use this
    # same coordinate system; their original resizing happens inside Predictor.
    # This cross-model comparison choice is recorded, rather than described
    # as a pixel-for-pixel reproduction of the authors' CNN explanation images.
    segments = slic(rgb.astype(np.float32)/255.0,
        n_segments=settings['n_segments'], compactness=settings['compactness'],
        enforce_connectivity=True, start_label=0, channel_axis=-1, convert2lab=True)
    ids = np.unique(segments)
    if not np.array_equal(ids, np.arange(len(ids))):
        raise ValueError('LIME requires contiguous segment IDs beginning at zero')
    explainer = lime_image.LimeImageExplainer(
        random_state=seed, kernel_width=settings['kernel_width'], feature_selection='none')
    explanation = explainer.explain_instance(
        rgb, classifier_fn=predictor, labels=(predicted_class,), top_labels=None,
        hide_color=settings['hide_color'], num_samples=settings['num_samples'],
        num_features=len(ids), batch_size=settings['batch_size'],
        distance_metric='cosine', segmentation_fn=lambda image: segments,
        random_seed=seed)
    limit = settings['display_features'] or len(ids)
    _, positive_mask = explanation.get_image_and_mask(
        predicted_class, positive_only=True, hide_rest=False,
        num_features=limit, min_weight=0.0)
    coefficients = np.zeros(len(ids), dtype=np.float32)
    for segment_id, weight in explanation.local_exp[predicted_class]:
        coefficients[int(segment_id)] = float(weight)
    # Released LIME 0.2.0.1 stores score/local_pred as scalars/arrays; newer
    # source stores dictionaries. Both forms are supported for one target class.
    def scalar(value):
        if isinstance(value, dict):
            value = value[predicted_class]
        number = float(np.asarray(value).reshape(-1)[0])
        return number if np.isfinite(number) else None
    details = {'segment_count': len(ids), 'seed': seed,
        'local_weighted_r2': scalar(explanation.score),
        'local_surrogate_prediction': scalar(explanation.local_pred),
        'surrogate_intercept': scalar(explanation.intercept),
        'positive_segment_count': int(sum(weight > 0 for weight in coefficients)),
        'selected_positive_segment_count': int(len(np.unique(segments[positive_mask > 0]))),
        'positive_pixel_fraction': float((positive_mask > 0).mean())}
    return segments, coefficients, positive_mask.astype(np.uint8), details


def run_lime(root, rows, audit, predictor, output, settings):
    from matplotlib import pyplot as plt
    for index, row in enumerate(rows, start=1):
        destination = output/Path(row['file']).stem
        if all((destination/name).is_file() for name in ['done.json', 'maps.npz', 'explanation.png', 'coefficients.json', 'segments.png']):
            print(f"[{index}/45] Already complete: {row['file']}")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        rgb = read_rgb(root, row)
        record = audit[row['file']]
        target = record['predicted_label']
        print(f"[{index}/45] LIME: {row['file']}; predicted {target}; correct={record['correct']}")
        segments, coefficients, mask, details = explain_one(
            rgb, predictor, target, image_seed(row['file'], settings['seed']), settings)
        # Coefficients are in probability units, not a proof of physical causality.
        np.savez_compressed(destination/'maps.npz', segments=segments,
                            coefficients=coefficients, signed_weights=coefficients[segments],
                            positive_mask=mask)
        Image.fromarray(rgb).save(destination/'original.png')
        plt.imsave(destination/'segments.png', mark_boundaries(rgb/255.0, segments))
        write_json(destination/'coefficients.json', {
            'explained_class': target, 'segments': [
                {'segment_id': int(i), 'coefficient': float(weight)}
                for i, weight in enumerate(coefficients)]})
        save_figure(rgb, mask, f"LIME: {'High' if target else 'Low'}; correct={record['correct']}",
                    destination/'explanation.png', binary=True)
        write_json(destination/'done.json', {**record, **details, 'method': 'lime',
                    'explained_class': target, 'native_width': rgb.shape[1], 'native_height': rgb.shape[0]})
    return summarize(root, rows, audit, output, method='lime')
