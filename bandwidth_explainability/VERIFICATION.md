# Preparation checks

Checks passed locally, without training or real-image model inference:

- Nine notebooks pass the notebook schema check. Every code cell and all
  shared Python modules compile; notebook outputs and execution counts are empty.
- All 45 original BMPs retain their source hashes and native size (770 pixels
  wide, 760 high). The portable manifest matches the original validation set.
  The original 176/45 split has no overlapping filenames.
- CNN, ViT-Tiny, DINOv2 and VMamba checkpoints restore with strict state-dict
  loading. VMamba's separately supplied ImageNet backbone also restores strictly.
- Original preprocessing produces the expected channel counts and input sizes;
  this check transforms an image but does not forward it through a classifier.
- Archived prediction files cover all 45 images and contain the expected
  correct counts: CNN 44, ViT-Tiny 43, DINOv2 43, VMamba 44.
- Synthetic attention matrices verify residual normalization and multiplication
  order. Both actual attention-capture paths work on random-weight transformer
  models with synthetic inputs, return the expected patch grids, preserve
  probabilities within tolerance, and remove hooks after use.
- Gradient rollout checks cover zero/negative gradients and finite empty maps.
  Both transformer adapters produce distinct target-specific maps on synthetic
  inputs with nonzero gradients at every attention layer. They preserve all
  parameter values and predictions, accumulate no parameter gradients, and
  restore hooks/forward methods afterward.
- The transfer archive is checked for ZIP corruption and complete file contents.

The local checks used the project's existing macOS environment (PyTorch 2.8.0,
timm 1.0.20). The proposed lab environment uses Python 3.11 and CUDA PyTorch.
LIME/scikit-image were not installed in the local environment, so full LIME
execution was not tested here. No original-image explanation or model training
was run. RTX 5090 execution and VMamba's CUDA kernel remain to be verified on
the lab computer using notebook 00 and each model's prediction audit.

Machine-readable details: `reference/preparation_checks.json`.
To repeat the preparation checks, with existing Python/model dependencies:

```bash
python tools/verify_without_explanations.py
```

This verification script regenerates `reference/preparation_checks.json`.
That report is informational and excluded from runtime fixed-file enforcement,
so repeating the check does not invalidate the package.
