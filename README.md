# Model training and explainability results — 12–16 October 2026

Initial snapshot: 2026-10-16T07:41:56.754573+00:00

Latest ResNet152 result update: 2026-10-19T00:44:52.831690+00:00

This repository records two separate experiments. Their datasets, labels, checkpoints, and accuracy measurements must be kept distinct.

## RDN training: nine models from scratch

The SUTD-PRCM compact RDN128 release contains 111,000 layouts: **97,200 training / 10,800 validation / 3,000 held-out test**. Class 0 is `argmax(compact_response) < 83`; class 1 is `argmax(compact_response) >= 83`. This compact-response target is not an exact reproduction of the source paper's raw-reflection 8.3 GHz classification.

All nine active models trained all layers from random initialization. Checkpoints were selected by minimum validation loss. The values below evaluate the selected checkpoint. DINOv2 was excluded from this experiment.

| Model | Epochs | Selected epoch | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| CNN | 30 | 10 | 58.11% | 55.40% | **55.73%** |
| ViT-Tiny | 20 | 2 | 51.26% | 51.26% | **51.57%** |
| ViT-Small | 20 | 20 | 52.62% | 52.06% | **51.73%** |
| ViT-Base | 20 | 18 | 52.89% | 51.52% | **51.93%** |
| ViT-Large | 5 | 5 | 51.26% | 51.26% | **51.57%** |
| ResNet50 | 20 | 11 | 69.95% | 66.57% | **68.07%** |
| ResNet101 | 20 | 11 | 70.91% | 67.19% | **67.93%** |
| ResNet152 | 30 | 11 | 70.45% | 66.83% | **68.90%** |
| VMamba | 5 | 4 | 64.66% | 64.71% | **65.33%** |

CNN and ResNet152 completed 30 total epochs. Five models completed 20 epochs. ViT-Large and VMamba were capped at **five total epochs** to limit computation; their original 20-epoch learning-rate schedule and directory names were retained for checkpoint compatibility. Different epoch budgets limit direct architecture comparisons. These are single-seed results (seed 42), not repeated-run estimates.

Training/validation curves, test confusion matrices, histories, metrics, test predictions, and exact saved run configurations are in [`rdn_training/results`](rdn_training/results). See [`results_summary.csv`](rdn_training/results_summary.csv). Some training-summary files retain `test_evaluated: false` from the moment training ended; subsequent held-out evaluation is recorded in each `test_metrics.json`.

**Completed ResNet152 continuation:** training resumed after epoch 20 and finished 30 total epochs. Epoch 11 remained the validation-selected checkpoint, so its test accuracy remains **68.90%**. Epochs 21–30 retained the original schedule at its minimum LR of `3e-6`; see [continuation details](rdn_training/RESNET152_EXTENSION.md). The `epochs_20` directory name is retained for checkpoint compatibility.

## Original bandwidth explainability: four previously trained models

This is the earlier **176 training / 45 validation / no independent test** experiment, with Low/High bandwidth labels. Its pretrained DINOv2 and VMamba classifiers are part of this experiment even though DINOv2 is excluded from the RDN experiment. Explanation generation does not retrain any model.

All four GPU prediction audits reproduce the archived class predictions: CNN 44/45, ViT-Tiny 43/45, DINOv2 43/45, and VMamba 44/45. These are checkpoint-selected validation accuracies, not held-out test scores.

| Model | Method | Images complete | Status |
|---|---|---:|---|
| cnn | lime | 45/45 | complete |
| dinov2 | attention_rollout | 45/45 | complete |
| dinov2 | gradient_attention_rollout | 45/45 | complete |
| dinov2 | lime | 45/45 | complete |
| vit_tiny | attention_rollout | 45/45 | complete |
| vit_tiny | gradient_attention_rollout | 45/45 | complete |
| vit_tiny | lime | 45/45 | complete |
| vmamba | lime | 45/45 | complete |

All eight explanation runs are complete, with 45 validation images per run (360 image-method results). This includes LIME for all four models, plus attention rollout and gradient attention rollout for ViT-Tiny and DINOv2. Each run includes its per-image outputs, summary, prediction audit, and class mean maps. Gradient rollout includes separate Low- and High-class targets.

LIME uses 1,000 perturbations per image, native-resolution SLIC regions, white masking, and the predicted class as its target. Outputs include figures, signed coefficients, raw maps, per-image metadata, prediction audits, and available class means. A completed image is included only when its `done.json` marker exists. See [`output_status.json`](bandwidth_explainability/output_status.json) and [`outputs`](bandwidth_explainability/outputs).

The supplied LIME code was corrected to remove a `progress_bar` argument unsupported by LIME 0.2.0.1. The original integrity manifest and the documented local change are retained under `bandwidth_explainability/reference/`.

## Contents and reproducibility

- `rdn_training/`: training notebooks, shared model/runtime code, configurations, numerical results and plots.
- `bandwidth_explainability/`: explanation notebooks/code and the currently completed explanation artifacts.
- `UPLOAD_MANIFEST.json`: paths, byte sizes, and SHA-256 hashes of uploaded artifacts.

Original datasets, model checkpoint weight files, Python environments, caches, server authentication files, and runtime connection files are not part of this results upload. For RDN, full training/validation prediction CSVs remain in the local experiment; test prediction CSVs are included here. Running notebooks elsewhere requires obtaining the original data and checkpoint files and following the relevant setup instructions. Some retained notebook outputs include local paths and platform information. Setup documents describe the original local environment and may reference files excluded from this upload.

LIME fidelity scores and the existing mean-brightness diagnostic should be considered when interpreting the bandwidth maps. Spatial explanations alone do not establish a physical mechanism or causality. Attention rollout is not class-specific; gradient rollout targets class logits.

## Sources

- [SUTD-PRCM dataset](https://github.com/veya2ztn/SUTD_PRCM_dataset) and [paper](https://arxiv.org/abs/2203.00002).
- [Original photonics explainability paper](https://doi.org/10.1002/adpr.202500284).
- [LIME](https://github.com/marcotcr/lime), [attention rollout](https://arxiv.org/abs/2005.00928), [gradient rollout implementation](https://github.com/jacobgil/vit-explain).
- [VMamba](https://github.com/MzeroMiko/VMamba), whose vendored MIT license is retained in each experiment's `licenses/` folder.
