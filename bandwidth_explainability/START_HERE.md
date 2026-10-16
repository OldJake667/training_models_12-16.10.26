# Original bandwidth explainability — lab package

Use this package to explain the **original 45 validation images** using the
already trained classifiers. It does not retrain any model. It is separate
from the full SUTD RDN training package and its resonance-index classification.
Here the two classes remain **0 = Low bandwidth, 1 = High bandwidth**.

Copy/extract the entire folder on the lab computer, including `data`,
`checkpoints`, `reference`, and `xai_code`. No model download is needed after
the Python environment and VMamba CUDA extension have been installed.

## Setup

Linux is recommended for all four models; Windows with WSL2 is another option.
CNN, ViT-Tiny and DINOv2 can also use native Windows. Use Python 3.11 in a fresh
environment. In a terminal opened inside this folder on Linux/WSL2:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m ipykernel install --user --name bandwidth-xai --display-name "Bandwidth XAI CUDA"
python -m jupyter lab
```

On native Windows use `py -3.11 -m venv .venv` then
`.venv\Scripts\Activate.ps1` in PowerShell; subsequent `python` commands are the
same. Select **Bandwidth XAI CUDA** in Jupyter. Verify the NVIDIA driver supports
the chosen CUDA runtime and that PyTorch detects the RTX 5090 before starting.
The pinned CUDA pair follows [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/).
The RTX 5090 environment itself has not been tested on this Mac.

### Additional setup for VMamba

Only VMamba needs `selective_scan_cuda`. On Linux/WSL2, install the CUDA 13.0
developer toolkit (`nvcc`), a supported C++ compiler and Git. PyTorch's wheel
provides a runtime, not the CUDA compiler. Activate the same environment:

```bash
git clone --branch v2.2.6.post3 --depth 1 https://github.com/state-spaces/mamba.git external_mamba
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps ./external_mamba
python -c "import selective_scan_cuda; print('VMamba scan extension imported')"
```

This source contains the Mamba-1 scan used by the supplied VMamba model and an
`sm_120` build target for CUDA >=12.8. See the [upstream build script](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/setup.py).
A successful import alone does not verify GPU execution: run the synthetic
kernel comparison in notebook 00, then the VMamba notebook's prediction audit.
VMamba LIME stops if CUDA or the fast extension is unavailable, avoiding a
silently slow reference scan. The supplied instructions do not cover native
Windows extension compilation.

## Notebooks to run

| Notebook | What it runs |
| --- | --- |
| `00_environment_and_checkpoint_check.ipynb` | File integrity, software/GPU checks, optional VMamba kernel check |
| `01_cnn_lime.ipynb` | LIME on all 45 images using the original CNN |
| `02_vit_tiny_lime.ipynb` | LIME using the original ViT-Tiny |
| `03_dinov2_lime.ipynb` | LIME using the original trained DINOv2-S/14 classifier |
| `04_vmamba_lime.ipynb` | LIME using the original VMamba-T backbone and trained head |
| `05_vit_tiny_attention_rollout.ipynb` | Attention rollout using the original ViT-Tiny |
| `06_dinov2_attention_rollout.ipynb` | Attention rollout using the original DINOv2 classifier |
| `07_vit_tiny_gradient_attention_rollout.ipynb` | Class-specific gradient rollout: Low and High maps for ViT-Tiny |
| `08_dinov2_gradient_attention_rollout.ipynb` | Class-specific gradient rollout: Low and High maps for DINOv2 |

Run notebook 00 first, then one model notebook at a time, top to bottom.
The actual LIME/rollout functions appear as executable code cells in each
notebook. Model restoration and preprocessing are in `xai_code/models.py`;
the original CNN/ViT architecture definitions are in `xai_code/scratch_models.py`.
All notebook execution counts and outputs are initially empty. Save any edited
notebook before starting a run so its source hash records those edits.

Each explanation notebook first loads its fixed checkpoint and predicts all
45 images. It writes `prediction_audit.csv` and compares filenames, labels and
predictions to the archived results before explaining anything. A changed
class prediction stops the notebook by default. Probability differences are
also reported; larger changes should be inspected even if the class agrees.
`ALLOW_PREDICTION_DIFFERENCES` is an explicit, recorded override for a deliberate
new-environment analysis, not a way to retain the old accuracy claim after
predictions change. Keep it `False` for the first run.

The saved best-validation-accuracy checkpoints correspond to:

| Model | Archived correct / validation images |
| --- | --- |
| CNN | 44 / 45 (97.78%) |
| ViT-Tiny | 43 / 45 (95.56%) |
| DINOv2 | 43 / 45 (95.56%) |
| VMamba | 44 / 45 (97.78%) |

These checkpoints were selected using the same validation set. The original
split was **176 training / 45 validation / 0 test**. Do not call these 45 images
an independent test set. Both correct and incorrect predictions are explained;
the two models with 43/45 do not have exactly the same errors.

## LIME method and paper comparison

The source paper is [Advancing Photonic Inverse Design with Interpretable Machine Learning](https://advanced.onlinelibrary.wiley.com/doi/10.1002/adpr.202500284),
especially sections 4.4 and 5.1. It applies LIME to its CNN, replicates grayscale
into three channels, uses SLIC with approximately 100 segments, compactness 10
and connectivity enforcement, generates 1,000 perturbations, replaces hidden
regions with white (255), and shows positive evidence for the predicted class.
Its class summary maps group explanations by predicted class.

This package follows those reported settings and extends LIME to our four
classifiers. Important explicit comparison choices:

- SLIC operates on the original native-resolution image, with identical
  segmentation and perturbation seeds across models for a given filename.
  Each perturbed image is then resized and normalized using that model's
  original validation preprocessing (CNN 256; other models 224).
  This common spatial coordinate system is a cross-model comparison choice;
  it is not a promise to reproduce the author's resized explanation pixels.
- The callback recomputes the full backbone and head for every perturbation,
  including DINOv2 and VMamba. Reusing original cached features would invalidate
  explanations of image regions.
- The surrogate fits all SLIC segments (`feature_selection='none'`), with
  LIME's cosine distance, kernel width 0.25 and default Ridge regressor. These
  supplementary surrogate settings are recorded choices where the paper's
  reported description does not fully specify them.
- By default every segment with a positive coefficient is shown. Set
  `display_features` to a positive integer to show only that many top positive
  segments. The output still preserves every signed coefficient.
- The explained target is the predicted class, including for misclassified
  images. Local weighted R² and the surrogate's unperturbed prediction are
  saved so you can assess how well the local linear model fits the perturbed
  model probabilities. Low fidelity or an empty positive mask should not be
  described as a persuasive explanation.
- The default per-image seed derives from its filename and base seed 42.
  Changing the base seed produces a separate output directory for a stability
  comparison. Do not choose seeds by how attractive a heatmap looks.

White masking can change global brightness as well as geometry. A train-fitted
mean-brightness rule already classified 43/45 validation images correctly in
our earlier diagnostic (included in `reference`). Thus spatial maps alone do
not establish that a network learned the intended physical mechanism.
The original paper used additional electromagnetic simulations to examine
physical interpretations. This package runs classifier explanations only.

## Attention rollout

Attention rollout is an additional transformer analysis, not the paper's LIME
method. Following [Abnar and Zuidema, 2020](https://arxiv.org/abs/2005.00928),
we average attention across heads, add the residual identity, normalize rows,
multiply across all 12 layers, and extract CLS-to-patch flow. No attention
values are discarded. Patch grids are 14×14 for ViT-Tiny and 16×16 for DINOv2.
Capture hooks are removed afterward, and captured-forward probabilities are
checked against the unchanged model's probabilities.

Standard rollout is **not class-specific**: it does not measure how a patch
raises the High or Low score, and does not incorporate head weights or gradient
attribution. The heatmaps show relative attention flow, normalized by each
image's maximum. Their magnitudes should not be compared as absolute class
evidence. Ordinary transformer rollout is not supplied for CNN or VMamba's
state-space blocks.

### Gradient attention rollout

Notebooks 07 and 08 add a class-specific variant for both transformers. Each
image receives **two maps**, one for the Low logit and one for the High logit,
regardless of which class was predicted. This makes it possible to inspect
both targets on the same misclassified image. The targets are the pre-softmax
class scores, not probabilities or the High-minus-Low decision margin.

The variant follows the [gradient-weighted rollout implementation](https://github.com/jacobgil/vit-explain/blob/main/vit_grad_rollout.py):
at each layer multiply attention by its gradient with respect to the chosen
class logit, average heads, clip negative weights, add the residual identity,
normalize rows, and multiply across layers. This package discards no small
weights (`discard_ratio=0`). It does not implement the full relevance
propagation method of Chefer et al.

Gradient computation does not retrain the model. All original weights remain
frozen and unchanged; only the input and attention computation need gradients.
Do not wrap gradient-rollout cells in `torch.inference_mode()`. For the custom
ViT-Tiny, the adapter temporarily replaces the attention forward with equivalent
explicit Q/K/V operations so the captured probability tensor is actually used
to compute the output. Simply hooking the weights returned by
`nn.MultiheadAttention` can capture a tensor outside the score's gradient path.
For DINOv2, fused attention is temporarily disabled and the actual attention
tensor is hooked. Both paths verify prediction agreement and restore the
original forward methods, fused settings and hooks afterward.

Each target heatmap is separately maximum-normalized for visibility; colors
cannot be interpreted as comparable amounts of High versus Low evidence.
Raw patch flow, positive gradient-weighted layer matrices, target logits,
maximum gradient magnitudes and empty-positive-map flags are saved. Negative
contributions are excluded by this variant. A map is a gradient-based account
of a selected score, not an electromagnetic simulation or proof of causality.

## Outputs and resuming

Results go into `outputs/<model>/<method>/<configuration_id>/`. The directory
identifier includes settings, checkpoint/data/code hashes, software versions
and device information. Re-running the same configuration skips complete
images and resumes incomplete images.

Each image gets its original image, a three-panel explanation figure,
machine-readable raw maps (`maps.npz`), and metadata. LIME also saves SLIC
boundaries and all signed coefficients. Rollout saves the raw patch flow and
per-layer head-averaged attention matrices. Each run saves prediction audits,
per-image summary CSV and class mean maps grouped by predicted class, including
errors. LIME means are mask-selection frequencies; rollout means average
per-image normalized attention flow. Member lists and image counts are saved
alongside the means. They are descriptive summaries, not physical validations.

Gradient-rollout runs also save separate Low/High heatmap PNGs and a combined
original/Low/High comparison figure for every image. Class means are saved for
each target, grouped separately by predicted class and true class, including
errors. Result directories use the method name `gradient_attention_rollout`,
separate from standard rollout and LIME. Existing standard-rollout notebooks
remain available for comparison.

Start with the default 1,000 LIME samples. If GPU memory runs out, reduce
`BATCH_SIZE`; do not change the model or preprocessing to make the explanations
fit. LIME performs about 45,000 model forwards per model across its 45 images;
the CPU reference VMamba path is unsuitable for that workload. Timing depends
on the lab computer and compiled kernel.

## Included provenance and verification

Only the 45 validation BMPs are copied; the full original 221-row split manifest
is included for split provenance, without old absolute computer paths.
Checkpoints are copies of the existing original-data best-accuracy outputs.
VMamba also includes its required ImageNet backbone checkpoint, since its
saved trained checkpoint contains only the two-class head. DINOv2's trained
checkpoint already contains its entire backbone and head. SHA-256 hashes cover
all supplied code, notebooks, images and checkpoints. Runtime integrity checks
enforce fixed-file hashes but allow notebooks to be edited and saved normally;
current notebook code-cell source hashes are recorded in each run configuration.

`VERIFICATION.md` describes checks performed during preparation. No model was
retrained and no real-image explanation was run while preparing this package.
The full CUDA/LIME execution remains to be performed on the lab computer.

VMamba's vendored implementation and license come from the earlier verified
project copy; see `licenses/VMamba_LICENSE.txt` and the [official VMamba repository](https://github.com/MzeroMiko/VMamba).
The paper links its author code at [lime_multiplexer](https://github.com/danielpeace/lime_multiplexer).
The packaged notebooks are new adapters to our existing trained models, not
a claim that previously completed explanation figures were found or recovered.
