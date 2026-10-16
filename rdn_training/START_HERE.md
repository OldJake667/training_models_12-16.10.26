# Full RDN notebooks for the lab GPU

These are new runnable notebooks, not the previous result-review notebooks.
They are supplied with empty outputs. No model training was performed while
preparing this package.

Copy or extract the **entire folder** to the lab computer. It contains all
111,000 layouts, the fixed full split, both pretrained backbones, supporting
Python files, and ten model notebooks. No paths refer to the original Mac.

## Environment

Native Linux with Ubuntu LTS is recommended. Windows with WSL2 + Ubuntu is
also suitable, especially for VMamba. The other nine notebooks can use native
Windows with a compatible PyTorch GPU environment. Run one model at a time.

Use Python 3.11 or 3.12 in a separate environment. On Linux/WSL2:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sutd-rdn --display-name "SUTD RDN CUDA"
python -m jupyter lab
```

The torch/torchvision pair is a proposed reproducible CUDA 13.0 environment,
not a configuration tested on the lab machine. Verify availability against
[PyTorch's installation selector](https://pytorch.org/get-started/locally/)
before installing. For PyTorch 2.12 CUDA 13.0, the stated minimum NVIDIA driver
is 580.65.06 on Linux or 580.88 on Windows; use a compatible current driver.
[PyTorch release guidance](https://pytorch.org/blog/pytorch-2-12-release-blog/).

For native Windows, use `py -3.12 -m venv .venv`, then
`.venv\Scripts\Activate.ps1`; the remaining `python -m ...` commands are the same.
Under WSL2, use the Windows NVIDIA driver, not a separate Linux display driver.
Keep the package inside WSL's Linux filesystem.
[NVIDIA WSL instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

Open `00_environment_and_data_check.ipynb` first and select the **SUTD RDN CUDA**
kernel. It checks the GPU, file integrity, and full split without training.

## VMamba kernel setup

Only VMamba needs this additional step. The supplied architecture and ImageNet
checkpoint are the same as the pilot's `vmamba_tiny_s1l8`. The accelerated scan
comes from `selective_scan_cuda` in Mamba. A wheel compiled for another PyTorch,
CUDA, or GPU generation may fail, so the following source build is proposed
for Linux/WSL2. It has not been executed on the RTX 5090 here.

Install a CUDA 13.0 developer toolkit with `nvcc`, a supported C++ compiler,
and Git first. PyTorch's wheel supplies its runtime, not the CUDA compiler.
With the notebook environment activated:

```bash
git clone --branch v2.2.6.post3 --depth 1 https://github.com/state-spaces/mamba.git external_mamba
MAMBA_FORCE_BUILD=TRUE MAX_JOBS=2 python -m pip install --no-build-isolation --no-deps ./external_mamba
python -c "import selective_scan_cuda; print('VMamba scan extension imported')"
```

This Mamba release contains the Mamba-1 kernel used by this VMamba
implementation, and its setup script includes the RTX 5090's `sm_120` target
with CUDA 12.8 or newer. **An import alone does not establish working GPU
execution**. Run the
optional kernel comparison cell in `00_environment_and_data_check.ipynb`.
It compares small synthetic inputs against the reference implementation and
checks backward execution without updating any model weights. If it fails,
resolve the kernel build before starting VMamba. The full-data VMamba notebook
stops if the accelerated scan is absent; it does not silently use the slow
reference implementation. Native Windows VMamba builds are not supported by
this package's installation instructions.

[Mamba kernel build source](https://github.com/state-spaces/mamba/blob/v2.2.6.post3/setup.py),
[VMamba project](https://github.com/MzeroMiko/VMamba).

## Dataset and target

Every model uses **97,200 training / 10,800 validation / 3,000 test** layouts.
The test set is the original released test set. The 90/10 division of the
released 108,000 training rows is our validation adaptation, not a prescribed
split from the paper. The source paper has inconsistent test counts; the
actual compact release contains 111,000 layouts.

These notebooks continue the pilot's **compact-release task**:

```python
peak_index = numpy.argmax(compact_response)  # first maximum; zero-based index
label = int(peak_index >= 83)
```

This predicts peak-index side, not bandwidth. The author's compact B1N response
is processed using `1 - abs(first complex branch)`, rounding and downsampling.
It is not an exact reproduction of the paper's raw reflection-maximum task
with an 8.3 GHz threshold. Do not equate index 83 with 8.3 GHz. If exact paper
task replication is required, obtain the raw spectra and prepare a separate,
verified target and manifest before training. The 39 released test labels that
disagree with the fixed cutoff have been documented in `data/dataset_summary.json`;
the same fixed cutoff is used in all three partitions here.

[Dataset repository](https://github.com/veya2ztn/SUTD_PRCM_dataset),
[Paper, Sections 3.3–4.1](https://arxiv.org/html/2203.00002).

## Model notebooks

| Notebook | Default training |
|---|---|
| `01_cnn.ipynb` | Original CNN; all weights; lower diagnosed learning rate |
| `02_vit_tiny.ipynb` | Original grayscale ViT-Tiny; all weights |
| `03_resnet50.ipynb` | ResNet-50 from scratch |
| `04_resnet101.ipynb` | ResNet-101 from scratch |
| `05_resnet152.ipynb` | ResNet-152 from scratch |
| `06_vit_small.ipynb` | ViT-Small/16 from scratch |
| `07_vit_base.ipynb` | ViT-Base/16 from scratch |
| `08_vit_large.ipynb` | ViT-Large/16 from scratch |
| `10_vmamba.ipynb` | VMamba-Tiny from random weights; all layers |

The DINOv2-style ViT-Small/14 experiment is excluded from the active comparison.
Its notebooks are preserved in `archived_notebooks/`, leaving nine active models.

All nine models default to a maximum of **20 epochs**. New automatically named
run folders include `_epochs_20`, keeping earlier runs separate. Completed
results and their saved training configurations are retained.

Exception: ViT-Large and VMamba have a **5-total-epoch stopping cap** in
`EPOCH_LIMITS.json` and their notebook configuration cells. The original
20-epoch configuration and learning-rate schedule are preserved for checkpoint
compatibility; `STOP_AFTER_EPOCH=5` limits the loop. The cap counts saved epochs,
so ViT-Large resumes from epoch 3 and runs only epochs 4 and 5. The original
folder names remain unchanged, and final summaries record the five-epoch cap.

The **configuration cell** controls seed, epochs, batch size, accumulation,
learning rate, precision, workers, and test evaluation. Recipes are starting
points for the full-data experiment, not previously validated full-data
hyperparameters. All nine scratch models use an effective training batch of 128
by default. Gradient accumulation does not make ResNet BatchNorm statistics
identical across physical batch sizes; physical and effective sizes are logged.
VMamba uses batch 4
with 32 accumulation steps and activation checkpointing to reduce GPU memory.
Their former batch of 512 applied only to linear-head training and is no longer used.

CNN keeps its 256x256 grayscale input. ViT-Tiny keeps 224x224 grayscale input.
The other models receive 224x224 RGB copies of the same grayscale layout.
Only fixed bilinear resizing and normalization are used. There are no random
crops, flips, rotations, or erasures. Upscaling adds no physical information.

Every active notebook trains all layers from random weights. No model loads a
pretrained checkpoint or uses feature caching. The old pretrained files are
retained for reference and are not used by these scratch notebooks. New VMamba
run folders use `from_scratch` to distinguish them from any prior pretrained runs.

## Running and results

1. Check the target description above, install dependencies, and run the environment notebook.
2. Open one model notebook. Review its configuration cell and use **Run All**.
3. The model/data preparation and training launch cells are executable. All
   epoch, validation, and checkpoint-selection code is visible in the notebook.
4. Run one seed at a time. Change `seed` to 43 or 44 for additional runs; the
   automatically generated folder name changes with the seed and training mode.
5. `RUN_FINAL_TEST=True` enables evaluation of the validation-selected checkpoint
   on the held-out test set after training. To evaluate an already completed run,
   set `RUN_TRAINING=False` and use **Run All** with the same configuration and
   `RUN_NAME`. This loads the saved checkpoint and runs no training epochs.

Results go to `results/<model>/<run_name>/`. Each completed run saves:

- `run_config.json`, environment and total/trainable parameter counts;
- `last_checkpoint.pt` with optimizer/scheduler/scaler/RNG state for epoch-boundary resume;
- `best_checkpoint.pt` selected solely by minimum validation loss;
- `history.csv` and `history.json`;
- training/validation accuracy and loss plots in PNG and PDF;
- training and validation predictions/metrics at the same selected checkpoint;
- final test metrics, predictions, and confusion matrix only when explicitly requested.

Online training curves include dropout and, where configured, label smoothing.
Final training and validation metrics use evaluation mode and unsmoothed loss.
The graph labels distinguish these measurements. Test data never enters
training, early stopping, feature normalization, or checkpoint selection.

Interrupted runs resume automatically from the last complete epoch. A partial
epoch is repeated. Frozen feature extraction also resumes. Configuration
changes require a new run name; the code refuses to overwrite a run with a
different configuration. A completed run loads its selected checkpoint instead
of retraining. Repeating the final-test cell reuses its saved measurement.

Default `num_workers=0` is notebook-friendly on both operating systems. On Linux,
increase to 2–4 after checking RAM and throughput. If memory runs out, restart
the kernel, reduce physical batch size, and increase accumulation to preserve
the intended effective batch. Use a new run name for changed settings.

Do not launch two notebooks writing the same run or feature-cache folder.
Set a different run name for recipe changes, not just for different model names.

## Verification limits

`VERIFICATION.json` records the checks actually performed during preparation.
They cover notebook structure/syntax, full-data integrity and partition mapping,
model architecture/state-dict compatibility, and checkpoint serialization.
There are no invented accuracy outputs. CUDA execution, VMamba kernel
compatibility, throughput and training convergence must be checked on the lab
computer. No full-data or subset training was run to prepare these notebooks.
