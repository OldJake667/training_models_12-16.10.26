# Run the notebooks on this laptop

Project folder: `/home/murat/Desktop/training_models`

## Open Jupyter

Open a normal Ubuntu Terminal and run:

```bash
cd ~/Desktop/training_models
./start_jupyter.sh
```

Jupyter opens in your browser. Keep that terminal open while you work. If the browser does not open automatically, use the local link printed in the terminal.

## Start a model

1. In `ready_notebooks`, open `00_environment_and_data_check.ipynb` and choose **Run → Run All Cells**. This verifies CUDA, data, and the VMamba scan kernel without training on the dataset.
2. Open one model notebook, starting with `01_cnn.ipynb`. Check that its kernel is **SUTD RDN CUDA**.
3. Review the configuration cell, then choose **Run → Run All Cells** to start training.
4. Save the notebook with **Ctrl + S**. Results and epoch checkpoints are also saved automatically under `results/<model>/<run_name>/`.
5. Before starting another model, choose **Kernel → Shut Down Kernel** in the previous notebook to release GPU memory. Closing its tab alone may leave its kernel running.

Run one model at a time. Keep the laptop plugged in and awake during training. The original recipes use a 20-epoch scheduler budget; model-specific execution limits are documented below. Early-stopping patience remains 20 epochs without improving validation loss. Interrupted training resumes from the last completed epoch when you run the same notebook again. New run folders include `_epochs_20` to keep them separate from earlier runs.

All nine active model notebooks use `RUN_FINAL_TEST=True`. **Run All** trains a new run, then evaluates its validation-selected checkpoint on the 3,000 held-out test samples. Completed runs automatically load their saved checkpoint instead of repeating training.

ViT-Large and VMamba now stop at **5 total epochs**, controlled by
`STOP_AFTER_EPOCH = 5` in their configuration cells. Their original
`CONFIG['max_epochs']=20`, scheduler, and run folder names are retained to
resume existing checkpoints without changing their identities. ViT-Large
has 3 saved epochs, so it needs only epochs 4 and 5. This is a shorter run of
the original learning-rate schedule. Its final metrics and completion record
will include the actual epoch limit. Close and reopen these two notebook
files and restart their kernels before **Run All** to load the updated loop.

All nine active models start from random weights and train every layer.
The DINOv2-style ViT-Small/14 experiment is excluded; its notebook copies are
retained in `archived_notebooks/`. `10_vmamba.ipynb` trains VMamba-Tiny from
scratch without pretrained files or cached features. Restart its kernel and
reopen the updated notebook before using **Run All**. It uses batch 4 with
32 accumulation steps, giving an effective batch of 128.

## Evaluate the completed CNN without training

The CNN checkpoint is already saved in `results/cnn/seed_42_all_layers`. To evaluate it without executing any training code, run:

```bash
cd ~/Desktop/training_models
.venv/bin/python tools/evaluate_saved_run.py results/cnn/seed_42_all_layers
```

This saves `test_metrics.json`, `test_predictions.csv`, and `test_confusion_matrix.png` in the existing run folder. Repeating the command reuses the saved test result. The same tool accepts other completed model run folders.

The completed CNN retains its original 100-epoch configuration and its 30 completed epochs. Use the command above to evaluate that saved run. Running the updated CNN notebook starts a separate 20-epoch experiment; it does not overwrite the completed CNN.

## Manuals and checks

- [Original setup and experiment manual](START_HERE.md)
- [Pretrained model notes](pretrained/README.md)
- [Local setup status](LOCAL_RUN_STATUS.md)
- GPU memory check results: `local_checks/gpu_batches.json`
- Executed environment and kernel comparison: `local_checks/environment_and_kernel_check.ipynb`
- Installed package versions: `requirements-local-lock.txt`

All ten models passed synthetic GPU checks at their original batch sizes. The `ready_notebooks` copies select the installed CUDA kernel automatically. Both notebook folders now use the 20-epoch limit and enable final test evaluation.

## VMamba build details

The environment uses Python 3.12, PyTorch 2.12.0 with CUDA 13.0, and torchvision 0.27.0 as proposed by the package. The NVIDIA driver is 595.91.07 on an RTX 5050 Laptop GPU with 8 GB of VRAM.

VMamba's source is Mamba `v2.2.6.post3`, commit `10b5d6358f27966f6a40e4bf0baa17a460688128`. Its kernel is built with a separate project-local CUDA 13.2 compiler because CUDA 13.0 headers conflict with this Ubuntu version's system math headers. The build targets `sm_120` with one compilation job. The compiler flag changes are recorded in `MAMBA_BUILD_FLAGS.patch`; the scan kernel sources are unchanged. `tools/build_vmamba.sh` records the build command and the linker alias needed by the compiler's runtime wheel.

## ResNet152 continuation to 30 total epochs

The ResNet152 notebook is prepared to continue its saved 20-epoch run through epoch 30 with `STOP_AFTER_EPOCH=30`. The original configuration, optimizer state, run folder, and scheduler are retained; the learning rate remains at its 3e-6 floor. See `RESNET152_EXTENSION.md` for resume instructions and the archived baseline. The published metrics still describe the completed 20-epoch baseline.
