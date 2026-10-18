# ResNet152: continuation from 20 to 30 total epochs

The scratch ResNet152 notebook is prepared to resume its saved run at epoch 21
and stop at 30 total epochs. Training has not been started during preparation.

- Open `ready_notebooks/05_resnet152.ipynb` in the original working project.
- Close and reopen the file to load the updated configuration, then restart its
  kernel and select Run All. Run one model at a time.
- `STOP_AFTER_EPOCH=30` controls the actual loop. Keep `CONFIG['max_epochs']=20`,
  `resume=True`, and `RUN_NAME=None` unchanged for checkpoint compatibility.
- The existing optimizer, scaler, RNG state, epoch history, and minimum-validation-loss
  selection resume from `last_checkpoint.pt`. The selected checkpoint can remain
  from the first 20 epochs if later validation loss does not improve.
- The original 20-epoch cosine schedule has reached its minimum learning rate,
  **3e-6**. Epochs 21–30 continue at that rate. This is not a new 30-epoch cosine
  schedule or an ImageNet-pretrained experiment.
- The directory remains `results/resnet152/seed_42_all_layers_epochs_20` to retain
  checkpoint identity. Final execution records and metrics report the actual
  completed epochs and the 30-epoch limit.

The full original 20-epoch run, including both checkpoints, was copied and
hash-verified under `local_checks/resnet152_before_30_epochs_*` on the working
computer. Its completion marker and previous test outputs were archived so the
continuation can train and then evaluate its selected checkpoint. Original
checkpoints remain local and are not uploaded to GitHub.

The published result table and `results_summary.csv` still describe the
completed 20-epoch run. No 30-epoch results exist until the continuation finishes.
