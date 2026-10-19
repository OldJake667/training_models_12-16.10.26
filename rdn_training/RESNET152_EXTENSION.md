# ResNet152: completed continuation to 30 epochs

The scratch ResNet152 run completed 30 total epochs. Epoch 11 remained the
minimum-validation-loss checkpoint; train accuracy is 70.45%, validation
accuracy 66.83%, and held-out test accuracy 68.90% on 3,000 samples.

The run resumed after epoch 20 using the original optimizer, scheduler, scaler,
RNG state, and checkpoint identity. Epochs 21–30 used the original scheduler's
minimum learning rate, **3e-6**. This was a continuation of the 20-epoch schedule,
rather than a newly initialized 30-epoch schedule.

The directory remains `results/resnet152/seed_42_all_layers_epochs_20` for
checkpoint compatibility. History, completion metadata, metrics, plots, and
`results_summary.csv` now record the actual 30 completed epochs. The original
20-epoch run and checkpoint files remain backed up locally; weights are not
uploaded.

The notebook exposes `STOP_AFTER_EPOCH=30` while preserving
`CONFIG['max_epochs']=20`, `resume=True`, and `RUN_NAME=None`. A completed run
loads the selected checkpoint. `RESNET152_EXTENSION_VERIFICATION.json` records
the earlier synthetic preparation check; `RESNET152_COMPLETION.json` records
the completed research run's publication audit.
