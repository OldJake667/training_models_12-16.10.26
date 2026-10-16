"""Regenerate clean notebooks from support code; does not execute any notebook."""
import ast
import hashlib
import json
from pathlib import Path
from pprint import pformat

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names):
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    nodes = {n.name: n for n in ast.parse(text).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
    chunks = []
    for name in names:
        node = nodes[name]
        start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
        chunks.append("\n".join(lines[start:node.end_lineno]))
    return "\n\n\n".join(chunks)


IMPORTS = '''from __future__ import annotations
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import timm
from tqdm.auto import tqdm
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix, f1_score, roc_auc_score)

# Works when Jupyter starts in the package root or its notebooks directory.
ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / "lab_code/runtime.py").is_file()
             and (p / "data/full_manifest.csv").is_file()), None)
if ROOT is None:
    raise FileNotFoundError("Extract the entire package and start Jupyter inside it.")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lab_code.runtime import (
    CLASS_NAMES, FrozenClassifier, FeatureDataset, load_full_data,
    environment_info, select_device, seed_everything, precision_context,
    make_loader, layout_dataset, build_training_loaders, cache_features,
    sha256_file, atomic_json, atomic_checkpoint, save_predictions, save_history,
    plot_history, capture_rng, restore_rng, config_identity,
)
print("Package:", ROOT)
'''

COMMON = {
    "seed": 42, "max_epochs": 20, "early_stopping_patience": 20,
    "minimum_improvement": 0.0, "optimizer": "AdamW",
    "learning_rate": 3e-4, "weight_decay": 0.05, "label_smoothing": 0.0,
    "warmup_epochs": 5, "minimum_lr_ratio": 0.01, "gradient_clip_norm": 1.0,
    "precision": "bf16", "num_workers": 0, "resume": True,
    "image_size": 224, "channels": 3, "normalization": "half",
    "train_backbone": True, "require_fast_vmamba": True,
    "batch_size": 32, "accumulation_steps": 4, "feature_batch_size": 32,
}

PROFILES = [
    ("01_cnn.ipynb", "CNN", "cnn", "ScratchCNN", "build_cnn",
     dict(image_size=256, channels=1, normalization="none", optimizer="Adam",
          learning_rate=1e-4, weight_decay=0.0, warmup_epochs=0),
     "Original three-convolution CNN with its original classifier. All layers train. "
     "The lower 1e-4 learning rate follows the earlier CNN diagnostics."),
    ("02_vit_tiny.ipynb", "ViT-Tiny", "vit_tiny", "VisionTransformerTiny", "build_vit_tiny",
     dict(channels=1, weight_decay=0.0),
     "Original grayscale ViT-Tiny: patch size 16, embedding size 192, 12 blocks and 3 heads. "
     "All layers train. Weight decay and label smoothing start at zero, informed by the diagnostics; "
     "the full-data recipe still needs validation."),
    ("03_resnet50.ipynb", "ResNet-50", "resnet50", "resnet50", "build_standard_model",
     dict(normalization="imagenet", weight_decay=1e-4),
     "Standard ResNet-50 from scratch, with a new two-class classifier. All layers train."),
    ("04_resnet101.ipynb", "ResNet-101", "resnet101", "resnet101", "build_standard_model",
     dict(normalization="imagenet", weight_decay=1e-4, batch_size=16, accumulation_steps=8),
     "Standard ResNet-101 from scratch. The standard name is ResNet-101, not ResNet-100. All layers train."),
    ("05_resnet152.ipynb", "ResNet-152", "resnet152", "resnet152", "build_standard_model",
     dict(normalization="imagenet", weight_decay=1e-4, batch_size=8, accumulation_steps=16),
     "Standard ResNet-152 from scratch with a two-class output. All layers train."),
    ("06_vit_small.ipynb", "ViT-Small", "vit_small", "vit_small_patch16_224", "build_standard_model",
     {}, "Standard ViT-Small/16 from scratch: embedding size 384, 12 blocks and 6 heads. All layers train."),
    ("07_vit_base.ipynb", "ViT-Base", "vit_base", "vit_base_patch16_224", "build_standard_model",
     dict(batch_size=16, accumulation_steps=8),
     "Standard ViT-Base/16 from scratch: embedding size 768, 12 blocks and 12 heads. All layers train."),
    ("08_vit_large.ipynb", "ViT-Large", "vit_large", "vit_large_patch16_224", "build_standard_model",
     dict(batch_size=8, accumulation_steps=16),
     "Standard ViT-Large/16 from scratch: embedding size 1024, 24 blocks and 16 heads. "
     "About 303 million parameters. The initial physical batch is deliberately small; profile it on the lab GPU."),
    ("10_vmamba.ipynb", "VMamba-Tiny from scratch", "vmamba", "vmamba_tiny_s1l8", "build_vmamba_from_scratch",
     dict(normalization="imagenet", train_backbone=True, batch_size=4,
          accumulation_steps=32, feature_batch_size=4, learning_rate=3e-4,
          weight_decay=0.05, warmup_epochs=5, initialization="random", activation_checkpointing=True),
     "Randomly initialized VMamba-Tiny with a new two-class head. All layers train on RDN labels; "
     "no ImageNet weights or cached features are used. Activation checkpointing reduces training "
     "memory at the cost of recomputing activations. The accelerated scan kernel is required."),
]


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text, role=None):
    cell = nbf.v4.new_code_cell(text.strip() + "\n")
    if role:
        cell.metadata["lab_role"] = role
    return cell


def limited_fit_source(runtime, model_key):
    source = definitions(runtime, ["fit"])
    limits = json.loads((ROOT / "EPOCH_LIMITS.json").read_text()) if (ROOT / "EPOCH_LIMITS.json").exists() else {}
    if model_key not in limits:
        return source
    source = source.replace(
        '    for epoch in range(start, config["max_epochs"]):',
        '    epoch_limit = min(config["max_epochs"], STOP_AFTER_EPOCH)\n'
        '    if epoch_limit <= 0 or start > epoch_limit:\n'
        '        raise ValueError("Stopping limit must be positive and cannot precede saved epochs")\n'
        '    atomic_json(folder / "execution_limit.json", {"stop_after_epoch": epoch_limit,\n'
        '                "original_epoch_budget": config["max_epochs"], "resume_schedule_preserved": True})\n'
        '    print(f"Training stops at {epoch_limit} total epochs; resuming after epoch {start}.")\n'
        '    for epoch in range(start, epoch_limit):')
    source = source.replace(
        '"epochs_completed": len(history), "model_info": model_info,',
        '"epochs_completed": len(history), "epoch_limit": epoch_limit, "model_info": model_info,')
    source = source.replace(
        '"epochs_completed": len(history), "test_evaluated": False}',
        '"epochs_completed": len(history), "epoch_limit": epoch_limit, "test_evaluated": False}')
    return source


def write_notebook(path, cells, model=None):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update(
        kernelspec={"display_name": "SUTD RDN CUDA", "language": "python", "name": "sutd-rdn"},
        language_info={"name": "python"},
        lab={"model": model, "training_executed": False, "dataset": "full RDN128 compact release",
             "split": {"train": 97200, "validation": 10800, "test": 3000}},
    )
    nbf.validate(notebook)
    nbf.write(notebook, path)


def main(models=None):
    folder = ROOT / "notebooks"
    folder.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / "lab_code/runtime.py"
    builders = ROOT / "lab_code/model_builders.py"
    scratch = ROOT / "lab_code/scratch_models.py"
    profiles = {}
    for filename, title, model_key, architecture, builder, overrides, description in PROFILES:
        config = {**COMMON, "model": model_key, "architecture": architecture, **overrides}
        profiles[model_key] = config
        if models is not None and model_key not in models:
            continue
        model_code = ""
        if model_key == "cnn":
            model_code += definitions(scratch, ["ScratchCNN"]) + "\n\n"
        if model_key == "vit_tiny":
            model_code += definitions(scratch, ["PatchEmbedding", "MLP", "TransformerBlock", "VisionTransformerTiny"]) + "\n\n"
        definition_file = builders
        if model_key in ("dinov2", "vmamba"):
            definition_file = ROOT / "lab_code/scratch_backbones.py"
            model_code += definitions(definition_file, ["ScratchBackboneClassifier"]) + "\n\n"
        model_code += definitions(definition_file, [builder]) + "\n\n"
        model_code += f"model = {builder}(ROOT, CONFIG)\n"
        model_code += '''model_info = {
    "architecture": CONFIG["architecture"],
    "total_parameters": sum(p.numel() for p in model.parameters()),
    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    "training_mode": "from_scratch",
    "physical_batch_size": CONFIG["batch_size"],
    "effective_batch_size": CONFIG["batch_size"] * CONFIG["accumulation_steps"],
    "input_size": CONFIG["image_size"], "input_channels": CONFIG["channels"],
    "pretrained": False,
}
identity["support_code_sha256"] = {
    name: sha256_file(ROOT / "lab_code" / name)
    for name in ("runtime.py", "scratch_models.py", "model_builders.py")
}
if CONFIG["model"] in ("dinov2", "vmamba"):
    identity["support_code_sha256"]["scratch_backbones.py"] = sha256_file(ROOT / "lab_code/scratch_backbones.py")
if CONFIG["model"] == "vmamba":
    identity["vmamba_source_sha256"] = sha256_file(ROOT / "lab_code/vmamba_official.py")
print(json.dumps(model_info, indent=2))
'''
        config_code = "CONFIG = " + pformat(config, sort_dicts=False, width=95) + '''

# For additional seeds, change CONFIG['seed'] to 43 or 44 before running later cells.
# For a changed recipe, set a distinct RUN_NAME (for example 'seed_42_lr_check').
RUN_NAME = None
REQUIRE_CUDA = True
RUN_TRAINING = True
RUN_FINAL_TEST = True  # Evaluate the validation-selected checkpoint after training.
'''
        limits = json.loads((ROOT / "EPOCH_LIMITS.json").read_text()) if (ROOT / "EPOCH_LIMITS.json").exists() else {}
        if model_key in limits:
            config_code += ("\n# Total epoch cap, including epochs already saved. Preserve CONFIG for checkpoint resume.\n"
                            f"STOP_AFTER_EPOCH = {limits[model_key]}\n")
        prepare = '''if CONFIG["batch_size"] <= 0 or CONFIG["accumulation_steps"] <= 0:
    raise ValueError("Batch size and accumulation steps must be positive")
if CONFIG["max_epochs"] <= 0 or CONFIG["early_stopping_patience"] <= 0:
    raise ValueError("Epoch budget and patience must be positive")
if CONFIG["num_workers"] < 0 or not 0 <= CONFIG["warmup_epochs"] < CONFIG["max_epochs"]:
    raise ValueError("Invalid worker count or warmup length")
device = select_device(require_cuda=REQUIRE_CUDA)
seed_everything(CONFIG["seed"])
partitions, identity = load_full_data(ROOT)
mode = "from_scratch" if CONFIG.get("initialization") == "random" else "all_layers"
run_name = RUN_NAME or f"seed_{CONFIG['seed']}_{mode}_epochs_{CONFIG['max_epochs']}"
RUN_DIR = ROOT / "results" / CONFIG["model"] / run_name
print(json.dumps(environment_info(device), indent=2))
print("Full split:", {k: len(v) for k, v in partitions.items()})
print("Results:", RUN_DIR)
'''
        launch = '''# This is the actual training launch. It runs the FULL training/validation split.
# The test partition is not loaded into any epoch loop.
if RUN_TRAINING:
    existing_config = RUN_DIR / "run_config.json"
    if existing_config.exists() and json.loads(existing_config.read_text()) != config_identity(CONFIG, identity):
        raise ValueError("This folder has different settings/data. Choose a new RUN_NAME before feature extraction.")
    model.to(device)
    training_model, loaders, cached_features = build_training_loaders(
        model, ROOT, partitions, CONFIG, device, identity
    )
    history = fit(training_model, loaders, CONFIG, device, RUN_DIR, identity, model_info)
else:
    print("Training is disabled by RUN_TRAINING; no model was trained.")
'''
        results = '''if (RUN_DIR / "TRAINING_COMPLETE.json").exists():
    summary = json.loads((RUN_DIR / "training_validation_metrics.json").read_text())
    print("Selected epoch:", summary["selected_epoch"])
    for split in ("train", "validation"):
        values = summary[split]
        print(f"{split}: accuracy={values['accuracy']:.2%}, "
              f"balanced accuracy={values['balanced_accuracy']:.2%}, macro F1={values['macro_f1']:.4f}")
    plot_history(RUN_DIR, show=True)
else:
    print("No completed training results yet.")
'''
        test_code = definitions(runtime, ["evaluate_test"]) + '''

if RUN_FINAL_TEST:
    # This cell can also be run after restarting the kernel and executing setup/model
    # cells. It does not require re-executing the training launch cell.
    cached_features = CONFIG["model"] in ("dinov2", "vmamba") and not CONFIG["train_backbone"]
    training_model = model.head if cached_features else model
    metrics = evaluate_test(model, training_model, ROOT, partitions, CONFIG, device,
                            RUN_DIR, identity, cached_features)
    print(json.dumps(metrics, indent=2))
else:
    print("Final test evaluation is disabled. Keep it disabled during tuning.")
'''
        cells = [
            md(f"# {title}: full RDN training\n\n{description}\n\n"
               "**97,200 training / 10,800 validation / 3,000 released test layouts.** "
               "Read `../START_HERE.md` and run the environment notebook first. These are clean executable "
               "training cells with no saved results. Run one notebook at a time."),
            md("The target continues the pilot: class 0 if the maximum compact-response index is below 83, "
               "class 1 otherwise. This is a peak-index classification task, not bandwidth, and it is not "
               "automatically the paper's raw reflection task at 8.3 GHz. The full split is fixed across models. "
               "Validation is our adaptation of the released training set.\n\n"
               "Hyperparameters below are starting points for a new full-data experiment. All nine scratch "
               "models start with effective batch size 128. All nine models use random initialization. "
               "Final test evaluation is enabled after training."),
            md("## Imports and portable paths"), code(IMPORTS, "imports"),
            md("## Configuration\n\nChange settings here before running subsequent cells. "
               "Changes to an existing run need a new `RUN_NAME`. All layers train from random weights; "
               "see the setup guide for the memory settings."), code(config_code, "configuration"),
            md("## Device and full-data checks\n\nOnly native arrays are memory-mapped; "
               "224/256-pixel inputs are prepared per batch. No random spatial augmentation is used."),
            code(prepare, "preparation"),
            md("## Model definition\n\nTotal parameters and trainable parameters are recorded separately."),
            code(model_code, "model_definition"),
            md("## Optimizer and epoch code\n\nThis code is used directly by the training loop. "
               "Accumulation accounts for the actual number of samples in the final batch. "
               "Validation always uses evaluation mode and unsmoothed cross entropy."),
            code(definitions(runtime, ["optimizer_and_scheduler"]), "optimizer_definition"),
            code(definitions(runtime, ["train_one_epoch"]), "training_epoch_definition"),
            code(definitions(runtime, ["evaluate"]), "evaluation_definition"),
            md("## Training and validation loop\n\nChoose the checkpoint solely by minimum validation "
               "loss. Save progress after each epoch and resume from the last complete epoch. "
               "The test set is never used in this function."),
            code(limited_fit_source(runtime, model_key), "fit_definition"),
            md("## Start or resume training\n\nAll models train from images and update every layer. "
               "This cell starts real training on the lab computer."),
            code(launch, "training_launch"),
            md("## Results and accuracy/loss graphs\n\nCurves show online training and evaluation-mode "
               "validation. Final train/validation metrics are evaluated at the same selected checkpoint."),
            code(results, "results"),
            md("## Final held-out test\n\nEnabled by `RUN_FINAL_TEST=True`; runs after training completes. "
               "This evaluates the selected checkpoint on all 3,000 original test layouts once, "
               "and reuses its saved result on repeated execution."), code(test_code, "final_test"),
            md("## References\n\n"
               "[SUTD-PRCM paper](https://arxiv.org/html/2203.00002) · "
               "[Dataset release](https://github.com/veya2ztn/SUTD_PRCM_dataset) · "
               "[timm](https://github.com/huggingface/pytorch-image-models) · "
               "[DINOv2](https://github.com/facebookresearch/dinov2) · "
               "[VMamba](https://github.com/MzeroMiko/VMamba)"),
        ]
        write_notebook(folder / filename, cells, model_key)
    (ROOT / "MODEL_CONFIGS.json").write_text(json.dumps(profiles, indent=2) + "\n")
    if models is None:
        create_preflight(folder)
    print(f"Created {len(PROFILES) if models is None else len(models)} model notebooks; none executed.")


def create_preflight(folder):
    cells = [md("# Environment and full RDN check\n\nRun this on the lab computer before opening "
                "a model notebook. No dataset training or model accuracy evaluation occurs here."),
             code(IMPORTS, "imports"),
             code('''device = select_device(require_cuda=True)
print(json.dumps(environment_info(device), indent=2))
partitions, identity = load_full_data(ROOT)
print("Verified full split:", {k: len(v) for k, v in partitions.items()})
print("Target: first argmax of compact response >= 83")
''', "preparation"),
             md("## Optional VMamba kernel check\n\nAfter installing the scan extension, set "
                "`CHECK_VMAMBA_KERNEL=True`. This checks forward and backward on small synthetic "
                "tensors, including BF16, against the reference scan. It never updates model weights."),
             code('''CHECK_VMAMBA_KERNEL = False
if CHECK_VMAMBA_KERNEL:
    from lab_code import vmamba_official as official
    if not official.WITH_SELECTIVESCAN_MAMBA:
        raise RuntimeError("Install selective_scan_cuda first; see START_HERE.md")
    # Shapes mirror VMamba's grouped B/C convention, with a short sequence.
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(42)
        shape = (1, 8, 16)
        values = [torch.randn(shape, device=device, dtype=dtype) * 0.1,
                  torch.randn(shape, device=device, dtype=dtype) * 0.1,
                  -torch.rand((8, 1), device=device, dtype=torch.float32),
                  torch.randn((1, 4, 1, 16), device=device, dtype=dtype) * 0.1,
                  torch.randn((1, 4, 1, 16), device=device, dtype=dtype) * 0.1,
                  torch.randn(8, device=device, dtype=torch.float32) * 0.1,
                  torch.randn(8, device=device, dtype=torch.float32) * 0.1]
        reference_values = [x.detach().clone().requires_grad_() for x in values]
        kernel_values = [x.detach().clone().requires_grad_() for x in values]
        reference = official.selective_scan_fn(*reference_values, delta_softplus=True,
                                               oflex=True, backend="torch")
        actual = official.selective_scan_fn(*kernel_values, delta_softplus=True,
                                            oflex=True, backend="mamba")
        tolerance = 0.02 if dtype == torch.bfloat16 else 1e-4
        torch.testing.assert_close(actual.float(), reference.float(), rtol=tolerance, atol=tolerance)
        reference.float().square().mean().backward()
        actual.float().square().mean().backward()
        for ref, out in zip(reference_values, kernel_values):
            assert out.grad is not None and torch.isfinite(out.grad).all()
            torch.testing.assert_close(out.grad.float(), ref.grad.float(),
                                       rtol=max(tolerance, 0.02), atol=max(tolerance, 1e-4))
        print("VMamba scan forward/backward passed:", dtype)
else:
    print("Optional VMamba kernel check skipped.")
''', "kernel_check")]
    write_notebook(folder / "00_environment_and_data_check.ipynb", cells)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=[p[2] for p in PROFILES])
    args = parser.parse_args()
    main(set(args.models) if args.models else None)
