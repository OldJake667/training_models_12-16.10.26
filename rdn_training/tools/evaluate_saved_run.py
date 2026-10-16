"""Evaluate a completed run's best checkpoint without invoking training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.path.insert(0, str(ROOT))

from lab_code import model_builders as builders, runtime
from lab_code import scratch_backbones


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_folder", type=Path, help="Completed results/<model>/<run_name> folder")
    args = parser.parse_args()
    folder = args.run_folder
    if not folder.is_absolute():
        folder = ROOT / folder
    if not (folder / "TRAINING_COMPLETE.json").is_file():
        raise RuntimeError(f"No completed training run in {folder}")
    contract = json.loads((folder / "run_config.json").read_text())
    config = contract["config"]
    partitions, identity = runtime.load_full_data(ROOT)
    identity["support_code_sha256"] = {
        name: runtime.sha256_file(ROOT / "lab_code" / name)
        for name in ("runtime.py", "scratch_models.py", "model_builders.py")
    }
    name = config["model"]
    scratch = config.get("initialization") == "random" and name in ("dinov2", "vmamba")
    if scratch:
        identity["support_code_sha256"]["scratch_backbones.py"] = runtime.sha256_file(ROOT / "lab_code/scratch_backbones.py")
    elif name in ("dinov2", "vmamba"):
        weight = "dinov2_small.safetensors" if name == "dinov2" else "vmamba_tiny_imagenet.pth"
        identity["pretrained_sha256"] = runtime.sha256_file(ROOT / "pretrained" / weight)
    if name == "vmamba":
        identity["vmamba_source_sha256"] = runtime.sha256_file(ROOT / "lab_code/vmamba_official.py")
    if runtime.config_identity(config, identity) != contract:
        raise ValueError("Saved run settings/data/support code do not match the current package")
    device = runtime.select_device(require_cuda=True)
    runtime.seed_everything(config["seed"])
    factory = {
        "cnn": builders.build_cnn,
        "vit_tiny": builders.build_vit_tiny,
        "dinov2": builders.build_dinov2,
        "vmamba": builders.build_vmamba,
    }.get(name, builders.build_standard_model)
    if scratch:
        factory = (scratch_backbones.build_dinov2_from_scratch if name == "dinov2"
                   else scratch_backbones.build_vmamba_from_scratch)
    model = factory(ROOT, config)
    cached = name in ("dinov2", "vmamba") and not config["train_backbone"]
    training_model = model.head if cached else model
    print(f"Evaluating saved {name} checkpoint; no training epochs will run.", flush=True)
    metrics = runtime.evaluate_test(model, training_model, ROOT, partitions, config,
                                    device, folder, identity, cached)
    print(f"Test accuracy: {metrics['accuracy']:.2%} ({metrics['records']} samples)")
    print(f"Selected epoch: {metrics['selected_epoch']}")
    print(f"Results: {folder / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
