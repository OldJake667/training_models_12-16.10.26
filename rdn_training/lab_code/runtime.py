"""Data loading and training utilities shared verbatim by the ten lab notebooks.

There is no training or downloading at import time. The notebooks contain the
epoch loop, validation loop, fit function, and explicit training launch cell.
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, roc_auc_score)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

CLASS_NAMES = ["PeakBelow83", "PeakAtOrAbove83"]
EXPECTED_COUNTS = {"train": 97200, "validation": 10800, "test": 3000}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def environment_info(device):
    packages = {}
    for name in ("torch", "torchvision", "timm", "numpy", "scikit-learn",
                 "matplotlib", "Pillow", "safetensors", "mamba-ssm", "triton"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_memory_bytes": torch.cuda.get_device_properties(device).total_memory
            if device.type == "cuda" else None}


def select_device(require_cuda=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # Perform a tiny CUDA operation to catch wheels without this GPU's support.
        value = torch.ones(1, device=device) + 1
        if value.item() != 2:
            raise RuntimeError("CUDA check failed")
        return device
    if require_cuda:
        raise RuntimeError("CUDA is unavailable. Select the GPU environment/kernel; "
                           "see START_HERE.md. Training has not started.")
    return torch.device("cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # VMamba custom kernels may not have deterministic implementations.
    torch.use_deterministic_algorithms(True, warn_only=True)


def precision_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unsupported on this GPU; set precision='fp32' or 'fp16'.")
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("precision must be bf16, fp16, or fp32")
    return torch.autocast(device_type="cuda", dtype=dtype)


def load_full_data(root):
    """Validate the exact packaged data, fixed target, and full split; no model evaluation."""
    root = Path(root)
    provenance = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for name in ("data/images.npy", "data/labels.npy", "data/spectra.npy",
                 "data/full_manifest.csv"):
        actual = sha256_file(root / name)
        if actual != provenance["files"][name]["sha256"]:
            raise ValueError(f"File differs from the supplied full release: {name}")
    images = np.load(root / "data/images.npy", mmap_mode="r", allow_pickle=False)
    labels = np.load(root / "data/labels.npy", mmap_mode="r", allow_pickle=False)
    spectra = np.load(root / "data/spectra.npy", mmap_mode="r", allow_pickle=False)
    if images.shape != (111000, 16, 16) or labels.shape != (111000,) or spectra.shape != (111000, 128):
        raise ValueError("Expected the full 111,000-layout RDN release, not the pilot collection")
    if not np.isin(images, [0, 1]).all() or not np.isfinite(spectra).all():
        raise ValueError("Invalid image pixels or response values")
    if not np.array_equal(labels, (spectra.argmax(axis=1) >= 83).astype(np.int64)):
        raise ValueError("Labels must use the fixed compact-response cutoff at index 83")
    with (root / "data/full_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 111000 or [int(r["array_index"]) for r in rows] != list(range(111000)):
        raise ValueError("The full manifest must map every array index exactly once")
    if len({r["sample_id"] for r in rows}) != 111000:
        raise ValueError("Duplicate sample identifiers")
    if len({r["geometry_sha256"] for r in rows}) != 111000:
        raise ValueError("Duplicate geometries require a grouped split")
    partitions = {}
    for split, expected in EXPECTED_COUNTS.items():
        partition = [r for r in rows if r["split"] == split]
        if len(partition) != expected:
            raise ValueError(f"{split}: expected {expected} layouts, got {len(partition)}")
        indices = np.asarray([int(r["array_index"]) for r in partition], dtype=np.int64)
        expected_source = "test" if split == "test" else "train"
        if any(r["source_split"] != expected_source for r in partition):
            raise ValueError("Released test samples cannot enter training or validation")
        if not np.array_equal(labels[indices], np.asarray([int(r["label"]) for r in partition])):
            raise ValueError("Manifest labels disagree with label array")
        if split == "test" and not np.array_equal(indices, np.arange(108000, 111000)):
            raise ValueError("The original 3,000 released test samples must be preserved")
        if split != "test" and (indices >= 108000).any():
            raise ValueError("Test samples found in training/validation")
        partitions[split] = partition
    identity = {name: provenance["files"][name]["sha256"]
                for name in ("data/images.npy", "data/labels.npy", "data/spectra.npy",
                             "data/full_manifest.csv")}
    identity["target"] = "compact_response_argmax_ge_83_first_maximum"
    return partitions, identity


class LayoutDataset(Dataset):
    """Native arrays stay on disk; images are resized only when a sample is loaded."""
    def __init__(self, data_dir, rows, image_size, channels, normalization):
        self.path = str(Path(data_dir) / "images.npy")
        self.rows = rows
        self.images = None
        self.channels = channels
        operations = [transforms.Resize((image_size, image_size),
                      interpolation=transforms.InterpolationMode.BILINEAR), transforms.ToTensor()]
        if normalization == "half":
            operations.append(transforms.Normalize([0.5] * channels, [0.5] * channels))
        elif normalization == "imagenet":
            if channels != 3:
                raise ValueError("ImageNet normalization requires three channels")
            operations.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        elif normalization != "none":
            raise ValueError(normalization)
        self.transform = transforms.Compose(operations)

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["images"] = None  # Windows/WSL workers reopen the memory map.
        return state

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode="r", allow_pickle=False)
        row = self.rows[index]
        image = Image.fromarray(self.images[int(row["array_index"])] * np.uint8(255))
        if self.channels == 3:
            image = image.convert("RGB")
        return self.transform(image), int(row["label"]), row["sample_id"]


class FeatureDataset(Dataset):
    def __init__(self, path, rows):
        self.path, self.rows, self.features = str(path), rows, None

    def __len__(self):
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["features"] = None
        return state

    def __getitem__(self, index):
        if self.features is None:
            self.features = np.load(self.path, mmap_mode="r", allow_pickle=False)
        row = self.rows[index]
        return torch.from_numpy(np.array(self.features[index], copy=True)), int(row["label"]), row["sample_id"]


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, shuffle, seed, workers, device):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      persistent_workers=False, worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed), drop_last=False)


def layout_dataset(root, rows, config):
    return LayoutDataset(Path(root) / "data", rows, config["image_size"],
                         config["channels"], config["normalization"])


class FrozenClassifier(nn.Module):
    """The same wrapper supports a frozen linear probe or full fine-tuning."""
    def __init__(self, backbone, feature_dim, train_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feature_dim, 2)
        self.train_backbone = train_backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad = train_backbone

    def train(self, mode=True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def extract_features(self, images):
        if self.train_backbone:
            return self.backbone(images)
        with torch.no_grad():
            return self.backbone(images)

    def forward(self, images):
        return self.head(self.extract_features(images))


def cache_features(model, root, rows, split, config, device, identity):
    """One frozen-backbone pass per split. The test cache is created only on explicit test evaluation."""
    weight_name = "dinov2_small.safetensors" if config["model"] == "dinov2" else "vmamba_tiny_imagenet.pth"
    signature = {"data": identity, "model": config["model"],
                 "weight_sha256": sha256_file(Path(root) / "pretrained" / weight_name),
                 "image_size": config["image_size"], "channels": config["channels"],
                 "normalization": config["normalization"], "interpolation": "PIL_bilinear",
                 "precision": config["precision"], "torch": torch.__version__,
                 "timm": importlib.metadata.version("timm"),
                 "runtime_sha256": sha256_file(Path(root) / "lab_code/runtime.py")}
    if config["model"] == "vmamba":
        signature["implementation_sha256"] = sha256_file(Path(root) / "lab_code/vmamba_official.py")
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = Path(root) / "feature_cache" / config["model"] / key[:20]
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / f"{split}.npy"
    metadata = folder / f"{split}.json"
    ids_hash = hashlib.sha256("\n".join(r["sample_id"] for r in rows).encode()).hexdigest()
    if final.exists() and metadata.exists():
        details = json.loads(metadata.read_text())
        if details["sample_ids_sha256"] != ids_hash or details["signature"] != signature:
            raise ValueError("Feature cache identity mismatch")
        array = np.load(final, mmap_mode="r", allow_pickle=False)
        if array.shape != (len(rows), model.head.in_features) or sha256_file(final) != details["sha256"]:
            raise ValueError("Invalid feature cache")
        print(f"Reusing {split} features: {len(rows):,} layouts")
        return final
    partial = folder / f"{split}.partial.npy"
    state_path = folder / f"{split}.partial.json"
    offset = 0
    if partial.exists() and state_path.exists():
        state = json.loads(state_path.read_text())
        if state["sample_ids_sha256"] == ids_hash and state["signature"] == signature:
            offset = int(state["completed_rows"])
    mode = "r+" if offset else "w+"
    features = np.lib.format.open_memmap(partial, mode=mode, dtype="float32",
                                         shape=(len(rows), model.head.in_features))
    model.eval()
    dataset = layout_dataset(root, rows[offset:], config)
    loader = make_loader(dataset, config["feature_batch_size"], False, config["seed"],
                         config["num_workers"], device)
    with torch.inference_mode():
        for images, _, _ in tqdm(loader, desc=f"{split} feature extraction", leave=True):
            with precision_context(device, config["precision"]):
                values = model.extract_features(images.to(device, non_blocking=True))
            values = values.float().cpu().numpy()
            if values.shape[1] != model.head.in_features or not np.isfinite(values).all():
                raise FloatingPointError("Invalid frozen-backbone features")
            features[offset:offset + len(values)] = values
            offset += len(values)
            features.flush()
            atomic_json(state_path, {"completed_rows": offset, "sample_ids_sha256": ids_hash,
                                     "signature": signature})
    del features
    partial.replace(final)
    atomic_json(metadata, {"signature": signature, "sample_ids_sha256": ids_hash,
                           "records": len(rows), "sha256": sha256_file(final)})
    state_path.unlink(missing_ok=True)
    return final


def build_training_loaders(model, root, partitions, config, device, identity):
    cached = config["model"] in ("dinov2", "vmamba") and not config["train_backbone"]
    datasets = {}
    for split in ("train", "validation"):
        rows = partitions[split]
        if cached:
            path = cache_features(model, root, rows, split, config, device, identity)
            datasets[split] = FeatureDataset(path, rows)
        else:
            datasets[split] = layout_dataset(root, rows, config)
    # Frozen features are reusable across seeds. Keep the backbone off GPU during head training.
    if cached:
        model.backbone.to("cpu")
        trainable_model = model.head
    else:
        trainable_model = model
    trainable_model.to(device)
    loaders = {split: make_loader(datasets[split], config["batch_size"], split == "train",
                                  config["seed"], config["num_workers"], device)
               for split in ("train", "validation")}
    return trainable_model, loaders, cached


def optimizer_and_scheduler(model, config):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(parameters, lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"])
    else:
        raise ValueError(config["optimizer"])
    warmup, epochs = config["warmup_epochs"], config["max_epochs"]
    minimum = config["minimum_lr_ratio"]
    def multiplier(epoch):
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = min(1.0, max(0.0, (epoch - warmup) / max(1, epochs - warmup - 1)))
        return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    return optimizer, scheduler


def train_one_epoch(model, loader, optimizer, scaler, config, device):
    """Accumulate sample-weighted gradients, including the smaller final batch/window."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"], reduction="sum")
    total_loss = total_correct = total_samples = window_samples = 0
    for step, (images, labels, _) in enumerate(tqdm(loader, desc="Training", leave=False)):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with precision_context(device, config["precision"]):
            logits = model(images)
            loss_sum = criterion(logits, labels)
        if not torch.isfinite(loss_sum):
            raise FloatingPointError("Non-finite training loss; no checkpoint overwritten")
        # Normalize by a fixed reference batch first, then correct for the actual
        # sample count at the update. This avoids very large FP16 scaled sums.
        scaler.scale(loss_sum / config["batch_size"]).backward()
        samples = len(labels)
        window_samples += samples
        total_samples += samples
        total_loss += loss_sum.detach().item()
        total_correct += (logits.argmax(1) == labels).sum().item()
        update = (step + 1) % config["accumulation_steps"] == 0 or step + 1 == len(loader)
        if update:
            scaler.unscale_(optimizer)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(config["batch_size"] / window_samples)
            if config["gradient_clip_norm"]:
                nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"],
                                         error_if_nonfinite=config["precision"] != "fp16")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            window_samples = 0
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


@torch.inference_mode()
def evaluate(model, loader, device, config):
    """Evaluation-mode metrics use unsmoothed cross entropy for every model."""
    model.eval()
    total_loss = total_samples = 0
    labels_all, probabilities_all, predictions_all, sample_ids = [], [], [], []
    for images, labels, ids in tqdm(loader, desc="Evaluation", leave=False):
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with precision_context(device, config["precision"]):
            logits = model(images)
        loss = nn.functional.cross_entropy(logits.float(), labels, reduction="sum")
        probabilities = logits.float().softmax(1)[:, 1]
        if not torch.isfinite(loss) or not torch.isfinite(probabilities).all():
            raise FloatingPointError("Non-finite evaluation output")
        total_loss += loss.item()
        total_samples += len(labels)
        labels_all.extend(labels.cpu().tolist())
        probabilities_all.extend(probabilities.cpu().tolist())
        predictions_all.extend(logits.argmax(1).cpu().tolist())
        sample_ids.extend(ids)
    y = np.asarray(labels_all, dtype=np.int64)
    probability = np.asarray(probabilities_all)
    predicted = np.asarray(predictions_all, dtype=np.int64)  # argmax ties choose class zero.
    metrics = {"records": total_samples, "loss": total_loss / total_samples,
               "accuracy": float(accuracy_score(y, predicted)),
               "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
               "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
               "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
               "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
               "classification_report": classification_report(y, predicted, labels=[0, 1],
                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)}
    return metrics, {"sample_id": sample_ids, "label": y.tolist(),
                     "probability_class_1": probability.tolist(), "prediction": predicted.tolist()}


def save_predictions(path, predictions):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        columns = list(predictions)
        writer.writerow(columns)
        writer.writerows(zip(*(predictions[c] for c in columns)))
    temporary.replace(path)


def save_history(folder, history):
    atomic_json(Path(folder) / "history.json", history)
    temporary = Path(folder) / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    temporary.replace(Path(folder) / "history.csv")
    plot_history(folder, history)


def plot_history(folder, history=None, show=False):
    folder = Path(folder)
    if history is None:
        history = json.loads((folder / "history.json").read_text())
    epochs = [r["epoch"] for r in history]
    for metric, label, scale in (("accuracy", "Accuracy (%)", 100), ("loss", "Cross-entropy loss", 1)):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, [scale * r[f"train_{metric}"] for r in history], label="Training (online)")
        ax.plot(epochs, [scale * r[f"validation_{metric}"] for r in history], label="Validation (evaluation mode)")
        ax.set(xlabel="Epoch", ylabel=label)
        if metric == "accuracy":
            ax.set_ylim(0, 100)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(folder / f"{metric}_curves.png", dpi=180)
        fig.savefig(folder / f"{metric}_curves.pdf")
        if show:
            plt.show()
        plt.close(fig)


def capture_rng(loader):
    numpy_state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": {"algorithm": numpy_state[0], "keys": numpy_state[1].tolist(),
                      "position": numpy_state[2], "has_gauss": numpy_state[3],
                      "cached_gaussian": numpy_state[4]},
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": loader.generator.get_state()}


def restore_rng(state, loader):
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state((values["algorithm"], np.asarray(values["keys"], dtype=np.uint32),
                         values["position"], values["has_gauss"], values["cached_gaussian"]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    loader.generator.set_state(state["loader"].cpu())


def config_identity(config, identity):
    # Paths may change when the package moves between computers; run meaning must not.
    return {"config": config, "data": identity}


def fit(model, loaders, config, device, folder, identity, model_info):
    """Train/validate only; restore the minimum-validation-loss checkpoint on completion."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    contract = config_identity(config, identity)
    contract_path = folder / "run_config.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("This run folder has different settings/data. Use another run_name.")
    atomic_json(contract_path, contract)
    atomic_json(folder / "environment.json", environment_info(device))
    atomic_json(folder / "model_info.json", model_info)
    latest, best = folder / "last_checkpoint.pt", folder / "best_checkpoint.pt"
    if (folder / "TRAINING_COMPLETE.json").exists():
        state = torch.load(best, map_location="cpu", weights_only=True)
        if state["contract"] != contract:
            raise ValueError("Completed checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        print("Training already complete; loaded the validation-selected checkpoint.")
        return json.loads((folder / "history.json").read_text())
    optimizer, scheduler = optimizer_and_scheduler(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config["precision"] == "fp16")
    history, start, best_loss, best_epoch, stale = [], 0, float("inf"), 0, 0
    elapsed_before = 0.0
    if latest.exists():
        if not config["resume"]:
            raise ValueError("Existing checkpoint found. Enable resume or use another run_name.")
        state = torch.load(latest, map_location="cpu", weights_only=True)
        if state["contract"] != contract:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history, start = state["history"], state["epoch"]
        best_loss, best_epoch, stale = state["best_loss"], state["best_epoch"], state["stale_epochs"]
        elapsed_before = state["training_seconds"]
        restore_rng(state["rng"], loaders["train"])
        print(f"Resuming after epoch {start}; interrupted partial epochs are repeated.")
    started = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start, config["max_epochs"]):
        if stale >= config["early_stopping_patience"]:
            break
        epoch_start = time.monotonic()
        lr = optimizer.param_groups[0]["lr"]
        training = train_one_epoch(model, loaders["train"], optimizer, scaler, config, device)
        validation, _ = evaluate(model, loaders["validation"], device, config)
        scheduler.step()
        history.append({"epoch": epoch + 1, "learning_rate": lr,
                        "train_loss": training["loss"], "train_accuracy": training["accuracy"],
                        "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
                        "epoch_seconds": time.monotonic() - epoch_start})
        improved = validation["loss"] < best_loss - config["minimum_improvement"]
        if improved:
            best_loss, best_epoch, stale = validation["loss"], epoch + 1, 0
        else:
            stale += 1
        state = {"contract": contract, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch + 1, "history": history, "best_loss": best_loss,
                 "best_epoch": best_epoch, "stale_epochs": stale,
                 "training_seconds": elapsed_before + time.monotonic() - started,
                 "rng": capture_rng(loaders["train"])}
        if improved:
            atomic_checkpoint(best, {"contract": contract, "model": model.state_dict(),
                                     "epoch": best_epoch, "validation_loss": best_loss})
        atomic_checkpoint(latest, state)
        save_history(folder, history)
        print(f"Epoch {epoch + 1:03d}: training {training['accuracy']:.2%}; "
              f"validation {validation['accuracy']:.2%}; val loss {validation['loss']:.5f}")
    if not best.exists():
        raise RuntimeError("No validation-selected checkpoint was created")
    selected = torch.load(best, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model"])
    # Evaluate BOTH train and validation at the SAME selected checkpoint in eval mode.
    summary = {"selection": "minimum_validation_loss", "selected_epoch": selected["epoch"],
               "epochs_completed": len(history), "model_info": model_info,
               "training_seconds": elapsed_before + time.monotonic() - started,
               "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device)
               if device.type == "cuda" else None,
               "test_evaluated": False}
    for split in ("train", "validation"):
        original_loader = loaders[split]
        ordered = make_loader(original_loader.dataset, config["batch_size"], False,
                               config["seed"], config["num_workers"], device)
        metrics, predictions = evaluate(model, ordered, device, config)
        summary[split] = metrics
        save_predictions(folder / f"{split}_predictions.csv", predictions)
    atomic_json(folder / "training_validation_metrics.json", summary)
    atomic_json(folder / "TRAINING_COMPLETE.json", {"selected_epoch": selected["epoch"],
                                                  "epochs_completed": len(history), "test_evaluated": False})
    return history


def evaluate_test(model, training_model, root, partitions, config, device, folder, identity, cached):
    """An explicitly called, cached final test evaluation. Never called by fit()."""
    folder = Path(folder)
    if not (folder / "TRAINING_COMPLETE.json").exists():
        raise RuntimeError("Complete training/validation selection before evaluating the test set")
    checkpoint = folder / "best_checkpoint.pt"
    checksum = sha256_file(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["contract"] != config_identity(config, identity):
        raise ValueError("Test checkpoint identity mismatch")
    output = folder / "test_metrics.json"
    if output.exists():
        stored = json.loads(output.read_text())
        if stored["checkpoint_sha256"] != checksum:
            raise ValueError("Test output exists for a different checkpoint; refusing to overwrite it")
        print("Using the existing final test evaluation.")
        return stored
    training_model.load_state_dict(state["model"])
    training_model.to(device)
    rows = partitions["test"]
    if cached:
        model.backbone.to(device)
        features = cache_features(model, root, rows, "test", config, device, identity)
        model.backbone.to("cpu")
        dataset = FeatureDataset(features, rows)
    else:
        dataset = layout_dataset(root, rows, config)
    loader = make_loader(dataset, config["batch_size"], False, config["seed"], config["num_workers"], device)
    metrics, predictions = evaluate(training_model, loader, device, config)
    train_y = [int(r["label"]) for r in partitions["train"]]
    majority_label = int(np.argmax(np.bincount(train_y, minlength=2)))
    metrics["training_majority_label"] = majority_label
    metrics["training_selected_majority_baseline_accuracy"] = float(np.mean(
        np.asarray(predictions["label"]) == majority_label))
    metrics["selected_epoch"] = state["epoch"]
    metrics["checkpoint_sha256"] = checksum
    save_predictions(folder / "test_predictions.csv", predictions)
    atomic_json(output, metrics)
    fig, ax = plt.subplots(figsize=(5, 4))
    matrix = np.asarray(metrics["confusion_matrix"])
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Predicted class", ylabel="True class")
    fig.tight_layout()
    fig.savefig(folder / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return metrics
