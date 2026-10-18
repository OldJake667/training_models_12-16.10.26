"""Verify notebook code, data, model definitions, and serialization; never train.

No forward pass on dataset images, optimizer step, fit call, feature extraction,
or accuracy evaluation is performed. Large random models are created on meta.
"""
import ast
import contextlib
import gc
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lab_code import model_builders as builders
from lab_code import runtime
from lab_code import scratch_backbones
from lab_code.scratch_models import ScratchCNN, VisionTransformerTiny


def verify():
    torch.set_num_threads(2)
    report = {"training_executed": False, "gpu_verified": False,
              "torch_used_for_local_checks": str(torch.__version__),
              "checks": [], "notebooks": {}, "models": {}}
    configs = json.loads((ROOT / "MODEL_CONFIGS.json").read_text())
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        roles = set()
        for number, cell in enumerate(nb.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{path.name}:cell{number}", "exec")
                assert cell.execution_count is None and cell.outputs == []
                roles.add(cell.metadata.get("lab_role"))
        if nb.metadata.lab.model is not None:
            required = {"imports", "configuration", "preparation", "model_definition",
                        "training_epoch_definition", "evaluation_definition", "fit_definition",
                        "training_launch", "results", "final_test"}
            assert required <= roles
            fit_cell = next(c for c in nb.cells if c.metadata.get("lab_role") == "fit_definition")
            fit_node = ast.parse(fit_cell.source).body[0]
            assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                           and n.func.id == "evaluate_test" for n in ast.walk(fit_node))
            config_cell = next(c for c in nb.cells if c.metadata.get("lab_role") == "configuration")
            namespace = {}
            exec(config_cell.source, namespace)
            assert namespace["RUN_TRAINING"] is True
            assert namespace["RUN_FINAL_TEST"] is True
            assert namespace["CONFIG"] == configs[nb.metadata.lab.model]
        report["notebooks"][path.name] = {"valid": True, "code_compiles": True, "outputs_empty": True}
    for path in (ROOT / "lab_code").glob("*.py"):
        compile(path.read_text(), str(path), "exec")
    report["checks"].append(f"{len(report['notebooks'])} valid, empty notebooks; all code cells and supporting modules compile")
    partitions, identity = runtime.load_full_data(ROOT)
    report["split"] = {k: len(v) for k, v in partitions.items()}
    report["checks"].append("Full-array hashes, 111000 records, fixed cutoff labels, manifest and released test isolation")
    for model_key, config in configs.items():
        dataset = runtime.layout_dataset(ROOT, partitions["train"][:2], config)
        tensor, label, sample_id = dataset[0]
        assert tensor.shape == (config["channels"], config["image_size"], config["image_size"])
        assert torch.isfinite(tensor).all()
        assert label in [0, 1]
        loader = runtime.make_loader(dataset, 2, False, 42, 0, torch.device("cpu"))
        batch, labels, ids = next(iter(loader))
        assert len(ids) == 2 and batch.shape[0] == 2
    report["checks"].append(f"On-demand preprocessing and full-data loader sample shapes for all {len(configs)} configurations")
    # Verify both preserved architecture definitions against the original definitions.
    original_path = ROOT.parent / "sutd_prcm_experiment/models/scratch_models.py"
    if original_path.exists():
        spec = importlib.util.spec_from_file_location("original_scratch_for_verification", original_path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        for name, cls in [("cnn", ScratchCNN), ("vit_tiny", VisionTransformerTiny)]:
            previous_cls = original.ScratchCNN if name == "cnn" else original.VisionTransformerTiny
            torch.manual_seed(42)
            previous = previous_cls(classes=2)
            torch.manual_seed(42)
            current = cls(classes=2)
            previous_state, current_state = previous.state_dict(), current.state_dict()
            assert previous_state.keys() == current_state.keys()
            assert all(torch.equal(previous_state[k], current_state[k]) for k in previous_state)
            report["models"][name] = {"total_parameters": sum(p.numel() for p in current.parameters()),
                                      "trainable_parameters": sum(p.numel() for p in current.parameters()),
                                      "original_seeded_state_dict_identical": True}
            del previous, current
            gc.collect()
        report["checks"].append("CNN and ViT-Tiny seeded state dicts exactly match previous architectures")
    # ViT's torch.linspace(...).item() requires CPU scalars even when its parameters are meta.
    original_linspace = torch.linspace
    def cpu_linspace(*args, **kwargs):
        kwargs["device"] = "cpu"
        return original_linspace(*args, **kwargs)
    torch.linspace = cpu_linspace
    try:
        for name in ("resnet50", "resnet101", "resnet152", "vit_small", "vit_base", "vit_large"):
            with torch.device("meta"):
                model = builders.build_standard_model(ROOT, configs[name])
            count = sum(p.numel() for p in model.parameters())
            report["models"][name] = {"total_parameters": count, "trainable_parameters": count,
                                      "architecture_created_on_meta": True}
            del model
    finally:
        torch.linspace = original_linspace
    report["checks"].append("All six larger standard architectures construct with two-class heads on meta")
    # Construct random backbones without reading any pretrained checkpoint.
    for name in ("dinov2", "vmamba"):
        if name not in configs:
            continue
        config = dict(configs[name])
        config["require_fast_vmamba"] = False  # CPU architecture inspection only.
        factory = (scratch_backbones.build_dinov2_from_scratch if name == "dinov2"
                   else scratch_backbones.build_vmamba_from_scratch)
        model = factory(ROOT, config)
        count = sum(p.numel() for p in model.parameters())
        trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trained == count
        model.train()
        assert model.backbone.training and all(p.requires_grad for p in model.parameters())
        report["models"][name] = {"total_parameters": count, "default_trainable_parameters": trained,
                                  "local_pretrained_file_loaded": False, "all_layers_trainable": True}
        del model
        gc.collect()
    report["checks"].append("ViT-Small/14 and VMamba construct from random weights with all layers trainable")
    # Execute actual notebook setup, model construction, and function definitions.
    # Explicitly disable launch/test cells and use meta for the six larger models.
    previous_directory = Path.cwd()
    os.chdir(ROOT)
    try:
        for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
            nb = nbformat.read(path, as_version=4)
            name = nb.metadata.lab.model
            if name is None:
                continue
            namespace = {}
            with contextlib.redirect_stdout(io.StringIO()):
                for cell in nb.cells:
                    if cell.cell_type != "code":
                        continue
                    role = cell.metadata.get("lab_role")
                    if role == "preparation":
                        namespace["REQUIRE_CUDA"] = False
                        namespace["RUN_TRAINING"] = False
                        namespace["RUN_FINAL_TEST"] = False
                        namespace["RUN_NAME"] = f"_verification_{name}"
                        namespace["CONFIG"]["require_fast_vmamba"] = False
                    if role == "model_definition" and name not in ("cnn", "vit_tiny", "dinov2", "vmamba"):
                        torch.linspace = cpu_linspace
                        try:
                            with torch.device("meta"):
                                exec(cell.source, namespace)
                        finally:
                            torch.linspace = original_linspace
                    else:
                        exec(cell.source, namespace)
            report["notebooks"][path.name]["setup_model_and_functions_executed_without_training"] = True
            assert namespace["RUN_TRAINING"] is False and namespace["RUN_FINAL_TEST"] is False
            del namespace
            gc.collect()
    finally:
        os.chdir(previous_directory)
    report["checks"].append("Actual notebook setup/model/function cells execute; training and test launch flags disabled")
    # Check safe checkpoint serialization and RNG round-trip without taking any optimizer step.
    dataset = runtime.layout_dataset(ROOT, partitions["train"][:2], configs["cnn"])
    loader = runtime.make_loader(dataset, 2, True, 42, 0, torch.device("cpu"))
    runtime.seed_everything(42)
    state = runtime.capture_rng(loader)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "serialization.pt"
        runtime.atomic_checkpoint(path, {"rng": state, "contract": runtime.config_identity(configs["cnn"], identity)})
        restored = torch.load(path, map_location="cpu", weights_only=True)
        runtime.restore_rng(restored["rng"], loader)
        assert torch.equal(state["torch"], torch.get_rng_state())
        assert torch.equal(state["loader"], loader.generator.get_state())
    report["checks"].append("weights_only checkpoint and RNG serialization round-trip; no optimizer steps")
    report["limitations"] = ["No CUDA hardware available for these checks",
                              "RTX 5090 and VMamba CUDA kernel require GPU preflight",
                              "No training, feature extraction, dataset model inference, or accuracy evaluation performed"]
    runtime.atomic_json(ROOT / "VERIFICATION.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    verify()
