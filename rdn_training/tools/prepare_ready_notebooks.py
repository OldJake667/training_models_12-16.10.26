"""Create local notebook copies with the installed kernel and verified batches."""
import json
import argparse
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--models", nargs="+")
args = parser.parse_args()
configs = json.loads((ROOT / "RTX5050_MODEL_CONFIGS.json").read_text())
original = json.loads((ROOT / "MODEL_CONFIGS.json").read_text())
destination = ROOT / "ready_notebooks"
destination.mkdir(exist_ok=True)
for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
    notebook = nbformat.read(path, as_version=4)
    model = notebook.metadata.lab.model
    if args.models and model not in args.models:
        continue
    if model is not None and model not in configs:
        raise RuntimeError(f"Finish the GPU check for {model} before preparing its notebook")
    notebook.metadata.kernelspec = {"display_name": "SUTD RDN CUDA", "language": "python",
                                    "name": "sutd-rdn"}
    for cell in notebook.cells:
        role = cell.metadata.get("lab_role")
        if role == "kernel_check":
            cell.source = cell.source.replace("CHECK_VMAMBA_KERNEL = False", "CHECK_VMAMBA_KERNEL = True")
        if role == "configuration" and model is not None:
            changes = {key: value for key, value in configs[model].items()
                       if value != original[model][key]}
            if changes:
                mode = "linear_probe" if not configs[model]["train_backbone"] else "all_layers"
                run_name = f"seed_42_rtx5050_b{configs[model]['batch_size']}_a{configs[model]['accumulation_steps']}_{mode}_epochs_{configs[model]['max_epochs']}"
                cell.source += f"\n# Batch settings checked on this RTX 5050 laptop.\nCONFIG.update({changes!r})\nRUN_NAME = {run_name!r}\n"
        if cell.cell_type == "code":
            compile(cell.source, f"{path.name}:{role}", "exec")
            cell.execution_count = None
            cell.outputs = []
    nbformat.validate(notebook)
    nbformat.write(notebook, destination / path.name)
    print("Prepared", destination / path.name)
