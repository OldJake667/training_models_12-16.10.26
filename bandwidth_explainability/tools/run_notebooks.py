"""Execute explanation notebooks sequentially, saving outputs during execution."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter/data"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter/runtime"))
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".ipython"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import nbformat
from nbclient import NotebookClient


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class SavingClient(NotebookClient):
    def __init__(self, notebook, path, status, **kwargs):
        super().__init__(notebook, **kwargs)
        self.path = path
        self.status = status
        self.last_save = 0.0

    def save(self):
        temporary = self.path.with_suffix(".ipynb.tmp")
        nbformat.write(self.nb, temporary)
        temporary.replace(self.path)
        self.status["last_saved_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(ROOT / "RUN_STATUS.json", self.status)
        self.last_save = time.monotonic()

    def process_message(self, msg, cell, cell_index):
        self.status["cell_index"] = cell_index
        try:
            return super().process_message(msg, cell, cell_index)
        finally:
            if time.monotonic() - self.last_save > 10:
                self.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", help="Notebook filenames; defaults to all nine in order")
    args = parser.parse_args()
    paths = ([ROOT / name for name in args.notebooks] if args.notebooks
             else sorted(ROOT.glob("[0-9][0-9]_*.ipynb")))
    status = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "state": "running", "completed": [], "pending": [p.name for p in paths],
              "dataset": "original bandwidth; 45 validation images; no retraining"}
    for path in paths:
        status.update(current_notebook=path.name, state="running")
        print(f"Starting {path.name}", flush=True)
        nb = nbformat.read(path, as_version=4)
        client = SavingClient(nb, path, status, kernel_name="bandwidth-xai", timeout=None,
                              resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
        try:
            client.execute()
            status["completed"].append(path.name)
            status["pending"].remove(path.name)
            print(f"Completed {path.name}", flush=True)
        except BaseException as error:
            status.update(state="failed", error=f"{type(error).__name__}: {error}")
            traceback.print_exc()
            raise
        finally:
            client.save()
    status.update(state="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(ROOT / "RUN_STATUS.json", status)
    print("All requested notebooks completed.", flush=True)


if __name__ == "__main__":
    main()
