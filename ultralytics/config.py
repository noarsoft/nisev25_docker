# config.py
# Loads config.json (next to this file) for the host-side scripts: init_data, run_gen, transfer, clear.
# Use another file with: STRIP_CONFIG=/path/to/config.json
# Adding a dataset: add "name": "file.csv" to "datasets" in config.json
# (the dataset also needs a loader in lib.py / strip_pre2.select_dataset)
# "run_datasets" (optional list) = datasets used by run_gen.py when --datasets is not given.
# Priority everywhere: command line option > config.json > built-in fallback.

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_REQUIRED = ("volume", "image", "data_dir", "export_dir", "defaults", "datasets")


def load():
    path = Path(os.environ.get("STRIP_CONFIG", HERE / "config.json"))
    try:
        with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig: Windows editors may add a BOM
            cfg = json.load(f)
    except FileNotFoundError:
        sys.exit(f"config not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"{path} is not valid json: {e}")
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        sys.exit(f"{path} is missing keys: {', '.join(missing)}")
    return cfg


def resolve(p):
    """relative paths in the config are relative to this folder, not to the current directory"""
    p = Path(p)
    return p if p.is_absolute() else HERE / p
