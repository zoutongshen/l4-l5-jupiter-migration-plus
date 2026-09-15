"""Run bookkeeping: configs, initial conditions, chunk assignment, chunk status."""

import json
import tomllib

import numpy as np
import pandas as pd

from trojans import paths


def load_config_file(path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_ic(ic_id: str) -> tuple[pd.DataFrame, dict]:
    directory = paths.initial_dir(ic_id)
    return pd.read_parquet(directory / "states.parquet"), json.loads((directory / "ic.json").read_text())


def assign_chunks(spkids, n_chunks: int, seed: int, subset: int | None = None) -> pd.DataFrame:
    """Seeded shuffle (optionally a random subset), dealt round-robin into chunks."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(np.asarray(spkids, dtype=np.int64))
    if subset is not None:
        order = order[:subset]
    return pd.DataFrame({"spkid": order, "chunk": np.arange(len(order)) % n_chunks})


def chunk_status(run_id: str, chunk: int, n_windows: int) -> str:
    """'new', 'partial' or 'done'."""
    latest = paths.chunk_dir(run_id, chunk) / "latest.json"
    if not latest.exists():
        return "new"
    return "done" if json.loads(latest.read_text())["window_index"] >= n_windows else "partial"
