"""Stage 3: merge a finished run's chunks into data/runs/<run_id>/merged/.

  events.parquet  one row per Trojan: fate and event times, joined with the
                  catalogue flags and labels (swarm, complete, disisto2013, ...)
  windows.h5      every window field as (n_windows, n_trojans), same column order
                  as the spkid dataset
  cost.json       CPU and wall time summed over chunk sessions
"""

import argparse
import json

import h5py
import numpy as np
import pandas as pd

from trojans import integrate, paths, runs
from trojans.outputs import write_json, write_provenance

CATALOGUE_COLUMNS = [
    "spkid", "full_name", "number", "H", "complete", "disisto2013", "condition_code", "state_source",
    "swarm", "sigma", "a", "e", "inc", "A", "B", "C",
    "v24_selected", "jpl_tjn", "mpc_trojan", "mpc_ln", "label_region", "v24_family",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true", help="overwrite an existing merged/ directory")
    args = parser.parse_args()

    run_dir = paths.run_dir(args.run_id)
    config = runs.load_config_file(run_dir / "config.toml")
    _, ic = runs.load_ic(config["run"]["ic_id"])
    settings = integrate.settings_from_config(config, ic["dt_yr"])
    n_chunks = config["run"]["n_chunks"]
    unfinished = [c for c in range(n_chunks) if runs.chunk_status(args.run_id, c, settings.n_windows) != "done"]
    if unfinished:
        raise SystemExit(f"chunks not finished: {unfinished}")

    chunk_dirs = [paths.chunk_dir(args.run_id, c) for c in range(n_chunks)]
    master = pd.read_parquet(paths.catalogue_dir(ic["catalogue_id"]) / "master_table.parquet", columns=CATALOGUE_COLUMNS)
    events = pd.concat([pd.read_parquet(d / "events.parquet") for d in chunk_dirs], ignore_index=True)
    events = events.merge(master, on="spkid", how="left", validate="one_to_one")

    sessions = [json.loads(line) for d in chunk_dirs for line in (d / "sessions.jsonl").read_text().splitlines()]
    cost = {
        "cpu_s": sum(s["cpu_s"] for s in sessions),
        "wall_s_max_chunk": max(s["wall_s"] for s in sessions),
        "n_sessions": len(sessions),
        "n_trojans": len(events),
        "t_end_yr": settings.n_windows * settings.checks_per_window * settings.check_interval_yr,
        "dt_yr": settings.dt_yr,
        "check_every_steps": settings.check_every_steps,
    }
    cost["cpu_s_per_trojan_myr"] = cost["cpu_s"] / (cost["n_trojans"] * cost["t_end_yr"] / 1e6)

    out = paths.make_output_dir(paths.merged_dir(args.run_id), args.resume)
    events.to_parquet(out / "events.parquet", index=False)
    with h5py.File(out / "windows.h5", "w") as merged:
        sources = [h5py.File(d / "windows.h5", "r") for d in chunk_dirs]
        try:
            t_end = sources[0]["t_end"][:]
            if any(not np.array_equal(s["t_end"][:], t_end) for s in sources):
                raise SystemExit("chunks disagree on window times")
            merged.attrs.update(sources[0].attrs)
            merged.create_dataset("t_end", data=t_end)
            merged.create_dataset("spkid", data=np.concatenate([s["spkid"][:] for s in sources]))
            for name in ["class", "n_samples", *integrate.WINDOW_FIELDS]:
                merged.create_dataset(name, data=np.concatenate([s[name][:] for s in sources], axis=1),
                                      chunks=True, compression="lzf")
        finally:
            for s in sources:
                s.close()
    write_json(out / "cost.json", cost)
    write_provenance(out)
    print(json.dumps(cost, indent=2))


if __name__ == "__main__":
    main()
