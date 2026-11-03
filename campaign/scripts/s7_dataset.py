"""Stage 7: one table of snapshot features and integration labels, for map and real objects.

Rows come from
  - grid runs (s7_run_map.py --source grid): test particles, source "grid";
  - the real catalogue (``--catalogue-id``): objects in the pilot (``in_run``) take
    their labels from the pilot's windows, the rest from the catalogue run
    (s7_run_map.py --source catalogue); source "real".
Every row has the snapshot features of ``trojans.definition`` and labels
``label_k<k>`` for each k in --k (first k windows of 10 kyr all tadpole of one
swarm). Real rows also carry the catalogue flags and the orbit condition code.

Writes data/runs/<out-id>/dataset.parquet.

    python campaign/scripts/s7_dataset.py --out-id definition_v1 --grid-runs map_grid \
        --catalogue-id 2026-09-15 --pilot-run pilot_10myr --catalogue-run map_catalogue
"""

import argparse
import json

import numpy as np
import pandas as pd

from trojans import coorbital, definition, paths, runs
from trojans.outputs import provenance, write_json

REAL_COLUMNS = ["H", "condition_code", "jpl_tjn", "mpc_trojan", "v24_selected", "in_run"]


def labels_for(spkids, windows_files, ks) -> pd.DataFrame:
    ids, classes = definition.read_classes(windows_files)
    frame = pd.DataFrame({"spkid": ids})
    for k in ks:
        if classes.shape[0] >= k:
            frame[f"label_k{k}"] = definition.label_from_classes(classes, k)
    return frame.set_index("spkid").reindex(spkids).reset_index()


def chunk_windows(run_id: str):
    files = sorted(paths.run_dir(run_id).glob("chunks/c*/windows.h5"))
    if not files:
        raise FileNotFoundError(f"no chunk windows under {paths.run_dir(run_id)}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-id", required=True)
    parser.add_argument("--grid-runs", nargs="*", default=[])
    parser.add_argument("--catalogue-id", default=None)
    parser.add_argument("--pilot-run", default="pilot_10myr")
    parser.add_argument("--catalogue-run", default=None, help="s7 run of the catalogue's ~in_run objects")
    parser.add_argument("--ic-id", default="2026-09-15_giants")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    states, _ = runs.load_ic(args.ic_id)
    jupiter = coorbital.jupiter_at_t0(states)
    parts = []

    for run_id in args.grid_runs:
        settings = json.loads((paths.run_dir(run_id) / "settings.json").read_text())
        if settings["ic_id"] != args.ic_id:
            raise SystemExit(f"{run_id} starts from {settings['ic_id']}, not {args.ic_id}")
        el = pd.read_parquet(paths.run_dir(run_id) / "initial_elements.parquet")
        rows = definition.features(el, jupiter)
        rows.insert(0, "spkid", el["spkid"].to_numpy())
        rows.insert(1, "source", "grid")
        rows.insert(2, "run_id", run_id)
        lab = labels_for(el["spkid"].to_numpy(), chunk_windows(run_id), args.k)
        parts.append(pd.concat([rows.reset_index(drop=True), lab.drop(columns="spkid")], axis=1))

    if args.catalogue_id:
        table = pd.read_parquet(paths.catalogue_dir(args.catalogue_id) / "master_table.parquet")
        rows = definition.features(table, jupiter)
        rows.insert(0, "spkid", table["spkid"].to_numpy())
        rows.insert(1, "source", "real")
        rows.insert(2, "run_id", np.where(table["in_run"], args.pilot_run, args.catalogue_run or ""))
        for col in REAL_COLUMNS:
            rows[col] = table[col].to_numpy()
        ids = table["spkid"].to_numpy()
        lab = labels_for(ids, [paths.merged_dir(args.pilot_run) / "windows.h5"], args.k)
        if args.catalogue_run:
            rest = labels_for(ids, chunk_windows(args.catalogue_run), args.k)
            lab = lab.set_index("spkid").combine_first(rest.set_index("spkid")).reindex(ids).reset_index()
        parts.append(pd.concat([rows.reset_index(drop=True), lab.drop(columns="spkid")], axis=1))

    if not parts:
        parser.error("nothing to do: give --grid-runs and/or --catalogue-id")
    data = pd.concat(parts, ignore_index=True)
    out = paths.make_output_dir(paths.run_dir(args.out_id), resume=args.resume)
    data.to_parquet(out / "dataset.parquet", index=False)
    write_json(out / "dataset.json", {"grid_runs": args.grid_runs, "catalogue_id": args.catalogue_id,
                                      "pilot_run": args.pilot_run, "catalogue_run": args.catalogue_run,
                                      "ic_id": args.ic_id, "k": args.k, "jupiter_t0": jupiter, **provenance()})
    label_cols = [c for c in data.columns if c.startswith("label_k")]
    print(data.groupby("source")[label_cols].agg(lambda s: dict(s.value_counts(dropna=False))).to_string())


if __name__ == "__main__":
    main()
