"""Stage 7: label single-epoch orbits by integrating them in the present-day system.

Two sources, both started at the pilot's t0 with the pilot's giants and settings:
  grid       test particles drawn uniformly around Jupiter's orbit (trojans.coorbital);
             these map where the tadpole label flips
  catalogue  the catalogue objects the pilot did not integrate (``~in_run``: inside
             the SBDB superset box but under no Trojan label); these are the real
             negatives the pilot never labelled
  targets    any objects with elements at t0 in a parquet (--targets), e.g. the
             screen's candidates from their Horizons states (s7_fetch_states.py)

Labels come from the same window classifier as every other run. A short run is
enough for "librating now"; longer runs give the same label held for longer.

    python campaign/scripts/s7_run_map.py --source grid --n 2000 --t-end-yr 1e5 --run-id map_smoke
    python campaign/scripts/s7_run_map.py --source catalogue --catalogue-id 2026-09-15 --run-id map_catalogue
"""

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from trojans import coorbital, initial, integrate, paths, runs
from trojans.outputs import provenance, write_json

DEFAULT_IC = "2026-09-15_giants"


def _init_worker():
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def run_chunk(task: dict) -> dict:
    settings = integrate.Settings(**task["settings"])
    root = paths.run_dir(task["run_id"])
    states = pd.read_parquet(root / "states.parquet")
    giants = json.loads((root / "settings.json").read_text())["giants"]
    ids = np.asarray(task["ids"], dtype=np.int64)
    out = paths.make_output_dir(paths.chunk_dir(task["run_id"], task["chunk"]), resume=task["resume"])
    if task["resume"] and (out / "latest.json").exists():
        runner = integrate.ChunkRunner.resume(ids, settings, giants, out)
    else:
        sim = initial.build_simulation(states, ids, settings.dt_yr)
        runner = integrate.ChunkRunner.start(sim, ids, settings, giants, out)
    cpu0, wall0 = time.process_time(), time.time()
    tag = f"[{task['run_id']} c{task['chunk']:03d}]"
    runner.run(log=lambda message: print(f"{tag} {message}", flush=True))
    session = {"chunk": task["chunk"], "n": len(ids), "cpu_s": time.process_time() - cpu0,
               "wall_s": time.time() - wall0,
               "n_alive": int((runner.events["removal_reason"] == integrate.ALIVE).sum())}
    with open(out / "sessions.jsonl", "a") as f:
        f.write(json.dumps(session) + "\n")
    return session


def catalogue_rest(catalogue_id: str) -> pd.DataFrame:
    table = pd.read_parquet(paths.catalogue_dir(catalogue_id) / "master_table.parquet")
    rest = table[~table["in_run"]]
    return rest[["spkid", "a", "e", "inc", "Omega", "omega", "M", "sigma"]].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source", choices=["grid", "catalogue", "targets"], required=True)
    parser.add_argument("--ic-id", default=DEFAULT_IC, help="giants and t0 come from here")
    parser.add_argument("--catalogue-id", default=None, help="source catalogue only")
    parser.add_argument("--targets", default=None, help="targets only: parquet with spkid and elements at t0")
    parser.add_argument("--n", type=int, default=None, help="grid only: number of test particles")
    parser.add_argument("--seed", type=int, default=coorbital.MapGrid.seed)
    parser.add_argument("--t-end-yr", type=float, default=1.0e5)
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    states, ic = runs.load_ic(args.ic_id)
    jupiter = coorbital.jupiter_at_t0(states)
    if args.source == "grid":
        if args.n is None:
            parser.error("--source grid needs --n")
        grid = coorbital.MapGrid(n=args.n, seed=args.seed)
        particles = coorbital.draw_elements(grid, jupiter)
        source_info = {"grid": asdict(grid)}
    elif args.source == "targets":
        if args.targets is None:
            parser.error("--source targets needs --targets")
        particles = pd.read_parquet(args.targets)[["spkid", "a", "e", "inc", "Omega", "omega", "M", "sigma"]]
        source_info = {"targets": args.targets}
    else:
        if args.catalogue_id is None:
            parser.error("--source catalogue needs --catalogue-id")
        particles = catalogue_rest(args.catalogue_id)
        source_info = {"catalogue_id": args.catalogue_id, "selection": "master_table ~in_run"}

    settings = integrate.Settings(t_end_yr=args.t_end_yr, dt_yr=ic["dt_yr"])
    root = paths.make_output_dir(paths.run_dir(args.run_id), resume=args.resume)
    if not args.resume:
        coorbital.states_with_particles(states, particles).to_parquet(root / "states.parquet", index=False)
        particles.to_parquet(root / "initial_elements.parquet", index=False)
        write_json(root / "settings.json", {
            "source": args.source, **source_info, "ic_id": args.ic_id, "t0_jd_tdb": ic.get("t0_jd_tdb"),
            "jupiter_t0": jupiter, "giants": ic["giants"], "settings": asdict(settings),
            "n_chunks": args.n_chunks, **provenance(),
        })

    ids = particles["spkid"].to_numpy(dtype=np.int64)
    tasks = [{"run_id": args.run_id, "chunk": k, "resume": args.resume, "ids": chunk.tolist(),
              "settings": asdict(settings)}
             for k, chunk in enumerate(np.array_split(ids, args.n_chunks))]

    print(f"{args.source}: {len(ids)} particles in {args.n_chunks} chunks on {args.workers} workers, "
          f"{args.t_end_yr:.3g} yr, dt = {settings.dt_yr:.4f} yr, {settings.n_windows} windows")
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        sessions = pool.map(run_chunk, tasks)
    cpu = sum(s["cpu_s"] for s in sessions)
    print(f"done: {sum(s['n_alive'] for s in sessions)}/{len(ids)} alive, {cpu / 3600:.2f} CPU-h, "
          f"{cpu / (len(ids) * args.t_end_yr / 1e6):.2f} CPU-s per particle-Myr")


if __name__ == "__main__":
    main()
