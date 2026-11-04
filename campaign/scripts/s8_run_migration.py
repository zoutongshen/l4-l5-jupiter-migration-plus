"""Stage 8: capture from a mirror-symmetric disc while Jupiter grows and migrates.

One arm per TOML config (campaign/configs/mig_*.toml). Jupiter alone (folded Sun),
circular and planar, follows the history in trojans/history.py; after the forcing
ends it stays fixed for [run] t_classify_yr so the window classifier can decide
who is a Trojan (s8_migration_report.py). The disc is built from mirror twins
(trojans/disc.py), so any L4/L5 difference was made by the history.

Config sections:
  [run]        run_id, t_classify_yr
  [jupiter]    a_start_au, a_end_au, and either t_mig_yr or nu0 (sets t_mig from the
               speed at the core mass, const_nu law); migration_law, t_mig_start_yr,
               m0_earth, t_grow_start_yr, t_grow_yr, growth_law.
               Or only track_csv = "tracks/<file>.csv" (columns t_yr, a_au, m_earth).
  [disc]       n_pairs, a_min_au, a_max_au, e_max, inc_max_deg, seed
  [snapshots]  fine_every_yr (until the forcing ends), every_yr (after; default one window)
  [parallel]   n_chunks, workers
  [integrator] / [diagnostics] / [escape]  optional overrides of integrate.Settings

    python campaign/scripts/s8_run_migration.py campaign/configs/mig_v2_pirani.toml
    python campaign/scripts/s8_run_migration.py <config> --run-id mig_smoke --n-pairs 250 --t-classify-yr 2e4
    python campaign/scripts/s8_run_migration.py <config> --resume
"""

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from trojans import constants, disc, integrate, paths, runs
from trojans.history import HistoryForcing, JupiterHistory
from trojans.outputs import provenance, write_json

PLANETS = ("jupiter",)


def _init_worker():
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def history_from_config(section: dict) -> JupiterHistory:
    j = dict(section)
    if "track_csv" in j:            # a tabulated track (t_yr, a_au, m_earth), path relative to configs/
        if set(j) != {"track_csv"}:
            raise ValueError("track_csv takes no other [jupiter] keys")
        path = Path(__file__).resolve().parents[1] / "configs" / j["track_csv"]
        t, a, m = np.loadtxt(path, delimiter=",", skiprows=1, unpack=True)
        return JupiterHistory.from_table(t, a, m * disc.M_EARTH)
    m0 = float(j.pop("m0_earth", 10.0)) * disc.M_EARTH
    if "m_final_earth" in j:
        j["m_final_msun"] = float(j.pop("m_final_earth")) * disc.M_EARTH
    a_start, a_end = float(j.pop("a_start_au")), float(j.pop("a_end_au"))
    if "nu0" in j:
        if "t_mig_yr" in j or j.get("migration_law", "const_nu") != "const_nu":
            raise ValueError("give either nu0 (const_nu law) or t_mig_yr, not both")
        j.pop("migration_law", None)
        return JupiterHistory.from_nu(float(j.pop("nu0")), a_start, a_end, m0_msun=m0, **j)
    return JupiterHistory(a_start_au=a_start, a_end_au=a_end, m0_msun=m0, **j)


def build(config: dict, overrides: argparse.Namespace):
    run = config["run"]
    history = history_from_config(config["jupiter"])
    d = dict(config["disc"])
    if overrides.n_pairs:
        d["n_pairs"] = overrides.n_pairs
    the_disc = disc.Disc(n_pairs=int(d["n_pairs"]), a_min=float(d["a_min_au"]), a_max=float(d["a_max_au"]),
                         e_max=float(d.get("e_max", disc.Disc.e_max)),
                         inc_max_deg=float(d.get("inc_max_deg", disc.Disc.inc_max_deg)),
                         seed=int(d.get("seed", disc.Disc.seed)))
    t_classify = overrides.t_classify_yr or float(run["t_classify_yr"])
    t_end = history.t_forcing_end_yr + t_classify
    fields = {**config.get("integrator", {}), **config.get("diagnostics", {}), **config.get("escape", {})}
    window = float(fields.get("window_yr", integrate.Settings.window_yr))
    snaps = config.get("snapshots", {})
    settings = integrate.Settings(
        t_end_yr=t_end, dt_yr=constants.timestep(history.a_min_au),
        snapshot_every_yr=float(snaps.get("every_yr", window)),
        snapshot_fine_every_yr=float(snaps.get("fine_every_yr", 0.0)),
        snapshot_fine_until_yr=history.t_forcing_end_yr, **fields)
    return history, the_disc, settings


def run_chunk(task: dict) -> dict:
    history = JupiterHistory(**task["history"])
    settings = integrate.Settings(**task["settings"])
    ids = np.asarray(task["ids"], dtype=np.int64)
    directory = paths.chunk_dir(task["run_id"], task["chunk"])
    # A chunk that never started is started fresh on --resume.
    out = paths.make_output_dir(directory, resume=task["resume"] and directory.exists())
    a0, m0 = history.a(0.0), history.mass(0.0)
    giants = {"jupiter": {"a": a0, "hill_radius": disc.hill_radius(a0, m0)}}
    forcing = HistoryForcing(history)
    if task["resume"] and (out / "latest.json").exists():
        runner = integrate.ChunkRunner.resume(ids, settings, giants, out, PLANETS, forcing)
    else:
        pos, vel = np.asarray(task["pos"]), np.asarray(task["vel"])
        sim = disc.build_simulation(pos, vel, ids, history, a0, settings.dt_yr)
        runner = integrate.ChunkRunner.start(sim, ids, settings, giants, out, PLANETS, forcing)
    cpu0, wall0 = time.process_time(), time.time()
    tag = f"[{task['run_id']} c{task['chunk']:03d}]"
    runner.run(log=lambda message: print(f"{tag} {message}", flush=True))
    session = {"chunk": task["chunk"], "n": len(ids), "cpu_s": time.process_time() - cpu0,
               "wall_s": time.time() - wall0,
               "n_alive": int((runner.events["removal_reason"] == integrate.ALIVE).sum())}
    with open(out / "sessions.jsonl", "a") as f:
        f.write(json.dumps(session) + "\n")
    return session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--run-id", help="override [run] run_id (smoke tests)")
    parser.add_argument("--n-pairs", type=int, help="override [disc] n_pairs (smoke tests)")
    parser.add_argument("--t-classify-yr", type=float, help="override [run] t_classify_yr (smoke tests)")
    parser.add_argument("--workers", type=int, help="override [parallel] workers")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print the arm and exit")
    args = parser.parse_args()

    config = runs.load_config_file(args.config)
    run_id = args.run_id or config["run"]["run_id"]
    history, the_disc, settings = build(config, args)
    par = config.get("parallel", {})
    n_chunks = int(par.get("n_chunks", 8))
    workers = args.workers or int(par.get("workers", 8))

    record = {"run_id": run_id, "config_file": str(args.config), "config": config,
              "history": history.summary(), "disc": asdict(the_disc), "settings": asdict(settings),
              "planets": PLANETS, "n_chunks": n_chunks}
    summary = history.summary()
    print(f"{run_id}: {summary['direction']} {history.a_start_au} -> {history.a_end_au} au "
          f"over {history.t_mig_yr / 1e3:.1f} kyr ({summary['mean_rate_au_per_myr']:+.1f} au/Myr), "
          f"nu start/mid/end {summary['nu_start']:.3f}/{summary['nu_mid']:.3f}/{summary['nu_end']:.3f}; "
          f"m {summary['m0_earth']:.3g} -> {summary['m_final_earth']:.0f} M_E over "
          f"[{history.t_grow_start_yr / 1e3:.0f}, {history.t_grow_end_yr / 1e3:.0f}] kyr; "
          f"t_end {settings.t_end_yr / 1e6:.3f} Myr, dt {settings.dt_yr:.4f} yr, "
          f"{2 * the_disc.n_pairs} particles in a [{the_disc.a_min}, {the_disc.a_max}] au, "
          f"e <= {the_disc.e_max}, I <= {the_disc.inc_max_deg} deg; {n_chunks} chunks on {workers} workers")
    if args.dry_run:
        return

    root = paths.run_dir(run_id)
    if args.resume:
        previous = json.loads((root / "settings.json").read_text())
        for key in ("history", "disc", "settings"):
            if json.dumps(previous[key], sort_keys=True) != json.dumps(record[key], sort_keys=True):
                raise SystemExit(f"--resume: {key} differs from {root / 'settings.json'}")
    else:
        paths.make_output_dir(root, resume=False)
        write_json(root / "settings.json", {**record, **provenance()})

    pos, vel = disc.disc_states(the_disc)
    ids = disc.particle_ids(the_disc.n_pairs)
    if not args.resume:
        np.savez(root / "disc_initial.npz", id=ids, pos=pos, vel=vel)

    # Twins stay in the same chunk.
    tasks = []
    for k, pairs in enumerate(np.array_split(np.arange(the_disc.n_pairs), n_chunks)):
        rows = np.sort(np.concatenate([2 * pairs, 2 * pairs + 1]))
        tasks.append({"run_id": run_id, "chunk": k, "resume": args.resume,
                      "ids": ids[rows].tolist(), "pos": pos[rows].tolist(), "vel": vel[rows].tolist(),
                      "history": asdict(history), "settings": asdict(settings)})

    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        sessions = pool.map(run_chunk, tasks)
    cpu = sum(s["cpu_s"] for s in sessions)
    n = sum(s["n"] for s in sessions)
    print(f"done: {sum(s['n_alive'] for s in sessions)}/{n} alive, {cpu / 3600:.2f} CPU-h, "
          f"{cpu / (n * settings.t_end_yr / 1e6):.2f} CPU-s per particle-Myr")


if __name__ == "__main__":
    main()
