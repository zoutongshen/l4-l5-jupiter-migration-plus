"""Stage 6: capture from a mirror-symmetric disc without migration (arms 0 and 1).

Arm "fixed": Jupiter at full mass from t = 0. Arm "grow": Jupiter grows from a
core to full mass over --t-grow-yr. In both arms Jupiter stays on a circular
orbit at today's a; no other planets. See trojans/disc.py for why the disc is
built from mirror twins.

Each chunk is an independent simulation (the particles are massless), run in its
own process; outputs go to data/runs/<run-id>/chunks/cNN/ in the campaign layout.

    python campaign/scripts/s6_run_disc.py --arm grow --n-pairs 500 --t-end-yr 2e4 --run-id disc_smoke
"""

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict

import numpy as np

from trojans import constants, disc, integrate, paths
from trojans.outputs import provenance, write_json

A_JUPITER = 5.2026          # au, today's
PLANETS = ("jupiter",)


def _init_worker():
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


def run_chunk(task: dict) -> dict:
    growth = disc.Growth(**task["growth"])
    settings = integrate.Settings(**task["settings"])
    ids = np.asarray(task["ids"], dtype=np.int64)
    out = paths.make_output_dir(paths.chunk_dir(task["run_id"], task["chunk"]), resume=task["resume"])
    m0 = growth.mass(0.0)
    giants = {"jupiter": {"a": A_JUPITER, "hill_radius": disc.hill_radius(A_JUPITER, m0)}}
    forcing = disc.GrowthForcing(growth, A_JUPITER)
    if task["resume"] and (out / "latest.json").exists():
        runner = integrate.ChunkRunner.resume(ids, settings, giants, out, PLANETS, forcing)
    else:
        pos, vel = np.asarray(task["pos"]), np.asarray(task["vel"])
        sim = disc.build_simulation(pos, vel, ids, growth, A_JUPITER, settings.dt_yr)
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
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--arm", choices=["fixed", "grow"], required=True)
    parser.add_argument("--n-pairs", type=int, required=True, help="disc particles = 2 x this")
    parser.add_argument("--t-end-yr", type=float, required=True)
    parser.add_argument("--t-grow-yr", type=float, default=1.0e5)
    parser.add_argument("--m0-earth", type=float, default=10.0, help="core mass at t = 0 (grow arm)")
    parser.add_argument("--a-min", type=float, default=disc.Disc.a_min)
    parser.add_argument("--a-max", type=float, default=disc.Disc.a_max)
    parser.add_argument("--e-max", type=float, default=disc.Disc.e_max)
    parser.add_argument("--inc-max-deg", type=float, default=disc.Disc.inc_max_deg)
    parser.add_argument("--seed", type=int, default=disc.Disc.seed)
    parser.add_argument("--n-chunks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    d = disc.Disc(n_pairs=args.n_pairs, a_min=args.a_min, a_max=args.a_max, e_max=args.e_max,
                  inc_max_deg=args.inc_max_deg, seed=args.seed)
    if args.arm == "fixed":
        growth = disc.Growth(m0_msun=constants.GIANT_MASS["jupiter"], t_grow_yr=0.0)
    else:
        growth = disc.Growth(m0_msun=args.m0_earth * disc.M_EARTH, t_grow_yr=args.t_grow_yr)
    settings = integrate.Settings(t_end_yr=args.t_end_yr, dt_yr=constants.timestep(A_JUPITER))

    pos, vel = disc.disc_states(d)
    ids = disc.particle_ids(d.n_pairs)
    root = paths.make_output_dir(paths.run_dir(args.run_id), resume=args.resume)
    write_json(root / "settings.json", {
        "arm": args.arm, "disc": asdict(d), "growth": asdict(growth), "settings": asdict(settings),
        "a_jupiter_au": A_JUPITER, "planets": PLANETS, "n_chunks": args.n_chunks, **provenance(),
    })
    np.savez(root / "disc_initial.npz", id=ids, pos=pos, vel=vel)

    # Twins stay in the same chunk.
    pair_chunks = np.array_split(np.arange(d.n_pairs), args.n_chunks)
    tasks = []
    for k, pairs in enumerate(pair_chunks):
        rows = np.sort(np.concatenate([2 * pairs, 2 * pairs + 1]))
        tasks.append({"run_id": args.run_id, "chunk": k, "resume": args.resume,
                      "ids": ids[rows].tolist(), "pos": pos[rows].tolist(), "vel": vel[rows].tolist(),
                      "growth": asdict(growth), "settings": asdict(settings)})

    print(f"{args.arm}: {2 * d.n_pairs} particles in {args.n_chunks} chunks on {args.workers} workers, "
          f"{args.t_end_yr:.3g} yr, dt = {settings.dt_yr:.4f} yr, {settings.n_windows} windows")
    with mp.get_context("spawn").Pool(args.workers, initializer=_init_worker) as pool:
        sessions = pool.map(run_chunk, tasks)
    cpu = sum(s["cpu_s"] for s in sessions)
    n = sum(s["n"] for s in sessions)
    print(f"done: {sum(s['n_alive'] for s in sessions)}/{n} alive, {cpu / 3600:.2f} CPU-h, "
          f"{cpu / (n * args.t_end_yr / 1e6):.2f} CPU-s per particle-Myr")


if __name__ == "__main__":
    main()
