"""Stage 9: Pirani+2019a simulations, one scenario and repeat per call.

Her set-up (docs/PIRANI_2019A_REPRODUCTION.md): MERCURIUS at dt = 50 d; the 44-ring disc
(4-26 au x 1000, e <= 0.01, i <= 0.01 deg, no twins); the giants follow her growth tracks
(configs/tracks/pirani19a/track_*.csv) through a migration force, with free e and i damped
below 30 M_E (trojans/pirani_forces.py); Adachi gas drag on the particles; gas gone at 3 Myr;
t = 0 -> 5 Myr. Each repeat k re-randomises the disc (seed = base seed + k), as her ten runs.

Config sections (configs/pir_*.toml):
  [run]        run_id, t_end_yr
  [giants]     jupiter = "pirani19a/track_....csv", saturn = ..., ... (simulation order = this order)
  [disc]       a_min_au, a_max_au, ring_au, n_per_ring, e_max, inc_max_deg, seed
  [integrator] name ("mercurius"), dt_days; other integrate.Settings fields
  [physics]    drag (bool), drag_every_yr, planet_density_g_cm3, sun_radius_au
  [snapshots]  fine_every_yr, fine_until_yr, every_yr
  [parallel]   n_chunks, workers

    python campaign/scripts/s9_run_pirani.py campaign/configs/pir_nominal.toml --repeat 0
    python campaign/scripts/s9_run_pirani.py <config> --planets-only          # giants alone, no disc (test 1)
    python campaign/scripts/s9_run_pirani.py <config> --n-per-ring 100 --run-id pir_smoke   # 1/10 disc
    python campaign/scripts/s9_run_pirani.py <config> --repeat 0 --resume
"""

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from trojans import constants, diagnostics, disc, gasdisc, integrate, paths, runs
from trojans.history import JupiterHistory, SystemHistory
from trojans.outputs import provenance, write_json
from trojans.pirani_forces import DragParams, PiraniForcing

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def system_from_config(config: dict) -> SystemHistory:
    giants = config["giants"]
    names = list(giants)
    if names[0] != "jupiter":
        raise ValueError("[giants] must list jupiter first")
    hs = []
    for n in names:
        t, a, m = np.loadtxt(CONFIGS / "tracks" / giants[n], delimiter=",", skiprows=1, unpack=True)
        hs.append(JupiterHistory.from_table(t, a, m * disc.M_EARTH))
    return SystemHistory(planets=tuple(names), histories=tuple(hs))


def build(config: dict, args):
    run = config["run"]
    system = system_from_config(config)
    d = dict(config["disc"])
    if args.n_per_ring:
        d["n_per_ring"] = args.n_per_ring
    ring = disc.RingDisc(a_min=float(d.get("a_min_au", 4.0)), a_max=float(d.get("a_max_au", 26.0)),
                         ring_au=float(d.get("ring_au", 0.5)), n_per_ring=int(d.get("n_per_ring", 1000)),
                         e_max=float(d.get("e_max", 0.01)), inc_max_deg=float(d.get("inc_max_deg", 0.01)),
                         seed=int(d.get("seed", 0)) + args.repeat)
    integ = dict(config.get("integrator", {}))
    name = integ.pop("name", "mercurius")
    dt = float(integ.pop("dt_days", 50.0)) / constants.JULIAN_YEAR_D
    phys = config.get("physics", {})
    snaps = config.get("snapshots", {})
    t_end = args.t_end_yr or float(run["t_end_yr"])
    settings = integrate.Settings(
        t_end_yr=t_end, dt_yr=dt, integrator=name, remove_hill_radii=0.0,
        sun_radius_au=float(phys.get("sun_radius_au", 0.005)),
        snapshot_every_yr=float(snaps.get("every_yr", 5e4)),
        snapshot_fine_every_yr=float(snaps.get("fine_every_yr", 1e4)),
        snapshot_fine_until_yr=float(snaps.get("fine_until_yr", 3.5e6)), **integ)
    drag = DragParams(enabled=bool(phys.get("drag", True)), every_yr=float(phys.get("drag_every_yr", 100.0)))
    density = float(phys.get("planet_density_g_cm3", 1.33))
    return system, ring, settings, drag, density


def _forcing(system, drag, density, integrator):
    return PiraniForcing(system, drag, planet_density_g_cm3=density, integrator=integrator)


def run_chunk(task: dict) -> dict:
    system = SystemHistory(**task["system"])
    settings = integrate.Settings(**task["settings"])
    drag = DragParams(**task["drag"])
    ids = np.asarray(task["ids"], dtype=np.int64)
    directory = paths.chunk_dir(task["run_id"], task["chunk"])
    out = paths.make_output_dir(directory, resume=task["resume"] and directory.exists())
    planets = system.planets
    a0 = [h.a(0.0) for h in system.histories]
    m0 = [h.mass(0.0) for h in system.histories]
    giants = {p: {"a": a, "hill_radius": disc.hill_radius(a, m)} for p, a, m in zip(planets, a0, m0)}
    forcing = _forcing(system, drag, task["density"], settings.integrator)
    if task["resume"] and (out / "latest.json").exists():
        runner = integrate.ChunkRunner.resume(ids, settings, giants, out, planets, forcing)
    else:
        pos, vel = np.asarray(task["pos"]), np.asarray(task["vel"])
        sim = disc.build_system(pos, vel, ids, planets, a0, m0, settings.dt_yr, integrator=settings.integrator)
        runner = integrate.ChunkRunner.start(sim, ids, settings, giants, out, planets, forcing)
    cpu0, wall0 = time.process_time(), time.time()
    tag = f"[{task['run_id']} c{task['chunk']:03d}]"
    runner.run(log=lambda message: print(f"{tag} {message}", flush=True))
    session = {"chunk": task["chunk"], "n": len(ids), "cpu_s": time.process_time() - cpu0,
               "wall_s": time.time() - wall0,
               "n_alive": int((runner.events["removal_reason"] == integrate.ALIVE).sum())}
    with open(out / "sessions.jsonl", "a") as f:
        f.write(json.dumps(session) + "\n")
    return session


def planets_only(system, settings, drag, density, out: Path, every_yr=1000.0):
    """Test 1: the giants alone (no disc), 0 -> t_end; a, e, i, m of each and the J-S resonant angles."""
    planets = system.planets
    a0 = [h.a(0.0) for h in system.histories]
    m0 = [h.mass(0.0) for h in system.histories]
    sim = disc.build_system(np.zeros((0, 3)), np.zeros((0, 3)), [], planets, a0, m0, settings.dt_yr,
                            integrator=settings.integrator)
    forcing = _forcing(system, drag, density, settings.integrator)

    class _Runner:                                   # what PiraniForcing needs from a runner
        s = settings
    runner = _Runner()
    forcing(sim, runner)
    rows = []
    started = time.time()
    n_steps = max(1, int(round(every_yr / settings.dt_yr / settings.check_every_steps)))
    while sim.t < settings.t_end_yr:
        for _ in range(n_steps):
            sim.steps(settings.check_every_steps)
            forcing(sim, runner)
        sun = sim.particles[0]
        row = {"t": sim.t}
        lam = {}
        for k, p in enumerate(planets):
            o = sim.particles[k + 1].orbit(primary=sun)
            row.update({f"a_{p}": o.a, f"e_{p}": o.e, f"inc_{p}": np.degrees(o.inc),
                        f"m_{p}": sim.particles[k + 1].m / disc.M_EARTH, f"track_a_{p}": system.histories[k].a(sim.t)})
            lam[p] = (o.l, o.pomega)
        if "saturn" in lam:
            lj, pj = lam["jupiter"]
            ls, ps = lam["saturn"]
            row["phi_js_1"] = np.degrees(np.mod(2 * ls - lj - pj + np.pi, 2 * np.pi) - np.pi)   # 1:2 (J:S periods)
            row["phi_js_2"] = np.degrees(np.mod(2 * ls - lj - ps + np.pi, 2 * np.pi) - np.pi)
            row["period_ratio_sj"] = (row["a_saturn"] / row["a_jupiter"]) ** 1.5
        rows.append(row)
    import pandas as pd
    df = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "planets_only.csv", index=False)
    print(f"planets only: {settings.t_end_yr / 1e6:.2f} Myr in {time.time() - started:.0f} s; written {out / 'planets_only.csv'}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--repeat", type=int, default=0, help="her ten re-randomised realisations: 0..9")
    parser.add_argument("--run-id", help="override the run id (default: <run_id>_r<repeat>)")
    parser.add_argument("--n-per-ring", type=int, help="override [disc] n_pairs per ring (smoke tests)")
    parser.add_argument("--t-end-yr", type=float, help="override [run] t_end_yr")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--n-chunks", type=int)
    parser.add_argument("--planets-only", action="store_true", help="giants alone, no disc (test 1)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = runs.load_config_file(args.config)
    system, ring, settings, drag, density = build(config, args)
    run_id = args.run_id or f"{config['run']['run_id']}_r{args.repeat}"
    par = config.get("parallel", {})
    n_chunks = args.n_chunks or int(par.get("n_chunks", 24))
    workers = args.workers or int(par.get("workers", 24))
    print(f"{run_id}: {', '.join(f'{p} {h.a(0.0):.2f}->{h.a(3e6):.2f} au, ->{h.mass(3e6) / disc.M_EARTH:.0f} M_E' for p, h in zip(system.planets, system.histories))}; "
          f"{ring.n} particles ({ring.n_rings} rings x {ring.n_per_ring}, {ring.a_min}-{ring.a_max} au, seed {ring.seed}); "
          f"{settings.integrator} dt {settings.dt_yr * 365.25:.1f} d, t_end {settings.t_end_yr / 1e6:.2f} Myr, "
          f"alpha {gasdisc.ALPHA}, drag {'on' if drag.enabled else 'off'}; {n_chunks} chunks on {workers} workers")
    if args.dry_run:
        return
    if args.planets_only:
        planets_only(system, settings, drag, density, REPORTS / f"{config['run']['run_id']}_planets")
        return

    record = {"run_id": run_id, "config_file": str(args.config), "config": config, "repeat": args.repeat,
              "system": asdict(system), "disc": asdict(ring), "settings": asdict(settings), "drag": asdict(drag),
              "planet_density_g_cm3": density, "alpha": gasdisc.ALPHA, "n_chunks": n_chunks}
    root = paths.run_dir(run_id)
    if args.resume:
        previous = json.loads((root / "settings.json").read_text())
        for key in ("system", "disc", "settings", "drag"):
            if json.dumps(previous.get(key), sort_keys=True) != json.dumps(record[key], sort_keys=True):
                raise SystemExit(f"--resume: {key} differs from {root / 'settings.json'}")
    else:
        paths.make_output_dir(root, resume=False)
        write_json(root / "settings.json", {**record, **provenance()})

    pos, vel, a0 = disc.ring_disc_states(ring)
    ids = disc.FIRST_PARTICLE_HASH + np.arange(ring.n, dtype=np.int64)
    if not args.resume:
        np.savez(root / "disc_initial.npz", id=ids, pos=pos, vel=vel, a0=a0)
    # interleave so every chunk samples every ring (equal cost per chunk)
    tasks = []
    for k in range(n_chunks):
        rows = np.arange(k, ring.n, n_chunks)
        tasks.append({"run_id": run_id, "chunk": k, "resume": args.resume, "ids": ids[rows].tolist(),
                      "pos": pos[rows].tolist(), "vel": vel[rows].tolist(), "system": asdict(system),
                      "settings": asdict(settings), "drag": asdict(drag), "density": density})
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        sessions = pool.map(run_chunk, tasks)
    cpu = sum(s["cpu_s"] for s in sessions)
    n = sum(s["n"] for s in sessions)
    print(f"done: {sum(s['n_alive'] for s in sessions)}/{n} alive, {cpu / 3600:.2f} CPU-h, "
          f"{cpu / (n * settings.t_end_yr / 1e6):.2f} CPU-s per particle-Myr")


def _init_worker():
    import os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"


if __name__ == "__main__":
    main()
