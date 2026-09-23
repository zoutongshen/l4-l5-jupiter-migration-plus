"""Stage 4: the Trojan mass ladder. One massive Trojan per simulation, IAS15.

Simon asked (2026-09-22) whether there is a mass below which a Trojan stays in
L4/L5 forever. There is: Gascheau (1843) / Routh (1875) make the equilateral
configuration linearly unstable above m_T = 0.0391 Msun (40.9 M_Jup for this
model), and Liberato & Winter (2020) put the practical limit at mu = 0.023313
(24.0 M_Jup). This measures where the knee actually falls, with the same
catalogue and the same initial conditions as the campaign.

Each rung integrates the SAME objects, changing only the mass each one carries.
One Trojan per simulation is the point: 100 Trojans at 1 M_Jup in one box is
100 M_Jup in the co-orbital region, which is a different experiment.

Why not the campaign's chunk pipeline: WHFast cannot do this. At 24 M_Jup it
loses |dE/E| ~ 9e-2 and at 10 M_Jup it disagrees with IAS15 about whether the
Trojan survives at all. IAS15 costs the same at six bodies (~22 s per object per
Myr) and holds |dE/E| ~ 1e-14, so the ladder runs on IAS15 and reuses only the
campaign's pure classification helpers.

    python campaign/scripts/s4_run_ladder.py --workers 8
"""

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd

from trojans import constants, diagnostics, initial, integrate, paths, runs
from trojans.outputs import write_json, write_parquet, write_provenance

# Rungs in Jupiter masses. 0 is the massless control that bridges to the pilot;
# 1e-9 is about (624) Hektor, the largest real Trojan; 24 is Liberato & Winter's
# practical co-orbital limit and 40.9 is Routh's linear one. The 50 rung sits
# above Routh and MUST come apart -- it is the check that the setup reproduces
# known physics, and nothing below it is worth believing until it does.
RUNGS_MJUP = [0.0, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 24.0, 30.0, 40.9, 50.0]

IC_ID = "2026-09-15_giants"
SEED = 20260915                  # the campaign seed, so the objects match the pilot's shuffle
REMOVE_HILL_RADII = 1.0          # Di Sisto+ 2014, as integrate.Settings
R_MAX_AU = 100.0

FATE_LIBRATING, FATE_LEFT, FATE_REMOVED = "librating", "left", "removed"

_STATE = {}


def _init_worker():
    states, ic = runs.load_ic(IC_ID)
    _STATE["states"] = states
    _STATE["ic"] = ic
    _STATE["hill"] = np.array([ic["giants"][p]["hill_radius"] for p in constants.GIANTS])


def run_one(task: tuple) -> dict:
    """Integrate one Trojan at one mass and report what became of it."""
    spkid, mjup, t_end_yr, sample_yr = task
    states, ic, hill = _STATE["states"], _STATE["ic"], _STATE["hill"]
    m_msun = mjup * constants.GIANT_MASS["jupiter"]
    mu_c = constants.G * constants.CENTRAL_MASS
    mu_j = constants.G * (constants.CENTRAL_MASS + constants.GIANT_MASS["jupiter"])

    sim = initial.build_simulation(states, [spkid], ic["dt_yr"], trojan_mass=m_msun)
    sim.integrator = "ias15"
    e0 = sim.energy()
    cpu0 = time.process_time()

    xyz = np.empty((sim.N, 3))
    vxyz = np.empty((sim.N, 3))

    sigma_prev = np.array([np.nan])
    t_leave = t_cross0 = t_removed = np.nan
    reason = ""
    s_min, s_max, sigma_0 = np.inf, -np.inf, np.nan
    n_samples, sum_sin, sum_cos = 0, 0.0, 0.0
    last = {}

    for t in np.arange(sample_yr, t_end_yr + 0.5 * sample_yr, sample_yr):
        sim.integrate(float(t))
        sim.serialize_particle_data(xyz=xyz, vxvyvz=vxyz)
        pos, vel = xyz[5] - xyz[0], vxyz[5] - vxyz[0]

        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
            reason, t_removed = "non_finite", float(t)
            break
        planets = xyz[1:5] - xyz[0]
        if (np.linalg.norm(pos - planets, axis=1) / hill).min() < REMOVE_HILL_RADII:
            reason, t_removed = "hill", float(t)
            break
        if np.linalg.norm(pos) > R_MAX_AU:
            reason, t_removed = "r_max", float(t)
            break

        el = diagnostics.elements(pos[None], vel[None], mu_c)
        jup = diagnostics.elements((xyz[1] - xyz[0])[None], (vxyz[1] - vxyz[0])[None], mu_j)
        sigma = diagnostics.resonant_angle(el["lam"], float(jup["lam"][0]))

        crossed180, crossed0 = integrate.sigma_crossings(sigma_prev, sigma)
        if crossed180[0] and np.isnan(t_leave):
            t_leave = float(t)
        if crossed0[0] and np.isnan(t_cross0):
            t_cross0 = float(t)
        sigma_prev = sigma
        if np.isnan(sigma_0):
            sigma_0 = float(sigma[0])
        s_min, s_max = min(s_min, float(sigma[0])), max(s_max, float(sigma[0]))
        rad = np.radians(float(sigma[0]))
        n_samples, sum_sin, sum_cos = n_samples + 1, sum_sin + np.sin(rad), sum_cos + np.cos(rad)
        last = {"a": float(el["a"][0]), "e": float(el["e"][0]), "inc": float(el["inc"][0])}

    # Classify exactly as the campaign does (integrate.classify_windows), so a
    # Trojan that turns into a horseshoe is not counted as still being in L4/L5.
    cmean = np.degrees(np.arctan2(sum_sin, sum_cos))
    cls = int(integrate.classify_windows(np.array([n_samples]), np.array([not np.isnan(t_cross0)]),
                                         np.array([not np.isnan(t_leave)]), np.array([cmean]))[0])
    name = integrate.CLASS_NAMES[cls]
    if reason:
        fate = FATE_REMOVED
    else:
        fate = FATE_LIBRATING if name in ("tadpole_L4", "tadpole_L5") else FATE_LEFT

    return {
        "spkid": int(spkid), "m_jup": float(mjup), "m_msun": float(m_msun),
        "mu": constants.coorbital_mu(m_msun),
        "swarm": "L4" if sigma_0 > 0 else "L5",
        "fate": fate, "libration_class": name, "removal_reason": reason,
        "t_leave_yr": t_leave, "t_cross0_yr": t_cross0, "t_removed_yr": t_removed,
        "sigma_min": s_min if np.isfinite(s_min) else np.nan,
        "sigma_max": s_max if np.isfinite(s_max) else np.nan,
        "sigma_cmean": float(cmean), "n_samples": n_samples,
        "a_final": last.get("a", np.nan), "e_final": last.get("e", np.nan),
        "inc_final": last.get("inc", np.nan),
        "dE_rel": float(abs((sim.energy() - e0) / e0)),
        "t_integrated_yr": float(sim.t), "cpu_s": time.process_time() - cpu0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="mass_ladder")
    parser.add_argument("--n-objects", type=int, default=100)
    parser.add_argument("--t-end-yr", type=float, default=1.0e6)
    parser.add_argument("--sample-yr", type=float, default=15.0,
                        help="libration period is ~148 yr; 15 yr resolves it with room to spare")
    parser.add_argument("--rungs", type=float, nargs="*", default=None,
                        help="rungs in Jupiter masses (default: the full ladder)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rungs = RUNGS_MJUP if args.rungs is None else list(args.rungs)
    out = paths.make_output_dir(paths.run_dir(args.run_id), resume=args.resume)
    table = out / "ladder.parquet"

    states, _ = runs.load_ic(IC_ID)
    spkids = runs.assign_chunks(states.loc[states["body"] == "trojan", "spkid"],
                               n_chunks=1, seed=SEED, subset=args.n_objects)["spkid"].to_numpy()

    tasks = [(int(s), float(m), args.t_end_yr, args.sample_yr) for m in rungs for s in spkids]
    done = pd.read_parquet(table) if args.resume and table.exists() else None
    if done is not None:
        seen = set(zip(done["spkid"], done["m_jup"]))
        tasks = [t for t in tasks if (t[0], t[1]) not in seen]
        print(f"resuming: {len(seen)} done, {len(tasks)} to go")

    write_provenance(out)
    write_json(out / "settings.json", {
        "ic_id": IC_ID, "seed": SEED, "n_objects": args.n_objects, "rungs_mjup": rungs,
        "t_end_yr": args.t_end_yr, "sample_yr": args.sample_yr, "integrator": "ias15",
        "remove_hill_radii": REMOVE_HILL_RADII, "r_max_au": R_MAX_AU,
        "routh_limit_mjup": constants.routh_trojan_mass_limit() / constants.GIANT_MASS["jupiter"],
        "liberato_winter_mu": constants.LIBERATO_WINTER_MU,
    })

    if not tasks:
        print("nothing to do")
        return

    rows = [] if done is None else done.to_dict("records")
    wall0 = time.time()
    with mp.Pool(args.workers, initializer=_init_worker, maxtasksperchild=50) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1), 1):
            rows.append(row)
            if i % 25 == 0 or i == len(tasks):
                write_parquet(table, pd.DataFrame(rows))
                rate = i / (time.time() - wall0)
                print(f"  {i}/{len(tasks)}  {rate*60:.1f}/min  "
                      f"eta {(len(tasks)-i)/rate/60:.1f} min", flush=True)

    frame = pd.DataFrame(rows)
    write_parquet(table, frame)
    cpu_h = frame["cpu_s"].sum() / 3600.0
    print(f"\n{len(frame)} runs, {cpu_h:.2f} CPU-h, {(time.time()-wall0)/60:.1f} min wall")
    print(f"written to {table}")
    print("\nsurviving fraction by rung:")
    for m, grp in frame.groupby("m_jup"):
        alive = (grp["fate"] == FATE_LIBRATING).mean()
        print(f"  {m:>8g} M_Jup  mu {grp['mu'].iloc[0]:.4e}  librating {alive:6.1%}  "
              f"max |dE/E| {grp['dE_rel'].max():.1e}")


if __name__ == "__main__":
    main()
