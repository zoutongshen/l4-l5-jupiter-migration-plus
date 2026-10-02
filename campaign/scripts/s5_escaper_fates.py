"""Stage 5a: follow each escaper of a run after it leaves, to its end state.

Escapers are objects whose first entry within 2.1 Hill radii of a giant was recorded
in the run (V24-selected, U <= 4). Two groups:
  trojan       V24 and JPL/MPC all call it a Trojan
  interloper   only V24's element box admits it (comet-like, e > 0.3; see
               docs/INITIAL_CONDITIONS_KS.md)

Phase 1, replay. The run's own setup (WHFast, same step, same check cadence) is rebuilt
for the escapers alone and integrated to each one's 2.1 R_H entry. Trojans are massless
test particles, so the replay reproduces the run's trajectory; the entry time is checked
against the run's (replay_dt_yr).

Phase 2, fate. At that moment the planets and the escaper are copied into a new
simulation with TRACE (close-encounter integrator; same step) and followed until
  ejected     r > EJECT_AU from the Sun on a hyperbolic barycentric orbit
  oort        r > EJECT_AU but still bound to the Solar System (barycentric e < 1)
  sun         collision with the Sun (folded with Mercury-Mars; physical solar radius)
  <planet>    collision with a giant (physical radius)
  survived    still bound at T_MAX_YR after escape
Every check (8 steps, ~1 yr) records encounters within 1 R_H of each giant, the closest
approach, the smallest perihelion, co-orbital episodes with Jupiter (|a - a_J| < 5% a_J
and 20 < |sigma| < 130 deg, held >= RECAPTURE_MIN_YR: a recapture into L4 or L5), and
time spent per orbital class:
  trojan_like  the co-orbital condition above
  jfc          2 < T_J < 3 and q < a_J
  centaur      a_J < q and a < a_N
  inner        T_J > 3 and a < a_J (asteroid-belt-like)
  outer        a > a_N
  other        anything else (e.g. T_J < 2 inside a_N)
Elements are heliocentric inside BARYCENTRIC_BEYOND_AU and barycentric (about the
Sun + giants) beyond it: far out, the Sun's reflex motion around the barycentre is
comparable to the object's speed and makes heliocentric elements meaningless (bound
orbits look hyperbolic). T_J uses ecliptic elements and Jupiter's current a.
The inner planets are folded into the Sun, so close approaches to the Sun inside
~1.5 au miss the terrestrial planets; min_q flags those.

Outputs
  data/runs/<out_id>/fates/<spkid>.json       one per escaper (resumable)
  data/runs/<out_id>/tracks/<spkid>.parquet   a, e, i, q, T_J, class every SAMPLE_YR
  campaign/reports/<out_id>/fates.csv         one row per escaper

    python campaign/scripts/s5_escaper_fates.py --run-id pilot_10myr --workers 8
"""

import argparse
import json
import multiprocessing as mp
import time

import numpy as np
import pandas as pd
import rebound

from trojans import constants, diagnostics, initial, paths, runs

EJECT_AU = 1000.0
T_MAX_YR = 1.0e8
SAMPLE_YR = 1.0e3
RECAPTURE_MIN_YR = 1.0e3
CHECK_EVERY_STEPS = 8
BARYCENTRIC_BEYOND_AU = 30.0
R_SUN_AU = 0.00465047
RADIUS_AU = {"jupiter": 4.7789e-4, "saturn": 4.0287e-4, "uranus": 1.7085e-4, "neptune": 1.6554e-4}
CLASSES = ["trojan_like", "jfc", "centaur", "inner", "outer", "other"]


# --- phase 1: replay ------------------------------------------------------------

def replay(run_id: str, targets: pd.DataFrame, log=print) -> dict:
    """Integrate the run's setup for the escapers alone; return each one's hand-off state."""
    run_dir = paths.run_dir(run_id)
    config = runs.load_config_file(run_dir / "config.toml")
    states, ic = runs.load_ic(config["run"]["ic_id"])
    settings = json.loads((run_dir / "settings.json").read_text())
    spkids = targets["spkid"].to_numpy()
    sim = initial.build_simulation(states, spkids, settings["dt_yr"])
    sim.ri_whfast.safe_mode = settings["safe_mode"]
    sim.ri_whfast.corrector = settings["corrector"]
    hill = np.array([ic["giants"][p]["hill_radius"] for p in constants.GIANTS])
    n_planets = 1 + len(constants.GIANTS)
    record = settings["record_hill_radii"]
    pending = set(int(s) for s in spkids)
    handoff = {}
    t_last = float(targets["t_hill_record"].max())
    while pending and sim.t <= t_last + 10 * settings["dt_yr"] * settings["check_every_steps"]:
        sim.steps(settings["check_every_steps"])
        n = sim.N
        xyz, vxyz = np.empty((n, 3)), np.empty((n, 3))
        hashes = np.empty(n, dtype=np.uint32)
        sim.serialize_particle_data(xyz=xyz, vxvyvz=vxyz, hash=hashes)
        planets = xyz[1:n_planets] - xyz[0]
        pos = xyz[n_planets:] - xyz[0]
        dist = np.linalg.norm(pos[:, None, :] - planets[None], axis=2) / hill[None]
        hit = dist.min(axis=1) < record
        for k in np.flatnonzero(hit):
            h = int(hashes[n_planets + k])
            handoff[h] = {
                "t_yr": sim.t,
                "massive": [(xyz[i].tolist(), vxyz[i].tolist()) for i in range(n_planets)],
                "masses": [sim.particles[i].m for i in range(n_planets)],
                "particle": (xyz[n_planets + k].tolist(), vxyz[n_planets + k].tolist()),
            }
            pending.discard(h)
        for k in np.flatnonzero(hit)[::-1]:
            sim.remove(hash=int(hashes[n_planets + k]))
        if hit.any():
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
            log(f"replay t = {sim.t / 1e6:7.3f} Myr  handed off {len(handoff)}/{len(spkids)}")
    for h in pending:
        log(f"replay: {h} never reached {record} R_H (run recorded "
            f"{targets.set_index('spkid').at[h, 't_hill_record']:.4g} yr)")
    return handoff


# --- phase 2: fate ----------------------------------------------------------------

def _jupiter_lam(sim):
    sun, jup = sim.particles[0], sim.particles[1]
    o = jup.orbit(primary=sun)
    return o.l, o.a


def follow(args) -> dict:
    spkid, group, swarm, h, dt_yr, t_max_yr, out_dir = args
    sim = rebound.Simulation()
    sim.G = constants.G
    sim.integrator = "trace"
    sim.dt = dt_yr
    names = ["sun", *constants.GIANTS]
    radii = [R_SUN_AU] + [RADIUS_AU[p] for p in constants.GIANTS]
    for (p, v), m, r in zip(h["massive"], h["masses"], radii):
        sim.add(m=m, r=r, x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2])
    p, v = h["particle"]
    sim.add(m=0.0, r=0.0, x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2])
    sim.N_active = len(names)
    sim.testparticle_type = 0
    sim.collision = "direct"
    sim.collision_resolve = "halt"
    sim.t = h["t_yr"]

    a_n = sim.particles[4].orbit(primary=sim.particles[0]).a
    hill_a = np.array([sim.particles[i].orbit(primary=sim.particles[0]).a for i in range(1, 5)])
    hill = hill_a * (np.array([constants.GIANT_MASS[g] for g in constants.GIANTS]) / (3 * constants.CENTRAL_MASS)) ** (1 / 3)
    mu = constants.G * constants.CENTRAL_MASS

    t0 = sim.t
    in_hill = np.zeros(4, bool)
    n_enc = np.zeros(4, int)
    min_hill = np.full(4, np.inf)
    min_q = np.inf
    min_r = np.inf
    max_i = 0.0
    time_in = dict.fromkeys(CLASSES, 0.0)
    episodes = []
    co_start, co_swarm = None, None
    track = []
    next_sample = t0
    fate = "survived"
    wall0 = time.time()
    check_dt = CHECK_EVERY_STEPS * dt_yr

    def close_episode(t_end):
        if co_start is not None and t_end - co_start >= RECAPTURE_MIN_YR:
            episodes.append({"swarm": co_swarm, "t_start_after_escape_yr": co_start - t0,
                             "duration_yr": t_end - co_start})

    while sim.t - t0 < t_max_yr:
        try:
            sim.steps(CHECK_EVERY_STEPS)
        except rebound.Escape:
            o = sim.particles[5].orbit(primary=sim.com(last=5))
            fate = "ejected" if o.e >= 1 else "oort"
            break
        except rebound.Collision:
            ps = sim.particles
            d = [np.sqrt((ps[i].x - ps[5].x) ** 2 + (ps[i].y - ps[5].y) ** 2 + (ps[i].z - ps[5].z) ** 2) - radii[i]
                 for i in range(5)]
            fate = names[int(np.argmin(d))]
            break
        ps = sim.particles
        sun = ps[0]
        rel = np.array([[ps[i].x - ps[5].x, ps[i].y - ps[5].y, ps[i].z - ps[5].z] for i in range(1, 5)])
        dh = np.linalg.norm(rel, axis=1) / hill
        min_hill = np.minimum(min_hill, dh)
        now_in = dh < 1.0
        n_enc += now_in & ~in_hill
        in_hill = now_in

        r_sun = np.sqrt((ps[5].x - sun.x) ** 2 + (ps[5].y - sun.y) ** 2 + (ps[5].z - sun.z) ** 2)
        o = ps[5].orbit(primary=sun if r_sun < BARYCENTRIC_BEYOND_AU else sim.com(last=5))
        q = o.a * (1 - o.e)  # also right for hyperbolic orbits (a < 0, e > 1)
        min_r = min(min_r, r_sun)
        if r_sun < BARYCENTRIC_BEYOND_AU:
            min_q = min(min_q, q)
        if r_sun > EJECT_AU:  # checked here: exit_max_distance does not fire under sim.steps()
            fate = "ejected" if o.e >= 1 else "oort"
            break
        if r_sun < 5.0 and q < R_SUN_AU:
            fate = "sun"  # perihelion inside the Sun; the 1-yr check would step past the impact
            break
        inc = np.degrees(o.inc)
        max_i = max(max_i, inc)
        lam_j, a_j = _jupiter_lam(sim)
        sigma = np.degrees((o.l - lam_j + np.pi) % (2 * np.pi) - np.pi) if o.e < 1 else np.nan
        coorb = o.e < 1 and abs(o.a - a_j) < 0.05 * a_j and 20.0 < abs(sigma) < 130.0
        if o.e < 1 and o.a > 0:
            t_j = a_j / o.a + 2 * np.sqrt(max(o.a / a_j * (1 - o.e**2), 0.0)) * np.cos(o.inc)
        else:
            t_j = np.nan
        if coorb:
            cls = "trojan_like"
        elif np.isfinite(t_j) and 2 < t_j < 3 and q < a_j:
            cls = "jfc"
        elif o.e < 1 and q > a_j and o.a < a_n:
            cls = "centaur"
        elif np.isfinite(t_j) and t_j > 3 and o.a < a_j:
            cls = "inner"
        elif o.e < 1 and o.a > a_n:
            cls = "outer"
        else:
            cls = "other"
        time_in[cls] += check_dt

        sw = ("L4" if sigma > 0 else "L5") if coorb else None
        if co_start is not None and sw != co_swarm:
            close_episode(sim.t)
            co_start, co_swarm = None, None
        if sw is not None and co_start is None:
            co_start, co_swarm = sim.t, sw

        if sim.t >= next_sample:
            track.append({"t_after_escape_yr": sim.t - t0, "a": o.a, "e": o.e, "inc": inc, "q": q,
                          "t_j": t_j, "cls": cls, "min_hill_now": float(dh.min())})
            next_sample += SAMPLE_YR
    close_episode(sim.t)

    result = {
        "spkid": int(spkid), "group": group, "swarm": swarm, "fate": fate,
        "t_escape_yr": t0, "t_fate_after_escape_yr": sim.t - t0,
        **{f"n_enc_{g}": int(n) for g, n in zip(constants.GIANTS, n_enc)},
        **{f"min_rh_{g}": float(x) for g, x in zip(constants.GIANTS, min_hill)},
        "min_q_au": float(min_q), "min_r_sun_au": float(min_r), "max_inc_deg": float(max_i),
        **{f"t_{c}_yr": time_in[c] for c in CLASSES},
        "n_recaptures": len(episodes),
        "recapture_swarms": ";".join(e["swarm"] for e in episodes),
        "recapture_years": float(sum(e["duration_yr"] for e in episodes)),
        "episodes": episodes, "wall_s": time.time() - wall0,
    }
    pd.DataFrame(track).to_parquet(out_dir / "tracks" / f"{spkid}.parquet", index=False)
    (out_dir / "fates" / f"{spkid}.json").write_text(json.dumps(result, indent=1))
    return result


# --- driver ------------------------------------------------------------------------

def escapers(run_id: str) -> pd.DataFrame:
    ev = pd.read_parquet(paths.merged_dir(run_id) / "events.parquet")
    sel = ev["v24_selected"] & (pd.to_numeric(ev["condition_code"], errors="coerce") <= 4) & ev["t_hill_record"].notna()
    t = ev.loc[sel, ["spkid", "swarm", "e", "inc", "label_region", "t_hill_record", "planet_hill_record"]].copy()
    t["group"] = np.where(t["label_region"] == "V24", "interloper", "trojan")
    return t.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    parser.add_argument("--out-id", default="escaper_fates_pilot")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--only", type=int, nargs="*", help="spkids to follow (default: all escapers)")
    parser.add_argument("--t-max-yr", type=float, default=T_MAX_YR)
    args = parser.parse_args()

    targets = escapers(args.run_id)
    if args.only:
        targets = targets[targets["spkid"].isin(args.only)]
    out_dir = paths.run_dir(args.out_id)
    paths.make_output_dir(out_dir, resume=out_dir.exists())
    for sub in ("fates", "tracks"):
        paths.make_output_dir(out_dir / sub, resume=(out_dir / sub).exists())
    done = {int(p.stem) for p in (out_dir / "fates").glob("*.json")}
    todo = targets[~targets["spkid"].isin(done)]
    print(f"{len(targets)} escapers ({(targets['group'] == 'trojan').sum()} Trojans, "
          f"{(targets['group'] == 'interloper').sum()} interlopers); {len(todo)} to follow")

    if len(todo):
        handoff = replay(args.run_id, todo)
        settings = json.loads((paths.run_dir(args.run_id) / "settings.json").read_text())
        jobs = [(int(r.spkid), r.group, r.swarm, handoff[int(r.spkid)], settings["dt_yr"], args.t_max_yr, out_dir)
                for r in todo.itertuples() if int(r.spkid) in handoff]
        replay_check = {s: handoff[s]["t_yr"] for s in handoff}
        (out_dir / "replay_check.json").write_text(json.dumps(
            {int(r.spkid): {"run_yr": float(r.t_hill_record), "replay_yr": replay_check.get(int(r.spkid))}
             for r in todo.itertuples()}, indent=1))
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for res in pool.imap_unordered(follow, jobs):
                print(f"{res['spkid']:>9} {res['group']:<10} {res['swarm']} -> {res['fate']:<9} after "
                      f"{res['t_fate_after_escape_yr'] / 1e6:8.3f} Myr  recaptures {res['n_recaptures']}"
                      f"  wall {res['wall_s']:6.0f} s", flush=True)

    rows = [json.loads(p.read_text()) for p in sorted((out_dir / "fates").glob("*.json"))]
    table = pd.DataFrame(rows).drop(columns=["episodes"])
    checks = json.loads((out_dir / "replay_check.json").read_text()) if (out_dir / "replay_check.json").exists() else {}
    table["replay_dt_yr"] = [(checks[str(s)]["replay_yr"] or np.nan) - checks[str(s)]["run_yr"]
                             if str(s) in checks else np.nan for s in table["spkid"]]
    report = paths.report_dir(args.out_id)
    paths.make_output_dir(report, resume=report.exists())
    table.to_csv(report / "fates.csv", index=False, float_format="%.6g")
    print(table.groupby(["group", "swarm", "fate"]).size().to_string())


if __name__ == "__main__":
    main()
