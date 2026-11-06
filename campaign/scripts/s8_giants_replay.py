"""Replay the giants of a stage-8 run alone and store them at the snapshot times.

The production snapshots keep only Jupiter. The disc particles are massless
(N_active = 1 + n_giants), so the giants' motion does not depend on them: the same
build (disc.build_system with no particles), WHFast settings, check cadence and
SystemForcing reproduce their orbits. Jupiter's stored a and lambda are used to
check the replay; the maximum differences are printed and saved.

    python campaign/scripts/s8_giants_replay.py configs/mig_c2_nu09_cold.toml
Writes campaign/reports/<run-id>/giants.npz: t, and per giant x, y, a, e (heliocentric).
"""

import argparse
import sys
import tomllib
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from s8_run_migration import build  # noqa: E402

from trojans import constants, disc, paths  # noqa: E402
from trojans.history import SystemForcing  # noqa: E402


class _Runner:
    """The two attributes SystemForcing writes."""
    m_jupiter = 0.0
    hill = None


def replay(config_path: Path):
    config = tomllib.loads(config_path.read_text())
    run_id = config["run"]["run_id"]
    overrides = argparse.Namespace(n_pairs=None, t_classify_yr=None)
    system, _, settings = build(config, overrides)
    chunk = sorted((paths.run_dir(run_id) / "chunks").iterdir())[0]
    with h5py.File(chunk / "snapshots.h5") as f:
        t_snap, ja, jlam = f["t"][:], f["jupiter_a"][:], f["jupiter_lam"][:]

    a0 = [h.a(0.0) for h in system.histories]
    m0 = [h.mass(0.0) for h in system.histories]
    sim = disc.build_system(np.empty((0, 3)), np.empty((0, 3)), [], system.planets, a0, m0, settings.dt_yr)
    sim.ri_whfast.safe_mode = settings.safe_mode
    sim.ri_whfast.corrector = settings.corrector
    if settings.safe_mode == 0:
        sim.ri_whfast.keep_unsynchronized = 1
    forcing, runner = SystemForcing(system), _Runner()
    forcing(sim, runner)

    n_g = len(system.planets)
    out = {k: np.full((len(t_snap), n_g), np.nan) for k in ("x", "y", "a", "e", "lam")}
    targets = np.rint(t_snap / settings.check_interval_yr).astype(np.int64)
    check, k = 0, 0

    def record(k):
        sun = sim.particles[0]
        for g in range(n_g):
            p = sim.particles[g + 1]
            o = p.orbit(primary=sun)
            out["x"][k, g], out["y"][k, g] = p.x - sun.x, p.y - sun.y
            out["a"][k, g], out["e"][k, g], out["lam"][k, g] = o.a, o.e, np.degrees(o.l) % 360

    while k < len(targets) and targets[k] == 0:
        record(k); k += 1
    while k < len(targets):
        sim.steps(settings.check_every_steps)
        if settings.safe_mode == 0:
            sim.synchronize()
        forcing(sim, runner)
        check += 1
        while k < len(targets) and targets[k] == check:
            record(k); k += 1

    dlam = np.abs((out["lam"][:, 0] - jlam % 360 + 180) % 360 - 180)
    da = np.abs(out["a"][:, 0] - ja)
    print(f"{run_id}: replayed {n_g} giants to {sim.t / 1e6:.3f} Myr; Jupiter vs snapshots: "
          f"max |da| = {da.max():.2e} au, max |dlambda| = {dlam.max():.2e} deg")
    path = paths.report_dir(run_id) / "giants.npz"
    np.savez(path, t=t_snap, planets=np.array(system.planets), check_da=da.max(), check_dlam=dlam.max(), **out)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("config", type=Path)
    replay(ap.parse_args().config)


if __name__ == "__main__":
    main()
