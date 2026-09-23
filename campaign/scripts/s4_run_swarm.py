"""Stage 4: does the Trojans' own mutual gravity matter? The self-gravitating swarm.

Every long-term Trojan study, this project's campaign included, treats Trojans as
massless test particles and neglects their mutual gravity. The mass ladder
(s4_run_ladder.py) shows a Trojan's own mass is irrelevant to its own dynamics by
ten orders of magnitude. That is a statement about ONE body, and leaves the
collective question open: 16,445 bodies pulling on each other is not the same as
one body pulling on Jupiter.

This runs the objects that carry the swarm's mass -- the largest per swarm, since
mass goes as roughly 10^(-0.6 H) and the brightest few hundred dominate -- with
their real H-derived masses and full mutual gravity, against the identical objects
massless. Any difference between the two is the size of the assumption everyone
makes. The expected answer is "none measurable"; nobody has published the check.

WHFast is correct here, unlike on the mass ladder: these masses are ~1e-12 Msun,
so there is no mass-ratio problem, and staying on WHFast keeps the comparison with
the campaign exact.

    python campaign/scripts/s4_run_swarm.py --n-per-swarm 100
"""

import argparse
import json
import time

import numpy as np
import pandas as pd

from trojans import catalogue as catalogue_mod
from trojans import constants, initial, integrate, paths, runs
from trojans.outputs import provenance, write_json

IC_ID = "2026-09-15_giants"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="swarm")
    parser.add_argument("--n-per-swarm", type=int, default=100)
    parser.add_argument("--t-end-yr", type=float, default=1.0e6)
    parser.add_argument("--massless", action="store_true",
                        help="the control: identical objects, no masses, no mutual gravity")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    states, ic = runs.load_ic(IC_ID)
    master = pd.read_parquet(paths.catalogue_dir(ic["catalogue_id"]) / "master_table.parquet")
    picked = catalogue_mod.largest_per_swarm(master, args.n_per_swarm)
    spkids = picked["spkid"].to_numpy(dtype="int64")
    masses = np.array([constants.mass_from_H(h) for h in picked["H"]])

    run_id = f"{args.run_id}_massless" if args.massless else f"{args.run_id}_selfgrav"
    out = paths.make_output_dir(paths.run_dir(run_id), resume=args.resume)
    settings = integrate.Settings(t_end_yr=args.t_end_yr, dt_yr=ic["dt_yr"])

    total = masses.sum()
    print(f"{len(spkids)} Trojans, H {picked['H'].min():.2f}-{picked['H'].max():.2f}, "
          f"{picked['swarm'].value_counts().to_dict()}")
    print(f"total mass {total:.3e} Msun = {total * constants.MSUN_KG / 5.972e24:.3e} M_Earth "
          f"({'IGNORED - massless control' if args.massless else 'self-gravitating'})")

    if args.resume:
        runner = integrate.ChunkRunner.resume(spkids, settings, ic["giants"], out)
    else:
        sim = initial.build_simulation(states, spkids, settings.dt_yr,
                                       trojan_mass=0.0 if args.massless else masses,
                                       self_gravity=not args.massless)
        print(f"N = {sim.N}, N_active = {sim.N_active}, testparticle_type = {sim.testparticle_type}")
        runner = integrate.ChunkRunner.start(sim, spkids, settings, ic["giants"], out)

    picked.to_parquet(out / "selection.parquet", index=False)
    write_json(out / "settings.json", {
        **{k: v for k, v in vars(args).items()},
        "ic_id": IC_ID, "catalogue_id": ic["catalogue_id"], "n": len(spkids),
        "total_mass_msun": float(total), "albedo": constants.TROJAN_ALBEDO,
        "density_kg_m3": constants.TROJAN_DENSITY_KG_M3, "integrator": "whfast",
    })
    session = {**provenance(), "resume": args.resume, "n_particles": len(spkids)}
    cpu0, wall0 = time.process_time(), time.time()
    runner.run(log=lambda message: print(f"[{run_id}] {message}", flush=True))
    session.update(cpu_s=time.process_time() - cpu0, wall_s=time.time() - wall0,
                   n_alive=int((runner.events["removal_reason"] == integrate.ALIVE).sum()))
    with open(out / "sessions.jsonl", "a") as f:
        f.write(json.dumps(session) + "\n")
    print(f"\n{session['n_alive']}/{len(spkids)} alive, {session['cpu_s']/3600:.2f} CPU-h")


if __name__ == "__main__":
    main()
